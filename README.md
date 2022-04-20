# VectorizedKernel
Running GPGPU kernels on CPU with auto-vectorization for SSE/AVX/AVX512 SIMD microarchitectures.

How does it work?

- User writes scalar-looking code (see below sample)
- createKernel factory function is given the user lambda function ```[](auto factory, auto idThread, kernelArgs){}``` and blueprint of the kernel parameters ```Vectorization::KernelArgs<int*>{}``` 
- Kernel is launched for N times, computed by ```simd```-sized steps (can be bigger than actual SIMD width of CPU for extra pipelining)
- When N is not integer multiple of simd, the remaining tail is computed with simd=1 automatically
- User only takes care of the algorithm while each operation is done in parallel in 8,16,32,64,.. steps

What must be given to the lambda function as parameters?

- first parameter: auto factory (this is used for constructing scalar-looking variables that do compute in parallel)
- second parameter: auto idThread (this is a scalar-looking parallel variable that holds per-work-item id values that are zero-based, up to N given from run method)
- all others: actual kernel arguments to be used for GPGPU computations (their blueprint is given like this: ```Vectorization::KernelArgs<your_arg_type>{}``` or this: ```Vectorization::KernelArgs<some_arg,some_other_arg>{}``` or anything with more template arguments to declare actual kernel arguments after the second parameter of lambda function )

Basic samples are found in wiki: https://github.com/tugrul512bit/VectorizedKernel/wiki.

### How do variables' methods work?

- contiguous memory read/write methods require a pointer to target pointer and an index object: variable.writeToContiguous(ptr,index)
- scatter/gather operations (similar to the contiguous read/write) requires: variable.writeTo(ptr,index) only difference is each element goes different its own index
- computation methods require firstOperand.methodName(secondOperand,result)
- ternary requires conditionObject.ternary(trueChoice,falseChoice,result)
- logical methods require firstOperand.logicalAnd(secondOperand,result)

Hello-world:

```C++
#include "VectorizedKernel.h"
#include <iostream>
int main()
{
	constexpr int simd = 8; // >= SIMD width of CPU (bigger  = more pipelining)

	auto kernelHelloWorld = Vectorization::createKernel<simd>([&](auto & factory, auto & idThread, float * bufferIn, float * bufferOut){

		const int currentSimdWidth = factory.width;
		auto tmp = factory.template generate<float>();
		tmp.readFromContiguous(bufferIn,idThread); // contiguous (first work-item in simd group decides where to read)
		tmp.add(0.33f,tmp);
		tmp.writeToContiguous(bufferOut,idThread); // contiguous (first work-item in simd group decides where to write)

	}, /* defining kernel parameter types */ Vectorization::KernelArgs<float*,float*>{});

	// size does not have to be multiple of simd
	const int n = 23;

	// better performance with aligned buffers
	float vecIn[n], vecOut[n];
	for(int i=0;i<n;i++)
	{
		vecIn[i]=i;
	}
	kernelHelloWorld.run(n,vecIn,vecOut);
	for(int i=0;i<n;i++)
	{
		std::cout<<vecOut[i]<<" ";
	}
	std::cout<<std::endl;
	return 0;
}

```

Mandelbrot generation sample that has more than 10x speedup (compared to scalar version) for avx512 cpu:

- 3.6GHz Fx8150 single thread + 1333MHz DDR3 RAM (simd=32): 91 cycles per pixel (~30 ms per 1000x1000 image).

- Godbolt.org AVX512 server single thread (simd=32): 19 cycles per pixel (-std=c++2a -O3 -march=cascadelake -mavx512f -mavx512bw -mprefer-vector-width=512  -ftree-vectorize -fno-math-errno) (less than 10 ms per 1000x1000 image)

```C++
#include <iostream>
#include <fstream>
#include <sstream>
#include <complex>
#include <vector>

using namespace std;

constexpr int frames=20;
constexpr float width = 2000;
constexpr float height = 2000;
constexpr int grainSize=width/4; // pixels
void createImage();
int getPoint(int x, int y);

#include "VectorizedKernel.h"
int main()
{
	createImage();

	return 0;
}


#include <stdint.h>  // <cstdint> is preferred in C++, but stdint.h works.

#ifdef _MSC_VER
# include <intrin.h>
#else
# include <x86intrin.h>
#endif

// optional wrapper if you don't want to just use __rdtsc() everywhere
inline
uint64_t readTSC() {
    // _mm_lfence();  // optionally wait for earlier insns to retire before reading the clock
    uint64_t tsc = __rdtsc();
    // _mm_lfence();  // optionally block later instructions until rdtsc retires
    return tsc;
}
void createImage()
{

	std::vector<int> img(height*width);


	// compute single-thread "scalar" CPU
	/*

	 int getPoint(int a, int b)
	{
		float x = static_cast<float>(b);
		float y = static_cast<float>(a);
		x = (x - width / 2 - width/4)/(width/3);
		y = (height / 2 - y)/(width/3);


		complex<float> c (x,y);

		complex <float> z(0, 0);
		size_t iter = 0;
		while (abs(z) < 2 && iter <= 35)
		{
			z = z * z + c;
			iter++;
		}
		if (iter < 34) return iter*255/33;
		else return 0;
	}

	for (size_t i = 0; i < height; i++)
	{
		for (size_t j = 0; j < width; j++)
		{
			img[j+i*width]= getPoint(i, j);
		}
	}
	 */

	// single-thread vectorized
	constexpr int simd = 32;
	auto kernel = Vectorization::createKernel<simd>([&](auto & factory, auto & idThread, int * img){
		auto j = factory.template generate<int>();
		idThread.modulus(width, j);

		auto i = factory.template generate<int>();
		idThread.div(width, i);

		const int vecWidth = factory.width;

		auto x0 = factory.template generate<float>();
		j.template castAndCopyTo<float>(x0);
		auto y0 = factory.template generate<float>();
		i.template castAndCopyTo<float>(y0);

		const auto heightDiv2 = factory.template generate<float>(height/2.0f);

		auto x = factory.template generate<float>();
		x0.sub(width/2.0f, x0);

		x0.sub(width/4.0f, x0);
		x0.div(width/3.0f,x);

		auto y =  factory.template generate<float>();
		heightDiv2.sub(y0,y0);
		y0.div(width/3.0f, y);

		const auto imagc = factory.template generate<float>(y);
		const auto realc = factory.template generate<float>(x);

		auto imagz = factory.template generate<float>(0);
		auto realz = factory.template generate<float>(0);

		// loop
		bool anyTrue = true;
		auto iteration = factory.template generate<int>(0);
		const auto iterationLimit = factory.template generate<int>(35);

		// allocate all re-used resources
		auto imagzSquared = factory.template generate<float>();
		auto absLessThan2 = factory.template generate<int>();
		auto tmp1 = factory.template generate<float>();
		auto whileLoopCondition = factory.template generate<int>();
		auto tmp2 = factory.template generate<int>();
		auto zzReal = factory.template generate<float>();

		auto zzImag = factory.template generate<float>();
		auto tmp3 = factory.template generate<float>();
		auto tmpAdd1 = factory.template generate<float>();
		auto tmpAdd2 = factory.template generate<float>();
		auto tmp4 = factory.template generate<int>();
		while(anyTrue)
		{

			// computing while loop condition start
            imagz.mul(imagz, imagzSquared);
			realz.fusedMultiplyAdd(realz,imagzSquared,tmp1);
			tmp1.lessThan(4.0f, absLessThan2);

			iteration.lessThanOrEquals(35, tmp2);
			absLessThan2.logicalAnd(tmp2, whileLoopCondition);
			anyTrue = whileLoopCondition.isAnyTrue();
			// computing while loop condition end

			// do complex multiplication z = z*z + c
			realz.fusedMultiplySub(realz,imagzSquared, zzReal);
			imagz.mul(realz, tmp3);
			realz.fusedMultiplyAdd(imagz,tmp3, zzImag);

			// if a lane has completed work, do not modify it
			zzReal.add(realc, tmpAdd1);
			zzImag.add(imagc, tmpAdd2);
			whileLoopCondition.ternary(tmpAdd1, realz, realz);
			whileLoopCondition.ternary(tmpAdd2, imagz, imagz);

			// increment iteration
			whileLoopCondition.ternary(1,0, tmp4);
			iteration.add(tmp4, iteration);
		}

		const auto thirtyFour = factory.template generate<int>(34);
		auto ifLessThanThirtyFour = factory.template generate<int>();
		iteration.lessThan(thirtyFour, ifLessThanThirtyFour);

		auto conditionalValue1 = factory.template generate<int>();
		iteration.mul(255, conditionalValue1);
		conditionalValue1.div(33, conditionalValue1);

		auto conditionalValue2 = factory.template generate<int>(0);
		auto returnValue = factory.template generate<int>(0);
		ifLessThanThirtyFour.ternary(conditionalValue1, conditionalValue2, returnValue);
		auto tmp5 = factory.template generate<int>();
		i.mul(width, tmp5);
		auto writeAddr = factory.template generate<int>(0);
		j.add(tmp5, writeAddr);

		returnValue.writeTo(img,writeAddr);

	},Vectorization::KernelArgs<int*>{});

	for(int i=0;i<10;i++)
	{
		auto t1 = readTSC();
		kernel.run(width*height,img.data());
		auto t2 = readTSC();
		std::cout<<(t2-t1)/(width*height)<<" cycles per pixel"<<std::endl;
	}
	// create string
	stringstream sstr;
	sstr << "P3" << endl << width << " " << height << endl << 255 << endl;
	for (size_t i = 0; i < height; i++)
	{
		for (size_t j = 0; j < width; j++)
		{
			sstr << img[j+i*width] << " 0 0" << endl;
		}
	}

	// write to file at once
	ofstream fout;
	fout.open("mandelbrot.ppm");
	if (fout.is_open())
	{
		cout << "File is opened!" << endl;
		fout << sstr.str();
		fout.close();
	}
	else
	{
		cout << "Could not open the file!" << endl;
	}
}

```

