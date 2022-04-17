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
- second parameter: auto idThread (this is a scalar-looking variable that holds per-work-item id values that are zero-based, up to N given from run method)
- all others: actual kernel arguments to be used for GPGPU computations (their blueprint is given like this: ```Vectorization::KernelArgs<your_arg_type>{}``` or this: ```Vectorization::KernelArgs<some_arg,some_other_arg>{}``` or anything with more template arguments to declare actual kernel arguments after the second parameter of lambda function )

Mandelbrot generation sample:

- 2.1GHz Fx8150 single thread + 1333MHz DDR3 RAM (simd=4 best width): 230 cycles per pixel.

- Godbolt.org AVX512 server single thread (simd=64 best width): 55 cycles per pixel (-std=c++2a  -O3 -march=cascadelake -mprefer-vector-width=512 -ftree-vectorize -fno-math-errno )

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
	for (size_t i = 0; i < height; i++)
	{
		for (size_t j = 0; j < width; j++)
		{
			img[j+i*width]= getPoint(i, j);
		}
	}
	 */

	// single-thread vectorized
	constexpr int simd = 64; // 8 for bulldozer, 16 for avx2 cpu, 64 for avx512
	auto kernel = Vectorization::createKernel<simd>([&](auto & factory, auto & idThread, int * img){



		const auto j = idThread.modulus(width);
		const auto i = idThread.div(width);
		const int vecWidth = factory.width;



		auto x0 = j.template cast<float>();
		auto y0 = i.template cast<float>();

		const auto heightDiv2 = factory.template generate<float>(height/2.0f);

		const auto x = x0.sub(width/2.0f).sub(width/4.0f).div(width/3.0f);
		const auto y = heightDiv2.sub(y0).div(width/3.0f);


		const auto imagc = factory.template generate<float>(y);
		const auto realc = factory.template generate<float>(x);



		auto imagz = factory.template generate<float>(0);
		auto realz = factory.template generate<float>(0);


		// loop
		bool anyTrue = true;
		auto iteration = factory.template generate<int>(0);
		const auto iterationLimit = factory.template generate<int>(35);
		while(anyTrue)
		{
			// an optimization for fma instruction
			const auto realzClone = factory.template generate<float>(realz);

			// computing while loop condition start
            		const auto imagzSquared = imagz.mul(imagz);
			const auto absLessThan2 = realz.fusedMultiplyAdd(realzClone,imagzSquared).lessThan(4.0f);
			const auto whileLoopCondition = absLessThan2.logicalAnd(iteration.lessThanOrEquals(35));
			anyTrue = whileLoopCondition.isAnyTrue();
			// computing while loop condition end

			// do complex multiplication z = z*z + c
			const auto zzReal = realz.fusedMultiplySub(realzClone,imagzSquared);
			const auto zzImag = realz.fusedMultiplyAdd(imagz,imagz.mul(realz));

			// if a lane has completed work, do not modify it
			realz = whileLoopCondition.ternary( zzReal.add(realc), realz);
			imagz = whileLoopCondition.ternary( zzImag.add(imagc), imagz);

			// increment iteration
			iteration = iteration.add(whileLoopCondition.ternary(1,0));
		}

		const auto thirtyFour = factory.template generate<int>(34);


		const auto ifLessThanThirtyFour = iteration.lessThan(thirtyFour);

		const auto conditionalValue1 = iteration.mul(255).div(33);
		const auto conditionalValue2 = factory.template generate<int>(0);

		const auto returnValue       = ifLessThanThirtyFour.ternary(conditionalValue1, conditionalValue2);
		const auto writeAddr = j.add(i.mul(width));

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

```

