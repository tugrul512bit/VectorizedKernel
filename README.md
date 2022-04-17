# VectorizedKernel
Running GPGPU kernels on CPU with auto-vectorization for SSE/AVX/AVX512 SIMD microarchitectures.

Mandelbrot generation sample:


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
	constexpr int simd = 32;
	auto kernel = Vectorization::createKernel<simd>([&](auto & factory, auto & idThread, int * img){



		const auto j = idThread.modulus(width);
		const auto i = idThread.div(width);
		const int vecWidth = factory.width;



		auto x0 = j.template cast<float>();
		auto y0 = i.template cast<float>();

		const auto widthDiv2 = factory.template generate<float>(width/2.0f);
		const auto widthDiv3 = factory.template generate<float>(width/3.0f);
		const auto widthDiv4 = factory.template generate<float>(width/4.0f);

		const auto heightDiv2 = factory.template generate<float>(height/2.0f);

		const auto x = x0.sub(widthDiv2).sub(widthDiv4).div(widthDiv3);
		const auto y = heightDiv2.sub(y0).div(widthDiv3);


		const auto imagc = factory.template generate<float>(y);
		const auto realc = factory.template generate<float>(x);



		auto imagz = factory.template generate<float>(0);
		auto realz = factory.template generate<float>(0);


		// loop
		bool anyTrue = true;
		auto iteration = factory.template generate<int>(0);
		const auto iterationLimit = factory.template generate<int>(35);
		const auto one = factory.template generate<int>(1);
		const auto zero = factory.template generate<int>(0);
		const auto four = factory.template generate<float>(4.0f);
		while(anyTrue)
		{

			// computing while loop condition start
			const auto absLessThan2 = realz.mul(realz).add(imagz.mul(imagz)).lessThan(4.0f);
			const auto whileLoopCondition = absLessThan2.logicalAnd(iteration.lessThanOrEquals(35));
			anyTrue = whileLoopCondition.isAnyTrue();
			// computing while loop condition end


			// do complex multiplication z = z*z + c
			const auto zzReal = realz.fusedMultiplySub(realz,imagz.mul(imagz));
			const auto zzImag = realz.fusedMultiplyAdd(imagz,imagz.mul(realz));

			// if a lane has completed work, do not modify it
			realz = whileLoopCondition.ternary( zzReal.add(realc), realz);
			imagz = whileLoopCondition.ternary( zzImag.add(imagc), imagz);


			// increment iteration
			iteration = iteration.add(whileLoopCondition.ternary(1,0)); // todo: ternary increment


		}

		const auto thirtyFour = factory.template generate<int>(34);


		const auto ifLessThanThirtyFour = iteration.lessThan(thirtyFour);
		const auto conditionalValue1Mul = factory.template generate<int>(255);
		const auto conditionalValue1Div = factory.template generate<int>(33);

		const auto conditionalValue1 = iteration.mul(conditionalValue1Mul).div(conditionalValue1Div);
		const auto conditionalValue2 = factory.template generate<int>(0);

		const auto returnValue       = ifLessThanThirtyFour.ternary(conditionalValue1, conditionalValue2);
		const auto writeAddr = j.add(i.mul(width));

		returnValue.writeTo(img,writeAddr);

	},Vectorization::KernelArgs<int*>{});

	auto t1 = readTSC();
	kernel.run(width*height,img.data());
	auto t2 = readTSC();
	std::cout<<(t2-t1)/(width*height)<<" cycles per pixel"<<std::endl;

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

With 2.1GHz Fx8150 single thread + 1333MHz DDR3 RAM (single channel), it takes 380 cycles per pixel (or 190 ns per pixel) (or ~5 ns per iteration of pixel)

For AVX512 server of Godbolt.org, it computes each pixel (35 iterations max) in 85 cycles. This means ~2 cycles per iteration per pixel or ~1 ns per iteration per pixel. 1000x1000 image takes ~40 milliseconds, single-thread, with zero explicit intrinsic instructions used.
