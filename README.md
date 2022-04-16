# VectorizedKernel
Running GPGPU kernels on CPU with auto-vectorization for SSE/AVX SIMD Architectures.

Mandelbrot generation sample:


```C++
#include <vector>
#include "VectorizedKernel.h"
int main()
{
	constexpr float width = 2000;
	constexpr float height = 2000;

    // buffer for writing pixel values (1 int per pixel)
	std::vector<int> img(height*width);
    
	constexpr int simd = 16; // good for avx512, not good for avx2/sse4
	auto kernel = Vectorization::createKernel<simd>([&](auto & factory, auto & idThread, int * img){

		// create a width variable that is 16 threads (not CPU threads) combined
		auto w = factory.template generate<int>();
		w.broadcast((int)width);
		auto j = idThread.modulus(w);
		auto i = idThread.div(w);
		const int vecWidth = factory.width;



		auto x = j.template cast<float>();
		auto y = i.template cast<float>();

		auto widthDiv2 = factory.template generate<float>();
		auto widthDiv3 = factory.template generate<float>();
		auto widthDiv4 = factory.template generate<float>();

		auto heightDiv2 = factory.template generate<float>();

		widthDiv2.broadcast(width/2.0f);
		widthDiv3.broadcast(width/3.0f);
		widthDiv4.broadcast(width/4.0f);

		heightDiv2.broadcast(height/2.0f);

		x = x.sub(widthDiv2).sub(widthDiv4).div(widthDiv3);
		y = heightDiv2.sub(y).div(widthDiv3);


		auto imagc = factory.template generate<float>();
		auto realc = factory.template generate<float>();

		realc.readFrom(x);
		imagc.readFrom(y);

		auto imagz = factory.template generate<float>();
		auto realz = factory.template generate<float>();

		realz.broadcast(0);
		imagz.broadcast(0);


		// loop
		bool anyTrue = true;
		auto iteration = factory.template generate<int>();
		auto iterationLimit = factory.template generate<int>();
		auto one = factory.template generate<int>();
		one.broadcast(1);
		auto zero = factory.template generate<int>();
		zero.broadcast(0);
		iteration.broadcast(0);
		iterationLimit.broadcast(35);
		auto two = factory.template generate<float>();
		two.broadcast(2.0f);
		while(anyTrue)
		{

		  // computing while loop condition start
		  auto absLessThan2 = realz.mul(realz).add(imagz.mul(imagz)).sqrt().lessThan(two);
		  auto whileLoopCondition = absLessThan2.logicalAnd(iteration.lessThanOrEquals(iterationLimit));
		  anyTrue = whileLoopCondition.isAnyTrue();
		  // computing while loop condition end


		  // do complex multiplication z = z*z + c
		  auto zzReal = realz.mul(realz).sub(imagz.mul(imagz));
		  auto zzImag = realz.mul(imagz).add(imagz.mul(realz));

		  // if a lane has completed work, do not modify it
		  realz = whileLoopCondition.ternary( zzReal.add(realc), realz);
		  imagz = whileLoopCondition.ternary( zzImag.add(imagc), imagz);


		  // increment iteration
		  iteration = iteration.add(whileLoopCondition.ternary(one,zero)); // todo: ternary increment


		}

		auto thirtyFour = factory.template generate<int>();
		thirtyFour.broadcast(34);

		auto ifLessThanThirtyFour = iteration.lessThan(thirtyFour);
		auto conditionalValue1Mul = factory.template generate<int>();
		auto conditionalValue1Div = factory.template generate<int>();
		conditionalValue1Mul.broadcast(255);
		conditionalValue1Div.broadcast(33);
		auto conditionalValue1 = iteration.mul(conditionalValue1Mul).div(conditionalValue1Div);
		auto conditionalValue2 = factory.template generate<int>();
		conditionalValue2.broadcast(0);
		auto returnValue       = ifLessThanThirtyFour.ternary(conditionalValue1, conditionalValue2);
		auto writeAddr = j.add(i.mul(w));

		returnValue.writeTo(img,writeAddr);

	},Vectorization::KernelArgs<int*>{});

	// run width*height threads (not real cpu threads, just simd lanes)
	kernel.run(width*height,img.data());
  
  // equivalent to (but much faster than) this:
  /*
  for (size_t i = 0; i < height; i++)
	{
		for (size_t j = 0; j < width; j++)
		{
			img[j+i*width]= getPoint(i, j);
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
  */
	return 0;
}
```

With 2.1GHz Fx8150 single thread, it takes less than 1 second 
to compute 1000x1000 pixels and write to disk in ppm format.
