/*
 * VectorizedKernel.h
 *
 *  Created on: Apr 16, 2022
 *      Author: tugrul
 */

#ifndef VECTORIZEDKERNEL_H_
#define VECTORIZEDKERNEL_H_



#include <iostream>
#include <string>
#include <functional>
#include <cmath>
#include <chrono>


namespace Vectorization
{

	class Bench
	{
	public:
		Bench(size_t * targetPtr)
		{
			target=targetPtr;
			t1 =  std::chrono::duration_cast< std::chrono::nanoseconds >(std::chrono::high_resolution_clock::now().time_since_epoch());
		}

		~Bench()
		{
			t2 =  std::chrono::duration_cast< std::chrono::nanoseconds >(std::chrono::high_resolution_clock::now().time_since_epoch());
			if(target)
			{
				*target= t2.count() - t1.count();
			}
			else
			{
				std::cout << (t2.count() - t1.count())/1000000000.0 << " seconds" << std::endl;
			}
		}
	private:
		size_t * target;
		std::chrono::nanoseconds t1,t2;
	};

	template<typename Type, int Simd>
	struct KernelData
	{

		alignas(32)
		Type data[Simd];



		KernelData(){}

		KernelData(const Type & broadcastedInit) noexcept
		{

			for(int i=0;i<Simd;i++)
			{
				data[i] = broadcastedInit;
			}
		}

		KernelData(const KernelData<Type,Simd> & vectorizedIit) noexcept
		{

			for(int i=0;i<Simd;i++)
			{
				data[i] = vectorizedIit.data[i];
			}
		}


		KernelData(KernelData&& dat){

			for(int i=0;i<Simd;i++)
			{
				data[i] = dat.data[i];
			}
	    }
		KernelData& operator=(const KernelData& dat){

			for(int i=0;i<Simd;i++)
			{
				data[i] = dat.data[i];
			}
	        return *this;
	    };
		KernelData& operator=(KernelData&& dat){

			for(int i=0;i<Simd;i++)
			{
				data[i] = dat.data[i];
			}
	        return *this;

	    };
	    ~KernelData(){

	    };

	    inline KernelData<Type,Simd> & assign()
		{
	    	return *this;
		}

	    // contiguous read element by element starting from beginning of ptr
		inline void readFrom(const Type * const __restrict__ ptr) noexcept
		{

			for(int i=0;i<Simd;i++)
			{
				data[i] = ptr[i];
			}
		}

		// contiguous write element by element starting from beginning of ptr
		inline void writeTo(Type * const __restrict__ ptr) const noexcept
		{

			for(int i=0;i<Simd;i++)
			{
				ptr[i] = data[i];
			}
		}

		// does scatter operation (every element writes its own targeted ptr element, decided by elements of id)
		inline void writeTo(Type * const __restrict__ ptr, const KernelData<int,Simd> & id) const noexcept
		{

			for(int i=0;i<Simd;i++)
			{
				ptr[id.data[i]] = data[i];
			}
		}

		// uses only first item of id to compute the starting point of target ptr element.
		// writes Simd number of elements to target starting from ptr + id.data[0]
		inline void writeToContiguous(Type * const __restrict__ ptr, const KernelData<int,Simd> & id) const noexcept
		{
			const int idx = id.data[0];
			for(int i=0;i<Simd;i++)
			{
				ptr[idx+i] = data[i];
			}
		}

		// does gather operation (every element reads its own sourced ptr element, decided by elements of id)
		inline void readFrom(Type * const __restrict__ ptr, const KernelData<int,Simd> & id) noexcept
		{

			for(int i=0;i<Simd;i++)
			{
				 data[i] = ptr[id.data[i]];
			}
		}

		// uses only first item of id to compute the starting point of source ptr element.
		// reads Simd number of elements from target starting from ptr + id.data[0]
		inline void readFromContiguous(Type * const __restrict__ ptr, const KernelData<int,Simd> & id) noexcept
		{
			const int idx = id.data[0];
			for(int i=0;i<Simd;i++)
			{
				data[i] = ptr[idx+i];
			}
		}

		template<typename F>
		inline void idCompute(const int id, const F & f) noexcept
		{

			for(int i=0;i<Simd;i++)
			{
				data[i] = f(id+i);
			}
		}

		// bool
		inline void lessThan(const KernelData<Type,Simd> & vec, KernelData<int,Simd> & result) const noexcept
		{
			for(int i=0;i<Simd;i++)
			{
				result.data[i] = data[i]<vec.data[i];
			}
		}

		// bool
		inline void lessThanOrEquals(const KernelData<Type,Simd> & vec, KernelData<int,Simd> & result) const noexcept
		{
			for(int i=0;i<Simd;i++)
			{
				result.data[i] = data[i]<=vec.data[i];
			}
		}

		// bool
		inline void lessThanOrEquals(const Type val, KernelData<int,Simd> & result) const noexcept
		{
			for(int i=0;i<Simd;i++)
			{
				result.data[i] = data[i]<=val;
			}
		}

		// bool
		inline void greaterThan(const KernelData<Type,Simd> & vec, KernelData<int,Simd> & result) const noexcept
		{
			for(int i=0;i<Simd;i++)
			{
				result.data[i] = data[i]>vec.data[i];
			}
		}

		// bool
		inline void greaterThan(const Type val, KernelData<int,Simd> & result) const noexcept
		{
			for(int i=0;i<Simd;i++)
			{
				result.data[i] = data[i]>val;
			}
		}

		// bool
		inline void equals(const KernelData<Type,Simd> & vec, KernelData<int,Simd> & result) const noexcept
		{
			for(int i=0;i<Simd;i++)
			{
				result.data[i] = data[i] == vec.data[i];
			}
		}

		// bool
		inline void equals(const Type val, KernelData<int,Simd> & result) const noexcept
		{
			for(int i=0;i<Simd;i++)
			{
				result.data[i] = data[i] == val;
			}
		}

		// bool
		inline void notEqual(const KernelData<Type,Simd> & vec, KernelData<int,Simd> & result) const noexcept
		{
			for(int i=0;i<Simd;i++)
			{
				result.data[i] = data[i] != vec.data[i];
			}
		}

		// bool
		inline  void notEqual(const Type val, KernelData<int,Simd> & result) const noexcept
		{
			for(int i=0;i<Simd;i++)
			{
				result.data[i] = data[i] != val;
			}
		}


		// bool
		inline  void logicalAnd(const KernelData<int,Simd> vec, KernelData<int,Simd> & result) const noexcept
		{
			for(int i=0;i<Simd;i++)
			{
				result.data[i] = data[i] && vec.data[i];
			}
		}

		// bool
		inline void logicalOr(const KernelData<int,Simd> vec, KernelData<int,Simd> & result) const noexcept
		{
			for(int i=0;i<Simd;i++)
			{
				result.data[i] = data[i] || vec.data[i];
			}
		}

		inline bool areAllTrue() const noexcept
		{
			bool result = true;


			for(int i=0;i<Simd;i++)
			{
				result = result && data[i];
			}
			return result;
		}

		inline bool isAnyTrue() const noexcept
		{
			bool result = false;


			for(int i=0;i<Simd;i++)
			{
				result = result || data[i];
			}
			return result;
		}

		template<typename ComparedType>
		inline void ternary(const KernelData<ComparedType,Simd> vec1, const KernelData<ComparedType,Simd> vec2, KernelData<ComparedType,Simd> & result) const noexcept
		{
			for(int i=0;i<Simd;i++)
			{
				result.data[i] = data[i]?vec1.data[i]:vec2.data[i];
			}
		}

		template<typename ComparedType>
		inline void ternary(const ComparedType val1, const ComparedType val2, KernelData<ComparedType,Simd> & result) const noexcept
		{
			for(int i=0;i<Simd;i++)
			{
				result.data[i] = data[i]?val1:val2;
			}
		}

		inline void broadcast(const Type val) noexcept
		{

			for(int i=0;i<Simd;i++)
			{
				data[i] = val;
			}
		}

		inline void readFrom(const KernelData<Type,Simd> & vec) noexcept
		{
			for(int i=0;i<Simd;i++)
			{
				data[i] = vec.data[i];
			}
		}

		template<typename NewType>
		inline void castAndCopyTo(KernelData<NewType,Simd> & result) const noexcept
		{
			for(int i=0;i<Simd;i++)
			{
				result.data[i] = (NewType)data[i];
			}
		}

		inline  void sqrt(KernelData<Type,Simd> & result) const noexcept
		{
			for(int i=0;i<Simd;i++)
			{
				result.data[i] = std::sqrt(data[i]);
			}
		}

		inline  void add(const KernelData<Type,Simd> & vec, KernelData<Type,Simd> & result) const noexcept
		{
			for(int i=0;i<Simd;i++)
			{
				result.data[i] = data[i] + vec.data[i];
			}
		}

		inline void sub(const KernelData<Type,Simd> & vec, KernelData<Type,Simd> & result) const noexcept
		{
			for(int i=0;i<Simd;i++)
			{
				result.data[i] = data[i] - vec.data[i];
			}
		}

		inline void div(const KernelData<Type,Simd> & vec, KernelData<Type,Simd> & result) const noexcept
		{
			for(int i=0;i<Simd;i++)
			{
				result.data[i] = data[i] / vec.data[i];
			}
		}

		inline void div(const Type & val, KernelData<Type,Simd> & result) const noexcept
		{
			for(int i=0;i<Simd;i++)
			{
				result.data[i] = data[i] / val;
			}
		}



		inline void mul(const KernelData<Type,Simd> & vec, KernelData<Type,Simd> & result) const noexcept
		{
			for(int i=0;i<Simd;i++)
			{
				result.data[i] = data[i] * vec.data[i];
			}
		}


		inline void fusedMultiplyAdd(const KernelData<Type,Simd> & vec1, const KernelData<Type,Simd> & vec2, KernelData<Type,Simd> & result) const noexcept
		{
			for(int i=0;i<Simd;i++)
			{
				result.data[i] = std::fma(data[i], vec1.data[i], vec2.data[i]);
			}
		}


		inline void fusedMultiplySub(const KernelData<Type,Simd> & vec1, const KernelData<Type,Simd> & vec2, KernelData<Type,Simd> & result) const noexcept
		{
			for(int i=0;i<Simd;i++)
			{
				result.data[i] = std::fma(data[i], vec1.data[i], -vec2.data[i]);
			}
		}

		inline void mul(const Type & val, KernelData<Type,Simd> & result) const noexcept
		{
			for(int i=0;i<Simd;i++)
			{
				result.data[i] = data[i] * val;
			}
		}

		inline void modulus(const KernelData<Type,Simd> & vec, KernelData<Type,Simd> & result) const noexcept
		{
			for(int i=0;i<Simd;i++)
			{
				result.data[i] = data[i] % vec.data[i];
			}
		}

		inline  void modulus(const Type & val, KernelData<Type,Simd> & result) const noexcept
		{
			for(int i=0;i<Simd;i++)
			{
				result.data[i] = data[i] % val;
			}
		}

		inline void leftShift(const KernelData<Type,Simd> & vec, KernelData<Type,Simd> & result) const noexcept
		{
			for(int i=0;i<Simd;i++)
			{
				result.data[i] = data[i] << vec.data[i];
			}
		}

		inline void rightShift(const KernelData<Type,Simd> & vec, KernelData<Type,Simd> & result) const noexcept
		{
			for(int i=0;i<Simd;i++)
			{
				result.data[i] = data[i] >> vec.data[i];
			}
		}

		// this function is not accelerated. use it sparsely.
		inline void pow(const KernelData<Type,Simd> & vec, KernelData<Type,Simd> & result) const noexcept
		{
			for(int i=0;i<Simd;i++)
			{
				result.data[i] = std::pow(data[i],vec.data[i]);
			}
		}

		// this function is not accelerated. use it sparsely.
		inline void exp(KernelData<Type,Simd> & result) const noexcept
		{
			for(int i=0;i<Simd;i++)
			{
				result.data[i] = std::exp(data[i]);
			}
		}

		// this function is not accelerated. use it sparsely.
		inline void log(KernelData<Type,Simd> & result) const noexcept
		{
			for(int i=0;i<Simd;i++)
			{
				result.data[i] = std::log(data[i]);
			}
		}

		// this function is not accelerated. use it sparsely.
		inline void log2(KernelData<Type,Simd> & result) const noexcept
		{
			for(int i=0;i<Simd;i++)
			{
				result.data[i] = std::log2(data[i]);
			}
		}
	};



	template<int mask>
	struct KernelDataFactory
	{
		KernelDataFactory():width(mask)
		{

		}


		template<typename Type>
		inline
		KernelData<Type,mask> generate() const
		{
			return KernelData<Type,mask>();
		}


		template<typename Type>
		inline
		KernelData<Type,mask> generate(const KernelData<Type,mask> & vec) const
		{
			return KernelData<Type,mask>(vec);
		}


		template<typename Type>
		inline
		KernelData<Type,mask> generate(const Type & val) const
		{
			return KernelData<Type,mask>(val);
		}
		const int width;
	};


	template<class...Args>
	struct KernelArgs
	{};

	template<int SimdWidth, typename F, typename... Args>
	class Kernel
	{
	public:


		Kernel(F&& kernelPrm):kernel(std::move(kernelPrm))
		{

		}


		void run(int n, Args... args)
		{
			const int nLoop = (n/SimdWidth);
			const KernelDataFactory<SimdWidth> factory;
			auto id = factory.template generate<int>();
			for(int i=0;i<nLoop;i++)
			{
				id.idCompute(i*SimdWidth,[](const int prm){ return prm;});
				kernel(factory, id, args...);
			}



			if((n/SimdWidth)*SimdWidth != n)
			{
				const KernelDataFactory<1> factoryLast;

				const int m = n%SimdWidth;
				auto id = factoryLast.template generate<int>();
				for(int i=0;i<m;i++)
				{

					id.idCompute(nLoop*SimdWidth+i,[](const int prm){ return prm;});
					kernel(factoryLast, id, args...);
				}
			}
		}
	private:
		F kernel;
	};

	template<int SimdWidth, typename F, class...Args>
	auto createKernel(F&& kernelPrm, KernelArgs<Args...> const& _prm_)
	{
		return Kernel<SimdWidth, F, Args...>(std::forward<F>(kernelPrm));
	}

}


#endif /* VECTORIZEDKERNEL_H_ */
