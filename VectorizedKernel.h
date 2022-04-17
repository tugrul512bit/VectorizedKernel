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
#include<cmath>


namespace Vectorization
{

	template<typename Type, int Simd>
	struct KernelData
	{
		alignas(32)
		Type data[Simd];

		KernelData(){}

		KernelData(const Type broadcastedInit)
		{
			for(int i=0;i<Simd;i++)
			{
				data[i] = broadcastedInit;
			}
		}

		KernelData(const KernelData<Type,Simd> & vectorizedIit)
		{
			for(int i=0;i<Simd;i++)
			{
				data[i] = vectorizedIit.data[i];
			}
		}


		inline void readFrom(const Type * const __restrict__ ptr) noexcept
		{

			for(int i=0;i<Simd;i++)
			{
				data[i] = ptr[i];
			}
		}

		inline void writeTo(Type * const __restrict__ ptr) const noexcept
		{

			for(int i=0;i<Simd;i++)
			{
				ptr[i] = data[i];
			}
		}

		inline void writeTo(Type * const __restrict__ ptr, const KernelData<int,Simd> vec) const noexcept
		{

			for(int i=0;i<Simd;i++)
			{
				ptr[vec.data[i]] = data[i];
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
		inline KernelData<int,Simd> lessThan(const KernelData<Type,Simd> vec) const noexcept
		{
			KernelData<int,Simd> result;

			for(int i=0;i<Simd;i++)
			{
				result.data[i] = data[i]<vec.data[i];
			}
			return result;
		}

		// bool
		inline KernelData<int,Simd> lessThanOrEquals(const KernelData<Type,Simd> vec) const noexcept
		{
			KernelData<int,Simd> result;

			for(int i=0;i<Simd;i++)
			{
				result.data[i] = data[i]<=vec.data[i];
			}
			return result;
		}

		// bool
		inline KernelData<int,Simd> lessThanOrEquals(const Type val) const noexcept
		{
			KernelData<int,Simd> result;

			for(int i=0;i<Simd;i++)
			{
				result.data[i] = data[i]<=val;
			}
			return result;
		}

		// bool
		inline KernelData<int,Simd> greaterThan(const KernelData<Type,Simd> vec) const noexcept
		{
			KernelData<int,Simd> result;

			for(int i=0;i<Simd;i++)
			{
				result.data[i] = data[i]>vec.data[i];
			}
			return result;
		}

		// bool
		inline KernelData<int,Simd> greaterThan(const Type val) const noexcept
		{
			KernelData<int,Simd> result;

			for(int i=0;i<Simd;i++)
			{
				result.data[i] = data[i]>val;
			}
			return result;
		}

		// bool
		inline KernelData<int,Simd> equals(const KernelData<Type,Simd> vec) const noexcept
		{
			KernelData<int,Simd> result;

			for(int i=0;i<Simd;i++)
			{
				result.data[i] = data[i] == vec.data[i];
			}
			return result;
		}

		// bool
		inline KernelData<int,Simd> equals(const Type val) const noexcept
		{
			KernelData<int,Simd> result;

			for(int i=0;i<Simd;i++)
			{
				result.data[i] = data[i] == val;
			}
			return result;
		}

		// bool
		inline KernelData<int,Simd> notEqual(const KernelData<Type,Simd> vec) const noexcept
		{
			KernelData<int,Simd> result;

			for(int i=0;i<Simd;i++)
			{
				result.data[i] = data[i] != vec.data[i];
			}
			return result;
		}

		// bool
		inline KernelData<int,Simd> notEqual(const Type val) const noexcept
		{
			KernelData<int,Simd> result;

			for(int i=0;i<Simd;i++)
			{
				result.data[i] = data[i] != val;
			}
			return result;
		}


		// bool
		inline KernelData<int,Simd> logicalAnd(const KernelData<int,Simd> vec) const noexcept
		{
			KernelData<int,Simd> result;


			for(int i=0;i<Simd;i++)
			{
				result.data[i] = data[i] && vec.data[i];
			}
			return result;
		}

		// bool
		inline KernelData<int,Simd> logicalOr(const KernelData<int,Simd> vec) const noexcept
		{
			KernelData<int,Simd> result;


			for(int i=0;i<Simd;i++)
			{
				result.data[i] = data[i] || vec.data[i];
			}
			return result;
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
		inline const KernelData<ComparedType,Simd> ternary(const KernelData<ComparedType,Simd> vec1, const KernelData<ComparedType,Simd> vec2) const noexcept
		{
			KernelData<ComparedType,Simd> result;

			for(int i=0;i<Simd;i++)
			{
				result.data[i] = data[i]?vec1.data[i]:vec2.data[i];
			}
			return result;
		}

		template<typename ComparedType>
		inline const KernelData<ComparedType,Simd> ternary(const ComparedType val1, const ComparedType val2) const noexcept
		{
			KernelData<ComparedType,Simd> result;

			for(int i=0;i<Simd;i++)
			{
				result.data[i] = data[i]?val1:val2;
			}
			return result;
		}

		inline void broadcast(const Type val) noexcept
		{

			for(int i=0;i<Simd;i++)
			{
				data[i] = val;
			}
		}

		inline void readFrom(const KernelData<Type,Simd> vec) noexcept
		{

			for(int i=0;i<Simd;i++)
			{
				data[i] = vec.data[i];
			}
		}

		template<typename NewType>
		inline const KernelData<NewType,Simd> cast() const noexcept
		{
			KernelData<NewType,Simd> result;

			for(int i=0;i<Simd;i++)
			{
				result.data[i] = (NewType)data[i];
			}
			return result;
		}

		inline const KernelData<Type,Simd> sqrt() const noexcept
		{
			KernelData<Type,Simd> result;

			for(int i=0;i<Simd;i++)
			{
				result.data[i] = std::sqrt(data[i]);
			}
			return result;
		}

		inline const KernelData<Type,Simd> add(const KernelData<Type,Simd> vec) const noexcept
		{
			KernelData<Type,Simd> result;

			for(int i=0;i<Simd;i++)
			{
				result.data[i] = data[i] + vec.data[i];
			}
			return result;
		}

		inline const KernelData<Type,Simd> sub(const KernelData<Type,Simd> vec) const noexcept
		{
			KernelData<Type,Simd> result;

			for(int i=0;i<Simd;i++)
			{
				result.data[i] = data[i] - vec.data[i];
			}
			return result;
		}

		inline const KernelData<Type,Simd> div(const KernelData<Type,Simd> vec) const noexcept
		{
			KernelData<Type,Simd> result;

			for(int i=0;i<Simd;i++)
			{
				result.data[i] = data[i] / vec.data[i];
			}
			return result;
		}

		inline const KernelData<Type,Simd> div(const Type val) const noexcept
		{
			KernelData<Type,Simd> result;

			for(int i=0;i<Simd;i++)
			{
				result.data[i] = data[i] / val;
			}
			return result;
		}

		inline const KernelData<Type,Simd> mul(const KernelData<Type,Simd> vec) const noexcept
		{
			KernelData<Type,Simd> result;

			for(int i=0;i<Simd;i++)
			{
				result.data[i] = data[i] * vec.data[i];
			}
			return result;
		}

		// returns current vector * vec1 + vec2
		inline const KernelData<Type,Simd> fusedMultiplyAdd(const KernelData<Type,Simd> vec1, const KernelData<Type,Simd> vec2) const noexcept
		{
			KernelData<Type,Simd> result;

			for(int i=0;i<Simd;i++)
			{
				result.data[i] = data[i] * vec1.data[i] + vec2.data[i];
			}
			return result;
		}

		// returns current vector * vec1 - vec2
		inline const KernelData<Type,Simd> fusedMultiplySub(const KernelData<Type,Simd> vec1, const KernelData<Type,Simd> vec2) const noexcept
		{
			KernelData<Type,Simd> result;

			for(int i=0;i<Simd;i++)
			{
				result.data[i] = data[i] * vec1.data[i] - vec2.data[i];
			}
			return result;
		}

		inline const KernelData<Type,Simd> mul(const Type val) const noexcept
		{
			KernelData<Type,Simd> result;

			for(int i=0;i<Simd;i++)
			{
				result.data[i] = data[i] * val;
			}
			return result;
		}

		inline const KernelData<Type,Simd> modulus(const KernelData<Type,Simd> vec) const noexcept
		{
			KernelData<Type,Simd> result;

			for(int i=0;i<Simd;i++)
			{
				result.data[i] = data[i] % vec.data[i];
			}
			return result;
		}

		inline const KernelData<Type,Simd> modulus(const Type val) const noexcept
		{
			KernelData<Type,Simd> result;

			for(int i=0;i<Simd;i++)
			{
				result.data[i] = data[i] % val;
			}
			return result;
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
			for(int i=0;i<nLoop;i++)
			{
				auto id = factory.template generate<int>();
				id.idCompute(i*SimdWidth,[](const int prm){ return prm;});
				kernel(factory, id, args...);
			}



			if((n/SimdWidth)*SimdWidth != n)
			{
				const KernelDataFactory<1> factoryLast;

				const int m = n%SimdWidth;
				for(int i=0;i<m;i++)
				{
					auto id = factoryLast.template generate<int>();
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
