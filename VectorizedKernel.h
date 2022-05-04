/*
 * VectorizedKernel.h
 *
 *  Created on: Apr 16, 2022
 *      Author: tugrul
 */

#ifndef VECTORIZEDKERNEL_H_
#define VECTORIZEDKERNEL_H_


#include <vector>
#include <iostream>
#include <cstring>
#include <functional>
#include <cmath>
#include <chrono>
#include <thread>
#include <atomic>

namespace Vectorization
{
#define CREATE_PRAGMA(x) _Pragma (#x)

#if defined(__INTEL_COMPILER)

#define VECTORIZED_KERNEL_METHOD __attribute__((always_inline))
#define VECTORIZED_KERNEL_LOOP CREATE_PRAGMA(simd)

#elif defined(__clang__)

#define VECTORIZED_KERNEL_METHOD inline
#define VECTORIZED_KERNEL_LOOP CREATE_PRAGMA(clang loop vectorize(assume_safety) vectorize_width(Simd))


#elif defined(__GNUC__) || defined(__GNUG__)

#define VECTORIZED_KERNEL_METHOD inline
#define VECTORIZED_KERNEL_LOOP

#elif defined(_MSC_VER)

#define VECTORIZED_KERNEL_METHOD __declspec(inline)
#define VECTORIZED_KERNEL_LOOP CREATE_PRAGMA(loop( ivdep ))

#elif

#define VECTORIZED_KERNEL_METHOD inline
#define VECTORIZED_KERNEL_LOOP CREATE_PRAGMA(notoptimized)

#endif

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


		alignas(64)
		Type data[Simd];



		VECTORIZED_KERNEL_METHOD
		KernelData(){}

		VECTORIZED_KERNEL_METHOD
		KernelData(const Type & broadcastedInit) noexcept
		{
			VECTORIZED_KERNEL_LOOP
			for(int i=0;i<Simd;i++)
			{
				data[i] = broadcastedInit;
			}
		}

		VECTORIZED_KERNEL_METHOD
		KernelData(const KernelData<Type,Simd> & vectorizedIit) noexcept
		{
			VECTORIZED_KERNEL_LOOP
			for(int i=0;i<Simd;i++)
			{
				data[i] = vectorizedIit.data[i];
			}
		}

		VECTORIZED_KERNEL_METHOD
		KernelData(KernelData&& dat) noexcept
		{
			VECTORIZED_KERNEL_LOOP
			for(int i=0;i<Simd;i++)
			{
				data[i] = dat.data[i];
			}
	    }

		VECTORIZED_KERNEL_METHOD
		KernelData& operator=(const KernelData& dat) noexcept
		{
			VECTORIZED_KERNEL_LOOP
			for(int i=0;i<Simd;i++)
			{
				data[i] = dat.data[i];
			}
	        return *this;
	    };

		VECTORIZED_KERNEL_METHOD
		KernelData& operator=(KernelData&& dat) noexcept
		{
			VECTORIZED_KERNEL_LOOP
			for(int i=0;i<Simd;i++)
			{
				data[i] = dat.data[i];
			}
	        return *this;

	    };

		VECTORIZED_KERNEL_METHOD
	    ~KernelData() noexcept
		{

	    };


	    // contiguous read element by element starting from beginning of ptr
		VECTORIZED_KERNEL_METHOD
		void readFrom(const Type * const __restrict__ ptr) noexcept
		{
			VECTORIZED_KERNEL_LOOP
			for(int i=0;i<Simd;i++)
			{
				data[i] = ptr[i];
			}
		}

	    // contiguous read element by element starting from beginning of ptr
		// masked read operation: if mask lane is set then read. if not set then don't read
		template<typename TypeMask>
		VECTORIZED_KERNEL_METHOD
		void readFromMasked(const Type * const __restrict__ ptr, const KernelData<TypeMask,Simd> & mask) noexcept
		{
			VECTORIZED_KERNEL_LOOP
			for(int i=0;i<Simd;i++)
			{
				data[i] = mask.data[i]?ptr[i]:data[i];
			}
		}

		// contiguous write element by element starting from beginning of ptr
		VECTORIZED_KERNEL_METHOD
		void writeTo(Type * const __restrict__ ptr) const noexcept
		{
			VECTORIZED_KERNEL_LOOP
			for(int i=0;i<Simd;i++)
			{
				ptr[i] = data[i];
			}
		}

		// contiguous write element by element starting from beginning of ptr
		// masked write: if mask lane is set then write, if not set then don't write
		template<typename TypeMask>
		VECTORIZED_KERNEL_METHOD
		void writeToMasked(Type * const __restrict__ ptr, const KernelData<TypeMask,Simd> & mask) const noexcept
		{

			for(int i=0;i<Simd;i++)
			{
				if(mask.data[i])
						ptr[i] = data[i];
			}
		}

		// does scatter operation (every element writes its own targeted ptr element, decided by elements of id)
		VECTORIZED_KERNEL_METHOD
		void writeTo(Type * const __restrict__ ptr, const KernelData<int,Simd> & id) const noexcept
		{

			for(int i=0;i<Simd;i++)
			{
				ptr[id.data[i]] = data[i];
			}
		}

		// does scatter operation (every element writes its own targeted ptr element, decided by elements of id)
		// masked write: if mask lane is set then write, if not set then don't write
		template<typename TypeMask>
		VECTORIZED_KERNEL_METHOD
		void writeToMasked(Type * const __restrict__ ptr, const KernelData<int,Simd> & id, const KernelData<TypeMask,Simd> & mask) const noexcept
		{

			for(int i=0;i<Simd;i++)
			{
				if(mask.data[i])
					ptr[id.data[i]] = data[i];
			}
		}

		// uses only first item of id to compute the starting point of target ptr element.
		// writes Simd number of elements to target starting from ptr + id.data[0]
		VECTORIZED_KERNEL_METHOD
		void writeToContiguous(Type * const __restrict__ ptr, const KernelData<int,Simd> & id) const noexcept
		{
			const int idx = id.data[0];
			VECTORIZED_KERNEL_LOOP
			for(int i=0;i<Simd;i++)
			{
				ptr[idx+i] = data[i];
			}
		}

		// uses only first item of id to compute the starting point of target ptr element.
		// writes Simd number of elements to target starting from ptr + id.data[0]
		// masked write: if mask lane is set then writes, if not set then does not write
		template<typename TypeMask>
		VECTORIZED_KERNEL_METHOD
		void writeToContiguousMasked(Type * const __restrict__ ptr, const KernelData<int,Simd> & id, const KernelData<TypeMask,Simd> & mask) const noexcept
		{
			const int idx = id.data[0];

			for(int i=0;i<Simd;i++)
			{
				if(mask.data[i])
					ptr[idx+i] = data[i];
			}
		}

		// does gather operation (every element reads its own sourced ptr element, decided by elements of id)
		VECTORIZED_KERNEL_METHOD
		void readFrom(Type * const __restrict__ ptr, const KernelData<int,Simd> & id) noexcept
		{

			for(int i=0;i<Simd;i++)
			{
				 data[i] = ptr[id.data[i]];
			}
		}

		// does gather operation (every element reads its own sourced ptr element, decided by elements of id)
		VECTORIZED_KERNEL_METHOD
		void readFrom(const Type * const __restrict__ ptr, const KernelData<int,Simd> & id) noexcept
		{

			for(int i=0;i<Simd;i++)
			{
				 data[i] = ptr[id.data[i]];
			}
		}

		// does gather operation (every element reads its own sourced ptr element, decided by elements of id)
		// masked operation: if mask lane is set, then it reads from pointer+id.data[i], if not set then it does not read anything
		template<typename TypeMask>
		VECTORIZED_KERNEL_METHOD
		void readFromMasked(Type * const __restrict__ ptr, const KernelData<int,Simd> & id, const KernelData<TypeMask,Simd> & mask) noexcept
		{

			for(int i=0;i<Simd;i++)
			{
				 data[i] = mask.data[i]?ptr[id.data[i]]:data[i];
			}
		}

		// does gather operation (every element reads its own sourced ptr element, decided by elements of id)
		// masked operation: if mask lane is set, then it reads from pointer+id.data[i], if not set then it does not read anything
		template<typename TypeMask>
		VECTORIZED_KERNEL_METHOD
		void readFromMasked(const Type * const __restrict__ ptr, const KernelData<int,Simd> & id, const KernelData<TypeMask,Simd> & mask) noexcept
		{

			for(int i=0;i<Simd;i++)
			{
				 data[i] = mask.data[i]?ptr[id.data[i]]:data[i];
			}
		}

		// uses only first item of id to compute the starting point of source ptr element.
		// reads Simd number of elements from target starting from ptr + id.data[0]
		VECTORIZED_KERNEL_METHOD
		void readFromContiguous(Type * const __restrict__ ptr, const KernelData<int,Simd> & id) noexcept
		{
			const int idx = id.data[0];
			VECTORIZED_KERNEL_LOOP
			for(int i=0;i<Simd;i++)
			{
				data[i] = ptr[idx+i];
			}
		}

		// uses only first item of id to compute the starting point of source ptr element.
		// reads Simd number of elements from target starting from ptr + id.data[0]
		// masked operation: if mask lane is set, then it reads from pointer+id.data[0], if not set then it does not read anything
		template<typename TypeMask>
		VECTORIZED_KERNEL_METHOD
		void readFromContiguousMasked(Type * const __restrict__ ptr, const KernelData<int,Simd> & id, const KernelData<TypeMask,Simd> & mask) noexcept
		{
			const int idx = id.data[0];
			VECTORIZED_KERNEL_LOOP
			for(int i=0;i<Simd;i++)
			{
				data[i] = mask.data[i]?ptr[idx+i]:data[i];
			}
		}

		template<typename F>
		VECTORIZED_KERNEL_METHOD
		void idCompute(const int id, const F & f) noexcept
		{
			VECTORIZED_KERNEL_LOOP
			for(int i=0;i<Simd;i++)
			{
				data[i] = f(id+i);
			}
		}

		// bool
        template<typename TypeMask>
        VECTORIZED_KERNEL_METHOD
		void lessThan(const KernelData<Type,Simd> & vec, KernelData<TypeMask,Simd> & result) const noexcept
		{
        	VECTORIZED_KERNEL_LOOP
			for(int i=0;i<Simd;i++)
			{
				result.data[i] = data[i]<vec.data[i];
			}
		}



		// bool
        template<typename TypeMask>
        VECTORIZED_KERNEL_METHOD
		void lessThanOrEquals(const KernelData<Type,Simd> & vec, KernelData<TypeMask,Simd> & result) const noexcept
		{
        	VECTORIZED_KERNEL_LOOP
			for(int i=0;i<Simd;i++)
			{
				result.data[i] = data[i]<=vec.data[i];
			}
		}

		// bool
        template<typename TypeMask>
        VECTORIZED_KERNEL_METHOD
		void lessThanOrEquals(const Type val, KernelData<TypeMask,Simd> & result) const noexcept
		{
        	VECTORIZED_KERNEL_LOOP
			for(int i=0;i<Simd;i++)
			{
				result.data[i] = data[i]<=val;
			}
		}

		// bool
        template<typename TypeMask>
        VECTORIZED_KERNEL_METHOD
		void greaterThan(const KernelData<Type,Simd> & vec, KernelData<TypeMask,Simd> & result) const noexcept
		{
        	VECTORIZED_KERNEL_LOOP
			for(int i=0;i<Simd;i++)
			{
				result.data[i] = data[i]>vec.data[i];
			}
		}

		// bool
        template<typename TypeMask>
        VECTORIZED_KERNEL_METHOD
		void greaterThan(const Type val, KernelData<TypeMask,Simd> & result) const noexcept
		{
        	VECTORIZED_KERNEL_LOOP
			for(int i=0;i<Simd;i++)
			{
				result.data[i] = data[i]>val;
			}
		}

		// bool
        template<typename TypeMask>
        VECTORIZED_KERNEL_METHOD
		void equals(const KernelData<Type,Simd> & vec, KernelData<TypeMask,Simd> & result) const noexcept
		{
        	VECTORIZED_KERNEL_LOOP
			for(int i=0;i<Simd;i++)
			{
				result.data[i] = data[i] == vec.data[i];
			}
		}

		// bool
        template<typename TypeMask>
        VECTORIZED_KERNEL_METHOD
		void equals(const Type val, KernelData<TypeMask,Simd> & result) const noexcept
		{
        	VECTORIZED_KERNEL_LOOP
			for(int i=0;i<Simd;i++)
			{
				result.data[i] = data[i] == val;
			}
		}

		// bool
        template<typename TypeMask>
        VECTORIZED_KERNEL_METHOD
		void notEqual(const KernelData<Type,Simd> & vec, KernelData<TypeMask,Simd> & result) const noexcept
		{
        	VECTORIZED_KERNEL_LOOP
			for(int i=0;i<Simd;i++)
			{
				result.data[i] = data[i] != vec.data[i];
			}
		}

		// bool
        template<typename TypeMask>
        VECTORIZED_KERNEL_METHOD
		void notEqual(const Type val, KernelData<TypeMask,Simd> & result) const noexcept
		{
        	VECTORIZED_KERNEL_LOOP
			for(int i=0;i<Simd;i++)
			{
				result.data[i] = data[i] != val;
			}
		}


		// bool
        template<typename TypeMask>
        VECTORIZED_KERNEL_METHOD
		void logicalAnd(const KernelData<TypeMask,Simd> vec, KernelData<TypeMask,Simd> & result) const noexcept
		{
        	VECTORIZED_KERNEL_LOOP
			for(int i=0;i<Simd;i++)
			{
				result.data[i] = data[i] && vec.data[i];
			}
		}

		// bool
        template<typename TypeMask>
        VECTORIZED_KERNEL_METHOD
		void logicalOr(const KernelData<TypeMask,Simd> vec, KernelData<TypeMask,Simd> & result) const noexcept
		{
        	VECTORIZED_KERNEL_LOOP
			for(int i=0;i<Simd;i++)
			{
				result.data[i] = data[i] || vec.data[i];
			}
		}

        VECTORIZED_KERNEL_METHOD
		bool areAllTrue() const noexcept
		{
			int result = 0;


			for(int i=0;i<Simd;i++)
			{
				result = result + (data[i]>0);
			}
			return result==Simd;
		}

        VECTORIZED_KERNEL_METHOD
		bool isAnyTrue() const noexcept
		{
			int result = 0;
			for(int i=0;i<Simd;i++)
			{
				result = result + (data[i]>0);
			}
			return result>0;
		}

        VECTORIZED_KERNEL_METHOD
		Type popCount() const noexcept
		{
			int count = 0;
			for(int i=0;i<Simd;i++)
			{
				count += (data[i]>0);
			}
			return count;
		}

        VECTORIZED_KERNEL_METHOD
		Type horizontalAdd() const noexcept
		{
			Type sum = Type(0);
			for(int i=0;i<Simd;i++)
			{
				sum += data[i];
			}
			return sum;
		}

		template<typename ComparedType>
		VECTORIZED_KERNEL_METHOD
		void ternary(const KernelData<ComparedType,Simd> vec1, const KernelData<ComparedType,Simd> vec2, KernelData<ComparedType,Simd> & result) const noexcept
		{
			VECTORIZED_KERNEL_LOOP
			for(int i=0;i<Simd;i++)
			{
				result.data[i] = data[i]?vec1.data[i]:vec2.data[i];
			}
		}

		template<typename ComparedType>
		VECTORIZED_KERNEL_METHOD
		void ternary(const ComparedType val1, const KernelData<ComparedType,Simd> vec2, KernelData<ComparedType,Simd> & result) const noexcept
		{
			VECTORIZED_KERNEL_LOOP
			for(int i=0;i<Simd;i++)
			{
				result.data[i] = data[i]?val1:vec2.data[i];
			}
		}

		template<typename ComparedType>
		VECTORIZED_KERNEL_METHOD
		void ternary(const KernelData<ComparedType,Simd> vec1, const ComparedType val2, KernelData<ComparedType,Simd> & result) const noexcept
		{
			VECTORIZED_KERNEL_LOOP
			for(int i=0;i<Simd;i++)
			{
				result.data[i] = data[i]?vec1.data[i]:val2;
			}
		}

		template<typename ComparedType>
		VECTORIZED_KERNEL_METHOD
		void ternary(const ComparedType val1, const ComparedType val2, KernelData<ComparedType,Simd> & result) const noexcept
		{
			VECTORIZED_KERNEL_LOOP
			for(int i=0;i<Simd;i++)
			{
				result.data[i] = data[i]?val1:val2;
			}
		}

		VECTORIZED_KERNEL_METHOD
		void broadcast(const Type val) noexcept
		{
			VECTORIZED_KERNEL_LOOP
			for(int i=0;i<Simd;i++)
			{
				data[i] = val;
			}
		}

		// in-place operation so the result variable must be different than current variable
		// gets value from a so-called thread (a lane) in the current SIMD
		// for main body of kernel launch, lane must not overflow Simd
		// for the tail the number of lanes is 1 so the only available lane is 0 that is itself
		// lane value[i] = lane value [id.data[i]]
		// this is a gather operation within the SIMD unit
		template<typename IntegerType>
		VECTORIZED_KERNEL_METHOD
		void gatherFromLane(const KernelData<IntegerType,Simd> & id, KernelData<Type,Simd> & result) const noexcept
		{
			for(int i=0;i<Simd;i++)
			{
				result.data[i] = data[id.data[i]];
			}
		}

		// similar to gatherFromLane but with constant index values for faster operation
		VECTORIZED_KERNEL_METHOD
		void transposeLanes(const int widthTranspose, KernelData<Type,Simd> & result) const noexcept
		{
			VECTORIZED_KERNEL_LOOP
			for(int i=0;i<widthTranspose;i++)
				for(int j=0;j<widthTranspose;j++)
			{
				result.data[i*widthTranspose+j] = data[j*widthTranspose+i];
			}
		}

		// in-place operation so the result variable must be different than current variable
		// shifts lanes (wraps around) left n times out-of-place
		// writes result to another result variable
		template<typename IntegerType>
		VECTORIZED_KERNEL_METHOD
		void lanesLeftShift(const IntegerType & n, KernelData<Type,Simd> & result) const noexcept
		{

			for(int i=0;i<Simd;i++)
			{
				const int j = (i+n)&(Simd-1);
				result.data[i] = data[j];
			}
		}

		// in-place operation so the result variable must be different than current variable
		// shifts lanes (wraps around) right n times out-of-place
		// writes result to another result variable
		// n must not be greater than Simd*2
		template<typename IntegerType>
		VECTORIZED_KERNEL_METHOD
		void lanesRightShift(const IntegerType & n, KernelData<Type,Simd> & result) const noexcept
		{

			for(int i=0;i<Simd;i++)
			{
				const int j = (i+2*Simd-n)&(Simd-1);
				result.data[i] = data[j];
			}
		}

		// shifts lanes (wraps around) left n times in-place
		template<typename IntegerType>
		VECTORIZED_KERNEL_METHOD
		void lanesLeftShift(const IntegerType & n) const noexcept
		{
			alignas(64)
			Type tmp[Simd];
			for(int i=0;i<Simd;i++)
			{
				tmp[i] = data[i];
			}
			for(int i=0;i<Simd;i++)
			{
				const int j = (i+n)&(Simd-1);
				data[i] = tmp[j];
			}
		}

		// shifts lanes (wraps around) left n times in-place
		// n must not be greater than Simd*2
		template<typename IntegerType>
		VECTORIZED_KERNEL_METHOD
		void lanesRightShift(const IntegerType & n) const noexcept
		{
			alignas(64)
			Type tmp[Simd];
			for(int i=0;i<Simd;i++)
			{
				tmp[i] = data[i];
			}
			for(int i=0;i<Simd;i++)
			{
				const int j = (i+2*Simd-n)&(Simd-1);
				data[i] = tmp[j];
			}
		}

		// gets value from a so-called thread in the current SIMD
		// for main body of kernel launch, lane must not overflow Simd
		// for the tail the number of lanes is 1 so the only available lane is 0 that is itself
		VECTORIZED_KERNEL_METHOD
		void broadcastFromLane(const int lane) noexcept
		{
            const Type bcast = data[lane];
            VECTORIZED_KERNEL_LOOP
			for(int i=0;i<Simd;i++)
			{
				data[i] = bcast;
			}
		}

		// same as broadcastFromLane(lane) but the target to copy is a result vector
		VECTORIZED_KERNEL_METHOD
		void broadcastFromLaneToVector(const int lane, KernelData<Type,Simd> & result) noexcept
		{
            const Type bcast = data[lane];
            VECTORIZED_KERNEL_LOOP
			for(int i=0;i<Simd;i++)
			{
				result.data[i] = bcast;
			}
		}

		VECTORIZED_KERNEL_METHOD
		void readFrom(const KernelData<Type,Simd> & vec) noexcept
		{
			VECTORIZED_KERNEL_LOOP
			for(int i=0;i<Simd;i++)
			{
				data[i] = vec.data[i];
			}
		}

		template<typename NewType>
		VECTORIZED_KERNEL_METHOD
		void castAndCopyTo(KernelData<NewType,Simd> & result) const noexcept
		{
			VECTORIZED_KERNEL_LOOP
			for(int i=0;i<Simd;i++)
			{
				result.data[i] = (NewType)data[i];
			}
		}

		template<typename NewType>
		VECTORIZED_KERNEL_METHOD
		void castBitwiseAndCopyTo(KernelData<NewType,Simd> & result) const noexcept
		{
			std::memcpy(result.data,data,sizeof(NewType)*Simd);
		}


		// optimized for [-any,+any] input range!!
		VECTORIZED_KERNEL_METHOD
		void cosFastFullRange(KernelData<Type,Simd> & result) const noexcept
		{
			// reduce range to [-pi,+pi] by modf(input, 2pi) - pi { at high precision }
			// divide by 4 (multiply 0.25)
			// compute on [-1,+1] range
			// compute T4(cos(x)) chebyshev (   8cos(x)^4 - 8cos(x)^2 + 1   )
			// return

            alignas(64)
            double wrapAroundHighPrecision[Simd];


            alignas(64)
            Type reducedData[Simd];

            alignas(64)
            Type xSqr[Simd];

            alignas(64)
            Type xSqrSqr[Simd];

            alignas(64)
            Type xSqrSqrSqr[Simd];

            alignas(64)
            Type xSqrSqrSqrSqr[Simd];

            // these have to be as high precision as possible to let wide-range of inputs be used
            constexpr double pi =  /*Type(std::acos(-1));*/ double(3.1415926535897932384626433832795028841971693993751058209749445923);
            constexpr Type piLowPrec = pi;
            constexpr double twoPi = double(2.0 * pi);
            constexpr double twoPiInv = double(1.0/twoPi);

			VECTORIZED_KERNEL_LOOP
			for(int i=0;i<Simd;i++)
			{
				wrapAroundHighPrecision[i] = data[i];
			}

			VECTORIZED_KERNEL_LOOP
			for(int i=0;i<Simd;i++)
			{
				reducedData[i] = wrapAroundHighPrecision[i] - twoPi * std::floor(wrapAroundHighPrecision[i] * twoPiInv);
			}

			VECTORIZED_KERNEL_LOOP
			for(int i=0;i<Simd;i++)
			{
				reducedData[i]=reducedData[i]<Type(0.0)?reducedData[i]-twoPi:reducedData[i];
			}

			VECTORIZED_KERNEL_LOOP
			for(int i=0;i<Simd;i++)
			{
				reducedData[i] = reducedData[i] - piLowPrec;
			}


			VECTORIZED_KERNEL_LOOP
			for(int i=0;i<Simd;i++)
			{
				reducedData[i] = reducedData[i]*Type(0.25);
			}


			VECTORIZED_KERNEL_LOOP
			for(int i=0;i<Simd;i++)
			{
				xSqr[i] = 	reducedData[i]*reducedData[i];
			}

			VECTORIZED_KERNEL_LOOP
			for(int i=0;i<Simd;i++)
			{
				xSqrSqr[i] = 	xSqr[i]*xSqr[i];
			}

            VECTORIZED_KERNEL_LOOP
			for(int i=0;i<Simd;i++)
			{
				xSqrSqrSqr[i] = 	xSqrSqr[i]*xSqr[i];
			}


            VECTORIZED_KERNEL_LOOP
			for(int i=0;i<Simd;i++)
			{
				xSqrSqrSqrSqr[i] = 	xSqrSqr[i]*xSqrSqr[i];
			}

			VECTORIZED_KERNEL_LOOP
			for(int i=0;i<Simd;i++)
			{
				result.data[i] = 	Type(-3.814697265625e-06)*xSqrSqrSqrSqr[i] +
									Type(-0.00133228302001953125)*xSqrSqrSqr[i] +
									Type(0.041629791259765625)*xSqrSqr[i] +
									Type(-0.49999141693115234375)*xSqr[i] +
									Type(0.999999523162841796875);
			}

			VECTORIZED_KERNEL_LOOP
			for(int i=0;i<Simd;i++)
			{
				result.data[i] = 	result.data[i]*result.data[i];
			}

			VECTORIZED_KERNEL_LOOP
			for(int i=0;i<Simd;i++)
			{
				result.data[i] = 	Type(8.0)*(result.data[i]*result.data[i] - result.data[i]) + Type(1.0);
			}

			VECTORIZED_KERNEL_LOOP
			for(int i=0;i<Simd;i++)
			{
				result.data[i] = 	-result.data[i];
			}
		}

		// only optimized for [-1,1] input range!!
		// Chebyshev Polynomial coefficients found by genetic algorithm running on 3 GPUs in 2 minutes
		VECTORIZED_KERNEL_METHOD
		void cosFast(KernelData<Type,Simd> & result) const noexcept
		{
            alignas(64)
            Type xSqr[Simd];

            alignas(64)
            Type xSqrSqr[Simd];

            alignas(64)
            Type xSqrSqrSqr[Simd];

            alignas(64)
            Type xSqrSqrSqrSqr[Simd];

			VECTORIZED_KERNEL_LOOP
			for(int i=0;i<Simd;i++)
			{
				xSqr[i] = 	data[i]*data[i];
			}

			VECTORIZED_KERNEL_LOOP
			for(int i=0;i<Simd;i++)
			{
				xSqrSqr[i] = 	xSqr[i]*xSqr[i];
			}

            VECTORIZED_KERNEL_LOOP
			for(int i=0;i<Simd;i++)
			{
				xSqrSqrSqr[i] = 	xSqrSqr[i]*xSqr[i];
			}


            VECTORIZED_KERNEL_LOOP
			for(int i=0;i<Simd;i++)
			{
				xSqrSqrSqrSqr[i] = 	xSqrSqr[i]*xSqrSqr[i];
			}

			VECTORIZED_KERNEL_LOOP
			for(int i=0;i<Simd;i++)
			{
				result.data[i] = 	Type(-3.814697265625e-06)*xSqrSqrSqrSqr[i] +
									Type(-0.00133228302001953125)*xSqrSqrSqr[i] +
									Type(0.041629791259765625)*xSqrSqr[i] +
									Type(-0.49999141693115234375)*xSqr[i] +
									Type(0.999999523162841796875);
			}
		}


		VECTORIZED_KERNEL_METHOD
		void cos(KernelData<Type,Simd> & result) const noexcept
		{
			VECTORIZED_KERNEL_LOOP
			for(int i=0;i<Simd;i++)
			{
				result.data[i] = 	std::cos(data[i]);
			}
		}

        // only optimized for [-any,any] input range!!
        // Chebyshev Polynomial coefficients found by genetic algorithm running on 768 GPU pipelines for 5 minutes
        VECTORIZED_KERNEL_METHOD
        void sinFastFullRange(KernelData<Type,Simd> & result) const noexcept
        {
            alignas(64)
			double wrapAroundHighPrecision[Simd];

            alignas(64)
			double wrapAroundHighPrecisionTmp[Simd];

            alignas(64)
			double wrapAroundHighPrecisionTmp2[Simd];

            alignas(64)
            Type reducedData[Simd];

            alignas(64)
            Type xSqr[Simd];

            alignas(64)
            Type xSqrSqr[Simd];

            alignas(64)
            Type xSqrSqr5[Simd];

            alignas(64)
            Type xSqrSqr8[Simd];

            alignas(64)
            Type tmp[Simd];


            // these have to be as high precision as possible to let wide-range of inputs be used
            constexpr double pi =  /*Type(std::acos(-1));*/ double(3.1415926535897932384626433832795028841971693993751058209749445923);
            constexpr Type piLowPrec = pi;
            constexpr double twoPi = double(2.0 * pi);
            constexpr double twoPiLowPrec = double(2.0 * pi);
            constexpr double twoPiInv = double(1.0/twoPi);

			VECTORIZED_KERNEL_LOOP
			for(int i=0;i<Simd;i++)
			{
				wrapAroundHighPrecision[i] = data[i];
			}


			VECTORIZED_KERNEL_LOOP
			for(int i=0;i<Simd;i++)
			{
				wrapAroundHighPrecisionTmp[i] =  wrapAroundHighPrecision[i] * twoPiInv;
			}

			VECTORIZED_KERNEL_LOOP
			for(int i=0;i<Simd;i++)
			{
				wrapAroundHighPrecisionTmp[i] =  std::floor(wrapAroundHighPrecisionTmp[i]);
			}

			VECTORIZED_KERNEL_LOOP
			for(int i=0;i<Simd;i++)
			{
				wrapAroundHighPrecisionTmp2[i] = twoPi * wrapAroundHighPrecisionTmp[i];
			}

			VECTORIZED_KERNEL_LOOP
			for(int i=0;i<Simd;i++)
			{
				reducedData[i] = wrapAroundHighPrecision[i] - wrapAroundHighPrecisionTmp2[i];
			}


			VECTORIZED_KERNEL_LOOP
			for(int i=0;i<Simd;i++)
			{
				tmp[i]=reducedData[i]<Type(0.0);
			}


			VECTORIZED_KERNEL_LOOP
			for(int i=0;i<Simd;i++)
			{
				xSqr[i]=reducedData[i]-twoPiLowPrec;
			}


			VECTORIZED_KERNEL_LOOP
			for(int i=0;i<Simd;i++)
			{
				reducedData[i]=tmp[i]?xSqr[i]:reducedData[i];
			}

			VECTORIZED_KERNEL_LOOP
			for(int i=0;i<Simd;i++)
			{
				reducedData[i] = reducedData[i] - piLowPrec;
			}


			VECTORIZED_KERNEL_LOOP
			for(int i=0;i<Simd;i++)
			{
				reducedData[i] = reducedData[i]*Type(0.2);
			}



            VECTORIZED_KERNEL_LOOP
            for(int i=0;i<Simd;i++)
            {
                xSqr[i] =   reducedData[i]*reducedData[i];
            }

            VECTORIZED_KERNEL_LOOP
            for(int i=0;i<Simd;i++)
            {
                xSqrSqr[i] =    xSqr[i]*xSqr[i];
            }

            VECTORIZED_KERNEL_LOOP
            for(int i=0;i<Simd;i++)
            {
                xSqrSqr5[i] =   xSqrSqr[i]*reducedData[i];
            }


            VECTORIZED_KERNEL_LOOP
            for(int i=0;i<Simd;i++)
            {
                xSqrSqr8[i] =   xSqrSqr[i]*xSqrSqr[i];
            }


            VECTORIZED_KERNEL_LOOP
            for(int i=0;i<Simd;i++)
            {
                xSqrSqr8[i] =   xSqrSqr8[i]*reducedData[i] ;
            }


            VECTORIZED_KERNEL_LOOP
            for(int i=0;i<Simd;i++)
            {
                tmp[i] =   xSqrSqr5[i]*xSqr[i] ;
            }

            VECTORIZED_KERNEL_LOOP
            for(int i=0;i<Simd;i++)
            {
                xSqr[i] =   xSqr[i]*reducedData[i];
            }

            VECTORIZED_KERNEL_LOOP
            for(int i=0;i<Simd;i++)
            {
                result.data[i]  =   Type(-0.0005664825439453125)*xSqrSqr8[i] +
                                    Type(0.001037120819091796875)*tmp[i] +
                                    Type(0.007439136505126953125)*xSqrSqr5[i] +
                                    Type(-0.166426181793212890625)*xSqr[i] +
                                    Type(0.999982357025146484375)*reducedData[i];
            }


            VECTORIZED_KERNEL_LOOP
            for(int i=0;i<Simd;i++)
            {
                xSqr[i] =   result.data[i]*result.data[i];
            }

            VECTORIZED_KERNEL_LOOP
            for(int i=0;i<Simd;i++)
            {
                xSqrSqr[i] =   xSqr[i]*result.data[i];
            }

            VECTORIZED_KERNEL_LOOP
            for(int i=0;i<Simd;i++)
            {
                xSqrSqr8[i] =   xSqr[i]*xSqr[i];
            }

            VECTORIZED_KERNEL_LOOP
            for(int i=0;i<Simd;i++)
            {
                xSqrSqr5[i] =   xSqrSqr8[i]*result.data[i];
            }

			VECTORIZED_KERNEL_LOOP
			for(int i=0;i<Simd;i++)
			{
				result.data[i] = 	Type(16.0)*xSqrSqr5[i]  -
						Type(20.0)*xSqrSqr[i] +
						Type(5.0)*result.data[i];
			}

			VECTORIZED_KERNEL_LOOP
			for(int i=0;i<Simd;i++)
			{
				result.data[i] = 	-result.data[i];
			}

        }

        // only optimized for [-1,1] input range!!
        // Chebyshev Polynomial coefficients found by genetic algorithm running on 768 GPU pipelines for 5 minutes
        VECTORIZED_KERNEL_METHOD
        void sinFast(KernelData<Type,Simd> & result) const noexcept
        {

            alignas(64)
            Type xSqr[Simd];

            alignas(64)
            Type xSqrSqr[Simd];

            alignas(64)
            Type xSqrSqr5[Simd];

            alignas(64)
            Type xSqrSqr8[Simd];

            alignas(64)
            Type tmp[Simd];

            VECTORIZED_KERNEL_LOOP
            for(int i=0;i<Simd;i++)
            {
                xSqr[i] =   data[i]*data[i];
            }

            VECTORIZED_KERNEL_LOOP
            for(int i=0;i<Simd;i++)
            {
                xSqrSqr[i] =    xSqr[i]*xSqr[i];
            }

            VECTORIZED_KERNEL_LOOP
            for(int i=0;i<Simd;i++)
            {
                xSqrSqr5[i] =   xSqrSqr[i]*data[i];
            }


            VECTORIZED_KERNEL_LOOP
            for(int i=0;i<Simd;i++)
            {
                xSqrSqr8[i] =   xSqrSqr[i]*xSqrSqr[i];
            }


            VECTORIZED_KERNEL_LOOP
            for(int i=0;i<Simd;i++)
            {
                xSqrSqr8[i] =   xSqrSqr8[i]*data[i] ;
            }


            VECTORIZED_KERNEL_LOOP
            for(int i=0;i<Simd;i++)
            {
                tmp[i] =   xSqrSqr5[i]*xSqr[i] ;
            }

            VECTORIZED_KERNEL_LOOP
            for(int i=0;i<Simd;i++)
            {
                xSqr[i] =   xSqr[i]*data[i];
            }

            VECTORIZED_KERNEL_LOOP
            for(int i=0;i<Simd;i++)
            {
                result.data[i]  =   Type(-0.0005664825439453125)*xSqrSqr8[i] +
                                    Type(0.001037120819091796875)*tmp[i] +
                                    Type(0.007439136505126953125)*xSqrSqr5[i] +
                                    Type(-0.166426181793212890625)*xSqr[i] +
                                    Type(0.999982357025146484375)*data[i];
            }

        }

		VECTORIZED_KERNEL_METHOD
		void sin(KernelData<Type,Simd> & result) const noexcept
		{
			VECTORIZED_KERNEL_LOOP
			for(int i=0;i<Simd;i++)
			{
				result.data[i] = 	std::sin(data[i]);
			}
		}

        // only optimized for [-1,1] input range!!
        // Chebyshev Polynomial coefficients found by genetic algorithm running on 768 GPU pipelines for 4 minutes
        VECTORIZED_KERNEL_METHOD
        void expFast(KernelData<Type,Simd> & result) const noexcept
        {

            alignas(64)
            Type xSqr2[Simd];

            alignas(64)
            Type xSqr3[Simd];

            alignas(64)
            Type xSqr4[Simd];

            alignas(64)
            Type xSqr5[Simd];

            alignas(64)
            Type xSqr6[Simd];

            VECTORIZED_KERNEL_LOOP
            for(int i=0;i<Simd;i++)
            {
                xSqr2[i] =   data[i]*data[i];
            }

            VECTORIZED_KERNEL_LOOP
            for(int i=0;i<Simd;i++)
            {
                xSqr3[i] =   xSqr2[i]*data[i];
            }

            VECTORIZED_KERNEL_LOOP
            for(int i=0;i<Simd;i++)
            {
                xSqr4[i] =   xSqr2[i]*xSqr2[i];
            }

            VECTORIZED_KERNEL_LOOP
            for(int i=0;i<Simd;i++)
            {
                xSqr5[i] =   xSqr3[i]*xSqr2[i];
            }

            VECTORIZED_KERNEL_LOOP
            for(int i=0;i<Simd;i++)
            {
                xSqr6[i] =   xSqr3[i]*xSqr3[i];
            }

            VECTORIZED_KERNEL_LOOP
            for(int i=0;i<Simd;i++)
            {
                result.data[i]  =   Type(0.0014438629150390625)*xSqr6[i] +
                                    Type(0.00861072540283203125)*xSqr5[i] +
                                    Type(0.041627407073974609375)*xSqr4[i] +
                                    Type(0.16657924652099609375)*xSqr3[i] +
                                    Type(0.5000095367431640625)*xSqr2[i]+
                                    Type(0.999999523162841796875)*data[i]+
									Type(0.999999523162841796875); // no typo here, genetic algorithm found same value
            }

        }

		VECTORIZED_KERNEL_METHOD
		void exp(KernelData<Type,Simd> & result) const noexcept
		{
			VECTORIZED_KERNEL_LOOP
			for(int i=0;i<Simd;i++)
			{
				result.data[i] = 	std::exp(data[i]);
			}
		}


		VECTORIZED_KERNEL_METHOD
		void sqrt(KernelData<Type,Simd> & result) const noexcept
		{
			VECTORIZED_KERNEL_LOOP
			for(int i=0;i<Simd;i++)
			{
				result.data[i] = std::sqrt(data[i]);
			}
		}

		VECTORIZED_KERNEL_METHOD
		void rsqrt(KernelData<Type,Simd> & result) const noexcept
		{
			VECTORIZED_KERNEL_LOOP
			for(int i=0;i<Simd;i++)
			{
				result.data[i] = 1.0f/std::sqrt(data[i]);
			}
		}

		VECTORIZED_KERNEL_METHOD
		void add(const KernelData<Type,Simd> & vec, KernelData<Type,Simd> & result) const noexcept
		{
			VECTORIZED_KERNEL_LOOP
			for(int i=0;i<Simd;i++)
			{
				result.data[i] = data[i] + vec.data[i];
			}
		}

		VECTORIZED_KERNEL_METHOD
		void add(const Type & val, KernelData<Type,Simd> & result) const noexcept
		{
			VECTORIZED_KERNEL_LOOP
			for(int i=0;i<Simd;i++)
			{
				result.data[i] = data[i] + val;
			}
		}

		VECTORIZED_KERNEL_METHOD
		void sub(const KernelData<Type,Simd> & vec, KernelData<Type,Simd> & result) const noexcept
		{
			VECTORIZED_KERNEL_LOOP
			for(int i=0;i<Simd;i++)
			{
				result.data[i] = data[i] - vec.data[i];
			}
		}

		VECTORIZED_KERNEL_METHOD
		void sub(const Type & val, KernelData<Type,Simd> & result) const noexcept
		{
			VECTORIZED_KERNEL_LOOP
			for(int i=0;i<Simd;i++)
			{
				result.data[i] = data[i] - val;
			}
		}

		VECTORIZED_KERNEL_METHOD
		void div(const KernelData<Type,Simd> & vec, KernelData<Type,Simd> & result) const noexcept
		{
			VECTORIZED_KERNEL_LOOP
			for(int i=0;i<Simd;i++)
			{
				result.data[i] = data[i] / vec.data[i];
			}
		}

		VECTORIZED_KERNEL_METHOD
		void div(const Type & val, KernelData<Type,Simd> & result) const noexcept
		{
			VECTORIZED_KERNEL_LOOP
			for(int i=0;i<Simd;i++)
			{
				result.data[i] = data[i] / val;
			}
		}


		VECTORIZED_KERNEL_METHOD
		void fusedMultiplyAdd(const KernelData<Type,Simd> & vec1, const KernelData<Type,Simd> & vec2, KernelData<Type,Simd> & result) const noexcept
		{
			VECTORIZED_KERNEL_LOOP
			for(int i=0;i<Simd;i++)
			{
				result.data[i] = (data[i]* vec1.data[i]+ vec2.data[i]);
			}
		}

		VECTORIZED_KERNEL_METHOD
		void fusedMultiplyAdd(const KernelData<Type,Simd> & vec1, const Type & val2, KernelData<Type,Simd> & result) const noexcept
		{
			VECTORIZED_KERNEL_LOOP
			for(int i=0;i<Simd;i++)
			{
				result.data[i] = (data[i]* vec1.data[i]+ val2);
			}
		}

		VECTORIZED_KERNEL_METHOD
		void fusedMultiplyAdd(const Type & val1, const KernelData<Type,Simd> & vec2, KernelData<Type,Simd> & result) const noexcept
		{
			VECTORIZED_KERNEL_LOOP
			for(int i=0;i<Simd;i++)
			{
				result.data[i] = (data[i]* val1+ vec2.data[i]);
			}
		}

		VECTORIZED_KERNEL_METHOD
		void fusedMultiplyAdd(const Type & val1, const Type & val2, KernelData<Type,Simd> & result) const noexcept
		{
			VECTORIZED_KERNEL_LOOP
			for(int i=0;i<Simd;i++)
			{
				result.data[i] = (data[i]* val1+ val2);
			}
		}

		VECTORIZED_KERNEL_METHOD
		void fusedMultiplySub(const KernelData<Type,Simd> & vec1, const KernelData<Type,Simd> & vec2, KernelData<Type,Simd> & result) const noexcept
		{
			VECTORIZED_KERNEL_LOOP
			for(int i=0;i<Simd;i++)
			{
				result.data[i] = (data[i]* vec1.data[i] -vec2.data[i]);
			}
		}

		VECTORIZED_KERNEL_METHOD
		void fusedMultiplySub(const Type & val1, const KernelData<Type,Simd> & vec2, KernelData<Type,Simd> & result) const noexcept
		{
			VECTORIZED_KERNEL_LOOP
			for(int i=0;i<Simd;i++)
			{
				result.data[i] = (data[i]* val1 -vec2.data[i]);
			}
		}

		VECTORIZED_KERNEL_METHOD
		void fusedMultiplySub(const KernelData<Type,Simd> & vec1, const Type & val2, KernelData<Type,Simd> & result) const noexcept
		{
			VECTORIZED_KERNEL_LOOP
			for(int i=0;i<Simd;i++)
			{
				result.data[i] = (data[i]* vec1.data[i] -val2);
			}
		}

		VECTORIZED_KERNEL_METHOD
		void fusedMultiplySub(const Type & val1, const Type & val2, KernelData<Type,Simd> & result) const noexcept
		{
			VECTORIZED_KERNEL_LOOP
			for(int i=0;i<Simd;i++)
			{
				result.data[i] = (data[i]* val1 -val2);
			}
		}

		VECTORIZED_KERNEL_METHOD
		void mul(const KernelData<Type,Simd> & vec, KernelData<Type,Simd> & result) const noexcept
		{
			VECTORIZED_KERNEL_LOOP
			for(int i=0;i<Simd;i++)
			{
				result.data[i] = data[i] * vec.data[i];
			}
		}

		VECTORIZED_KERNEL_METHOD
		void mul(const Type & val, KernelData<Type,Simd> & result) const noexcept
		{
			VECTORIZED_KERNEL_LOOP
			for(int i=0;i<Simd;i++)
			{
				result.data[i] = data[i] * val;
			}
		}


		VECTORIZED_KERNEL_METHOD
		void modulus(const KernelData<Type,Simd> & vec, KernelData<Type,Simd> & result) const noexcept
		{
			VECTORIZED_KERNEL_LOOP
			for(int i=0;i<Simd;i++)
			{
				result.data[i] = data[i] % vec.data[i];
			}
		}

		VECTORIZED_KERNEL_METHOD
		void modulus(const Type & val, KernelData<Type,Simd> & result) const noexcept
		{
			VECTORIZED_KERNEL_LOOP
			for(int i=0;i<Simd;i++)
			{
				result.data[i] = data[i] % val;
			}
		}

		VECTORIZED_KERNEL_METHOD
		void leftShift(const KernelData<Type,Simd> & vec, KernelData<Type,Simd> & result) const noexcept
		{
			VECTORIZED_KERNEL_LOOP
			for(int i=0;i<Simd;i++)
			{
				result.data[i] = data[i] << vec.data[i];
			}
		}

		VECTORIZED_KERNEL_METHOD
		void leftShift(const Type & val, KernelData<Type,Simd> & result) const noexcept
		{
			VECTORIZED_KERNEL_LOOP
			for(int i=0;i<Simd;i++)
			{
				result.data[i] = data[i] << val;
			}
		}

		VECTORIZED_KERNEL_METHOD
		void rightShift(const KernelData<Type,Simd> & vec, KernelData<Type,Simd> & result) const noexcept
		{
			VECTORIZED_KERNEL_LOOP
			for(int i=0;i<Simd;i++)
			{
				result.data[i] = data[i] >> vec.data[i];
			}
		}

		VECTORIZED_KERNEL_METHOD
		void rightShift(const Type & val, KernelData<Type,Simd> & result) const noexcept
		{
			VECTORIZED_KERNEL_LOOP
			for(int i=0;i<Simd;i++)
			{
				result.data[i] = data[i] >> val;
			}
		}

		// this function is not accelerated. use it sparsely.
		VECTORIZED_KERNEL_METHOD
		void pow(const KernelData<Type,Simd> & vec, KernelData<Type,Simd> & result) const noexcept
		{
			VECTORIZED_KERNEL_LOOP
			for(int i=0;i<Simd;i++)
			{
				result.data[i] = std::pow(data[i],vec.data[i]);
			}
		}

		// this function is not accelerated. use it sparsely.
		// x^y = x.pow(y,result)
		VECTORIZED_KERNEL_METHOD
		void pow(const Type & val, KernelData<Type,Simd> & result) const noexcept
		{
			VECTORIZED_KERNEL_LOOP
			for(int i=0;i<Simd;i++)
			{
				result.data[i] = std::pow(data[i],val);
			}
		}

		// this function is not accelerated. use it sparsely.
		// computes y^x = x.powFrom(y,result) is called
		VECTORIZED_KERNEL_METHOD
		void powFrom(const Type & val, KernelData<Type,Simd> & result) const noexcept
		{
			VECTORIZED_KERNEL_LOOP
			for(int i=0;i<Simd;i++)
			{
				result.data[i] = std::pow(val,data[i]);
			}
		}




		// this function is not accelerated. use it sparsely.
		VECTORIZED_KERNEL_METHOD
		void log(KernelData<Type,Simd> & result) const noexcept
		{
			VECTORIZED_KERNEL_LOOP
			for(int i=0;i<Simd;i++)
			{
				result.data[i] = std::log(data[i]);
			}
		}

		// this function is not accelerated. use it sparsely.
		VECTORIZED_KERNEL_METHOD
		void log2(KernelData<Type,Simd> & result) const noexcept
		{
			VECTORIZED_KERNEL_LOOP
			for(int i=0;i<Simd;i++)
			{
				result.data[i] = std::log2(data[i]);
			}
		}


		VECTORIZED_KERNEL_METHOD
		void bitwiseXor(const KernelData<Type,Simd> & vec, KernelData<Type,Simd> & result) const noexcept
		{
			VECTORIZED_KERNEL_LOOP
			for(int i=0;i<Simd;i++)
			{
				result.data[i] = data[i] ^ vec.data[i];
			}
		}

		VECTORIZED_KERNEL_METHOD
		void bitwiseXor(const Type & val, KernelData<Type,Simd> & result) const noexcept
		{
			VECTORIZED_KERNEL_LOOP
			for(int i=0;i<Simd;i++)
			{
				result.data[i] = data[i] ^ val;
			}
		}

		VECTORIZED_KERNEL_METHOD
		void bitwiseAnd(const KernelData<Type,Simd> & vec, KernelData<Type,Simd> & result) const noexcept
		{
			VECTORIZED_KERNEL_LOOP
			for(int i=0;i<Simd;i++)
			{
				result.data[i] = data[i] & vec.data[i];
			}
		}

		VECTORIZED_KERNEL_METHOD
		void bitwiseAnd(const Type & val, KernelData<Type,Simd> & result) const noexcept
		{
			VECTORIZED_KERNEL_LOOP
			for(int i=0;i<Simd;i++)
			{
				result.data[i] = data[i] & val;
			}
		}



		VECTORIZED_KERNEL_METHOD
		void bitwiseOr(const KernelData<Type,Simd> & vec, KernelData<Type,Simd> & result) const noexcept
		{
			VECTORIZED_KERNEL_LOOP
			for(int i=0;i<Simd;i++)
			{
				result.data[i] = data[i] | vec.data[i];
			}
		}

		VECTORIZED_KERNEL_METHOD
		void bitwiseOr(const Type & val, KernelData<Type,Simd> & result) const noexcept
		{
			VECTORIZED_KERNEL_LOOP
			for(int i=0;i<Simd;i++)
			{
				result.data[i] = data[i] | val;
			}
		}
		VECTORIZED_KERNEL_METHOD
		void bitwiseNot(KernelData<Type,Simd> & result) const noexcept
		{
			VECTORIZED_KERNEL_LOOP
			for(int i=0;i<Simd;i++)
			{
				result.data[i] = ~data[i];
			}
		}

		VECTORIZED_KERNEL_METHOD
		void logicalNot(KernelData<Type,Simd> & result) const noexcept
		{
			VECTORIZED_KERNEL_LOOP
			for(int i=0;i<Simd;i++)
			{
				result.data[i] = !data[i];
			}
		}

		VECTORIZED_KERNEL_METHOD
		void factorial(KernelData<Type,Simd> & result) const noexcept
		{
			alignas(64)
			Type tmpC[Simd];
			alignas(64)
			Type tmpD[Simd];
			alignas(64)
            Type tmpE[Simd];

			VECTORIZED_KERNEL_LOOP
			for(int i=0;i<Simd;i++)
			{
				tmpC[i]=data[i];
			}

			VECTORIZED_KERNEL_LOOP
			for(int i=0;i<Simd;i++)
			{
				tmpD[i]=data[i]-(Type)1;
			}
			int mask[Simd];


			int anyWorking = true;
			while(anyWorking)
			{
				anyWorking = false;

				VECTORIZED_KERNEL_LOOP
				for(int i=0;i<Simd;i++)
				{
					mask[i] = (tmpD[i]>0);
				}

				VECTORIZED_KERNEL_LOOP
				for(int i=0;i<Simd;i++)
				{
					anyWorking += mask[i];
				}

				VECTORIZED_KERNEL_LOOP
				for(int i=0;i<Simd;i++)
				{
					tmpE[i] =  tmpC[i] * tmpD[i];
				}

				VECTORIZED_KERNEL_LOOP
				for(int i=0;i<Simd;i++)
				{
					tmpC[i] = mask[i] ? tmpE[i] : tmpC[i];
				}

				VECTORIZED_KERNEL_LOOP
				for(int i=0;i<Simd;i++)
				{
					tmpD[i]--;
				}

			}

			VECTORIZED_KERNEL_LOOP
			for(int i=0;i<Simd;i++)
			{
				result.data[i] = tmpC[i]?tmpC[i]:1;
			}
		}
	};


	template<typename Type,int Simd,int ArraySize>
	struct KernelDataArray
	{
		KernelData<Type,Simd> arr[ArraySize];
		KernelData<Type,Simd> & operator[](const int index)
		{
			return arr[index];
		}
	};


	template<int CurrentSimd>
	struct KernelDataFactory
	{
		KernelDataFactory():width(CurrentSimd)
		{

		}


		template<typename Type>
		inline
		KernelData<Type,CurrentSimd> generate() const
		{
			return KernelData<Type,CurrentSimd>();
		}


		template<typename Type>
		inline
		KernelData<Type,CurrentSimd> generate(const KernelData<Type,CurrentSimd> & vec) const
		{
			return KernelData<Type,CurrentSimd>(vec);
		}


		// size has to be compile-time known otherwise it won't work
		template<typename Type,int Size>
		inline
		KernelDataArray<Type,CurrentSimd,Size> generateArray() const
		{
			return KernelDataArray<Type,CurrentSimd,Size>();
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

		template<int numThreads, int loadBalanceResolution = 100>
		void runMultithreadedLoadBalanced(int n, Args... args)
		{
			const int nLoop = (n/SimdWidth);
			const KernelDataFactory<SimdWidth> factory;

			// simple work scheduling.
			std::vector<std::thread> threads;
			const int nChunk = 1 + (nLoop/numThreads)/loadBalanceResolution;
			std::atomic<int> index;
			index.store(0);
			for(int ii=0;ii<numThreads;ii++)
			{
				threads.emplace_back([&,ii](){

					bool work = true;
					while(work)
					{
						work   = false;
						const int curIndex = index.fetch_add(nChunk);
						work   = (curIndex<nLoop);

						for(int j=0;j<nChunk;j++)
						{
							const int i = curIndex+j;
							if(i>=nLoop)
								break;

							auto id = factory.template generate<int>();
							id.idCompute(i*SimdWidth,[](const int prm){ return prm;});
							kernel(factory, id, args...);
						}
					}
				});
			}

			for(int i=0;i<threads.size();i++)
			{
				threads[i].join(); // this is a synchronization point for the data changes
			}






			// then do the tail computation serially (assume simd is not half of a big work)
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

		template<int numThreads>
		void runMultithreaded(int n, Args... args)
		{
			const int nLoop = (n/SimdWidth);
			const KernelDataFactory<SimdWidth> factory;


#ifdef _OPENMP
#include<omp.h>
			// distribute to threads by openmp
			#pragma omp parallel for num_threads(numThreads)
			for(int i=0;i<nLoop;i++)
			{
				auto id = factory.template generate<int>();
				id.idCompute(i*SimdWidth,[](const int prm){ return prm;});
				kernel(factory, id, args...);
			}
#else
			// simple work scheduling.
			std::vector<std::thread> threads;
			const int nChunk = numThreads>0?(1 + nLoop/numThreads):nLoop;

			for(int ii=0;ii<nLoop;ii+=nChunk)
			{
				threads.emplace_back([&,ii](){
					for(int j=0;j<nChunk;j++)
					{
						const int i = ii+j;
						if(i>=nLoop)
							break;

						auto id = factory.template generate<int>();
						id.idCompute(i*SimdWidth,[](const int prm){ return prm;});
						kernel(factory, id, args...);
					}
				});
			}

			for(int i=0;i<threads.size();i++)
			{
				threads[i].join(); // this is a synchronization point for the data changes
			}

#endif




			// then do the tail computation serially (assume simd is not half of a big work)
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
		std::vector<double> threadPerformances;
		std::vector<double> threadPerformancesOld;
	};

	template<int SimdWidth, typename F, class...Args>
	auto createKernel(F&& kernelPrm, KernelArgs<Args...> const& _prm_)
	{
		return Kernel<SimdWidth, F, Args...>(std::forward<F>(kernelPrm));
	}

}


#endif /* VECTORIZEDKERNEL_H_ */
