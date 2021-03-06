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
		    double wrapAroundHighPrecisionTmp[Simd];

		    alignas(64)
		    double reducedData[Simd];


		    alignas(64)
		    double reducedDataTmp[Simd];

		    alignas(64)
		    Type xSqr[Simd];


            alignas(64)
            Type resultData[Simd];


		    // these have to be as high precision as possible to let wide-range of inputs be used
		    constexpr double pi =  /*Type(std::acos(-1));*/ double(3.1415926535897932384626433832795028841971693993751058209749445923);

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
		        wrapAroundHighPrecisionTmp[i] = wrapAroundHighPrecision[i] * twoPiInv;
		    }

		    VECTORIZED_KERNEL_LOOP
		    for(int i=0;i<Simd;i++)
		    {
		        wrapAroundHighPrecisionTmp[i] = std::floor(wrapAroundHighPrecisionTmp[i]);
		    }

		    VECTORIZED_KERNEL_LOOP
		    for(int i=0;i<Simd;i++)
		    {
		        wrapAroundHighPrecisionTmp[i] = twoPi*wrapAroundHighPrecisionTmp[i];
		    }

		    VECTORIZED_KERNEL_LOOP
		    for(int i=0;i<Simd;i++)
		    {
		        reducedData[i] = wrapAroundHighPrecision[i] - wrapAroundHighPrecisionTmp[i];
		    }


		    VECTORIZED_KERNEL_LOOP
		    for(int i=0;i<Simd;i++)
		    {
		        reducedDataTmp[i] = reducedData[i]-twoPi;
		    }

		    VECTORIZED_KERNEL_LOOP
		    for(int i=0;i<Simd;i++)
		    {
		        reducedData[i]=reducedData[i]<double(0.0)?reducedDataTmp[i]:reducedData[i];
		    }

		    VECTORIZED_KERNEL_LOOP
		    for(int i=0;i<Simd;i++)
		    {
		        reducedData[i] = reducedData[i] - pi;
		    }


		    VECTORIZED_KERNEL_LOOP
		    for(int i=0;i<Simd;i++)
		    {
		        reducedData[i] = reducedData[i]*double(0.25);
		    }

		    VECTORIZED_KERNEL_LOOP
		    for(int i=0;i<Simd;i++)
		    {
		        reducedData[i] = 	reducedData[i]*reducedData[i];
		    }

			VECTORIZED_KERNEL_LOOP
			for(int i=0;i<Simd;i++)
			{
				xSqr[i] = 	reducedData[i];
			}



			VECTORIZED_KERNEL_LOOP
			for(int i=0;i<Simd;i++)
			{
				resultData[i] = 	Type(2.375724425540681750135263e-05);
			}


			VECTORIZED_KERNEL_LOOP
			for(int i=0;i<Simd;i++)
			{
				resultData[i] = 	resultData[i]*xSqr[i] + Type(-0.001387603183718333355045615);
			}

			VECTORIZED_KERNEL_LOOP
			for(int i=0;i<Simd;i++)
			{
				resultData[i] = 	resultData[i]*xSqr[i] + Type(0.04166606225906388516477818);
			}

			VECTORIZED_KERNEL_LOOP
			for(int i=0;i<Simd;i++)
			{
				resultData[i] = 	resultData[i]*xSqr[i] + Type(-0.4999999068460709850114654);
			}

			VECTORIZED_KERNEL_LOOP
			for(int i=0;i<Simd;i++)
			{
				resultData[i] = 	resultData[i]*xSqr[i] + Type(0.9999999771350314148321559);
			}


			VECTORIZED_KERNEL_LOOP
			for(int i=0;i<Simd;i++)
			{
				xSqr[i] = 	resultData[i]*resultData[i];
			}


			VECTORIZED_KERNEL_LOOP
			for(int i=0;i<Simd;i++)
			{
				resultData[i] = 	Type(8.0)*xSqr[i] - Type(8.0);
			}


			VECTORIZED_KERNEL_LOOP
			for(int i=0;i<Simd;i++)
			{
				resultData[i] = 	resultData[i]*xSqr[i] + Type(1.0);
			}

			VECTORIZED_KERNEL_LOOP
			for(int i=0;i<Simd;i++)
			{
				result.data[i] = 	-resultData[i];
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
            Type resultData[Simd];

			VECTORIZED_KERNEL_LOOP
			for(int i=0;i<Simd;i++)
			{
				xSqr[i] = 	data[i]*data[i];
			}

			VECTORIZED_KERNEL_LOOP
			for(int i=0;i<Simd;i++)
			{
				resultData[i] = 	Type(2.375724425540681750135263e-05);
			}


			VECTORIZED_KERNEL_LOOP
			for(int i=0;i<Simd;i++)
			{
				resultData[i] = 	resultData[i]*xSqr[i] + Type(-0.001387603183718333355045615);
			}

			VECTORIZED_KERNEL_LOOP
			for(int i=0;i<Simd;i++)
			{
				resultData[i] = 	resultData[i]*xSqr[i] + Type(0.04166606225906388516477818);
			}

			VECTORIZED_KERNEL_LOOP
			for(int i=0;i<Simd;i++)
			{
				resultData[i] = 	resultData[i]*xSqr[i] + Type(-0.4999999068460709850114654);
			}

			VECTORIZED_KERNEL_LOOP
			for(int i=0;i<Simd;i++)
			{
				result.data[i] = 	resultData[i]*xSqr[i] + Type(0.9999999771350314148321559);
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

			// reduce range to [-pi,+pi] by modf(input, 2pi) - pi { at high precision }
			// divide by 5 (multiply 0.2)
			// compute on [-1,+1] range
			// compute T5(cos(x)) chebyshev (   16sin(x)^5 - 20sin(x)^3 + 5sin(x)   )
			// return

		    alignas(64)
		    double wrapAroundHighPrecision[Simd];

		    alignas(64)
		    double wrapAroundHighPrecisionTmp[Simd];

		    alignas(64)
		    double reducedData[Simd];


		    alignas(64)
		    double reducedDataTmp[Simd];

		    alignas(64)
		    Type xSqr[Simd];

		    alignas(64)
		    Type x[Simd];

            alignas(64)
            Type resultData[Simd];


		    // these have to be as high precision as possible to let wide-range of inputs be used
		    constexpr double pi =  /*Type(std::acos(-1));*/ double(3.1415926535897932384626433832795028841971693993751058209749445923);

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
		        wrapAroundHighPrecisionTmp[i] = wrapAroundHighPrecision[i] * twoPiInv;
		    }

		    VECTORIZED_KERNEL_LOOP
		    for(int i=0;i<Simd;i++)
		    {
		        wrapAroundHighPrecisionTmp[i] = std::floor(wrapAroundHighPrecisionTmp[i]);
		    }

		    VECTORIZED_KERNEL_LOOP
		    for(int i=0;i<Simd;i++)
		    {
		        wrapAroundHighPrecisionTmp[i] = twoPi*wrapAroundHighPrecisionTmp[i];
		    }

		    VECTORIZED_KERNEL_LOOP
		    for(int i=0;i<Simd;i++)
		    {
		        reducedData[i] = wrapAroundHighPrecision[i] - wrapAroundHighPrecisionTmp[i];
		    }


		    VECTORIZED_KERNEL_LOOP
		    for(int i=0;i<Simd;i++)
		    {
		        reducedDataTmp[i] = reducedData[i]-twoPi;
		    }

		    VECTORIZED_KERNEL_LOOP
		    for(int i=0;i<Simd;i++)
		    {
		        reducedData[i]=reducedData[i]<double(0.0)?reducedDataTmp[i]:reducedData[i];
		    }

		    VECTORIZED_KERNEL_LOOP
		    for(int i=0;i<Simd;i++)
		    {
		        reducedData[i] = reducedData[i] - pi;
		    }


		    VECTORIZED_KERNEL_LOOP
		    for(int i=0;i<Simd;i++)
		    {
		        reducedData[i] = reducedData[i]*double(0.2);
		    }

		    VECTORIZED_KERNEL_LOOP
		    for(int i=0;i<Simd;i++)
		    {
		        x[i] = 	reducedData[i];
		    }

			VECTORIZED_KERNEL_LOOP
			for(int i=0;i<Simd;i++)
			{
				xSqr[i] = 	x[i]*x[i];
			}




	        VECTORIZED_KERNEL_LOOP
			for(int i=0;i<Simd;i++)
			{
				resultData[i] =    Type(2.543240487540288086165674e-06)*xSqr[i] + Type(-0.0001980781510937390521576162);
			}

			VECTORIZED_KERNEL_LOOP
			for(int i=0;i<Simd;i++)
			{
				resultData[i] =    resultData[i]*xSqr[i] + Type(0.008333159571549231259268709);
			}

			VECTORIZED_KERNEL_LOOP
			for(int i=0;i<Simd;i++)
			{
				resultData[i] =    resultData[i]*xSqr[i] + Type(-0.1666666483147380972695828);
			}

			VECTORIZED_KERNEL_LOOP
			for(int i=0;i<Simd;i++)
			{
				resultData[i] =    resultData[i]*xSqr[i] + Type(0.9999999963401755564973428);
			}

			VECTORIZED_KERNEL_LOOP
			for(int i=0;i<Simd;i++)
			{
				resultData[i] = 	resultData[i]*x[i];
			}

			VECTORIZED_KERNEL_LOOP
			for(int i=0;i<Simd;i++)
			{
				x[i] = 	resultData[i];
			}

			VECTORIZED_KERNEL_LOOP
			for(int i=0;i<Simd;i++)
			{
				xSqr[i] = 	resultData[i]*resultData[i];
			}


			VECTORIZED_KERNEL_LOOP
			for(int i=0;i<Simd;i++)
			{
				resultData[i] = 	Type(16.0)*xSqr[i] - Type(20.0);
			}


			VECTORIZED_KERNEL_LOOP
			for(int i=0;i<Simd;i++)
			{
				resultData[i] = 	resultData[i]*xSqr[i] + Type(5.0);
			}

			VECTORIZED_KERNEL_LOOP
			for(int i=0;i<Simd;i++)
			{
				resultData[i] = 	resultData[i]*x[i];
			}

			VECTORIZED_KERNEL_LOOP
			for(int i=0;i<Simd;i++)
			{
				result.data[i] = 	-resultData[i];
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
            Type resultData[Simd];

            VECTORIZED_KERNEL_LOOP
            for(int i=0;i<Simd;i++)
            {
                xSqr[i] =   data[i]*data[i];
            }







            VECTORIZED_KERNEL_LOOP
            for(int i=0;i<Simd;i++)
            {
            	resultData[i] =    Type(2.543240487540288086165674e-06)*xSqr[i] + Type(-0.0001980781510937390521576162);
            }

            VECTORIZED_KERNEL_LOOP
            for(int i=0;i<Simd;i++)
            {
            	resultData[i] =    resultData[i]*xSqr[i] + Type(0.008333159571549231259268709);
            }

            VECTORIZED_KERNEL_LOOP
            for(int i=0;i<Simd;i++)
            {
            	resultData[i] =    resultData[i]*xSqr[i] + Type(-0.1666666483147380972695828);
            }

            VECTORIZED_KERNEL_LOOP
            for(int i=0;i<Simd;i++)
            {
            	resultData[i] =    resultData[i]*xSqr[i] + Type(0.9999999963401755564973428);
            }

            VECTORIZED_KERNEL_LOOP
            for(int i=0;i<Simd;i++)
            {
                result.data[i]  = resultData[i] * data[i];
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

		// 4x faster (AVX512) than std::pow for normalized range [0,1]
		// still faster with higher range but gets slower
		// tolerance = 1.0/inverseTolerance

		template<int inverseTolerance=1000>
		VECTORIZED_KERNEL_METHOD
		void cubeRootNewtonRaphson(KernelData<Type,Simd> & result) const noexcept
		{
		    // f_err(x) = x*x*x - N
		    // f'_err(x) = 3*x*x
		    // x = x - (x*x*x - N)/(3*x*x)
		    // x = x - (x - N/(x*x))/3

            alignas(64)
            Type xd[Simd];

            alignas(64)
            Type resultData[Simd];


            alignas(64)
            Type xSqr[Simd];

            alignas(64)
            Type nDivXsqr[Simd];

            alignas(64)
            Type diff[Simd];

            VECTORIZED_KERNEL_LOOP
            for(int i=0;i<Simd;i++)
            {
                xd[i]=data[i]<=Type(0.000001);
            }

            VECTORIZED_KERNEL_LOOP
            for(int i=0;i<Simd;i++)
            {
                resultData[i]=xd[i]?Type(1.0):data[i];
            }
            // Newton-Raphson Iterations in parallel
            bool work = true;
            while(work)
            {
            	VECTORIZED_KERNEL_LOOP
                for(int i=0;i<Simd;i++)
                {
                    xSqr[i]=resultData[i]*resultData[i];
                }

            	VECTORIZED_KERNEL_LOOP
                for(int i=0;i<Simd;i++)
                {
                    nDivXsqr[i]=data[i]/xSqr[i];
                }

            	VECTORIZED_KERNEL_LOOP
                for(int i=0;i<Simd;i++)
                {
                    nDivXsqr[i]=resultData[i]-nDivXsqr[i];
                }

            	VECTORIZED_KERNEL_LOOP
                for(int i=0;i<Simd;i++)
                {
                    nDivXsqr[i]=nDivXsqr[i]/Type(3.0);
                }

            	VECTORIZED_KERNEL_LOOP
                for(int i=0;i<Simd;i++)
                {
                    diff[i]=resultData[i]-nDivXsqr[i];
                }

            	VECTORIZED_KERNEL_LOOP
                for(int i=0;i<Simd;i++)
                {
                    diff[i]=resultData[i]-diff[i];
                }

            	VECTORIZED_KERNEL_LOOP
                for(int i=0;i<Simd;i++)
                {
                    diff[i]=std::abs(diff[i]);
                }

            	VECTORIZED_KERNEL_LOOP
                for(int i=0;i<Simd;i++)
                {
                    diff[i]=diff[i]>Type(1.0/inverseTolerance);
                }


                Type check = Type(0);
                VECTORIZED_KERNEL_LOOP
                for(int i=0;i<Simd;i++)
                {
                   check += diff[i];
                }
                work = (check>Type(0.0));

                VECTORIZED_KERNEL_LOOP
                for(int i=0;i<Simd;i++)
                {
                    resultData[i]=resultData[i]-nDivXsqr[i];
                }
            }
            VECTORIZED_KERNEL_LOOP
            for(int i=0;i<Simd;i++)
            {
                result.data[i]=xd[i]?Type(0.0):resultData[i];
            }

		}

		// [0,1] range --> 6e-5 average error
		VECTORIZED_KERNEL_METHOD
		void cubeRootFast(KernelData<Type,Simd> & result) const noexcept
		{
            alignas(64)
            Type xd[Simd];

            alignas(64)
            Type resultData[Simd];


            alignas(64)
            Type scaling1[Simd];

            alignas(64)
            Type scaling2[Simd];

            VECTORIZED_KERNEL_LOOP
            for(int i=0;i<Simd;i++)
            {

                Type inp = data[i];
                // scaling + descaling for range [1,999]
                Type scaling = (inp>Type(0.333))?Type(1.0f):Type(3.0f);
                scaling = (inp>Type(0.111))?scaling:Type(1.0/0.111);
                scaling = (inp>Type(0.035))?scaling:Type(1.0/0.035);
                scaling = (inp>Type(0.011))?scaling:Type(1.0/0.011);
                scaling = (inp>Type(0.003))?scaling:Type(1.0/0.003);
                scaling = (inp>Type(0.001))?scaling:Type(1.0/0.001);
                scaling = (inp>Type(0.0003))?scaling:Type(1.0/0.0003);
                scaling = (inp>Type(0.0001))?scaling:Type(1.0/0.0001);
                scaling1[i] = scaling;
            }


            VECTORIZED_KERNEL_LOOP
            for(int i=0;i<Simd;i++)
            {

                Type inp = data[i];
                // scaling + descaling for range [1,999]
                Type scaling = (inp>Type(0.333))?Type(1.0f):Type(std::pow(3.0f,1.0/3.0));
                scaling = (inp>Type(0.111))?scaling:Type(std::pow(1.0/0.111,1.0/3.0));
                scaling = (inp>Type(0.035))?scaling:Type(std::pow(1.0/0.035,1.0/3.0));
                scaling = (inp>Type(0.011))?scaling:Type(std::pow(1.0/0.011,1.0/3.0));
                scaling = (inp>Type(0.003))?scaling:Type(std::pow(1.0/0.003,1.0/3.0));
                scaling = (inp>Type(0.001))?scaling:Type(std::pow(1.0/0.001,1.0/3.0));
                scaling = (inp>Type(0.0003))?scaling:Type(std::pow(1.0/0.0003,1.0/3.0));
                scaling = (inp>Type(0.0001))?scaling:Type(std::pow(1.0/0.0001,1.0/3.0));
                scaling2[i] = scaling;
            }


            VECTORIZED_KERNEL_LOOP
            for(int i=0;i<Simd;i++)
            {
            	resultData[i] =    data[i]*scaling1[i];
            }

            VECTORIZED_KERNEL_LOOP
            for(int i=0;i<Simd;i++)
            {
            	xd[i] =    resultData[i]-Type(1.0);
            }

            VECTORIZED_KERNEL_LOOP
            for(int i=0;i<Simd;i++)
            {
            	resultData[i] =    Type(-55913.0/4782969.0);
            }

            VECTORIZED_KERNEL_LOOP
            for(int i=0;i<Simd;i++)
            {
            	resultData[i] =   resultData[i] * xd[i] +  Type(21505.0/1594323.0);
            }

            VECTORIZED_KERNEL_LOOP
            for(int i=0;i<Simd;i++)
            {
            	resultData[i] =   resultData[i] * xd[i] +  Type(-935.0/59049.0);
            }

            VECTORIZED_KERNEL_LOOP
            for(int i=0;i<Simd;i++)
            {
            	resultData[i] =   resultData[i] * xd[i] +  Type(374.0/19683.0);
            }

            VECTORIZED_KERNEL_LOOP
            for(int i=0;i<Simd;i++)
            {
            	resultData[i] =   resultData[i] * xd[i] +  Type(-154.0/6561.0);
            }

            VECTORIZED_KERNEL_LOOP
			for(int i=0;i<Simd;i++)
			{
				resultData[i] =   resultData[i] * xd[i] +  Type(22.0/729.0);
			}

            VECTORIZED_KERNEL_LOOP
			for(int i=0;i<Simd;i++)
			{
				resultData[i] =   resultData[i] * xd[i] +  Type(-10.0/243.0);
			}

            VECTORIZED_KERNEL_LOOP
			for(int i=0;i<Simd;i++)
			{
				resultData[i] =   resultData[i] * xd[i] +  Type(5.0/81.0);
			}

            VECTORIZED_KERNEL_LOOP
			for(int i=0;i<Simd;i++)
			{
				resultData[i] =   resultData[i] * xd[i] +  Type(-1.0/9.0);
			}

            VECTORIZED_KERNEL_LOOP
			for(int i=0;i<Simd;i++)
			{
				resultData[i] =   resultData[i] * xd[i] +  Type(1.0/3.0);
			}

            VECTORIZED_KERNEL_LOOP
			for(int i=0;i<Simd;i++)
			{
				resultData[i] =   resultData[i] * xd[i] +  Type(1.0);
			}

            VECTORIZED_KERNEL_LOOP
			for(int i=0;i<Simd;i++)
			{
				result.data[i] =   resultData[i]/scaling2[i];
			}
		}

		// [-0.5,0.5] range --> 12 ulps
		VECTORIZED_KERNEL_METHOD
		void atanhFast(KernelData<Type,Simd> & result) const noexcept
		{
            alignas(64)
            Type xSqr[Simd];

            alignas(64)
            Type resultData[Simd];

            VECTORIZED_KERNEL_LOOP
            for(int i=0;i<Simd;i++)
            {
            	xSqr[i] =    data[i]*data[i];
            }

            VECTORIZED_KERNEL_LOOP
            for(int i=0;i<Simd;i++)
            {
            	resultData[i] =    Type(0.0666667);
            }

            VECTORIZED_KERNEL_LOOP
            for(int i=0;i<Simd;i++)
            {
            	resultData[i] =   resultData[i] * xSqr[i] +  Type(0.0769231);
            }

            VECTORIZED_KERNEL_LOOP
            for(int i=0;i<Simd;i++)
            {
            	resultData[i] =   resultData[i] * xSqr[i] +  Type(0.0909091);
            }

            VECTORIZED_KERNEL_LOOP
            for(int i=0;i<Simd;i++)
            {
            	resultData[i] =   resultData[i] * xSqr[i] +  Type(0.111111);
            }

            VECTORIZED_KERNEL_LOOP
            for(int i=0;i<Simd;i++)
            {
            	resultData[i] =   resultData[i] * xSqr[i] +  Type(0.142857);
            }

            VECTORIZED_KERNEL_LOOP
			for(int i=0;i<Simd;i++)
			{
				resultData[i] =   resultData[i] * xSqr[i] +  Type(0.2);
			}

            VECTORIZED_KERNEL_LOOP
			for(int i=0;i<Simd;i++)
			{
				resultData[i] =   resultData[i] * xSqr[i] +  Type(0.333333);
			}

            VECTORIZED_KERNEL_LOOP
			for(int i=0;i<Simd;i++)
			{
				resultData[i] =   resultData[i] * xSqr[i] +  Type(1.0);
			}

            VECTORIZED_KERNEL_LOOP
			for(int i=0;i<Simd;i++)
			{
				result.data[i] =   resultData[i] * data[i];
			}

		}

        // Chebyshev Polynomial coefficients found by genetic algorithm running on 768 GPU pipelines for 1 hour
        VECTORIZED_KERNEL_METHOD
        void expFastFullRange(KernelData<Type,Simd> & result) const noexcept
        {

        	// integer power
            alignas(64)
            Type resultData[Simd];

            alignas(64)
            Type scaledData[Simd];

            constexpr Type scalingDown = Type(1.0/35.0);


            VECTORIZED_KERNEL_LOOP
            for(int i=0;i<Simd;i++)
            {
            	scaledData[i] =    data[i]*scalingDown;
            }

            VECTORIZED_KERNEL_LOOP
            for(int i=0;i<Simd;i++)
            {
            	resultData[i] =    Type(0.9999999822082630274167059)*scaledData[i] + Type(0.8185161762850601263608041);
            }

            VECTORIZED_KERNEL_LOOP
            for(int i=0;i<Simd;i++)
            {
            	resultData[i] =    resultData[i]*scaledData[i] + Type(-0.9999999710986964274184174);
            }

            VECTORIZED_KERNEL_LOOP
            for(int i=0;i<Simd;i++)
            {
            	resultData[i] =    resultData[i]*scaledData[i] + Type(-0.9999999864918285297221701);
            }

            VECTORIZED_KERNEL_LOOP
            for(int i=0;i<Simd;i++)
            {
            	resultData[i] =    resultData[i]*scaledData[i] + Type(0.2452281592191898340615808);
            }

            VECTORIZED_KERNEL_LOOP
            for(int i=0;i<Simd;i++)
            {
            	resultData[i] =    resultData[i]*scaledData[i] + Type(0.4370558371259694041555122);
            }

            VECTORIZED_KERNEL_LOOP
            for(int i=0;i<Simd;i++)
            {
            	resultData[i] =    resultData[i]*scaledData[i] + Type(-0.1317165581945598518132101);
            }

            VECTORIZED_KERNEL_LOOP
            for(int i=0;i<Simd;i++)
            {
            	resultData[i] =    resultData[i]*scaledData[i] + Type(0.005196983233269669710807648);
            }

            constexpr Type scalingUp = std::exp((Type)35);

            VECTORIZED_KERNEL_LOOP
            for(int i=0;i<Simd;i++)
            {
            	result.data[i] =    resultData[i]*scalingUp;
            }
        }

        // only optimized for [-1,1] input range!!
        // Chebyshev Polynomial coefficients found by genetic algorithm running on 768 GPU pipelines for 1 hour
        VECTORIZED_KERNEL_METHOD
        void expFast(KernelData<Type,Simd> & result) const noexcept
        {

            alignas(64)
            Type resultData[Simd];



            VECTORIZED_KERNEL_LOOP
            for(int i=0;i<Simd;i++)
            {
            	resultData[i] =    Type(0.0001972591916103993980868836)*data[i] + Type(0.001433947376170863208244555);
            }

            VECTORIZED_KERNEL_LOOP
            for(int i=0;i<Simd;i++)
            {
            	resultData[i] =    resultData[i]*data[i] + Type(0.008338950118885968265658448);
            }

            VECTORIZED_KERNEL_LOOP
            for(int i=0;i<Simd;i++)
            {
            	resultData[i] =    resultData[i]*data[i] + Type(0.04164162895364054151059463);
            }

            VECTORIZED_KERNEL_LOOP
            for(int i=0;i<Simd;i++)
            {
            	resultData[i] =    resultData[i]*data[i] + Type(0.1666645212581130408580066);
            }

            VECTORIZED_KERNEL_LOOP
            for(int i=0;i<Simd;i++)
            {
            	resultData[i] =    resultData[i]*data[i] + Type(0.5000045184212300597437206);
            }

            VECTORIZED_KERNEL_LOOP
            for(int i=0;i<Simd;i++)
            {
            	resultData[i] =    resultData[i]*data[i] + Type(0.9999999756072401879691824);
            }

            VECTORIZED_KERNEL_LOOP
            for(int i=0;i<Simd;i++)
            {
            	result.data[i] =    resultData[i]*data[i] + Type(0.999999818912344906607359);
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



#if defined(_OPENMP)


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
