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
#include <string>
#include <functional>
#include <cmath>
#include <chrono>
#include <thread>

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


		alignas(64)
		Type data[Simd];



		__attribute__((always_inline))
		KernelData(){}

		__attribute__((always_inline))
		KernelData(const Type & broadcastedInit) noexcept
		{

			for(int i=0;i<Simd;i++)
			{
				data[i] = broadcastedInit;
			}
		}

		__attribute__((always_inline))
		KernelData(const KernelData<Type,Simd> & vectorizedIit) noexcept
		{

			for(int i=0;i<Simd;i++)
			{
				data[i] = vectorizedIit.data[i];
			}
		}

		__attribute__((always_inline))
		KernelData(KernelData&& dat){

			for(int i=0;i<Simd;i++)
			{
				data[i] = dat.data[i];
			}
	    }

		__attribute__((always_inline))
		KernelData& operator=(const KernelData& dat){

			for(int i=0;i<Simd;i++)
			{
				data[i] = dat.data[i];
			}
	        return *this;
	    };

		__attribute__((always_inline))
		KernelData& operator=(KernelData&& dat){

			for(int i=0;i<Simd;i++)
			{
				data[i] = dat.data[i];
			}
	        return *this;

	    };

		__attribute__((always_inline))
	    ~KernelData(){

	    };

		__attribute__((always_inline))
		inline KernelData<Type,Simd> & assign()
		{
	    	return *this;
		}

	    // contiguous read element by element starting from beginning of ptr
		__attribute__((always_inline))
		inline void readFrom(const Type * const __restrict__ ptr) noexcept
		{

			for(int i=0;i<Simd;i++)
			{
				data[i] = ptr[i];
			}
		}

	    // contiguous read element by element starting from beginning of ptr
		// masked read operation: if mask lane is set then read. if not set then don't read
		template<typename TypeMask>
		__attribute__((always_inline))
		inline void readFromMasked(const Type * const __restrict__ ptr, const KernelData<TypeMask,Simd> & mask) noexcept
		{

			for(int i=0;i<Simd;i++)
			{
				data[i] = mask.data[i]?ptr[i]:data[i];
			}
		}

		// contiguous write element by element starting from beginning of ptr
		__attribute__((always_inline))
		inline void writeTo(Type * const __restrict__ ptr) const noexcept
		{

			for(int i=0;i<Simd;i++)
			{
				ptr[i] = data[i];
			}
		}

		// contiguous write element by element starting from beginning of ptr
		// masked write: if mask lane is set then write, if not set then don't write
		template<typename TypeMask>
		__attribute__((always_inline))
		inline void writeToMasked(Type * const __restrict__ ptr, const KernelData<TypeMask,Simd> & mask) const noexcept
		{

			for(int i=0;i<Simd;i++)
			{
				if(mask.data[i])
						ptr[i] = data[i];
			}
		}

		// does scatter operation (every element writes its own targeted ptr element, decided by elements of id)
		__attribute__((always_inline))
		inline void writeTo(Type * const __restrict__ ptr, const KernelData<int,Simd> & id) const noexcept
		{

			for(int i=0;i<Simd;i++)
			{
				ptr[id.data[i]] = data[i];
			}
		}

		// does scatter operation (every element writes its own targeted ptr element, decided by elements of id)
		// masked write: if mask lane is set then write, if not set then don't write
		template<typename TypeMask>
		__attribute__((always_inline))
		inline void writeToMasked(Type * const __restrict__ ptr, const KernelData<int,Simd> & id, const KernelData<TypeMask,Simd> & mask) const noexcept
		{

			for(int i=0;i<Simd;i++)
			{
				if(mask.data[i])
					ptr[id.data[i]] = data[i];
			}
		}

		// uses only first item of id to compute the starting point of target ptr element.
		// writes Simd number of elements to target starting from ptr + id.data[0]
		__attribute__((always_inline))
		inline void writeToContiguous(Type * const __restrict__ ptr, const KernelData<int,Simd> & id) const noexcept
		{
			const int idx = id.data[0];
			for(int i=0;i<Simd;i++)
			{
				ptr[idx+i] = data[i];
			}
		}

		// uses only first item of id to compute the starting point of target ptr element.
		// writes Simd number of elements to target starting from ptr + id.data[0]
		// masked write: if mask lane is set then writes, if not set then does not write
		template<typename TypeMask>
		__attribute__((always_inline))
		inline void writeToContiguousMasked(Type * const __restrict__ ptr, const KernelData<int,Simd> & id, const KernelData<TypeMask,Simd> & mask) const noexcept
		{
			const int idx = id.data[0];
			for(int i=0;i<Simd;i++)
			{
				if(mask.data[i])
					ptr[idx+i] = data[i];
			}
		}

		// does gather operation (every element reads its own sourced ptr element, decided by elements of id)
		__attribute__((always_inline))
		inline void readFrom(Type * const __restrict__ ptr, const KernelData<int,Simd> & id) noexcept
		{

			for(int i=0;i<Simd;i++)
			{
				 data[i] = ptr[id.data[i]];
			}
		}

		// does gather operation (every element reads its own sourced ptr element, decided by elements of id)
		// masked operation: if mask lane is set, then it reads from pointer+id.data[i], if not set then it does not read anything
		template<typename TypeMask>
		__attribute__((always_inline))
		inline void readFromMasked(Type * const __restrict__ ptr, const KernelData<int,Simd> & id, const KernelData<TypeMask,Simd> & mask) noexcept
		{

			for(int i=0;i<Simd;i++)
			{
				 data[i] = mask.data[i]?ptr[id.data[i]]:data[i];
			}
		}

		// uses only first item of id to compute the starting point of source ptr element.
		// reads Simd number of elements from target starting from ptr + id.data[0]
		__attribute__((always_inline))
		inline void readFromContiguous(Type * const __restrict__ ptr, const KernelData<int,Simd> & id) noexcept
		{
			const int idx = id.data[0];
			for(int i=0;i<Simd;i++)
			{
				data[i] = ptr[idx+i];
			}
		}

		// uses only first item of id to compute the starting point of source ptr element.
		// reads Simd number of elements from target starting from ptr + id.data[0]
		// masked operation: if mask lane is set, then it reads from pointer+id.data[0], if not set then it does not read anything
		template<typename TypeMask>
		__attribute__((always_inline))
		inline void readFromContiguousMasked(Type * const __restrict__ ptr, const KernelData<int,Simd> & id, const KernelData<TypeMask,Simd> & mask) noexcept
		{
			const int idx = id.data[0];
			for(int i=0;i<Simd;i++)
			{
				data[i] = mask.data[i]?ptr[idx+i]:data[i];
			}
		}

		template<typename F>
		__attribute__((always_inline))
		inline void idCompute(const int id, const F & f) noexcept
		{

			for(int i=0;i<Simd;i++)
			{
				data[i] = f(id+i);
			}
		}

		// bool
        template<typename TypeMask>
        __attribute__((always_inline))
		inline void lessThan(const KernelData<Type,Simd> & vec, KernelData<TypeMask,Simd> & result) const noexcept
		{
			for(int i=0;i<Simd;i++)
			{
				result.data[i] = data[i]<vec.data[i];
			}
		}



		// bool
        template<typename TypeMask>
        __attribute__((always_inline))
		inline void lessThanOrEquals(const KernelData<Type,Simd> & vec, KernelData<TypeMask,Simd> & result) const noexcept
		{
			for(int i=0;i<Simd;i++)
			{
				result.data[i] = data[i]<=vec.data[i];
			}
		}

		// bool
        template<typename TypeMask>
        __attribute__((always_inline))
		inline void lessThanOrEquals(const Type val, KernelData<TypeMask,Simd> & result) const noexcept
		{
			for(int i=0;i<Simd;i++)
			{
				result.data[i] = data[i]<=val;
			}
		}

		// bool
         template<typename TypeMask>
         __attribute__((always_inline))
		inline void greaterThan(const KernelData<Type,Simd> & vec, KernelData<TypeMask,Simd> & result) const noexcept
		{
			for(int i=0;i<Simd;i++)
			{
				result.data[i] = data[i]>vec.data[i];
			}
		}

		// bool
        template<typename TypeMask>
        __attribute__((always_inline))
		inline void greaterThan(const Type val, KernelData<TypeMask,Simd> & result) const noexcept
		{
			for(int i=0;i<Simd;i++)
			{
				result.data[i] = data[i]>val;
			}
		}

		// bool
        template<typename TypeMask>
        __attribute__((always_inline))
		inline void equals(const KernelData<Type,Simd> & vec, KernelData<TypeMask,Simd> & result) const noexcept
		{
			for(int i=0;i<Simd;i++)
			{
				result.data[i] = data[i] == vec.data[i];
			}
		}

		// bool
        template<typename TypeMask>
        __attribute__((always_inline))
		inline void equals(const Type val, KernelData<TypeMask,Simd> & result) const noexcept
		{
			for(int i=0;i<Simd;i++)
			{
				result.data[i] = data[i] == val;
			}
		}

		// bool
        template<typename TypeMask>
        __attribute__((always_inline))
		inline void notEqual(const KernelData<Type,Simd> & vec, KernelData<TypeMask,Simd> & result) const noexcept
		{
			for(int i=0;i<Simd;i++)
			{
				result.data[i] = data[i] != vec.data[i];
			}
		}

		// bool
        template<typename TypeMask>
        __attribute__((always_inline))
		inline  void notEqual(const Type val, KernelData<TypeMask,Simd> & result) const noexcept
		{
			for(int i=0;i<Simd;i++)
			{
				result.data[i] = data[i] != val;
			}
		}


		// bool
        template<typename TypeMask>
        __attribute__((always_inline))
		inline  void logicalAnd(const KernelData<TypeMask,Simd> vec, KernelData<TypeMask,Simd> & result) const noexcept
		{
			for(int i=0;i<Simd;i++)
			{
				result.data[i] = data[i] && vec.data[i];
			}
		}

		// bool
        template<typename TypeMask>
        __attribute__((always_inline))
		inline void logicalOr(const KernelData<TypeMask,Simd> vec, KernelData<TypeMask,Simd> & result) const noexcept
		{
			for(int i=0;i<Simd;i++)
			{
				result.data[i] = data[i] || vec.data[i];
			}
		}

        __attribute__((always_inline))
		inline bool areAllTrue() const noexcept
		{
			int result = 0;


			for(int i=0;i<Simd;i++)
			{
				result = result + (data[i]>0);
			}
			return result==Simd;
		}

        __attribute__((always_inline))
		inline bool isAnyTrue() const noexcept
		{
			int result = 0;
			for(int i=0;i<Simd;i++)
			{
				result = result + (data[i]>0);
			}
			return result>0;
		}

		template<typename ComparedType>
		__attribute__((always_inline))
		inline void ternary(const KernelData<ComparedType,Simd> vec1, const KernelData<ComparedType,Simd> vec2, KernelData<ComparedType,Simd> & result) const noexcept
		{
			for(int i=0;i<Simd;i++)
			{
				result.data[i] = data[i]?vec1.data[i]:vec2.data[i];
			}
		}

		template<typename ComparedType>
		__attribute__((always_inline))
		inline void ternary(const ComparedType val1, const KernelData<ComparedType,Simd> vec2, KernelData<ComparedType,Simd> & result) const noexcept
		{
			for(int i=0;i<Simd;i++)
			{
				result.data[i] = data[i]?val1:vec2.data[i];
			}
		}

		template<typename ComparedType>
		__attribute__((always_inline))
		inline void ternary(const KernelData<ComparedType,Simd> vec1, const ComparedType val2, KernelData<ComparedType,Simd> & result) const noexcept
		{
			for(int i=0;i<Simd;i++)
			{
				result.data[i] = data[i]?vec1.data[i]:val2;
			}
		}

		template<typename ComparedType>
		__attribute__((always_inline))
		inline void ternary(const ComparedType val1, const ComparedType val2, KernelData<ComparedType,Simd> & result) const noexcept
		{
			for(int i=0;i<Simd;i++)
			{
				result.data[i] = data[i]?val1:val2;
			}
		}

		__attribute__((always_inline))
		inline void broadcast(const Type val) noexcept
		{

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
		__attribute__((always_inline))
		inline void gatherFromLane(const KernelData<IntegerType,Simd> & id, KernelData<Type,Simd> & result) const noexcept
		{
			for(int i=0;i<Simd;i++)
			{
				result.data[i] = data[id.data[i]];
			}
		}

		// similar to gatherFromLane but with constant index values for faster operation
		__attribute__((always_inline))
		inline void transposeLanes(const int widthTranspose, KernelData<Type,Simd> & result) const noexcept
		{
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
		__attribute__((always_inline))
		inline void lanesLeftShift(const IntegerType & n, KernelData<Type,Simd> & result) const noexcept
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
		__attribute__((always_inline))
		inline void lanesRightShift(const IntegerType & n, KernelData<Type,Simd> & result) const noexcept
		{
			for(int i=0;i<Simd;i++)
			{
				const int j = (i+2*Simd-n)&(Simd-1);
				result.data[i] = data[j];
			}
		}

		// shifts lanes (wraps around) left n times in-place
		template<typename IntegerType>
		__attribute__((always_inline))
		inline void lanesLeftShift(const IntegerType & n) const noexcept
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
		__attribute__((always_inline))
		inline void lanesRightShift(const IntegerType & n) const noexcept
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
		__attribute__((always_inline))
		inline void broadcastFromLane(const int lane) noexcept
		{
            const Type bcast = data[lane];
			for(int i=0;i<Simd;i++)
			{
				data[i] = bcast;
			}
		}

		// same as broadcastFromLane(lane) but the target to copy is a result vector
		__attribute__((always_inline))
		inline void broadcastFromLaneToVector(const int lane, KernelData<Type,Simd> & result) noexcept
		{
            const Type bcast = data[lane];
			for(int i=0;i<Simd;i++)
			{
				result.data[i] = bcast;
			}
		}

		__attribute__((always_inline))
		inline void readFrom(const KernelData<Type,Simd> & vec) noexcept
		{
			for(int i=0;i<Simd;i++)
			{
				data[i] = vec.data[i];
			}
		}

		template<typename NewType>
		__attribute__((always_inline))
		inline void castAndCopyTo(KernelData<NewType,Simd> & result) const noexcept
		{
			for(int i=0;i<Simd;i++)
			{
				result.data[i] = (NewType)data[i];
			}
		}

		template<typename NewType>
		__attribute__((always_inline))
		inline void castBitwiseAndCopyTo(KernelData<NewType,Simd> & result) const noexcept
		{
			for(int i=0;i<Simd;i++)
			{
				result.data[i] = *reinterpret_cast<const NewType*>(&data[i]);
			}
		}


		__attribute__((always_inline))
		inline  void sqrt(KernelData<Type,Simd> & result) const noexcept
		{
			for(int i=0;i<Simd;i++)
			{
				result.data[i] = std::sqrt(data[i]);
			}
		}

		__attribute__((always_inline))
		inline  void add(const KernelData<Type,Simd> & vec, KernelData<Type,Simd> & result) const noexcept
		{
			for(int i=0;i<Simd;i++)
			{
				result.data[i] = data[i] + vec.data[i];
			}
		}

		__attribute__((always_inline))
		inline  void add(const Type & val, KernelData<Type,Simd> & result) const noexcept
		{
			for(int i=0;i<Simd;i++)
			{
				result.data[i] = data[i] + val;
			}
		}

		__attribute__((always_inline))
		inline void sub(const KernelData<Type,Simd> & vec, KernelData<Type,Simd> & result) const noexcept
		{
			for(int i=0;i<Simd;i++)
			{
				result.data[i] = data[i] - vec.data[i];
			}
		}

		__attribute__((always_inline))
		inline void sub(const Type & val, KernelData<Type,Simd> & result) const noexcept
		{
			for(int i=0;i<Simd;i++)
			{
				result.data[i] = data[i] - val;
			}
		}

		__attribute__((always_inline))
		inline void div(const KernelData<Type,Simd> & vec, KernelData<Type,Simd> & result) const noexcept
		{
			for(int i=0;i<Simd;i++)
			{
				result.data[i] = data[i] / vec.data[i];
			}
		}

		__attribute__((always_inline))
		inline void div(const Type & val, KernelData<Type,Simd> & result) const noexcept
		{
			for(int i=0;i<Simd;i++)
			{
				result.data[i] = data[i] / val;
			}
		}


		__attribute__((always_inline))
		inline void fusedMultiplyAdd(const KernelData<Type,Simd> & vec1, const KernelData<Type,Simd> & vec2, KernelData<Type,Simd> & result) const noexcept
		{
			for(int i=0;i<Simd;i++)
			{
				result.data[i] = (data[i]* vec1.data[i]+ vec2.data[i]);
			}
		}

		__attribute__((always_inline))
		inline void fusedMultiplyAdd(const KernelData<Type,Simd> & vec1, const Type & val2, KernelData<Type,Simd> & result) const noexcept
		{
			for(int i=0;i<Simd;i++)
			{
				result.data[i] = (data[i]* vec1.data[i]+ val2);
			}
		}

		__attribute__((always_inline))
		inline void fusedMultiplyAdd(const Type & val1, const KernelData<Type,Simd> & vec2, KernelData<Type,Simd> & result) const noexcept
		{
			for(int i=0;i<Simd;i++)
			{
				result.data[i] = (data[i]* val1+ vec2.data[i]);
			}
		}

		__attribute__((always_inline))
		inline void fusedMultiplyAdd(const Type & val1, const Type & val2, KernelData<Type,Simd> & result) const noexcept
		{
			for(int i=0;i<Simd;i++)
			{
				result.data[i] = (data[i]* val1+ val2);
			}
		}

		__attribute__((always_inline))
		inline void fusedMultiplySub(const KernelData<Type,Simd> & vec1, const KernelData<Type,Simd> & vec2, KernelData<Type,Simd> & result) const noexcept
		{
			for(int i=0;i<Simd;i++)
			{
				result.data[i] = (data[i]* vec1.data[i] -vec2.data[i]);
			}
		}

		__attribute__((always_inline))
		inline void fusedMultiplySub(const Type & val1, const KernelData<Type,Simd> & vec2, KernelData<Type,Simd> & result) const noexcept
		{
			for(int i=0;i<Simd;i++)
			{
				result.data[i] = (data[i]* val1 -vec2.data[i]);
			}
		}

		__attribute__((always_inline))
		inline void fusedMultiplySub(const KernelData<Type,Simd> & vec1, const Type & val2, KernelData<Type,Simd> & result) const noexcept
		{
			for(int i=0;i<Simd;i++)
			{
				result.data[i] = (data[i]* vec1.data[i] -val2);
			}
		}

		__attribute__((always_inline))
		inline void fusedMultiplySub(const Type & val1, const Type & val2, KernelData<Type,Simd> & result) const noexcept
		{
			for(int i=0;i<Simd;i++)
			{
				result.data[i] = (data[i]* val1 -val2);
			}
		}

		__attribute__((always_inline))
		inline void mul(const KernelData<Type,Simd> & vec, KernelData<Type,Simd> & result) const noexcept
		{
			for(int i=0;i<Simd;i++)
			{
				result.data[i] = data[i] * vec.data[i];
			}
		}

		__attribute__((always_inline))
		inline void mul(const Type & val, KernelData<Type,Simd> & result) const noexcept
		{
			for(int i=0;i<Simd;i++)
			{
				result.data[i] = data[i] * val;
			}
		}


		__attribute__((always_inline))
		inline void modulus(const KernelData<Type,Simd> & vec, KernelData<Type,Simd> & result) const noexcept
		{
			for(int i=0;i<Simd;i++)
			{
				result.data[i] = data[i] % vec.data[i];
			}
		}

		__attribute__((always_inline))
		inline  void modulus(const Type & val, KernelData<Type,Simd> & result) const noexcept
		{
			for(int i=0;i<Simd;i++)
			{
				result.data[i] = data[i] % val;
			}
		}

		__attribute__((always_inline))
		inline void leftShift(const KernelData<Type,Simd> & vec, KernelData<Type,Simd> & result) const noexcept
		{
			for(int i=0;i<Simd;i++)
			{
				result.data[i] = data[i] << vec.data[i];
			}
		}

		__attribute__((always_inline))
		inline void leftShift(const Type & val, KernelData<Type,Simd> & result) const noexcept
		{
			for(int i=0;i<Simd;i++)
			{
				result.data[i] = data[i] << val;
			}
		}

		__attribute__((always_inline))
		inline void rightShift(const KernelData<Type,Simd> & vec, KernelData<Type,Simd> & result) const noexcept
		{
			for(int i=0;i<Simd;i++)
			{
				result.data[i] = data[i] >> vec.data[i];
			}
		}

		__attribute__((always_inline))
		inline void rightShift(const Type & val, KernelData<Type,Simd> & result) const noexcept
		{
			for(int i=0;i<Simd;i++)
			{
				result.data[i] = data[i] >> val;
			}
		}

		// this function is not accelerated. use it sparsely.
		__attribute__((always_inline))
		inline void pow(const KernelData<Type,Simd> & vec, KernelData<Type,Simd> & result) const noexcept
		{
			for(int i=0;i<Simd;i++)
			{
				result.data[i] = std::pow(data[i],vec.data[i]);
			}
		}

		// this function is not accelerated. use it sparsely.
		// x^y = x.pow(y,result)
		__attribute__((always_inline))
		inline void pow(const Type & val, KernelData<Type,Simd> & result) const noexcept
		{
			for(int i=0;i<Simd;i++)
			{
				result.data[i] = std::pow(data[i],val);
			}
		}

		// this function is not accelerated. use it sparsely.
		// computes y^x = x.powFrom(y,result) is called
		__attribute__((always_inline))
		inline void powFrom(const Type & val, KernelData<Type,Simd> & result) const noexcept
		{
			for(int i=0;i<Simd;i++)
			{
				result.data[i] = std::pow(val,data[i]);
			}
		}

		// this function is not accelerated. use it sparsely.
		__attribute__((always_inline))
		inline void exp(KernelData<Type,Simd> & result) const noexcept
		{
			for(int i=0;i<Simd;i++)
			{
				result.data[i] = std::exp(data[i]);
			}
		}


		// this function is not accelerated. use it sparsely.
		__attribute__((always_inline))
		inline void log(KernelData<Type,Simd> & result) const noexcept
		{
			for(int i=0;i<Simd;i++)
			{
				result.data[i] = std::log(data[i]);
			}
		}

		// this function is not accelerated. use it sparsely.
		__attribute__((always_inline))
		inline void log2(KernelData<Type,Simd> & result) const noexcept
		{
			for(int i=0;i<Simd;i++)
			{
				result.data[i] = std::log2(data[i]);
			}
		}


		__attribute__((always_inline))
		inline void bitwiseXor(const KernelData<Type,Simd> & vec, KernelData<Type,Simd> & result) const noexcept
		{
			for(int i=0;i<Simd;i++)
			{
				result.data[i] = data[i] ^ vec.data[i];
			}
		}

		__attribute__((always_inline))
		inline void bitwiseXor(const Type & val, KernelData<Type,Simd> & result) const noexcept
		{
			for(int i=0;i<Simd;i++)
			{
				result.data[i] = data[i] ^ val;
			}
		}

		__attribute__((always_inline))
		inline void bitwiseAnd(const KernelData<Type,Simd> & vec, KernelData<Type,Simd> & result) const noexcept
		{
			for(int i=0;i<Simd;i++)
			{
				result.data[i] = data[i] & vec.data[i];
			}
		}

		__attribute__((always_inline))
		inline void bitwiseAnd(const Type & val, KernelData<Type,Simd> & result) const noexcept
		{
			for(int i=0;i<Simd;i++)
			{
				result.data[i] = data[i] & val;
			}
		}

		__attribute__((always_inline))
		inline void bitwiseOr(const KernelData<Type,Simd> & vec, KernelData<Type,Simd> & result) const noexcept
		{
			for(int i=0;i<Simd;i++)
			{
				result.data[i] = data[i] | vec.data[i];
			}
		}

		__attribute__((always_inline))
		inline void bitwiseOr(const Type & val, KernelData<Type,Simd> & result) const noexcept
		{
			for(int i=0;i<Simd;i++)
			{
				result.data[i] = data[i] | val;
			}
		}
		__attribute__((always_inline))
		inline void bitwiseNot(KernelData<Type,Simd> & result) const noexcept
		{
			for(int i=0;i<Simd;i++)
			{
				result.data[i] = ~data[i];
			}
		}

		__attribute__((always_inline))
		inline void logicalNot(KernelData<Type,Simd> & result) const noexcept
		{
			for(int i=0;i<Simd;i++)
			{
				result.data[i] = !data[i];
			}
		}

		__attribute__((always_inline))
		inline void factorial(KernelData<Type,Simd> & result) const noexcept
		{
			alignas(64)
			Type tmpC[Simd];
			alignas(64)
			Type tmpD[Simd];
			alignas(64)
            Type tmpE[Simd];
			for(int i=0;i<Simd;i++)
			{
				tmpC[i]=data[i];
			}
			for(int i=0;i<Simd;i++)
			{
				tmpD[i]=data[i]-(Type)1;
			}
			int mask[Simd];


			int anyWorking = true;
			while(anyWorking)
			{
				anyWorking = false;

				for(int i=0;i<Simd;i++)
				{
					mask[i] = (tmpD[i]>0);
				}

				for(int i=0;i<Simd;i++)
				{
					anyWorking += mask[i];
				}

				for(int i=0;i<Simd;i++)
				{
					tmpE[i] =  tmpC[i] * tmpD[i];
				}

				for(int i=0;i<Simd;i++)
				{
					tmpC[i] = mask[i] ? tmpE[i] : tmpC[i];
				}

				for(int i=0;i<Simd;i++)
				{
					tmpD[i]--;
				}

			}

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
			// simple work scheduling. todo: do load-balance with core affinity + dedicated thread
			std::vector<std::thread> threads;
			const int nChunk = numThreads>0?(1 + nLoop/numThreads):nLoop; // each thread is like simd*100 GPU threads
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
	};

	template<int SimdWidth, typename F, class...Args>
	auto createKernel(F&& kernelPrm, KernelArgs<Args...> const& _prm_)
	{
		return Kernel<SimdWidth, F, Args...>(std::forward<F>(kernelPrm));
	}

}


#endif /* VECTORIZEDKERNEL_H_ */
