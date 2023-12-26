# FunSizedTensor

## What is FunSizedTensor?

FunSizedTensor is a header-only C++ library (currently requiring C++17) that uses [variadic templates](https://en.wikipedia.org/wiki/Variadic_template) to allow succinctly expressing simple tensor operations of any dimension, making use of [expression templates](https://en.wikipedia.org/wiki/Expression_templates) to avoid materializing intermediate temporaries. The evaluation loop nests use a recursive [cache-oblivious](https://en.wikipedia.org/wiki/Cache-oblivious_algorithm) structure.

## Usage

**A simple example**
```cpp
namespace fst = fun_sized_tensor;

// Defines a tensor where the outer dimension is defined as runtime, and in
// this case, the tensor will be 6765 x 100.
fst::tensor_odyn<float, fst::dims<100>> t1(6765);

// Defines a tensor where all of the dimensions are fixed at compile time, and
// in this case, the tensor will be 100 x 1024.
fst::tensor<float, fst::dims<100, 1024>> t1b;

// Fills the tensor with the value 2.0f.
t1b = 2.0f;

// Define another tensor with a dynamic outer dimension; other constructor
// parameters are passed through to the underlying container, std::vector by
// default, so this fills the tensor with 2.0f.
fst::tensor_odyn<ST, fst::dims<1024>> t1c(6765, 2.0f);

// Defines the index types used to express the tensor operations. The numbers
// for each index defines the loop ordering. In this case, _k will be the inner
// loop and _i will be the outer loop.
fst::index<2> _i;
fst::index<1> _j;
fst::index<0> _k;

// A simple matrix multiplication.
t1(_i, _j) = t1b(_j, _k)*t1c(_i, _k);

// A matrix multiplication, where the _k loop (the inner loop) is unrolled by
// some factor (the default is in the FUN_SIZED_TENSOR_UNROLL_COUNT preprocessor
// macro). In this case, += is used, so the results are added to the existing
// values in t1.
t1(_i, _j).unroll(_k) += t1b(_j, _k)*t1c(_i, _k);

// A matrix multiplication, where the compiler is asked to vectorize the inner
// loop and unroll the middle loop.
t1(_i, _j).vectorize(_k).unroll(_j) += t1b(_j, _k)*t1c(_i, _k);

// The operations use a cache-oblivious looping structure (based on the well-known
// work by Frigo et al.). The largest dimension is recursively split (so long
// as the largest dimension is greater than some limit, for which the default is
// in the preprocessor macro FUN_SIZED_TENSOR_RECURSE_SIZE). The recursion
// dimension-size threshold can be changed, like this:
t1(_i, _j).unroll(_k).recurse<6>() += t1b(_j, _k)*t1c(_i, _k);

// Note that you can combine indexed tensor expressions with scalars, like this:
float a = 2.4, b = 3.4, c = 4.4, d = 5.4, e = 8.2;
t1(_i, _j) = 1 / (b*t1b(_j, _k)+a)*(c - t1c(_i, _k) - d) / e;

// There are a number of convenience operations on the tensors themselves:
t1 = 5.0f;
t1 *= 5.0f;
t1 /= 2.0f;
t1 += 2.0f;
t1 -= 1.0f;
```

## CMake Variables

In addition to the standard CMake variable, the following CMake variables can be used to configure the library:

| Variable | Type | Default | Description |
| --- | --- | --- | --- |
| `FUN_SIZED_TENSOR_USE_LIBCXX` | BOOL | OFF | When ON, use select -stdlib=libc++ when compiling and linking. |
| `FUN_SIZED_TENSOR_SET_COMPILER_RPATH` | BOOL | OFF | When ON, add to the build-time linking RPATH all non-system-default C++ compiler linking directories. This is useful when using a compiler that is not installed in the default system location (and the RPATH is set at build time so that unit tests can run from the build directory). |

## Maturity and Stability

This library is in an "I just got something working" state. No interface or functional stability should be presumed at this point in time. There are no official releases.

## Support

This library is being developed as a hobby of the author. Bugs should be reported using the repository bug tracker. Contributions are welcome.
