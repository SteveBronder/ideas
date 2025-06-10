# Summary

## My Ideal Matrix Library

I want to write my perfect matrix/tensor library. Eigen is great, but there are a few things that are not possible in there project that I would like to fix with this.

1. The matrices should be usable in a constexpr context. Often times we have to do some precomputation before running an algorithm.

```c++
static constexpr double mat_data[4] = {1.0, 2.0, 3.0, 4.0};
static constexpr double vec_data[2] = {1.0, 2.0};
// Using default options
using opt = yae::Options<const double>;
// Make constexpr matrices of 2x2 and 2x1
constexpr yae::Matrix<opt, 2, 2> mat(mat_data);
constexpr yae::Vector<opt, 2> vec(vec_data);
constexpr yae::Vector<opt, 2> res = mat * vec;
```

2. CPU and GPU code should support expression fusion.

```c++
using yae::Tensor, yae::Dynamic, yae::Index, yae::Options;
// Use a cuda backend and allocator
using opt = Options<double,
  Index,
  yae::cuda_unified_allocator<double>,
  yae::GPUEngine<yae::DefaultGPU>>;
// Dynamic sized Tensor allocated in alloc with dims [5, 2, 8]
yae::cuda_unified_allocator alloc{};
// Tensors are associated with a device
yae::cuda_device gpu_device{0};
Tensor<opt, Dynamic, 2, 8> ten1(alloc, gpu_device, 5, 2, 8);
ten1.set_random();
// Each tensor must be on the same device
Tensor<opt, Dynamic, 8, 3> ten2(alloc, gpu_device, 5, 8, 3);
ten2.set_random();
// Follow numpy rules to contract on last axis of ten1 and second to last axis of ten2 for [5, 2, 3]
// batch 5 and contract (2x8) * (8x3) = (5, 2, 3)
Tensor<opt, Dynamic, 2, 3> res = ten1 * ten2;
```

3. The matrices should be allocator aware. One of the largest benefits of C++ is being able to manage your own memory. Along with this, fixed size matrices should have an option to be allocated on the heap.

```c++
// Make a polymorphic allocator with an underlying buffer to handle new resources
using poly_alloc_t = std::pmr::polymorphic_allocator;
auto mr = std::pmr::monotonic_buffer_resource();
auto alloc = poly_alloc_t{&mbr};
using yae::Tensor, yae::Dynamic, yae::Options;
using opt = Options<double, yae::Index, poly_alloc>;
// partially Dynamic sized Tensor allocated in alloc with dims [5, 2, 8]
Tensor<opt, Dynamic, 2, 8> ten1(alloc, 5, 2, 8);
ten1.set_random();
auto other_mr = std::pmr::unsynchronized_pool_resource();
auto other_alloc = poly_alloc_t{&mbr};
Tensor<opt, Dynamic, 8, Dynamic> ten2(other_alloc, 5, 8, 3);
ten2.set_random();
// For two matrices with different allocators, users need to say which one to use
// Throws error about now knowing which allocator to use
Tensor<opt, Dynamic, 2, Dynamic> res = ten1 * ten2;
// Tell the operation which allocator to use for the result
Tensor<opt, Dynamic, 2, Dynamic> res = (ten1 * ten2).allocator(other_alloc);

// Even with all known sizes, allow dynamic allocation of memory
using dynamic_opt = Options<double, Index, poly_alloc, yae::force_dynamic>;
Tensor<dynamic_opt, 5, 8, 10000> ten3(other_alloc, 5, 8, 10000);
ten3.set_random();
// Tell the operation which allocator to use for the result
Tensor<dynamic_opt, 5, 2, 10000> res = (ten1 * ten3).allocator(other_alloc);

```

4. It should have a nice API like Eigen

- I'd like to make it somewhere between blaze and Eigen. Like Eigen, I like the idea of forcing users to use an array wrapper for array like operations. Unlike Eigen, I prefer we use free functions instead of member functions. In the example below we make a tensor, use the pipe operator to perform array an exponential and sum, and then the sum sum is computed and added to an expression for a tensor multiplication.

```c++
using yae::Tensor, yae::Dynamic, yae::Index, yae::Options;
using opt = Options<double>;
Tensor<opt, Dynamic, 2, 8> ten1(5, 2, 8);
ten1.set_random();
Tensor<opt, Dynamic, 8, Dynamic> ten2(5, 8, 3);
// Use C++23 pipe operator for chaining array ops together
// NOTE: This is an expression because of auto
auto ten1_exp_sum = yae::to_array(ten1) | yae::exp | yae::sum;
auto res = ten1 * ten2 + ten1_exp_sum;
```

5. It should be extensible by users. For a particular CPU or GPU the user should be able to override operations and information given to the program so that they are able to fully utilize their hardware.

```c++
namespace yae {
// Overload for Intel Raptor Cove i9 13900KF
// See https://en.wikichip.org/wiki/intel/core_i9/i9-13900kf
struct i913900KF;
using yae::Instructions;
template <>
struct CpuEngine<i913900KF> {
  static constexpr std::string_view name = "i9-13900KF";
  static constexpr std::size_t total_cores = 24;
  // p stands for performance core
  static constexpr std::size_t p_total_cores = 8;
  static constexpr std::size_t p_l1_dcache_num = 8;
  static constexpr std::size_t p_l1_dcache_size = 49152;
  static constexpr std::size_t p_l1_icache_num = 8;
  static constexpr std::size_t p_l1_icache_size = 32768;
  static constexpr std::size_t p_l2_cache_num = 2;
  static constexpr std::size_t p_l2_cache_size = 8.389e+6;
  static constexpr std::size_t p_l3_cache_num = 3;
  static constexpr std::size_t p_l3_cache_size = 8.389e+6;
  // e stands for efficiency core
  static constexpr std::size_t e_total_cores = 16;
  static constexpr std::size_t e_l1_dcache_num = 16;
  static constexpr std::size_t e_l1_dcache_size = 32768;
  static constexpr std::size_t e_l1_icache_num = 16;
  static constexpr std::size_t e_l1_icache_size = 65536;
  static constexpr std::size_t e_l2_cache_num = 4;
  static constexpr std::size_t e_l2_cache_size = 4.194e+6;
  static constexpr std::size_t e_l3_cache_num = 4;
  static constexpr std::size_t e_l3_cache_size = 3.146e+6;

  static constexpr std::size_t total_threads = 32;
  static constexpr std::size_t prefetchers = 8;
  static constexpr std::array instruction_sets{General,
    SSE, SSE2, SSE3, SSE4_1, SSE4_2, SSSE3, AVX2, BMI1, BMI2, F16C, AVX512F,
    AVX512CD, AVX512PF, AVX512ER, AVX512BW, AVX512DQ, AVX512VL, AVX512_IFMA,
    AVX512_VBMI, AVX512_VBMI2, AVX512_BITALG, AVX512_VPOPCNTDQ, AVX512_4FMAPS,
    AVX512_4VNNIW, AVX512_FP16, AVX512_BF16, AVX512_VNNI, AVX512_VP2INTERSECT,
    VAES,  VPCLMULQDQ, GFNI};
  // This should be in a seperate SysInfo struct
  static constexpr std::size_t page_size = 16384;
};

template <typename T>
concept Enginei913900KF =
  is_expression<T> &&
  is_same<CpuEngine<i913900KF>, engine_type_t<T>>;

// Use concept to overspecialize dot product for above
template <Enginei913900KF Expr1, Enginei913900KF Expr2>
constexpr inline auto dot_product(Expr1&& expr1, Expr2&& expr2) {
  // My very clever dot product code...
}
}

using yae::Vector, yae::Dynamic, yae::Index, yae::Options;
using opt = Options<double,
  Index,
  yae::default_alloctor<double>,
  yae::CPUEngine<yae::i913900KF>>;
Vector<opt, Dynamic> vec_1(5);
vec_1.set_random();
Vector<opt, Dynamic> vec_2(5);
vec_2.set_random();
// Tensor multiplication can use the info above.
double res = dot_product(vec_1, vec_2);
```

A utility will be available to auto generate the CPU information structs users given their CPU.

6. Smartly reusing memory when possible.

When an object is a temporary, the internals should be smart enough to know we can use the temporaries memory in place.

```c++
using yae::Matrix, yae::Dynamic, yae::Options;
using opt = Options<double>;
Matrix<opt, Dynamic, Dynamic>  mat_1(5, 10);
mat_1.set_random();
Matrix<opt, Dynamic, Dynamic>  mat_2(5, 10);
mat_2.set_random();

using yae::to_array;
// An expression taking ownership of mat_1
auto mat_1_mutate = to_array(std::move(mat_1)) | yae::exp;
// mat_3 will use the memory from mat_1_mutate
auto mat_3 = to_array(mat_2) + to_array(std::move(mat_1_mutate));
```

---

## Motivation

Eigen is a fantastic package, but over the years I've found a few issues with it that appear unresolvable.

1. It has had a very slow turnaround time for new versions (the latest release was 3 years ago!)
2. The use of `auto` is very dangerous
3. It's was originally written in C++03, which means Eigen's code has a lot of workarounds for expression templating that both limit the code and make it very difficult to parse and compile.
3. Because of (3) constexpr matrix types will never be a thing (it has been tried and reverted sadly)
4. GPU support is second class, sitting in their unsupported Tensor library
5. It's rather hard to extend, for instance if you know a better optimization for your CPU's dot product, too bad!
6. Though Eigen has the `Map` class, it can be difficult to use your own allocator and requires a lot of extra boilerplate
7. Eigen does not support perfect forwarding within its expression templates, which does not allow nice memory reuse and can lead to bugs in hanging refs

```c++
inline auto foo(const Eigen::MatrixXd& x) {
  // When foo(x) is eventually evaluated, 3.0 will be gone!
  return X.array() + 3.0;
}
```

The points in the summary directly address these. The goal of this library is to address those shortcomings.

---

## Guide-level explanation

### Introducing new named concepts

- **`Tensor<Options, std::integral... Dims>`**
  - The core multi-dimensional array type.

- **Template parameters**

  - `Options<>`
  - Packs the compile time config options for a particular tensor.

  - `Dims...`
    Dimension sizes: compile-time constants for fixed axes, or the sentinel `Dynamic` for runtime sizing.
    - **Fixed vs. dynamic dimensions**

      - **Fixed-size** `(Dims != Dynamic ) &&...`
        - By default allocates all storage on stack and supports `constexpr` evaluation.
        - Users can set a flag `ForceDynamic` in `Options<>` that will force fixed sized matrices to be allocated in an allocator.
      - **Dynamic-size** `(Dims == Dynamic ) ||...`
        - If any dimension is dynamic, the matrix is considered dynamic and must be evaluated at runtime.
        - Requires passing sizes at runtime and allocates via the chosen allocator.

- **Allocator aware**
  - By default, `Options<>` uses a default allocator users do not need to pass. To control where a tensor allocates its storage, supply your own allocator type in `Options` and pass an allocator instance to the constructor.

```cpp
// Make a tensor allocated in a monotonic buffer resource
using PolyAlloc = std::pmr::polymorphic_allocator<double>;
std::pmr::monotonic_buffer_resource mr;
PolyAlloc alloc(&mr);
// 5×2×8 dynamic tensor allocated in user’s memory resource
yae::Tensor<yae::Options<double, yae::Index, PolyAlloc>,
    Dynamic, 2, 8> t(alloc, 5, 2, 8);
```

When two operand tensors carry different allocators, operations will throw a compile or runtime exception until you disambiguate which allocator to use for the result.

```text
error: cannot multiply tensors with different allocators;
        please call `(A * B).allocator(evaluation_allocator)` to pick result allocator
```

- **Owning vs. mapping (replacing `Eigen::Map`)**
  Passing a raw pointer to the constructor tells `Tensor` to **map** external storage instead of allocating:

In this example code `mmap_tensor` makes a view of `data`.

```cpp
static constexpr double data[] = {1,2,3,4};
// Like Eigen::Map<const Matrix<double,2,2>>
constexpr yae::Tensor<yae::Options<const double>, 2, 2>
  mmap_tensor(data);
// Dynamic only possible at runtime
yae::Tensor<yae::Options<const double>, yae::Dynamic, yae::Dynamic>
  dyn_mmap_tensor(data, 2, 2);
```

- **Replacing `Eigen::Map`**
Instead of `Eigen::Map<Matrix<double,R,C>>(ptr)`, we can pass a pointer to a Tensor and it will assume it does not manage that memory.

```cpp
// For compile time sized tensors we do not pass sizes
yae::Tensor<yae::Options<double>, 5, 10>  M(ptr);
// For dynamic sizes we must pass sizes at runtime
yae::Tensor<yae::Options<double>, Dynamic, Dynamic>  M(ptr, 5, 10);
```

You get the same zero-copy “view” semantics without a separate Map class or boilerplate.

### Introducing New Named Concepts

- **`Options<Scalar, Index, Allocator, Engine, Layout, AccessorPolicy>`**
  - The compile-time “policy bundle” that drives every tensor’s element type, indexing, memory resource, execution backend, data layout, and accessor behavior.

```cpp
template<
  typename Scalar,
  typename Allocator = std::allocator<Scalar>,
  typename Engine    = CpuEngine<DefaultCPU>,
  typename Index     = index_t, // Defaults to std::size_t
  typename Layout    = std::layout_left, // Column major by default
  typename AccessorPolicy = std::default_accessor<Scalar>
>
struct Options { ... };
```

The layout and accessor policy come from c++23's `mdspan` which is used in the tensor for managing access to memory.

The Engine must satisfy either the `CpuInfo` concept or the `GpuInfo` concept.

- **`CpuInfo` concept**

The `CpuInfo` concept gives the minimum values that must be passed to be accepted as an engine for the CPU.

```cpp
template <typename Info>
concept CpuInfo = requires(Info x) {
  Info::total_cores;
  Info::l1_dcache_num;
  Info::l1_dcache_size;
  Info::l1_icache_num;
  Info::l1_icache_size;
  Info::l2_cache_num;
  Info::l2_cache_size;
  Info::l3_cache_num;
  Info::l3_cache_size;
  Info::total_threads;
  Info::prefetchers;
  Info::instruction_sets;
};
```

- **`CpuEngine<Info>`**
  Ties an `Info` model (e.g. `DefaultCPU`) to the vector-intrinsics implementation. Because functions that use concepts choose the most specific candidate function, writing your own overload for `Engine<MyCpu>` will allow you to overload functions for your device.

- **`GpuInfo` concept**

The `GpuInfo` concept gives the minimum values that must be passed to be accepted as an engine for the GPU.

```cpp
template <typename Info>
concept GpuInfo = requires(Info x) {
  Info::vendor;
  Info::compute_units;
  Info::compute_capability;
  Info::max_work_item_dims;
  Info::max_work_item_sizes;
  Info::max_work_group_size;
  Info::device_preferred_work_group_size_multiple;
  Info::kernel_preferred_work_group_size_multiple;
  Info::warp_size;
  Info::min_alignment;
  Info::global_cache_size;
  Info::global_memory_cache_lize_size;
  Info::local_mem_size;
  Info::registers_per_block;
}
```

- **`GpuEngine<Info>`**

Ties an `Info` model (e.g. `DefaultGPU`) to the vector-intrinsics implementation. Because functions that use concepts choose the most specific candidate function, writing your own overload for `Engine<MyGpu>` will allow you to overload functions for your device.

### Examples

1. **Default options**
   - Using nothing but the defaults yields a plain CPU tensor, 64-byte aligned, `std::allocator<double>`, and `index_t` for indices:

```cpp
using Opt = yae::Options<double>;
// Uses CpuEngine<DefaultCPU>, stack allocates 3×3 doubles, aligned to 64 bytes
yae::Tensor<Opt, 3, 3> A;
```

2. **Custom allocator + GPU engine**

The below example creates a dynamically sized tensor using cuda unified memory.

```cpp
using alloc_t = std::pmr::polymorphic_allocator<float>;
auto cuda_resource = yae::cuda_unified_memory_resource;
using GOpt = yae::Options<
              float,
              alloc_t,
              yae::GPUEngine<yae::DefaultGPU>,
              std::uint32_t
            >;
alloc_t alloc(&cuda_resource);
// B’s storage lives in my_resource and uses the cuda backend
yae::Tensor<GOpt, Dynamic, 64, 64> B(alloc, 128, 64, 64);
```

3. **Injecting a custom CPU descriptor**

In this example we make a custom CPU overload and pass it to a tensor. When evaluating the tensor, cache sizes, core count, and the number of prefetchers are utilized for optimal iteration, GEMM, and decompositions.

```cpp
// Make a dummy struct for specializing the CpuEngine
struct MyCPU {};

template<> struct CpuEngine<MyCPU> { /* tuning parameters... */ };

using MyOpt = yae::Options<double, std::allocator<double>, CpuEngine<MyCPU>>;
```

4. **Layout and accessor policies**

   Swap row-major vs. column-major or add bounds-checking:

```cpp
using SafeOpt = yae::Options<
                  double,
                  std::allocator<double>,
                  CpuEngine<DefaultCPU>,
                  index_t,
                  std::layout_right,
                  yae::checked_accessor<double>
                >;
```

5. Force dynamic allocation and padding

In the below example, we set the options so that the tensors using this config are always allocated with an allocator and are padded to the maximum size of a SIMD packet.

```cpp
// I'm still unsure where to put the options for packing, force dynamic, etc. Maybe in other sub struct?
using DynOpt = yae::Options<double,
  std::allocator<double>,
  CpuEngine<DefaultCPU<yae::packing, yae::force_dynamic>>,
  index_t,
  std::layout_left,
  >;
```

### How Eigen programmers should *think* about `Options`

- **Plug-and-play policy**
  Instead of juggling separate Map/Matrix types or global flags, you carry *one* `Options` typedef through your code. Swapping memory resource or engine is just a template alias change.

```cpp
// Starting with base options we can modify the options we want to change using type traits
using base_opt = yae::Options<>;
// Change allocator to std::polymorphic_allocator
using polymorphic_opt =
  yae::change_allocator_t<base_opt, std::polymorphic_allocator<double>>;
// Change to std::layout_right
using row_major_opt = yae::set_row_major_t<polymorphic_opt>;
```

- **Explicit resource control**
  No hidden `new`/`delete`. If you mix two tensors with different allocators and no policy for resolution between the two, you’ll get a runtime or compile-time error:

```text
error: cannot assign Tensor<...,PoolA> to Tensor<...,PoolB>;
        mismatched Allocator policies
```

- **Compile-time tuning**
  Cache sizes, page sizes, and prefetchers all live in your chosen `Info` type. Either the engine
  specialization will choose an optimal setup or users can override functions in `yae` explicitly.

### Migration guidance & error samples

- **From Eigen**

  ```cpp
  // Eigen:
  Eigen::Matrix<double,4,4> M;
  Eigen::Map<Eigen::MatrixXd> V(ptr, n, n);
  ```

  becomes

  ```cpp
  // yae:
  using Opt = yae::Options<double>;
  yae::Matrix<Opt,4,4> M;
  yae::Matrix<yae::Options<const double>, Dynamic, Dynamic> V(ptr, n, n);
  ```

### Teaching existing vs. new users

- **Existing Eigen users**
  - You keep familiar shape aliases (`Matrix`, `Map` → `Tensor<...,R,C>`) but adopt a uniform policy-based API. No more free-standing `.array()` or `Eigen::internal` hacks.
- **New users**
  - Learn *one* composable `Options` bundle instead of separate types. The template parameters are orthogonal: pick your scalar, allocator, engine, and you’re done.

- **Array-style transformations**

  - **`to_array(expr)`**: view a tensor as an “array expression” for element-wise ops
  - **Pipe operator** (`|`): chain array transforms like `exp`, `sum`, etc., returning a lazy expression

---

### Examples

1. **Compile-time (`constexpr`) tensors**
   - Because the data pointer and dimensions are `constexpr`, you can do everything at compile time and even `static_assert` on results:

```cpp
static constexpr double mat_data[4] = {1,2,3,4};
using Opt = yae::Options<const double>;
constexpr yae::Matrix<Opt,2,2>  M(mat_data);
constexpr yae::Vector<Opt,2>     v({1,2});
constexpr yae::Vector<Opt,2> r = M * v;  // computed at compile time
static_assert(r(1,0)==7.0);
```

2. **Allocator-aware, dynamic tensors**

```cpp
using PolyAlloc = std::pmr::polymorphic_allocator<double>;
std::pmr::monotonic_buffer_resource  mr;
PolyAlloc                             alloc(&mr);
// 5×2×8 tensor in user-supplied memory
yae::Tensor<yae::Options<double,yae::Index,PolyAlloc>,
            Dynamic,2,8> t1(alloc,5,2,8);
t1.set_random();
```

   When you combine two tensors with different allocators, you’ll get a clear compile-time error. You can then explicitly choose the result allocator:

```text
error: cannot multiply tensors with different allocators;
      please specify result allocator via `.allocator(...)`
```

3. **Expression fusion on CPU and GPU**
   - On CUDA-backed engines you still get fused kernels—no intermediate copies:

```cpp
using Opt = yae::Options<double,Index,
          yae::cuda_unified_allocator<double>,
          yae::GPUEngine<yae::DefaultGPU>>;
yae::Tensor<Opt,Dynamic,2,8> A(alloc,5,2,8), B(alloc,5,8,3);
auto C = A * B;  // launches one batched GEMM on device
```

4. **Free-function, Eigen-like API with pipes**
   - Element-wise transforms and reductions are free functions, chained via `|`:

```cpp
auto sum_exp = yae::to_array(T) | yae::exp | yae::sum;
auto R       = T1 * T2 + sum_exp;
```

5. **User-extensible engines and intrinsics**
   - You can specialize `CpuEngine<MyCPU>` or overload `yae` for your microarchitecture. A helper can auto-generate your CPU’s cache sizes and instruction sets for you.

6. **Smart memory reuse**
   - Temporaries get moved and reused in place when safe:

```cpp
auto X = to_array(std::move(M)) | yae::exp;  // reuses M’s buffer
```

---

## Reference-level explanation

**Overview of Code-generation via Expression Templates**

When you write a chained tensor expression, nothing computes immediately. Instead, each operator builds a node in a compile-time expression tree. Only when you assign or explicitly call evaluate do we “walk” that tree and emit real loops.

#### 1. Lazy expression trees

Take:

```cpp
auto expr = yae::to_array(A) | yae::exp | yae::sum;
```

Under the hood you get a type resembling:

```cpp
SumExpr<
  ExpExpr<
    ArrayExpr<Tensor<double>>
  >
>;
```

Each `{*}Expr` node holds references or temporaries of its child node(s) + a functor (`exp`, `sum`, etc.).

An example exponential expression looks like the following

```cpp
template<Expression Expr>
struct ExpExpr {
  // Keep track of the number of underlying expressions that actually holding data
  static constexpr std::size_t arity = Expr::arity;
  // own_expr_t decides if we should reference or own the incoming object
  own_expr_t<Expr&&> expr_;
  constexpr ExpExpr(Expr&& expr) : expr_(std::forward<Expr>(expr)) {}
  // For single coeff evaluation
  template <std::integral... Idxs>
  constexpr operator()(Idxs... idxs) {
    return std::exp(expr(idxs...));
  }
  // For evaluating packets of expr at a time with SIMD
  template <std::integral Idx>
  constexpr packet(Idx idx) {
    using instr_set = get_instr_set<Expr>;
    return yae::exp<instr_set>(expr.packet(i));
  }
  // For jit kernels on the GPU
  constexpr kernel() requires(is_gpu_engine<Expr>) {
    using vendor_t = get_vendor<Expr>;
    // Can be options like 'fast' for using `__expf` instead of `expf`
    using gpu_opts = get_gpu_opts<Expr>;
    // adds exp(x[i]) to the gpu kernel
    return yae::exp<vendor_t, gpu_opts>(expr);
  }
}
```

Each expression keeps a compile time value indicating the overall arity of the expression. For instance, an expression that adds four matrices together will have an arity of 4.

```cpp
auto expr = arr_1 + arr_2 + arr_3 + arr_4; // binary op
/*
Generates the following Expression structure with `Expr::arity = 4`
AddExpr<
  AddExpr<
    AddExpr<
      ArrayExpr<Tensor<double>>,
      ArrayExpr<Tensor<double>>
    >,
    ArrayExpr<Tensor<double>>
  >,
  ArrayExpr<Tensor<double>>
*/
```

(Note from steve) I'm stil working this out, but I think there is a way to use the arity of the expression along with the dimensions of the problem to setup how to best utilize the prefetchers.

#### 2. Materialization via `evaluate()`

As soon as you do an assignment to a tensor the `yae` library invokes an evaluator to assign the right hand side.

```cpp
// yae::evaluate(out, expr);
Tensor<opt> out = expr;            // assignment
// yae::evaluate(out, left + right);
Tensor<opt> res = left + right; // binary op
```

The tensor classes `operator=` looks like the following in pseudocode, where a `tensor_evaluator` is a generic tree-walker that processes the expression tree.

```cpp
template<UserOptions Opt, index_type_t<Opt>... Ds>
template<Expression Expr>
auto& Tensor<Opt,Ds...>::operator=(Expr&& expr) {
  resize_this_if_needed(*this, expr);
  tensor_evaluator(*this, std::forward<Expr>(expr));
  return *this;
}
```

That `tensor_evaluator` is a **generic tree-walker** which:

1. Pulls compile-time info (size, SIMD width, prefetchers) from your chosen `CpuEngine<Info>`.
2. Emits one fused, vectorized loop over the entire result tensor.

### 3. Generic CPU tree-walker pseudocode

Below is a simplified pseudocode for the `evaluate()` function that processes the expression tree for an array like function.

For our example we will think about an add expression with two inner add expressions:

```cpp
AddExpr<
  AddExpr<
    ArrayExpr<Tensor<double>>,
    ArrayExpr<Tensor<double>>
  >,
  ArrayExpr<Tensor<double>>
```

```cpp
template<CPUTensor Out, CPUExpression Expr>
constexpr void array_evaluate(Out&& out, Expr&& expr) {
  using Info = get_engine<Expr>;
  using inst_set = get_instruction_set<Info>;
  using Scalar = typename Expr::Scalar; // e.g. double, float, etc.
  using index_t = get_index_t<Expr>;
  constexpr index_t simd_width = simd_width_v<Info, Scalar>; // lanes per vector
  constexpr index_t num_prefetchers = Info::prefetchers; // prefetch count
  const index_t N = expr.size();
  // Number of tlb pages we will need per underlying tensor
  const index_t num_bytes = N * sizeof(Scalar);
  const index_t num_pages = 1 + ((num_bytes - 1) / Info::page_size);
  // for large matrices always the tlb page size, otherwise less
  const index_t page_size = num_bytes < Info::page_size ? num_bytes : Info::page_size;
  const index_t packets_per_page = (simd_width * sizeof(Scalar)) / page_size;
  for (index_t packet_i = 0; packet_i < packets_per_page; ++packet_i) {
    // TODO: Break this up further to operate on batches of pages for the
    //  number of prefetchers available.
    for (index_t page_i = 0; page_i < num_pages; ++page_i) {
      index_t base_idx = page_i * page_size;
      index_t idx      = base_idx + packet_i * simd_width;
      assign(out.packet(idx), expr.packet(idx));
    }
  }
  // 4) Handle any remaining
  index_t processed = packets_per_page * simd_width * num_pages;  // total elems done above
  for (index_t i = processed; i < N; ++i) {
    assign(out(i), expr(i));  // fallback to scalar eval
  }
}
```

### GPU Kernel Fusion

GPU kernel fusion would operate much like Stan Math's [OpenCL kernel fusion](https://github.com/stan-dev/math/blob/develop/stan/math/opencl/kernel_generator/elt_function_cl.hpp).

The expressions for GPU kernels will accumulate the components of a kernel into separate components of the overall kernel. Then at the end the kernel components will be accumulated into one string and compiled at runtime via a runtime driver.

```cpp
struct kernel_parts {
  std::string includes;  // any function definitions - as if they were included
                         // at the start of kernel source
  std::string declarations;    // declarations of any local variables
  std::string initialization;  // the code for initializations done by all
                               // threads, even if they have no work
  std::string
      body_prefix;   // the code that should be run at the start of the kernel
                     // body (before the code for arguments of an operation)
  std::string body;  // the body of the kernel - code executing operations
  std::string body_suffix;  // the code that should be run at the end of the
                            // kernel body
  std::string
      reduction_1d;  // the code for reductions within work group by all
                     // threads, even if they have no work. Run once per column.
  std::string
      reduction_2d;  // the code for reductions within work group by all
                     // threads, even if they have no work. Run only once.
  std::string args;  // kernel arguments

  kernel_parts operator+(const kernel_parts& other) {
    return {includes + other.includes,
            declarations += other.declarations,
            initialization + other.initialization,
            body_prefix + other.body_prefix,
            body + other.body,
            body_suffix + other.body_suffix,
            reduction_1d + other.reduction_1d,
            reduction_2d + other.reduction_2d,
            args + other.args};
  }

  kernel_parts& operator+=(const kernel_parts& other) {
    includes += other.includes;
    declarations += other.declarations;
    initialization += other.initialization;
    body_prefix += other.body_prefix;
    body += other.body;
    body_suffix += other.body_suffix;
    reduction_1d += other.reduction_1d;
    reduction_2d += other.reduction_2d;
    args += other.args;
    return *this;
  }
};

inline std::ostream& operator<<(std::ostream& os, kernel_parts& parts) {
  os << "args:" << std::endl;
  os << parts.args.substr(0, parts.args.size() - 2) << std::endl;
  os << "Decl:" << std::endl;
  os << parts.declarations << std::endl;
  os << "Init:" << std::endl;
  os << parts.initialization << std::endl;
  os << "body:" << std::endl;
  os << parts.body << std::endl;
  os << "body_suffix:" << std::endl;
  os << parts.body_suffix << std::endl;
  os << "reduction_1d:" << std::endl;
  os << parts.reduction_1d << std::endl;
  os << "reduction_2d:" << std::endl;
  os << parts.reduction_2d << std::endl;
  return os;
}
```

An example base expression for elementwise expressions looks like the following. You can see a full impl [here](https://github.com/stan-dev/math/blob/develop/stan/math/opencl/kernel_generator/elt_function_cl.hpp#L1) in Stan Math.

```cpp
template<class T>
concept StringLike = std::is_convertible_v<T, std::string_view>;

template <GpuExpression Derived, GpuExpression... Exprs>
class elt_function : public gpu_operation<Derived, Exprs...> {
 public:
  using Scalar = scalar_type_t<Derived>;
  using base = gpu_operation<Derived, Exprs...>;
  using base::var_name_;

  /**
   * Constructor
   * @tparam Str A type that is convertible to a string view
   * @param fun function
   * @param args argument expression(s)
   */
  template <StringLike Str>
  constexpr elt_function(Str&& fun, T&&... args)  // NOLINT
      : base(std::forward<T>(args)...), fun_(std::forward<Str>(fun)) {}

  /**
   * Generates kernel code for this expression.
   * @param row_index_name row index variable name
   * @param col_index_name column index variable name
   * @param var_names_arg variable names of the nested expressions
   * @return part of kernel with code for this expression
   */
  template <StringLike StrRow, StringLike StrCol, StringLike... Names>
  constexpr inline kernel_parts generate(
      StrRow&& /*row_index_name*/, StrCol&& /*col_index_name */,
      Names&&... var_names_arg) const {
    kernel_parts res{};

    for (const char* incl : base::derived().includes) {
      res.includes += incl;
    }
    std::array<std::string, sizeof...(T)> var_names_arg_arr
        = {(std::forward<Names>(var_names_arg) + ", ")...};
    std::string var_names_list = std::accumulate(
        var_names_arg_arr.begin(), var_names_arg_arr.end(), std::string());
    res.body = type_str<Scalar>() + " " + var_name_ + " = " + fun_ + "((double)"
               + var_names_list.substr(0, var_names_list.size() - 2) + ");\n";
    return res;
  }
 protected:
  std::string fun_;
};
```

More research needs to be done on kernel runtime generation. For a cuda target [NVRTC](https://docs.nvidia.com/cuda/archive/10.1/pdf/NVRTC_User_Guide.pdf) is the most likely driver target.

# Drawbacks

This project would be a huge amount of work. I would only do it with buy in / interest from a lot of people at flatiron.

The overall picture for this project looks something like the following (+ all the tests for the following)

1. Build out base tensor types
2. CPU version running
3. SIMD Packet math for common instruction sets
4. Hooking up to BLAS for GEMM and decompositions
5. Basic GPU setup
6. Kernel Compilation
7. cuBLAS for gpu GEMM and decompositions
8. Useful docs for all of the above

Each of these 8 steps can each be a huge amount of work.

The one good thing is that we can actually utilize a lot of other open source projects as the base for this

1. Eigen's packet math can be modified slightly to fit this framework
2. Stan's OpenCL backend already does the kernel generation. We just need to change it for CUDA.

# Rationale and alternatives

- Why is this design the best in the space of possible designs?

I chose my design based off of Eigen and Blaze, which imo have nice APIs. This design doc is an attempt to handle some of Eigen's short comings.

- What other designs have been considered and what is the rationale for not choosing them?

I've tried making pull requests for several of these things in Eigen. The authors of Eigen do not seem interested in allocator aware matrices. And because of some internal choices they are unable to have matrices that can operate at compile time.

- What is the impact of not doing this?

Nothing that bad. We will stay with Eigen being pretty much the only used matrix library in C++.

# Unresolved questions

- What parts of the design do you expect to resolve through the RFC process before this gets merged?
- What parts of the design do you expect to resolve through the implementation of this feature before stabilization?
- What related issues do you consider out of scope for this RFC that could be addressed in the future independently of the solution that comes out of this RFC?
