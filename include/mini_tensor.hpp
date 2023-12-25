//          Copyright (C) Hal Finkel 2023.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE_1_0.txt or copy at
//          https://www.boost.org/LICENSE_1_0.txt)
// SPDX-License-Identifier: BSL-1.0

#include <array>
#include <algorithm>
#include <functional>
#include <numeric>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#ifndef MINI_TENSOR_HPP_INCLUDED
#define MINI_TENSOR_HPP_INCLUDED

#ifndef MINI_TENSOR_RECURSE_SIZE
#define MINI_TENSOR_RECURSE_SIZE 16
#endif

#ifndef MINI_TENSOR_UNROLL_COUNT
#define MINI_TENSOR_UNROLL_COUNT 4
#endif

namespace mini_tensor {
namespace detail {
template <std::size_t I, typename T0, typename... Ts>
inline constexpr decltype(auto) ith_value(T0 t0, Ts... ts) {
  if constexpr (I > 0)
    return ith_value<I-1>(ts...);
  else
    return t0;
}

template <std::size_t V, std::size_t... Vs, std::size_t... Is>
inline constexpr std::size_t first_index_match_idx(std::index_sequence<Is...>) {
  std::size_t r{};
  (void) ((V == Vs ? (r = Is, true) : false) || ...);
  return r;
}

template <std::size_t V, std::size_t... Vs>
inline constexpr std::size_t first_index_match_idx() {
  return first_index_match_idx<V, Vs...>(std::make_index_sequence<sizeof...(Vs)>{});
}
} // namespace detail

template <std::size_t... Dims>
struct dims {
  static constexpr std::size_t rank = sizeof...(Dims);

  template <typename... Ts,
            typename = std::enable_if_t<(sizeof...(Ts) == sizeof...(Dims)+1) &&
                                        (std::is_convertible_v<Ts, std::size_t> && ...)>>
  constexpr static std::size_t storage_index(Ts... idxs) {
    return wld<Dims..., 1>::compute_index(idxs...);
  }

  constexpr static std::size_t base_size() {
    return (Dims * ... * 1);
  }

  template <std::size_t SlotI>
  constexpr static std::size_t dim_size() {
    return detail::ith_value<SlotI>(Dims...);
  }

private:
  template <std::size_t... DimsW>
  struct wld {
    static_assert(std::is_same_v<std::index_sequence<DimsW...>,
                                 std::index_sequence<Dims..., 1>>);

    template <typename... Ts,
              typename = std::enable_if_t<(std::is_convertible_v<Ts, std::size_t> && ...)>>
    constexpr static std::size_t compute_index(Ts... idxs) {
      // Computing idxs[-1] + Dims[-1](*idxs[-2] + Dims[-2]*(idxs[-3] + Dims[-3]*(idxs[-4] + ...
      //  = idxs[-1] + Dims[-1]*idxs[-2] + Dims[-1]*Dims[-2]*idxs[-3] + Dims[-1]*...*Dims[-3]*idxs[-4] + ...
      // where the negative index values above indicate reverse indexing.

      // To effectively iterate over a pack in reverse order, we'll use the
      // assignment trick, see https://www.foonathan.net/2020/05/fold-tricks/

      std::size_t d{1}, r{0}, u{};
      (u = ... = (r += idxs*(d *= DimsW), 0));
      return r;
    }
  };
};

template <std::size_t I>
struct index {
  static constexpr std::size_t i = I;
};

template <typename T>
struct is_index {
  static constexpr bool value = false;
};

template <std::size_t I>
struct is_index<index<I>> {
  static constexpr bool value = true;
};

template <typename T>
inline constexpr bool is_index_v = is_index<T>::value;

namespace detail {
// Split a dims object into the head dimension and the tail dimensions.
template <typename T> struct dims_split_head {};
template <std::size_t DimO, std::size_t... Dims>
struct dims_split_head<dims<DimO, Dims...>> {
  constexpr static std::size_t outer_size = DimO;
  using inner_dims_t = dims<Dims...>;
};

struct indexed_exp {};

template <typename TEL, typename TER, typename... IndexTs>
struct indexed_binop_exp : public indexed_exp {
  indexed_binop_exp(const TEL &tel, const TER &ter) : tel(tel), ter(ter) {}

  using value_type = std::common_type_t<typename TEL::value_type,
                                        typename TER::value_type>;

  static constexpr std::size_t max_index() {
    return std::max(TEL::max_index(), TER::max_index());
  };

  template <std::size_t I>
  static constexpr bool has_index() {
    return TEL::template has_index<I>() || TER::template has_index<I>();
  }

  template <std::size_t LI>
  std::size_t dim_size_for_index_use() const {
    if constexpr (TEL::template has_index<LI>())
      return tel.template dim_size_for_index_use<LI>();
    else
      return ter.template dim_size_for_index_use<LI>();
  }

  const TEL &tel;
  const TER &ter;
};

template <typename TEL, typename TER, typename... IndexTs>
struct indexed_add_exp : public indexed_binop_exp<TEL, TER, IndexTs...> {
  indexed_add_exp(const TEL &tel, const TER &ter) :
    indexed_binop_exp<TEL, TER, IndexTs...>(tel, ter) {}

  template <typename... Ts>
  decltype(auto) eval(Ts... idxs) const {
    return this->tel.eval(idxs...) + this->ter.eval(idxs...);
  }
};

template <typename TEL, typename TER, typename... IndexTs>
struct indexed_sub_exp : public indexed_binop_exp<TEL, TER, IndexTs...> {
  indexed_sub_exp(const TEL &tel, const TER &ter) :
    indexed_binop_exp<TEL, TER, IndexTs...>(tel, ter) {}

  template <typename... Ts>
  decltype(auto) eval(Ts... idxs) const {
    return this->tel.eval(idxs...) - this->ter.eval(idxs...);
  }
};

template <typename TEL, typename TER, typename... IndexTs>
struct indexed_mul_exp : public indexed_binop_exp<TEL, TER, IndexTs...> {
  indexed_mul_exp(const TEL &tel, const TER &ter) :
    indexed_binop_exp<TEL, TER, IndexTs...>(tel, ter) {}

  template <typename... Ts>
  decltype(auto) eval(Ts... idxs) const {
    return this->tel.eval(idxs...) * this->ter.eval(idxs...);
  }
};

template <typename IEL, typename IER,
          typename = std::enable_if_t<std::is_base_of_v<indexed_exp, IEL> &&
                                      std::is_base_of_v<indexed_exp, IER>>>
decltype(auto) operator + (const IEL &iel, const IER &ier) {
  return indexed_add_exp<IEL, IER>(iel, ier);
}

template <typename IEL, typename IER,
          typename = std::enable_if_t<std::is_base_of_v<indexed_exp, IEL> &&
                                      std::is_base_of_v<indexed_exp, IER>>>
decltype(auto) operator - (const IEL &iel, const IER &ier) {
  return indexed_sub_exp<IEL, IER>(iel, ier);
}

template <typename IEL, typename IER,
          typename = std::enable_if_t<std::is_base_of_v<indexed_exp, IEL> &&
                                      std::is_base_of_v<indexed_exp, IER>>>
decltype(auto) operator * (const IEL &iel, const IER &ier) {
  return indexed_mul_exp<IEL, IER>(iel, ier);
}

template <typename TE, typename... IndexTs>
struct indexed_unaryop_exp : public indexed_exp {
  indexed_unaryop_exp(const TE &te) : te(te) {}

  using value_type = typename TE::value_type;

  static constexpr std::size_t max_index() {
    return TE::max_index();
  };

  template <std::size_t I>
  static constexpr bool has_index() {
    return TE::template has_index<I>();
  }

  template <std::size_t LI>
  std::size_t dim_size_for_index_use() const {
    return te.template dim_size_for_index_use<LI>();
  }

  const TE &te;
};

template <typename TE, typename... IndexTs>
struct indexed_neg_exp : public indexed_unaryop_exp<TE, IndexTs...> {
  indexed_neg_exp(const TE &te) :
    indexed_unaryop_exp<TE, IndexTs...>(te) {}

  template <typename... Ts>
  decltype(auto) eval(Ts... idxs) const {
    return -this->te.eval(idxs...);
  }
};

template <typename TE, typename... IndexTs>
struct indexed_pos_exp : public indexed_unaryop_exp<TE, IndexTs...> {
  indexed_pos_exp(const TE &te) :
    indexed_unaryop_exp<TE, IndexTs...>(te) {}

  template <typename... Ts>
  decltype(auto) eval(Ts... idxs) const {
    return +this->te.eval(idxs...);
  }
};

template <typename IE,
          typename = std::enable_if_t<std::is_base_of_v<indexed_exp, IE>>>
decltype(auto) operator - (const IE &ie) {
  return indexed_neg_exp<IE>(ie);
}

template <typename IE,
          typename = std::enable_if_t<std::is_base_of_v<indexed_exp, IE>>>
decltype(auto) operator + (const IE &ie) {
  return indexed_pos_exp<IE>(ie);
}

template <typename TE, typename... IndexTs>
struct indexed_scalar_mul_exp : public indexed_unaryop_exp<TE, IndexTs...> {
  indexed_scalar_mul_exp(const TE &te, const typename TE::value_type &v) :
    indexed_unaryop_exp<TE, IndexTs...>(te), v(v) {}

  template <typename... Ts>
  decltype(auto) eval(Ts... idxs) const {
    return v * this->te.eval(idxs...);
  }

  typename TE::value_type v;
};

template <typename TE, typename... IndexTs>
struct indexed_scalar_divl_exp : public indexed_unaryop_exp<TE, IndexTs...> {
  indexed_scalar_divl_exp(const TE &te, const typename TE::value_type &v) :
    indexed_unaryop_exp<TE, IndexTs...>(te), v(v) {}

  template <typename... Ts>
  decltype(auto) eval(Ts... idxs) const {
    return v / this->te.eval(idxs...);
  }

  typename TE::value_type v;
};

template <typename TE, typename... IndexTs>
struct indexed_scalar_divr_exp : public indexed_unaryop_exp<TE, IndexTs...> {
  indexed_scalar_divr_exp(const TE &te, const typename TE::value_type &v) :
    indexed_unaryop_exp<TE, IndexTs...>(te), v(v) {}

  template <typename... Ts>
  decltype(auto) eval(Ts... idxs) const {
    return this->te.eval(idxs...) / v;
  }

  typename TE::value_type v;
};

template <typename TE, typename... IndexTs>
struct indexed_scalar_add_exp : public indexed_unaryop_exp<TE, IndexTs...> {
  indexed_scalar_add_exp(const TE &te, const typename TE::value_type &v) :
    indexed_unaryop_exp<TE, IndexTs...>(te), v(v) {}

  template <typename... Ts>
  decltype(auto) eval(Ts... idxs) const {
    return v + this->te.eval(idxs...);
  }

  typename TE::value_type v;
};

template <typename TE, typename... IndexTs>
struct indexed_scalar_subl_exp : public indexed_unaryop_exp<TE, IndexTs...> {
  indexed_scalar_subl_exp(const TE &te, const typename TE::value_type &v) :
    indexed_unaryop_exp<TE, IndexTs...>(te), v(v) {}

  template <typename... Ts>
  decltype(auto) eval(Ts... idxs) const {
    return v - this->te.eval(idxs...);
  }

  typename TE::value_type v;
};

template <typename TE, typename... IndexTs>
struct indexed_scalar_subr_exp : public indexed_unaryop_exp<TE, IndexTs...> {
  indexed_scalar_subr_exp(const TE &te, const typename TE::value_type &v) :
    indexed_unaryop_exp<TE, IndexTs...>(te), v(v) {}

  template <typename... Ts>
  decltype(auto) eval(Ts... idxs) const {
    return this->te.eval(idxs...) - v;
  }

  typename TE::value_type v;
};

template <typename IE,
          typename = std::enable_if_t<std::is_base_of_v<indexed_exp, IE>>>
decltype(auto) operator + (const IE &ie, const typename IE::value_type &v) {
  return indexed_scalar_add_exp<IE>(ie, v);
}

template <typename IE,
          typename = std::enable_if_t<std::is_base_of_v<indexed_exp, IE>>>
decltype(auto) operator + (const typename IE::value_type &v, const IE &ie) {
  return indexed_scalar_add_exp<IE>(ie, v);
}

template <typename IE,
          typename = std::enable_if_t<std::is_base_of_v<indexed_exp, IE>>>
decltype(auto) operator - (const IE &ie, const typename IE::value_type &v) {
  return indexed_scalar_subr_exp<IE>(ie, v);
}

template <typename IE,
          typename = std::enable_if_t<std::is_base_of_v<indexed_exp, IE>>>
decltype(auto) operator - (const typename IE::value_type &v, const IE &ie) {
  return indexed_scalar_subl_exp<IE>(ie, v);
}

template <typename IE,
          typename = std::enable_if_t<std::is_base_of_v<indexed_exp, IE>>>
decltype(auto) operator * (const IE &ie, const typename IE::value_type &v) {
  return indexed_scalar_mul_exp<IE>(ie, v);
}

template <typename IE,
          typename = std::enable_if_t<std::is_base_of_v<indexed_exp, IE>>>
decltype(auto) operator * (const typename IE::value_type &v, const IE &ie) {
  return indexed_scalar_mul_exp<IE>(ie, v);
}

template <typename IE,
          typename = std::enable_if_t<std::is_base_of_v<indexed_exp, IE>>>
decltype(auto) operator / (const IE &ie, const typename IE::value_type &v) {
  return indexed_scalar_divr_exp<IE>(ie, v);
}

template <typename IE,
          typename = std::enable_if_t<std::is_base_of_v<indexed_exp, IE>>>
decltype(auto) operator / (const typename IE::value_type &v, const IE &ie) {
  return indexed_scalar_divl_exp<IE>(ie, v);
}

template <typename TT, typename... IndexTs>
struct indexed_const_tensor_exp : public indexed_exp {
  indexed_const_tensor_exp(const TT &tref) : tref(tref) {}

  using value_type = typename TT::value_type;

  static constexpr std::size_t max_index() {
    std::size_t r = 0;
    ((IndexTs::i > r ? r = IndexTs::i, 0 : 0), ...);
    return r;
  };

  template <std::size_t I>
  static constexpr bool has_index() {
    return ((IndexTs::i == I) || ...);
  }

  template <std::size_t LI>
  std::size_t dim_size_for_index_use() const {
    return tref.template dim_size<first_index_match_idx<LI, IndexTs::i...>()>();
  }

  template <typename... Ts>
  decltype(auto) eval(Ts... idxs) const {
    return tref(ith_value<IndexTs::i>(idxs...)...);
  }

protected:
  const TT &tref;
};

struct opt_base {
  template <std::size_t I>
  static constexpr bool should_vec() {
    return false;
  }

  template <std::size_t I>
  static constexpr bool should_unroll() {
    return false;
  }

  static constexpr std::size_t get_unroll() {
    return 0;
  }

  static constexpr bool has_rec() {
    return false;
  }

  static constexpr std::size_t get_rec() {
    return 0;
  }
};

template<typename... IndexTs>
struct opt_vec : public opt_base {
  template <std::size_t I>
  static constexpr bool should_vec() {
    return ((IndexTs::i == I) || ...);
  }
};

template<std::size_t UCount, typename... IndexTs>
struct opt_unroll : public opt_base {
  template <std::size_t I>
  static constexpr bool should_unroll() {
    return ((IndexTs::i == I) || ...);
  }

  static constexpr std::size_t get_unroll() {
    return UCount;
  }
};

template<std::size_t Rec>
struct opt_rec : public opt_base {
  static constexpr bool has_rec() {
    return true;
  }

  static constexpr std::size_t get_rec() {
    return Rec;
  }
};

template <typename... Opts>
struct opts_list {
  template <std::size_t I>
  static constexpr bool should_vec() {
    return (Opts::template should_vec<I>() || ...);
  }

  template<typename... OptIndexTs>
  using with_vec = opts_list<Opts..., opt_vec<OptIndexTs...>>;

  template <std::size_t I>
  static constexpr bool should_unroll() {
    return (Opts::template should_unroll<I>() || ...);
  }

  template <std::size_t I>
  static constexpr std::size_t get_unroll() {
    return (std::size_t(0) + ... +
            (Opts::template should_unroll<I>() ? Opts::get_unroll() : 0));
  }

  template<std::size_t UCount, typename... OptIndexTs>
  using with_unroll = opts_list<Opts..., opt_unroll<UCount, OptIndexTs...>>;

  static constexpr std::size_t get_rec() {
    if (!(Opts::has_rec() || ...))
      return MINI_TENSOR_RECURSE_SIZE;

    return (std::size_t(0) + ... + Opts::get_rec());
  }

  template <std::size_t Rec>
  using with_rec = opts_list<Opts..., opt_rec<Rec>>;
};

template <typename TT, typename OptsL, typename... IndexTs>
struct indexed_tensor_exp : public indexed_exp {
  indexed_tensor_exp(TT &tref) : tref(tref) {}

  using value_type = typename TT::value_type;

  static constexpr std::size_t max_index() {
    std::size_t r = 0;
    ((IndexTs::i > r ? r = IndexTs::i, 0 : 0), ...);
    return r;
  };

  template <std::size_t I>
  static constexpr bool has_index() {
    return ((IndexTs::i == I) || ...);
  }

  template <std::size_t LI>
  std::size_t dim_size_for_index_use() const {
    return tref.template dim_size<first_index_match_idx<LI, IndexTs::i...>()>();
  }

  template <typename... Ts>
  decltype(auto) eval(Ts... idxs) const {
    return tref(ith_value<IndexTs::i>(idxs...)...);
  }

  indexed_tensor_exp<TT, OptsL, IndexTs...> & operator = (const indexed_tensor_exp<TT, OptsL, IndexTs...> &ite) {
    tref = ite.tref;
    return *this;
  }

  template <typename TT2, typename OptsL2>
  indexed_tensor_exp<TT, OptsL, IndexTs...> & operator = (const indexed_tensor_exp<TT2, OptsL2, IndexTs...> &ite) {
    tref = ite.tref;
    return *this;
  }

  template <typename TT2, typename OptsL2>
  indexed_tensor_exp<TT, OptsL, IndexTs...> & operator += (const indexed_tensor_exp<TT2, OptsL2, IndexTs...> &ite) {
    tref += ite.tref;
    return *this;
  }

  template <typename TT2, typename OptsL2>
  indexed_tensor_exp<TT, OptsL, IndexTs...> & operator -= (const indexed_tensor_exp<TT2, OptsL2, IndexTs...> &ite) {
    tref -= ite.tref;
    return *this;
  }

private:
  template <std::size_t LIdxMax, std::size_t LIdx, typename TET, typename... VTs>
  void do_assign_loop(const TET &ie, VTs ...vs) {
    auto loop_body = [&](std::size_t i) {
      if constexpr (LIdx > 0) {
        do_assign_loop<LIdxMax, LIdx-1, TET, std::size_t, VTs...>(ie, i, vs...);
      } else {
        tref(ith_value<IndexTs::i>(i, vs...)...) += ie.eval(i, vs...);
      }
    };

    // The offset of the lower bound is at 2*LIdx, plus the additional (LIdxMax-LIdx) = LIdx+LidxMax.

    if constexpr (OptsL::template should_vec<LIdx>() &&
                  OptsL::template should_unroll<LIdx>()) {
      constexpr std::size_t uc = OptsL::template get_unroll<LIdx>();
#if defined(__clang__)
#pragma unroll uc
#if defined(__INTEL_LLVM_COMPILER)
#pragma ivdep
#endif // __INTEL_LLVM_COMPILER
#pragma clang loop vectorize(assume_safety) interleave(assume_safety)
#elif defined(__NVCOMPILER) || defined(__INTEL_COMPILER)
#pragma unroll uc
#pragma ivdep
#elif defined(__GNUC__)
// Note: This does not work because GCC cannot deal with the fact that uc
// might not be a constant express when this branch is not evaluated.
// #pragma GCC unroll uc
#pragma GCC ivdep
#elif defined(_MSC_VER)
#pragma loop(ivdep)
#endif
      for (std::size_t i = ith_value<LIdx+LIdxMax>(vs...); i < ith_value<LIdx+LIdxMax+1>(vs...); ++i)
        loop_body(i);

    } else if constexpr (OptsL::template should_vec<LIdx>()) {
#if defined(__clang__)
#if defined(__INTEL_LLVM_COMPILER)
#pragma ivdep
#endif // __INTEL_LLVM_COMPILER
#pragma clang loop vectorize(assume_safety) interleave(assume_safety)
#elif defined(__NVCOMPILER) || defined(__INTEL_COMPILER)
#pragma ivdep
#elif defined(__GNUC__)
#pragma GCC ivdep
#elif defined(_MSC_VER)
#pragma loop(ivdep)
#endif
      for (std::size_t i = ith_value<LIdx+LIdxMax>(vs...); i < ith_value<LIdx+LIdxMax+1>(vs...); ++i)
        loop_body(i);
    } else if constexpr (OptsL::template should_unroll<LIdx>()) {
      constexpr std::size_t uc = OptsL::template get_unroll<LIdx>();
#if defined(__clang__)
#pragma unroll uc
#elif defined(__GNUC__)
// Note: This does not work because GCC cannot deal with the fact that uc
// might not be a constant express when this branch is not evaluated.
// #pragma GCC unroll uc
#endif
      for (std::size_t i = ith_value<LIdx+LIdxMax>(vs...); i < ith_value<LIdx+LIdxMax+1>(vs...); ++i)
        loop_body(i);
    } else {
      for (std::size_t i = ith_value<LIdx+LIdxMax>(vs...); i < ith_value<LIdx+LIdxMax+1>(vs...); ++i)
        loop_body(i);
    }
  }

  template <std::size_t LIdxMax, typename TET, typename... VTs, std::size_t... Is>
  void do_assign_loop_rec(const TET &ie, std::index_sequence<Is...> is, VTs ...vs) {
    if constexpr (OptsL::get_rec()) {
      // Use a cache-oblivious looping structure (based on the well-known
      // work by Frigo et al.). Split the largest dimension (so long as the
      // largest dimension is greater than Rec).

      std::size_t bounds[sizeof...(VTs)] = { vs... };

      std::size_t ms = 0, msi = 0;
      for (std::size_t i = 0; i < LIdxMax; i += 2) {
        std::size_t s = bounds[i+1] - bounds[i];
        if (s > ms) {
          ms = s;
          msi = i;
        }
      }

      if (ms > OptsL::get_rec()) {
        std::size_t bounds_l[sizeof...(VTs)] = { vs... };
        std::size_t bounds_u[sizeof...(VTs)] = { vs... };

        bounds_l[msi+1] -= (ms+1)/2;
        bounds_u[msi]   += ms/2;

        do_assign_loop_rec<LIdxMax, TET>(ie, is, bounds_l[Is]...);
        do_assign_loop_rec<LIdxMax, TET>(ie, is, bounds_u[Is]...);
        return;
      }
    }

    do_assign_loop<LIdxMax, LIdxMax, TET, VTs...>(ie, vs...);
  }

  template <std::size_t LIdxMax, typename TET, typename... VTs>
  void do_assign_loop_rec(const TET &ie, VTs ...vs) {
    do_assign_loop_rec<LIdxMax>(ie, std::index_sequence_for<VTs...>{}, vs...);
  }

  template <std::size_t LIdxMax, std::size_t LIdx, typename TET, typename... VTs>
  void do_assign_loop_setup(const TET &ie, VTs ...vs) {
    static_assert(has_index<LIdx>() || TET::template has_index<LIdx>());
    std::size_t n = has_index<LIdx>() ?
      dim_size_for_index_use<LIdx>() : ie.template dim_size_for_index_use<LIdx>();

    if constexpr (LIdx > 0)
      do_assign_loop_setup<LIdxMax, LIdx-1, TET, std::size_t, std::size_t, VTs...>(ie, 0, n, vs...);
    else
      do_assign_loop_rec<LIdxMax, TET, std::size_t, std::size_t, VTs...>(ie, 0, n, vs...);
  }

public:
  template <typename TET, typename = std::enable_if_t<std::is_base_of_v<indexed_exp, TET>>>
  indexed_tensor_exp<TT, OptsL, IndexTs...> & operator = (const TET &ie) {
    std::fill(tref.begin(), tref.end(), 0);
    do_assign_loop_setup<TET::max_index(), TET::max_index(), TET>(ie);
    return *this;
  }

  template <typename TET, typename = std::enable_if_t<std::is_base_of_v<indexed_exp, TET>>>
  indexed_tensor_exp<TT, OptsL, IndexTs...> & operator += (const TET &ie) {
    do_assign_loop_setup<TET::max_index(), TET::max_index(), TET>(ie);
    return *this;
  }

  template <typename TET, typename = std::enable_if_t<std::is_base_of_v<indexed_exp, TET>>>
  indexed_tensor_exp<TT, OptsL, IndexTs...> & operator -= (const TET &ie) {
    do_assign_loop_setup<TET::max_index(), TET::max_index(), TET>(-ie);
    return *this;
  }

  template <typename... OptIndexTs,
            typename = std::enable_if_t<(is_index_v<OptIndexTs> && ...)>>
  decltype(auto) vectorize(OptIndexTs... idxs) {
    return indexed_tensor_exp<TT, typename OptsL::template with_vec<OptIndexTs...>, IndexTs...>(tref);
  }

  template <std::size_t UCount = MINI_TENSOR_UNROLL_COUNT,
            typename... OptIndexTs,
            typename = std::enable_if_t<(is_index_v<OptIndexTs> && ...)>>
  decltype(auto) unroll(OptIndexTs... idxs) {
    return indexed_tensor_exp<TT, typename OptsL::template with_unroll<UCount, OptIndexTs...>, IndexTs...>(tref);
  }

  template <std::size_t Rec>
  decltype(auto) recurse() {
    return indexed_tensor_exp<TT, typename OptsL::template with_rec<Rec>, IndexTs...>(tref);
  }

protected:
  TT &tref;
};

// CRTP base class for the tensor types.
template <typename TT, typename DimsT, typename ContainerT>
struct tensor_base {
private:
  TT &base() { return *static_cast<TT *>(this); }
  const TT &base() const { return *static_cast<const TT *>(this); }

public:
  using value_type = typename ContainerT::value_type;
  using size_type = std::size_t;
  using difference_type = std::ptrdiff_t;
  using reference = typename ContainerT::reference;
  using const_reference = typename ContainerT::const_reference;
  using pointer = typename ContainerT::pointer;
  using const_pointer = typename ContainerT::const_pointer;
  using iterator = typename ContainerT::iterator;
  using const_iterator = typename ContainerT::const_iterator;
  using reverse_iterator = typename ContainerT::reverse_iterator;
  using const_reverse_iterator = typename ContainerT::const_reverse_iterator;

  std::size_t size() const { return base().container.size(); }
  std::size_t max_size() const { return base().container.max_size(); }
  bool empty() const { return base().container.empty(); }

  reference front() { return base().container.front(); }
  const_reference front() const { return base().container.front(); }
  reference back() { return base().container.back(); }
  const_reference back() const { return base().container.back(); }
  pointer data() { return base().container.data(); }
  const_pointer data() const { return base().container.data(); }

  iterator begin() { return base().container.begin(); }
  const_iterator begin() const { return base().container.begin(); }
  const_iterator cbegin() const { return base().container.cbegin(); }
  iterator end() { return base().container.end(); }
  const_iterator end() const { return base().container.end(); }
  const_iterator cend() const { return base().container.cend(); }

  reverse_iterator rbegin() { return base().container.rbegin(); }
  const_reverse_iterator rbegin() const { return base().container.rbegin(); }
  const_reverse_iterator crbegin() const { return base().container.crbegin(); }
  reverse_iterator rend() { return base().container.rend(); }
  const_reverse_iterator rend() const { return base().container.rend(); }
  const_reverse_iterator crend() const { return base().container.crend(); }

  tensor_base<TT, DimsT, ContainerT> & operator = (const tensor_base<TT, DimsT, ContainerT> &t2) {
    if (&*t2.begin() != &*begin()) // Elide self assignment.
      std::copy(t2.begin(), t2.end(), begin());
    return *this;
  }

  template <typename TT2, typename ContainerT2>
  tensor_base<TT, DimsT, ContainerT> & operator = (const tensor_base<TT2, DimsT, ContainerT2> &t2) {
    std::copy(t2.begin(), t2.end(), begin());
    return *this;
  }

  template <typename TT2, typename ContainerT2>
  tensor_base<TT, DimsT, ContainerT> & operator += (const tensor_base<TT2, DimsT, ContainerT2> &t2) {
    std::transform(begin(), end(), t2.begin(), begin(), std::plus<>{});
    return *this;
  }

  template <typename TT2, typename ContainerT2>
  tensor_base<TT, DimsT, ContainerT> & operator -= (const tensor_base<TT2, DimsT, ContainerT2> &t2) {
    std::transform(begin(), end(), t2.begin(), begin(), std::minus<>{});
    return *this;
  }

  tensor_base<TT, DimsT, ContainerT> & operator = (value_type v) {
    std::fill(begin(), end(), v);
    return *this;
  }

  tensor_base<TT, DimsT, ContainerT> & operator += (value_type v) {
    std::transform(begin(), end(), begin(), [&v](auto &x) { return x + v; });
    return *this;
  }

  tensor_base<TT, DimsT, ContainerT> & operator -= (value_type v) {
    std::transform(begin(), end(), begin(), [&v](auto &x) { return x - v; });
    return *this;
  }

  tensor_base<TT, DimsT, ContainerT> & operator *= (value_type v) {
    std::transform(begin(), end(), begin(), [&v](auto &x) { return x * v; });
    return *this;
  }

  tensor_base<TT, DimsT, ContainerT> & operator /= (value_type v) {
    std::transform(begin(), end(), begin(), [&v](auto &x) { return x / v; });
    return *this;
  }

  template <typename... Ts,
            typename = std::enable_if_t<(sizeof...(Ts) == DimsT::rank+1) &&
                                        (std::is_convertible_v<Ts, std::size_t> && ...)>>
  constexpr static std::size_t storage_index(Ts... idxs) {
    return DimsT::storage_index(idxs...);
  }

  template <typename... Ts,
            typename = std::enable_if_t<(sizeof...(Ts) == DimsT::rank+1) &&
                                        (std::is_convertible_v<Ts, std::size_t> && ...)>>
  reference operator () (Ts... idxs) {
    return base().container[DimsT::storage_index(idxs...)];
  }

  template <typename... Ts,
            typename = std::enable_if_t<(sizeof...(Ts) == DimsT::rank+1) &&
                                        (std::is_convertible_v<Ts, std::size_t> && ...)>>
  const_reference operator () (Ts... idxs) const {
    return base().container[DimsT::storage_index(idxs...)];
  }

  template <typename... Ts,
            typename = std::enable_if_t<(sizeof...(Ts) == DimsT::rank+1) &&
                                        (is_index_v<Ts> && ...)>>
  indexed_tensor_exp<TT, opts_list<>, Ts...> operator () (Ts... idxs) {
    return indexed_tensor_exp<TT, opts_list<>, Ts...>(base());
  }

  template <typename... Ts,
            typename = std::enable_if_t<(sizeof...(Ts) == DimsT::rank+1) &&
                                        (is_index_v<Ts> && ...)>>
  indexed_const_tensor_exp<TT, Ts...> operator () (Ts... idxs) const {
    return indexed_const_tensor_exp<TT, Ts...>(base());
  }
};
} // namespace detail

template <typename T, typename DimsT,
          template <typename, typename...> typename ContainerT = std::vector,
          typename... ContainerTTs>
struct tensor_odyn :
  public detail::tensor_base<tensor_odyn<T, DimsT, ContainerT, ContainerTTs...>,
                             DimsT, ContainerT<T, ContainerTTs...>> {
  using Base = detail::tensor_base<tensor_odyn<T, DimsT, ContainerT, ContainerTTs...>,
                                   DimsT, ContainerT<T, ContainerTTs...>>;
  friend Base;
  using Base::operator=;

  using dims_t = DimsT;

  template <typename... Ps>
  tensor_odyn(std::size_t dimo, Ps&&... params) :
    container(dimo*DimsT::base_size(), std::forward<Ps>(params)...) {}

  template <std::size_t SlotI>
  std::size_t dim_size() const {
    if constexpr (!SlotI)
      return container.size() / DimsT::base_size();
    else
      return DimsT::template dim_size<SlotI-1>();
  }

protected:
  ContainerT<T, ContainerTTs...> container;
};

template <typename T, typename DimsT,
          template <typename, std::size_t, typename...> typename ContainerT = std::array,
          typename... ContainerTTs>
struct tensor :
  public detail::tensor_base<tensor<T, DimsT, ContainerT, ContainerTTs...>,
                             typename detail::dims_split_head<DimsT>::inner_dims_t,
                             ContainerT<T, DimsT::base_size(), ContainerTTs...>> {
  using Base = detail::tensor_base<tensor<T, DimsT, ContainerT, ContainerTTs...>,
                                   typename detail::dims_split_head<DimsT>::inner_dims_t,
                                   ContainerT<T, DimsT::base_size(), ContainerTTs...>>;
  friend Base;
  using Base::operator=;

  using dims_t = typename detail::dims_split_head<DimsT>::inner_dims_t;

  template <typename... Ps>
  tensor(Ps&&... params) : container(std::forward<Ps>(params)...) {}

  template <std::size_t SlotI>
  static constexpr std::size_t dim_size() {
    return DimsT::template dim_size<SlotI>();
  }

protected:
  ContainerT<T, DimsT::base_size(), ContainerTTs...> container;
};
} // namespace mini_tensor

#endif // MINI_TENSOR_HPP_INCLUDED

