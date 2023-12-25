//          Copyright (C) Hal Finkel 2023.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE_1_0.txt or copy at
//          https://www.boost.org/LICENSE_1_0.txt)
// SPDX-License-Identifier: BSL-1.0

#include "../include/mini_tensor.hpp"

#include <chrono>
#include <iostream>

namespace mt = mini_tensor;

template <typename ST, std::size_t LI, std::size_t LJ, std::size_t LK, bool NoAlias = false>
static void tests(std::size_t ntrials) {
  {
    ST v = 2.0f;
    mt::tensor_odyn<ST, mt::dims<100>> t1(6765);
    mt::tensor<ST, mt::dims<100, 1024>> t1b;
    t1b = v;
    mt::tensor_odyn<ST, mt::dims<1024>> t1c(6765, v);

    mt::index<LI> _i;
    mt::index<LJ> _j;
    mt::index<LK> _k;

    std::chrono::nanoseconds::rep all_durations{};

    for (std::size_t trial = 0; trial < ntrials; ++trial) {
      t1 = 0;
      auto start = std::chrono::steady_clock::now();

      if constexpr (NoAlias)
        t1(_i, _j).vectorize(_k).unroll(_j) += (t1b(_j, _k)*t1c(_i, _k));
      else
        t1(_i, _j).unroll(_k) += t1b(_j, _k)*t1c(_i, _k);

      auto end = std::chrono::steady_clock::now();
      auto duration =
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
      all_durations += duration;
    }

    std::cout << "mt" << NoAlias << ": " << LI << ", " << LJ << ", " << LK << ": " <<
                 (all_durations / ntrials) << " ns\n";
  }
}

template <typename ST>
static void tests_p(std::size_t ntrials) {
#ifndef MINI_TENSOR_TEST_210_ONLY
  tests<ST, 0, 1, 2>(ntrials);
  tests<ST, 1, 2, 0>(ntrials);
#endif
  tests<ST, 2, 1, 0>(ntrials);
  tests<ST, 2, 1, 0, true>(ntrials);
#ifndef MINI_TENSOR_TEST_210_ONLY
  tests<ST, 0, 2, 1>(ntrials);
  tests<ST, 2, 0, 1>(ntrials);
  tests<ST, 1, 0, 2>(ntrials);
#endif
}

template <typename ST>
static void tests_n1(std::size_t ntrials) {
    ST v = 2.0f;
    mt::tensor_odyn<ST, mt::dims<100>> t1(6765);
    mt::tensor<ST, mt::dims<100, 1024>> t1b;
    t1b = v;
    mt::tensor_odyn<ST, mt::dims<1024>> t1c(6765, v);

    std::chrono::nanoseconds::rep all_durations{};

    for (std::size_t trial = 0; trial < ntrials; ++trial) {
      t1 = 0;
      auto start = std::chrono::steady_clock::now();

      ST *t1d = t1.data();
      const ST *t1bd = t1b.data();
      const ST *t1cd = t1c.data();

      for (std::size_t i = 0; i < 6765; ++i)
      for (std::size_t j = 0; j < 100; ++j)
      for (std::size_t k = 0; k < 1024; ++k)
        t1d[100*i + j] += t1bd[100*j +  k]*t1cd[1024*i + k];

      auto end = std::chrono::steady_clock::now();
      auto duration =
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
      all_durations += duration;
    }

    std::cout << "n:  2, 1, 0: " << (all_durations / ntrials) << " ns\n";
}

template <typename ST>
static void tests_n2(std::size_t ntrials) {
    ST v = 2.0f;
    mt::tensor_odyn<ST, mt::dims<100>> t1(6765);
    mt::tensor<ST, mt::dims<100, 1024>> t1b;
    t1b = v;
    mt::tensor_odyn<ST, mt::dims<1024>> t1c(6765, v);

    std::chrono::nanoseconds::rep all_durations{};

    for (std::size_t trial = 0; trial < ntrials; ++trial) {
      t1 = 0;
      auto start = std::chrono::steady_clock::now();

      ST *t1d = t1.data();
      const ST *t1bd = t1b.data();
      const ST *t1cd = t1c.data();

      for (std::size_t k = 0; k < 1024; ++k)
      for (std::size_t j = 0; j < 100; ++j)
      for (std::size_t i = 0; i < 6765; ++i)
        t1d[100*i + j] += t1bd[100*j +  k]*t1cd[1024*i + k];

      auto end = std::chrono::steady_clock::now();
      auto duration =
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
      all_durations += duration;
    }

    std::cout << "n:  0, 1, 2: " << (all_durations / ntrials) << " ns\n";
}

template <typename ST>
static void tests_n(std::size_t ntrials) {
  tests_n1<ST>(ntrials);
#ifndef MINI_TENSOR_TEST_210_ONLY
  tests_n2<ST>(ntrials);
#endif
}

int main() {
  const std::size_t ntrials = 10;

  std::cout << "Testing with float...\n";
  tests_p<float>(ntrials);
  tests_n<float>(ntrials);
  std::cout << "Testing with double...\n";
  tests_p<double>(ntrials);
  tests_n<double>(ntrials);
  std::cout << "Testing with long double...\n";
  tests_p<long double>(ntrials);
  tests_n<long double>(ntrials);

  return 0;
}

