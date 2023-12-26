//          Copyright (C) Hal Finkel 2023.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE_1_0.txt or copy at
//          https://www.boost.org/LICENSE_1_0.txt)
// SPDX-License-Identifier: BSL-1.0

#include "../include/fun_sized_tensor.hpp"

#include <iostream>

namespace fst = fun_sized_tensor;

template <typename ST>
static bool tests() {
  {
    std::cout << "TEST 1: ";
    fst::tensor_odyn<ST, fst::dims<100>> t1(35);
    if (t1.size() != 3500 ||
        t1.storage_index(6, 5) != 605) {
      std::cout << "FAILED!\n";
      return false;
    }
    std::cout << "PASSED.\n";
  }

  {
    std::cout << "TEST 2: ";
    fst::tensor_odyn<ST, fst::dims<20, 10>> t1(222, 1.0f);
    if (t1.size() != 44400 ||
        t1.storage_index(5, 6, 7) != 1067 ||
        t1(5, 6, 7) != ST(1)) {
      std::cout << "FAILED!\n";
      return false;
    }
    std::cout << "PASSED.\n";
  }

  {
    std::cout << "TEST 3: ";
    fst::tensor<ST, fst::dims<444, 20, 10>> t1;
    t1 = 5;
    if (t1.size() != 88800 ||
        t1.storage_index(5, 6, 7) != 1067 ||
        t1(5, 6, 7) != ST(5)) {
      std::cout << "FAILED!\n";
      return false;
    }
    std::cout << "PASSED.\n";
  }

  fst::index<0> _i;
  fst::index<1> _j;
  fst::index<2> _k;

  {
    std::cout << "TEST 4: ";
    fst::tensor_odyn<ST, fst::dims<100>> t1(35);
    fst::tensor_odyn<ST, fst::dims<100>> t1b(35, 2.0f);

    t1(_i, _j) = t1b(_i, _j);
    if (t1(1, 1) != ST(2)) {
      std::cout << "FAILED!\n";
      return false;
    }
    std::cout << "PASSED.\n";
  }

  {
    std::cout << "TEST 5: ";
    fst::tensor_odyn<ST, fst::dims<100>> t1(35);
    fst::tensor_odyn<ST, fst::dims<100>> t1c(35, 8.0f);
    t1 = t1c;
    t1 += t1c;
    t1 += t1c;
    t1 -= t1c;
    if (t1(1, 1) != ST(16)) {
      std::cout << "FAILED!\n";
      return false;
    }
    std::cout << "PASSED.\n";
  }

  {
    std::cout << "TEST 6: ";
    fst::tensor_odyn<ST, fst::dims<100>> t1(35);
    fst::tensor_odyn<ST, fst::dims<100>> t1c(35, 8.0f);
    t1(_i, _j) = t1c(_i, _j);
    t1(_i, _j) += t1c(_i, _j);
    t1(_i, _j) -= t1c(_i, _j);
    if (t1(1, 1) != ST(8)) {
      std::cout << "FAILED!\n";
      return false;
    }
    std::cout << "PASSED.\n";
  }

  {
    std::cout << "TEST 7: ";
    fst::tensor_odyn<ST, fst::dims<100>> t1(35);
    t1 = 5.0f;
    ST v1 = t1(1, 1);
    t1 *= 5.0f;
    ST v2 = t1(1, 1);
    t1 /= 2.0f;
    ST v3 = t1(1, 1);
    t1 += 2.0f;
    ST v4 = t1(1, 1);
    t1 -= 1.0f;
    ST v5 = t1(1, 1);

    if (v1 != ST(5) ||
        v2 != ST(25) ||
        v3 != ST(12.5) ||
        v4 != ST(14.5) ||
        v5 != ST(13.5)) {
      std::cout << "FAILED!\n";
      return false;
    }
    std::cout << "PASSED.\n";
  }

  {
    std::cout << "TEST 8: ";
    fst::tensor_odyn<ST, fst::dims<100>> t1(35);
    fst::tensor_odyn<ST, fst::dims<35>> t1b(100, 2.0f);

    t1(_i, _j) = t1b(_j, _i);

    if (t1(1, 1) != ST(2)) {
      std::cout << "FAILED!\n";
      return false;
    }
    std::cout << "PASSED.\n";
  }

  {
    std::cout << "TEST 9: ";
    fst::tensor_odyn<ST, fst::dims<100>> t1(35);
    fst::tensor<ST, fst::dims<100, 35>> t1b;
    t1b = 2.0f;

    t1(_i, _j) = t1b(_j, _i);

    if (t1(1, 1) != ST(2)) {
      std::cout << "FAILED!\n";
      return false;
    }
    std::cout << "PASSED.\n";
  }

  {
    std::cout << "TEST 10: ";
    fst::tensor<ST, fst::dims<35, 100>> t1;
    fst::tensor<ST, fst::dims<100, 35>> t1b;
    t1b = 2.0f;

    t1(_i, _j) = t1b(_j, _i);

    if (t1(1, 1) != ST(2)) {
      std::cout << "FAILED!\n";
      return false;
    }
    std::cout << "PASSED.\n";
  }

  {
    std::cout << "TEST 11: ";
    fst::tensor_odyn<ST, fst::dims<100>> t1(35);
    fst::tensor_odyn<ST, fst::dims<35>> t1b(100, 2.0f);

    t1(_i, _j) = -t1b(_j, _i);

    if (t1(1, 1) != ST(-2)) {
      std::cout << "FAILED!\n";
      return false;
    }
    std::cout << "PASSED.\n";
  }

  {
    std::cout << "TEST 11: ";
    fst::tensor_odyn<ST, fst::dims<100>> t1(35);
    fst::tensor_odyn<ST, fst::dims<35>> t1b(100, 2.0f);

    t1(_i, _j) = +t1b(_j, _i);

    if (t1(1, 1) != ST(2)) {
      std::cout << "FAILED!\n";
      return false;
    }
    std::cout << "PASSED.\n";
  }

  {
    std::cout << "TEST 12: ";
    fst::tensor_odyn<ST, fst::dims<100>> t1(35);
    fst::tensor_odyn<ST, fst::dims<50>> t1b(35, 2.0f);
    fst::tensor_odyn<ST, fst::dims<100>> t1c(50, 2.0f);

    t1(_i, _j) = t1b(_i, _k)*t1c(_k, _j);

    if (t1(1, 1) != ST(200) ||
        std::accumulate(t1.begin(), t1.end(), 0.f) != ST(700000)) {
      std::cout << "FAILED!\n";
      return false;
    }
    std::cout << "PASSED.\n";
  }

  {
    std::cout << "TEST 13: ";
    ST v = 2.0f;
    fst::tensor_odyn<ST, fst::dims<100>> t1(35);
    fst::tensor_odyn<ST, fst::dims<50>> t1b(35, v);
    fst::tensor_odyn<ST, fst::dims<100>> t1c(50, v);

    ST a = 2.4, d = 5.4;
    t1(_i, _j) = (t1b(_i, _k)+a)*(t1c(_k, _j) - d);
    ST cv = (v + a)*(v - d);

    ST diff = t1(1, 1) - ST(50*cv);
    if (diff*diff > 1e-6) {
      std::cout << "FAILED!\n";
      return false;
    }
    std::cout << "PASSED.\n";
  }

  {
    std::cout << "TEST 14: ";
    ST v = 2.0f;
    fst::tensor_odyn<ST, fst::dims<100>> t1(35);
    fst::tensor_odyn<ST, fst::dims<50>> t1b(35, v);
    fst::tensor_odyn<ST, fst::dims<100>> t1c(50, v);

    ST a = 2.4, b = 3.4, c = 4.4, d = 5.4, e = 8.2;
    t1(_i, _j) = 1 / (b*t1b(_i, _k)+a)*(c - t1c(_k, _j) - d) / e;
    ST cv = 1 / (b * v + a)*(c - v - d) / e;

    ST diff = t1(1, 1) - ST(50*cv);
    if (diff*diff > 1e-6) {
      std::cout << "FAILED!\n";
      return false;
    }
    std::cout << "PASSED.\n";
  }

  {
    std::cout << "TEST 15: ";
    fst::tensor_odyn<ST, fst::dims<100>> t1(35);
    fst::tensor_odyn<ST, fst::dims<50>> t1b(35, 2.0f);
    fst::tensor_odyn<ST, fst::dims<100>> t1c(50, 2.0f);

    t1(_i, _j).unroll(_j) = t1b(_i, _k)*t1c(_k, _j);

    if (t1(1, 1) != ST(200) ||
        std::accumulate(t1.begin(), t1.end(), 0.f) != ST(700000)) {
      std::cout << "FAILED!\n";
      return false;
    }
    std::cout << "PASSED.\n";
  }

  {
    std::cout << "TEST 16: ";
    fst::tensor_odyn<ST, fst::dims<100>> t1(35);
    fst::tensor_odyn<ST, fst::dims<50>> t1b(35, 2.0f);
    fst::tensor_odyn<ST, fst::dims<100>> t1c(50, 2.0f);

    t1(_i, _j).unroll(_j, _i) = t1b(_i, _k)*t1c(_k, _j);

    if (t1(1, 1) != ST(200) ||
        std::accumulate(t1.begin(), t1.end(), 0.f) != ST(700000)) {
      std::cout << "FAILED!\n";
      return false;
    }
    std::cout << "PASSED.\n";
  }

  {
    std::cout << "TEST 17: ";
    fst::tensor_odyn<ST, fst::dims<100>> t1(35);
    fst::tensor_odyn<ST, fst::dims<50>> t1b(35, 2.0f);
    fst::tensor_odyn<ST, fst::dims<100>> t1c(50, 2.0f);

    t1(_i, _j).unroll(_i).template recurse<6>() = t1b(_i, _k)*t1c(_k, _j);

    if (t1(1, 1) != ST(200) ||
        std::accumulate(t1.begin(), t1.end(), 0.f) != ST(700000)) {
      std::cout << "FAILED!\n";
      return false;
    }
    std::cout << "PASSED.\n";
  }

  {
    std::cout << "TEST 18: ";
    fst::tensor_odyn<ST, fst::dims<100>> t1(35);
    fst::tensor_odyn<ST, fst::dims<50>> t1b(35, 2.0f);
    fst::tensor_odyn<ST, fst::dims<100>> t1c(50, 2.0f);

    t1(_i, _j).vectorize(_i).unroll(_i) = t1b(_i, _k)*t1c(_k, _j);

    if (t1(1, 1) != ST(200) ||
        std::accumulate(t1.begin(), t1.end(), 0.f) != ST(700000)) {
      std::cout << "FAILED!\n";
      return false;
    }
    std::cout << "PASSED.\n";
  }

  return true;
}

int main() {
  std::cout << "Testing with float...\n";
  if (!tests<float>())
    return 1;
  std::cout << "Testing with double...\n";
  if (!tests<double>())
    return 1;
  std::cout << "Testing with long double...\n";
  if (!tests<long double>())
    return 1;

  return 0;
}

