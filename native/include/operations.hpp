/**
 * @file   operations.hpp
 * @author Sébastien Rouault <sebastien.rouault@epfl.ch>
 *
 * @section LICENSE
 *
 * Copyright © 2018-2019 Sébastien ROUAULT.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 * @section DESCRIPTION
 *
 * Implementation of common operations.
**/

#pragma once

#include "common.hpp"
#include "threadpool.hpp"

// -------------------------------------------------------------------------- //
// CPU computation helpers

/** Sum-reduction of coordinate-wise squared-difference of two vectors.
 * @param x Input vector X
 * @param y Input vector Y
 * @param d Vector space dimension
 * @return Output scalar
**/
template<class T> static T reduce_sum_squared_difference(T const* x, T const* y, size_t d) {
    ::std::atomic<T> asum{0};
    parallel_for(pool, 0, d, [&](size_t start, size_t stop) {
        T sum = 0;
        for (size_t i = start; i < stop; ++i) { // Coordinates to work on
            auto d = x[i] - y[i];
            sum += d * d;
        }
        T tot = 0;
        while (!asum.compare_exchange_strong(tot, tot + sum, ::std::memory_order_relaxed));
    });
    return asum.load(::std::memory_order_relaxed);
}

/** Selection vector arithmetic mean.
 * @param g List of vectors
 * @param o Output vector
 * @param d Vector space dimension
 * @param a List of indexes to vectors to average
 * @param s Size of the list of indexes
**/
template<class T> static void selection_average(T const* g, T* o, size_t d, size_t const* a, size_t s) {
    parallel_for(pool, 0, d, [&](size_t start, size_t stop) {
        for (size_t i = start; i < stop; ++i) { // Coordinates to work on
            T sum = 0;
            for (size_t j = 0; j < s; ++j)
                sum += *(g + d * a[j] + i);
            o[i] = sum / static_cast<T>(s);
        }
    });
}
