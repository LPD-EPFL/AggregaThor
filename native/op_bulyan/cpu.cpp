/**
 * @file   cpu.cpp
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
 * Bulyan over Multi-Krum GAR, CPU kernel implementation.
 *
 * Based on the algorithm introduced in the following paper:
 *   El Mhamdi El Mahdi, Guerraoui Rachid, and Rouault Sébastien.
 *   The Hidden Vulnerability of Distributed Learning in Byzantium.
 *   In Dy, J. and Krause, A. (eds.), Proceedings of the 35th International
 *   Conference on Machine Learning, volume 80 of Proceedings of Machine
 *   Learning  Research, pp. 3521-3530, Stockholmsmässan, Stockholm Sweden,
 *   10-15 Jul 2018. PMLR. URL http://proceedings.mlr.press/v80/mhamdi18a.html.
**/

#include <limits>

#include <common.hpp>
#include <array.hpp>
#include <threadpool.hpp>
#include <operations.hpp>
#include "decl.hpp"

// -------------------------------------------------------------------------- //
// Kernel implementation
namespace OP_NAME {

template<class T> class Kernel<CPUDevice, T>: public Static {
public:
    static void process(OpKernelContext& context, size_t const n, size_t const f, size_t const d, size_t const m, Tensor const& input, Tensor& outpt) {
        auto const kn = n * (n - 1);
        auto const ln = kn / 2;
        auto const t  = n - 2 * f - 2;
        auto const b  = t - 2 * f;
        Tensor temp0; // Intermediate vectors
        OP_REQUIRES_OK(&context, context.allocate_temp(DataTypeToEnum<T>::value, TensorShape{static_cast<long long>(d * t)}, &temp0));
        auto inputs = input.flat<T>().data();
        auto inters = temp0.flat<T>().data();
        auto output = outpt.flat<T>().data();
        auto distances = vlarray<T>(n * n); // With 'distance[i * n + j]' representing the distance between vectors i & j from score of vector i
        auto scores    = vlarray<T>(n);
        auto ranks     = vlarray<size_t>(ln); // Indexes for 'distances'/'scores', so that 'distances[ranks[i]]'/'scores[ranks[i]]' increases with 'i' ('nan' is treated as '+inf')
        { // Initial Krum pass
            auto flat_dist     = vlarray<T>(ln);
            auto pos_to_gradid = vlarray<size_t>(kn);
            { // Distance computations
                auto dstcur = flat_dist.get();
                auto poscur = pos_to_gradid.get();
                for (size_t i = 0; i < n - 1; ++i) {
                    auto x = inputs + i * d;
                    distances[i * (n + 1)] = ::std::numeric_limits<T>::max();
                    for (size_t j = i + 1; j < n; ++j) {
                        auto y = inputs + j * d;
                        auto dist = reduce_sum_squared_difference(x, y, d);
                        distances[i * n + j] = dist;
                        distances[j * n + i] = dist;
                        *(dstcur++) = dist;
                        *(poscur++) = i;
                        *(poscur++) = j;
                    }
                }
                distances[n * n - 1] = ::std::numeric_limits<T>::max();
            }
            { // Initial score computations and distance pruning
                { // Compute 'ranks'
                    for (size_t i = 0; i < ln; ++i)
                        ranks[i] = i;
                    ::std::sort(ranks.get(), ranks.get() + ln, [&](size_t a, size_t b) {
                        auto&& x = flat_dist[a];
                        if (unlikely(!::std::isfinite(x)))
                            return false;
                        auto&& y = flat_dist[b];
                        if (unlikely(!::std::isfinite(y)))
                            return true;
                        return x < y;
                    });
                }
                parallel_for(pool, 0, n, [&](size_t start, size_t stop) {
                    for (size_t i = start; i < stop; ++i) {
                        // Score computation
                        T score = 0;
                        size_t count = n - f - 2;
                        auto cursor = ranks.get();
                        for (; count > 0; ++cursor) {
                            auto index = *cursor;
                            if (pos_to_gradid[2 * index] == i || pos_to_gradid[2 * index + 1] == i) { // Associated distance concerns current gradient
                                score += flat_dist[index];
                                --count;
                            }
                        }
                        scores[i] = score;
                        // Distance pruning
                        count += f + 1;
                        for (; count > 0; ++cursor) {
                            auto index = *cursor;
                            auto a = pos_to_gradid[2 * index];
                            auto b = pos_to_gradid[2 * index + 1];
                            if (a == i) { // Associated distance concerns current gradient
                                distances[i * n + b] = 0;
                                --count;
                            } else if (b == i) { // Associated distance concerns current gradient
                                distances[i * n + a] = 0;
                                --count;
                            }
                        }
                    }
                });
            }
        }
        { // Selection loop
            for (size_t i = 0; i < n; ++i) // Initialize 'ranks'
                ranks[i] = i;
            for (size_t k = 0;;) {
                // Compute ranks
                ::std::sort(ranks.get(), ranks.get() + n, [&](size_t a, size_t b) {
                    auto&& x = scores[a];
                    if (unlikely(!::std::isfinite(x)))
                        return false;
                    auto&& y = scores[b];
                    if (unlikely(!::std::isfinite(y)))
                        return true;
                    return x < y;
                });
                // Average the 'm - k' smallest-scoring gradients as the output of Krum
                selection_average(inputs, inters + k * d, d, ranks.get(), m - k);
                if (++k >= t) // Check if done
                    break;
                { // Remove the smallest-scoring gradient
                    auto id = ranks[0];
                    scores[id] = ::std::numeric_limits<T>::max(); // Virtually remove the gradient from selection
                    for (size_t i = 0; i < n; ++i) { // Update the scores
                        if (i == id)
                            continue;
                        scores[i] -= distances[i * n + id]; // Valid since distances have been pruned
                    }
                }
            }
        }
        // Averaged-median coordinate-by-coordinate
        Array<Array<T>> grads{inters, {t, d}};
        parallel_for(pool, 0, d, [&](size_t start, size_t stop) {
            for (size_t x = start; x < stop; ++x) { // Coordinates to work on
                typename decltype(grads)::Iter axis, aend;
                ::std::tie(axis, aend) = grads.axis(0, x);
                auto length = aend - axis;
                auto median = axis + length / 2ul;
                ::std::nth_element(axis, median, aend);
                auto zero = *median;
                ::std::nth_element(axis, axis + b, aend, [&](T x, T y) {
                    auto dx = x - zero;
                    if (dx < 0)
                        dx = -dx;
                    auto dy = y - zero;
                    if (dy < 0)
                        dy = -dy;
                    return dx < dy;
                });
                auto average = axis[0];
                for (size_t i = 1; i < b; ++i)
                    average += axis[i];
                output[x] = average / static_cast<T>(b);
            }
        });
    }
};

// Explicit instantiations
template class Kernel<CPUDevice, float>;
template class Kernel<CPUDevice, double>;

}
