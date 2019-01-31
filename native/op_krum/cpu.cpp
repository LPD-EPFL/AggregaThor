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
 * Multi-Krum GAR, CPU kernel implementation.
 *
 * Based on the algorithm introduced in the following paper:
 *   Blanchard Peva, El Mhamdi El Mahdi, Guerraoui Rachid, and Stainer Julien.
 *   Machine learning with adversaries: Byzantine tolerant gradient descent.
 *   In Advances in Neural Information Processing Systems 30, pp.118–128.
 *   Curran Associates, Inc., 2017.
**/

#include <algorithm>
#include <atomic>
#include <cmath>

#include <common.hpp>
#include <threadpool.hpp>
#include <operations.hpp>
#include "decl.hpp"

// -------------------------------------------------------------------------- //
// Kernel implementation
namespace OP_NAME {

template<class T> class Kernel<CPUDevice, T>: public Static {
public:
    static void process(OpKernelContext& context [[gnu::unused]], size_t const n, size_t const f, size_t const d, size_t const m, Tensor const& input, Tensor& output) {
        auto data_in  = input.flat<T>().data();
        auto data_out = output.flat<T>().data();
        auto const cn = n;
        auto const kn = n * (n - 1);
        auto const ln = n * (n - 1) / 2;
        size_t pos_to_gradid[kn]; // FIXME: Replace VLA (that are GNU extension) with 'vlarray'
        T distances[ln];
        { // Distance computations
            auto cursor = distances;
            auto poscur = pos_to_gradid;
            for (size_t i = 0; i < n - 1; ++i) {
                auto x = data_in + i * d;
                for (size_t j = i + 1; j < n; ++j) {
                    auto y = data_in + j * d;
                    *(cursor++) = reduce_sum_squared_difference(x, y, d);
                    *(poscur++) = i;
                    *(poscur++) = j;
                }
            }
        }
        T scores[cn];
        { // Score computations
            size_t ranks[ln]; // Indexes for 'distances', so that 'distances[ranks[i]]' increases with 'i' ('nan' is treated as '+inf')
            { // Compute 'ranks'
                for (size_t i = 0; i < ln; ++i)
                    ranks[i] = i;
                T* dist_ptr = distances;
                ::std::sort(ranks, ranks + ln, [dist_ptr](size_t a, size_t b) {
                    auto&& x = dist_ptr[a];
                    if (unlikely(!::std::isfinite(x)))
                        return false;
                    auto&& y = dist_ptr[b];
                    if (unlikely(!::std::isfinite(y)))
                        return true;
                    return x < y;
                });
            }
            for (size_t i = 0; i < n; ++i) { // Compute 'scores'
                T score = 0;
                size_t count = n - f - 2;
                for (auto* cursor = ranks; count > 0; ++cursor) {
                    auto index = *cursor;
                    if (pos_to_gradid[2 * index] == i || pos_to_gradid[2 * index + 1] == i) { // Associated distance concerns current gradient
                        score += distances[index];
                        --count;
                    }
                }
                scores[i] = score;
            }
        }
        { // Select the 'm' smallest scoring gradients and average them
            size_t selected[cn]; // Index of the selected gradients
            { // Compute 'selected'
                for (size_t i = 0; i < cn; ++i)
                    selected[i] = i;
                T* scores_ptr = scores;
                ::std::nth_element(selected, selected + m, selected + n, [scores_ptr](size_t a, size_t b) {
                    auto&& x = scores_ptr[a];
                    if (unlikely(!::std::isfinite(x)))
                        return false;
                    auto&& y = scores_ptr[b];
                    if (unlikely(!::std::isfinite(y)))
                        return true;
                    return x < y;
                });
            }
            selection_average(data_in, data_out, d, selected, m);
        }
    }
};

// Explicit instantiations
template class Kernel<CPUDevice, float>;
template class Kernel<CPUDevice, double>;

}
