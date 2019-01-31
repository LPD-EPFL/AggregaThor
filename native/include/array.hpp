/**
 * @file   array.hpp
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
 * Multi-dimensional array helper.
**/

#pragma once

// Compiler version check
#if __cplusplus < 201103L
    #error This translation unit requires at least a C++11 compiler
#endif
#ifndef __GNUC__
    #error This translation unit requires a GNU C++ compiler
#endif

// External headers
#include <algorithm>
#include <condition_variable>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <future>
#include <iterator>
#include <limits>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>
extern "C" {
#include <unistd.h>
}

// Internal headers
#include "common.hpp"

// -------------------------------------------------------------------------- //

/** Axis iterator class.
 * @param Elem Single element class
**/
template<class Elem> class Iterator {
private:
    /** Aliased classes.
    **/
    using This = Iterator<Elem>;
private:
    Elem*  cursor; // Current element, may not be dereferencable
    size_t stride; // Stride of the iterated axis
public:
    /** Default constructor (cannot be dereferenced).
    **/
    Iterator() noexcept: cursor{nullptr}, stride{0} {}
    /** Position/axis constructor.
     * @param stride Associated stride
     * @param cursor Current position
    **/
    Iterator(Elem* cursor, size_t stride) noexcept: cursor{cursor}, stride{stride} {}
public:
    /** Aliased iterator traits.
    **/
    using difference_type   = size_t;
    using value_type        = Elem;
    using pointer           = Elem*;
    using reference         = Elem&;
    using iterator_category = ::std::random_access_iterator_tag;
public: // ValueSwappable concept
    /** Swap two elements.
     * @param iter Pointed element to swap with
    **/
    void swap(This& iter) noexcept {
        auto temp = ::std::move(*iter.cursor);
        *iter.cursor = ::std::move(*cursor);
        *cursor = ::std::move(temp);
    }
public: // RandomAccessIterator concept
    /** Displacements.
    **/
    This& operator+=(size_t n) noexcept {
        cursor += n * stride;
        return *this;
    }
    This& operator-=(size_t n) noexcept {
        cursor -= n * stride;
        return *this;
    }
    This operator+(size_t n) noexcept {
        return This{cursor + n * stride, stride};
    }
    This operator-(size_t n) noexcept {
        return This{cursor - n * stride, stride};
    }
    This& operator++() noexcept {
        cursor += stride;
        return *this;
    }
    This operator++(int) noexcept {
        auto copy = *this;
        cursor += stride;
        return copy;
    }
    This& operator--() noexcept {
        cursor -= stride;
        return *this;
    }
    This operator--(int) noexcept {
        auto copy = *this;
        cursor -= stride;
        return copy;
    }
    /** Distance.
    **/
    size_t operator-(This const& x) const noexcept {
        return (cursor - x.cursor) / static_cast<ptrdiff_t>(stride);
    }
    /** Dereferencing.
    **/
    Elem& operator*() const noexcept {
        return *cursor;
    }
    Elem& operator[](size_t n) const noexcept {
        return *(cursor + n * stride);
    }
    /** Comparison.
    **/
    bool operator==(This const& x) const noexcept {
        return x.cursor == cursor;
    }
    bool operator!=(This const& x) const noexcept {
        return x.cursor != cursor;
    }
    bool operator<(This const& x) const noexcept {
        return (x.cursor - cursor) > 0l;
    }
    bool operator>(This const& x) const noexcept {
        return (x.cursor - cursor) < 0l;
    }
    bool operator<=(This const& x) const noexcept {
        return (x.cursor - cursor) >= 0l;
    }
    bool operator>=(This const& x) const noexcept {
        return (x.cursor - cursor) <= 0l;
    }
};

/** Selector helper class.
 * @param Elem Single element class
**/
template<class Elem> class Selector {
private:
    /** Aliased classes.
    **/
    using This = Selector<Elem>;
private:
    Elem* array; // First array element
    size_t const* stride; // Remaining strides
public:
    /** Selector constructor.
     * @param array  First array element
     * @param stride Stride for that level
    **/
    Selector(Elem* array, size_t const* stride) noexcept: array{array}, stride{stride} {}
    /** Array element access, no bound checking.
     * @param i Index for the current level
    **/
    This operator[](size_t i) const noexcept {
        return This{array + i * stride[0], stride + 1};
    }
    /** Current element read.
     * @return Reference to the current element
    **/
    operator Elem&() const noexcept {
        return *array;
    }
    Elem& get() const noexcept {
        return *array;
    }
    /** Current element write.
     * @param value Value to write
     * @return Reference to the current element
    **/
    Elem& operator=(Elem const& value) const {
        return *array = value;
    }
    Elem& operator=(Elem&& value) const {
        return *array = ::std::move(value);
    }
};

/** Simple array (row-major order) access helper class.
 * @param Type (Array(s) of) single element class
**/
template<class Type> class Array {
private:
    /** Type parser helper class.
    **/
    template<class X> class Parser {
    public:
        using Elem = X;
    public:
        static constexpr size_t depth = 1; // Number of strides to reserve
    };
    template<class X> class Parser<Array<X>> {
    private:
        using Parsed = Parser<X>;
    public:
        using Elem = typename Parsed::Elem;
    public:
        static constexpr auto depth = Parsed::depth + 1; // Number of strides to reserve
    };
    /** Parsed class.
    **/
    using Parsed = Parser<Type>;
public:
    /** Aliased classes.
    **/
    using Elem = typename Parsed::Elem;
    using Iter = Iterator<Elem>;
    using Slct = Selector<Elem>;
private:
    static constexpr auto depth = Parsed::depth;
private:
    Elem*  array;  // Pointer over the first array element
    size_t length; // Total length
    size_t dim[depth];    // Array dimensions
    size_t stride[depth]; // Strides of this array
public:
    /** Array, dimensions and strides constructor.
     * @param array Base array
     * @param dim   Array of dimensions
    **/
    Array(Elem* array, size_t const (&dim)[depth]) noexcept: array{array} {
        size_t length = 1;
        for (size_t i = 0; i < depth; ++i) {
            this->dim[i] = dim[i];
            length *= dim[i];
        }
        this->length = length;
        if (depth == 1) {
            stride[0] = 1;
        } else {
            stride[depth - 1] = 1;
            for (size_t i = depth - 1; i > 0; --i)
                stride[i - 1] = stride[i] * dim[i];
            stride[0] = stride[1] * dim[1];
        }
    }
public:
    /** Get the array dimensions.
     * @return Array dimensions
    **/
    decltype(dim) const& dims() const noexcept {
        return dim;
    }
    /** Get the array strides.
     * @return Array strides
    **/
    decltype(stride) const& strides() const noexcept {
        return stride;
    }
public:
    /** Array element access, no bound checking.
     * @param i Array index for the current level
     * @return Selector helper instance
    **/
    Slct operator[](size_t i) const noexcept {
        return Slct{array + i * stride[0], stride + 1};
    }
    /** Array axis iteration, no bound checking.
     * @param a Axis index
     * @param s Section index
     * @return Start iterator, end iterator
    **/
    ::std::tuple<Iter, Iter> axis(size_t a, size_t s) const noexcept {
        auto start = array;
        size_t p = length / dim[a];
        for (size_t i = 0; i < depth; ++i) { // Compute start address
            if (i == a)
                continue;
            p /= dim[i];
            auto d = s / p;
            start += d * stride[i];
            s -= d * p;
        }
        return ::std::make_tuple(Iter{start, stride[a]}, Iter{start + dim[a] * stride[a], stride[a]});
    }
};
