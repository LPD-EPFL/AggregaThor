/**
 * @file   common.hpp
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
 * Common declarations.
**/

#pragma once

#include <cstdint>
#include <memory>
#include <type_traits>

// -------------------------------------------------------------------------- //
// Compiler version check

#if __cplusplus < 201103L
    #error This translation unit requires at least a C++11 compiler
#endif
#ifndef __GNUC__
    #error This translation unit requires a GNU C++ compiler
#endif

// -------------------------------------------------------------------------- //
// Macro helpers

/** Define a proposition as likely true.
 * @param prop Proposition
**/
#undef likely
#define likely(prop) \
    __builtin_expect((prop) ? 1 : 0, 1)

/** Define a proposition as likely false.
 * @param prop Proposition
**/
#undef unlikely
#define unlikely(prop) \
    __builtin_expect((prop) ? 1 : 0, 0)

/** Op name management.
**/
#define CONVERT(X)   #X
#define MERGE(X, Y)  X##Y
#define TO_STRING(X) CONVERT(X)
#define TO_OPNAME(X) MERGE(X, Op)
#define TO_IFNAME(X) MERGE(X, If)

// -------------------------------------------------------------------------- //
// Exception tree
namespace Exception {

/** Defines a simple exception.
 * @param name   Exception name
 * @param parent Parent exception
 * @param text   Explanatory string
**/
#define EXCEPTION(name, parent, text) \
    class name: public parent { \
    public: \
        /** Return the explanatory string. \
         * @return Explanatory string \
        **/ \
        virtual char const* what() const noexcept { \
            return "native: " text; \
        } \
    }

/** User-defined exception root.
**/
EXCEPTION(Any, ::std::exception, "user-defined exception");

}
// -------------------------------------------------------------------------- //
// Class helpers

/** Non constructible class.
**/
class Static {
public:
    /** Deleted copy constructor/assignment.
    **/
    Static(Static const&) = delete;
    Static& operator=(Static const&) = delete;
    /** Deleted default constructor.
    **/
    Static() = delete;
};

/** Simple 'free'-calling deleter class.
**/
class Free final {
public:
    /** Default copy constructor/assignment.
    **/
    Free(Free const&) = default;
    Free& operator=(Free const&) = default;
    /** Default constructor.
    **/
    Free() noexcept {};
public:
    /** Free-calling deleter.
     * @param ptr Pointer to the memory to free
    **/
    void operator()(void* ptr) const noexcept {
        ::free(ptr);
    }
};

// -------------------------------------------------------------------------- //
// Function helpers

/** Allocate a "variable" (as in "non-constexpr const size") length array of the given size.
 * @param Type Element class
 * @param size Number of elements
 * @return Pointer on the first element of the array
**/
template<class Type> static ::std::unique_ptr<Type[], Free> vlarray(size_t size) {
    auto align = alignof(typename ::std::remove_extent<Type>::type[size]); // Always a power of 2
    if (align < sizeof(void*))
        align = sizeof(void*);
    auto alloc = ::aligned_alloc(align, size * sizeof(Type));
    if (unlikely(!alloc))
        throw ::std::bad_alloc{};
    return ::std::unique_ptr<Type[], Free>{reinterpret_cast<typename ::std::unique_ptr<Type[], Free>::pointer>(alloc)}; // Constructors are all marked noexcept
}
