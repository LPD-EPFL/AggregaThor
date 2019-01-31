/**
 * @file   threadpool.hpp
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
 * Another thread pool management class, with parallel for-loop helper.
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
#include <iostream>
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

/** Single-threaded, call-once, non-owning job class.
**/
class AbstractJob {
public:
    /** Aliased classes.
    **/
    using Mutex   = ::std::mutex;
    using CondVar = ::std::condition_variable;
private:
    Mutex lock; // Access serialization
    bool  done; // Job completion state
    CondVar cv; // Completion notification
public:
    /** Deleted copy constructor/assignment.
    **/
    AbstractJob(AbstractJob const&) = delete;
    AbstractJob& operator=(AbstractJob const&) = delete;
    /** Non-completed constructor.
    **/
    AbstractJob(): lock{}, done{false}, cv{} {}
protected:
    virtual void invoke() = 0;
public:
    /** Invoke the job transfer function, mark the job as completed, to be called at most once.
    **/
    void operator()() {
        invoke();
        mark();
    }
public:
    /** Mark the job as completed.
    **/
    void mark() {;
        { // Mark as completed
            ::std::lock_guard<Mutex> guard{lock};
            done = true;
        }
        cv.notify_all(); // Notify waiting threads
    }
    /** Wait for the job to complete.
    **/
    void wait() {
        ::std::unique_lock<Mutex> guard{lock};
        while (!done)
            cv.wait(guard);
    }
};

/** Non-owning thread pool class.
**/
class ThreadPool {
public:
    /** Aliased classes.
    **/
    using Mutex   = ::std::mutex;
    using CondVar = ::std::condition_variable;
    using Jobs    = ::std::queue<AbstractJob*>;
    using Threads = ::std::vector<::std::thread>;
    /** Status enum class.
    **/
    enum class Status {
        running, // Workers wait for job
        detached // Workers must terminate
    };
private:
    size_t  size;    // Number of workers
    Status  status;  // Current status
    Mutex   lock;    // Local lock
    CondVar cv;      // New jobs to process
    Jobs    jobs;    // Jobs to process
    Threads threads; // Worker threads
private:
    static void entry_point(ThreadPool&);
    static size_t get_default_nbworker() noexcept;
public:
    ThreadPool(size_t arg0 = 0);
    ~ThreadPool();
public:
    /** Get the number of workers in this pool.
     * @return Number of workers
    **/
    size_t get_size() noexcept {
        return size;
    }
public:
    void submit(AbstractJob&);
};

// Shared thread pool
extern ThreadPool pool;

// -------------------------------------------------------------------------- //

/** Simple job wrapper class.
 * @param Func Callable class (deducted)
 * @param Res  Return type (deducted)
**/
template<class Func, class Res = typename ::std::remove_reference<decltype(::std::declval<Func>()(::std::declval<size_t>(), ::std::declval<size_t>()))>::type> class SimpleJob final: public AbstractJob {
public:
    Func&  func;  // Callable instance
    Res    store; // Stored result
    size_t start; // Opaque start position (included)
    size_t stop;  // Opaque stop position (excluded)
public:
    /** Bind constructor.
     * @param func  Callable instance to bind
     * @param start Opaque start address (included)
     * @param stop  Opaque stop position (excluded)
    **/
    SimpleJob(Func& func, size_t start, size_t stop): func{func}, start{start}, stop{stop} {}
protected:
    /** Invoke the function.
    **/
    virtual void invoke() {
        store = func(start, stop);
    }
public:
    /** Access the stored result, undefined result if called before the job completed.
     * @return Reference to the stored result
    **/
    Res& get() noexcept {
        return store;
    }
};
template<class Func> class SimpleJob<Func, void> final: public AbstractJob {
public:
    Func&  func;  // Callable instance
    size_t start; // Opaque start position (included)
    size_t stop;  // Opaque stop position (excluded)
public:
    /** Bind constructor.
     * @param func  Callable instance to bind
     * @param start Opaque start position (included)
     * @param stop  Opaque stop position (excluded)
    **/
    SimpleJob(Func& func, size_t start, size_t stop): func{func}, start{start}, stop{stop} {}
protected:
    /** Invoke the wrapped transfer function.
    **/
    virtual void invoke() {
        func(start, stop);
    }
};

/** Auto-balancing parallel job submitter, wait for all jobs to complete before returning.
 * @param pool  Used thread pool
 * @param start Start position (included)
 * @param stop  Stop position (excluded)
 * @param func  Callable instance to call in parallel, taking (size_t start, size_t stop) -> Any
**/
template<class Func> static void parallel_for(ThreadPool& pool, size_t start, size_t stop, Func&& func) {
    using Job = SimpleJob<Func>;
    using Mem = typename ::std::aligned_storage<alignof(Job), sizeof(Job)>::type;
    // Reserve jobs
    auto nbworkers = pool.get_size();
    auto length    = stop - start;
    auto count = nbworkers > length ? length : nbworkers;
    auto jobs  = vlarray<Mem>(count);
    // Submit jobs
    auto cursor = start;
    for (size_t i = 0; i < count; ++i) {
        size_t length = stop - cursor;
        size_t remain = count - i;
        size_t take = length / remain + ((length % remain) << 1) / remain;
        pool.submit(*(new(jobs.get() + i) Job(func, cursor, cursor + take)));
        cursor += take;
    }
    // Wait all jobs
    for (size_t i = 0; i < count; ++i)
        reinterpret_cast<Job&>(jobs[i]).wait();
}
