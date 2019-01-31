/**
 * @file   native.cpp
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
 * (Deprecated) Native (i.e. C++) implementation of several gradient aggregation rules (GAR).
**/

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
#include <functional>

#include <iostream> // NOTE: Debug

// -------------------------------------------------------------------------- //

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

/** Define one or several attributes.
 * @param type... Attribute names
**/
#undef as
#define as(type...) \
    __attribute__((type))

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
    /** Limited move constructor.
     * @param job Job to "move"
    **/
    AbstractJob(AbstractJob&& job) noexcept: lock{}, done{job.done}, cv{} {}
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

// FIXME: Correct handling of lvalue-references (which must be wrapped before passed to 'bind'), currently using raw pointers in implementations
/** Single-threaded, call-once, non-owning job class.
 * @param ... Wrapped callable class
**/
template<class...> class Job;
template<class... Args> class Job<void(Args...)> final: public AbstractJob {
    /** Aliased classes.
    **/
    using This = Job<void(Args...)>;
    using Func = void (*)(Args...);
    using Wrap = decltype(::std::bind(::std::declval<Func>(), ::std::declval<Args>()...));
public:
    Wrap func; // Wrapped function
public:
    /** Limited move constructor.
     * @param job Job to "move"
    **/
    Job(This&& job) noexcept: func{::std::move(job.func)} {}
    /** Standalone constructor.
     * @param func Function to wrap
     * @param ...  Arguments to bind
    **/
    Job(Func&& func, Args&&... args): func{::std::bind(::std::move(func), ::std::forward<Args>(args)...)} {}
protected:
    /** Invoke the wrapped transfer function.
    **/
    virtual void invoke() {
        func();
    }
};
template<class Res, class... Args> class Job<Res(Args...)> final: public AbstractJob {
private:
    /** Aliased classes.
    **/
    using This = Job<Res(Args...)>;
    using Func = Res (*)(Args...);
    using Wrap = decltype(::std::bind(::std::declval<Func>(), ::std::declval<Args>()...));
public:
    Wrap func; // Wrapped function
    Res store; // Stored result
public:
    /** Limited move constructor.
     * @param job Job to "move"
    **/
    Job(This&& job) noexcept: func{::std::move(job.func)}, store{::std::move(job.store)} {}
    /** Standalone constructor.
     * @param func Function to wrap
     * @param ...  Arguments to bind
    **/
    Job(Func&& func, Args&&... args): func{::std::bind(::std::move(func), ::std::forward<Args>(args)...)} {}
protected:
    /** Invoke the wrapped transfer function.
    **/
    virtual void invoke() {
        store = func();
    }
public:
    /** Access the stored result, underfined result if called before the job completed.
     * @return Reference to the stored result
    **/
    Res& get() noexcept {
        return store;
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
    /** Lock guard class.
    **/
    template<class Lock> using Guard = ::std::unique_lock<Lock>;
    /** Lock "reverse" guard class.
    **/
    template<class Lock> class Forsake {
    private:
        Lock& lock; // Bound lock
    public:
        /** Deleted copy constructor/assignment.
        **/
        Forsake(Forsake const&) = delete;
        Forsake& operator=(Forsake const&) = delete;
        /** Lock release constructor.
         * @param lock Lock to bind
        **/
        Forsake(Lock& lock): lock{lock} {
            lock.unlock();
        }
        /** Lock acquire destructor.
        **/
        ~Forsake() {
            lock.lock();
        }
    };
private:
    size_t  size;    // Number of workers
    Status  status;  // Current status
    Mutex   lock;    // Local lock
    CondVar cv;      // New jobs to process
    Jobs    jobs;    // Jobs to process
    Threads threads; // Worker threads
private:
    /** Worker thread entry point.
     * @param self Bound thread pool
    **/
    static void entry_point(ThreadPool& self) {
        Guard<Mutex> guard{self.lock};
        while (true) { // Main loop
            if (self.status == Status::detached) // Must terminate
                return;
            if (!self.jobs.empty()) { // Pick and run job
                auto& job = *self.jobs.front(); // Get job info
                self.jobs.pop(); // Remove job
                { // Run job
                    Forsake<Mutex> forsake{self.lock};
                    job();
                }
                continue;
            }
            self.cv.wait(guard); // Wait for event
        }
    }
    /** Get the default number of worker threads to use.
     * @return Default number of worker threads
    **/
    static size_t get_default_nbworker() noexcept {
        auto res = ::sysconf(_SC_NPROCESSORS_ONLN);
        if (unlikely(res <= 0))
            return 8; // Arbitrary default
        return res;
    }
public:
    /** Thread pool begin constructor.
     * @param nbworkers Number of worker threads to use (optional, 0 for auto)
    **/
    ThreadPool(size_t nbworkers = 0): size{nbworkers}, status{Status::running}, lock{}, cv{}, jobs{}, threads{} {
        if (nbworkers == 0)
            size = get_default_nbworker();
        { // Worker threads creation
            threads.reserve(size);
            for (decltype(size) i = 0; i < size; ++i)
                threads.emplace_back(entry_point, ::std::ref(*this));
        }
    }
    /** Notify then detach worker threads destructor.
    **/
    ~ThreadPool() {
        { // Mark ending
            Guard<Mutex> guard{lock};
            status = Status::detached;
        }
        cv.notify_all();
        for (auto&& thread: threads) // Detach workers
            thread.detach();
    }
public:
    /** Get the number of workers in this pool.
     * @return Number of workers
    **/
    auto get_size() noexcept {
        return size;
    }
    /** Submit a new job.
     * @param job Job to submit
    **/
    void submit(AbstractJob& job) {
        { // Submit one job
            Guard<Mutex> guard{lock};
            jobs.push(::std::addressof(job));
        }
        cv.notify_one(); // Notify one worker
    }
};

// Shared thread pool
ThreadPool pool;

// -------------------------------------------------------------------------- //

/** Parallel-for helper class.
 * @param ... Argument classes to forward
**/
template<class... Args> class For {
private:
    /** Aliased classes.
    **/
    using This  = For<Args...>;
    using Func  = void (*)(size_t, size_t, Args...);
    using Jobs  = ::std::vector<Job<void(size_t, size_t, Args...)>>;
public:
    /** Deleted copy constructor/assignment.
    **/
    For(This const&) = delete;
    For& operator=(This const&) = delete;
    /** Deleted default constructor.
    **/
    For() = delete;
public:
    /** Parallel-run submitter.
     * @param pool  Used thread pool
     * @param start Start position (included)
     * @param stop  Stop position (excluded)
     * @param func  Function to call
     * @param ...  Data to bind (optional)
    **/
    static void run(ThreadPool& pool, size_t start, size_t stop, Func&& func, Args&&... args) {
        Jobs jobs; // Submitted jobs
        // Reserve jobs
        size_t nbworkers = pool.get_size();
        size_t length    = stop - start;
        jobs.reserve(nbworkers > length ? nbworkers : length);
        // Submit jobs
        size_t cursor = start;
        for (size_t i = 0; i < nbworkers; ++i) {
            size_t length = stop - cursor;
            size_t remain = nbworkers - i;
            size_t take = length / remain + ((length % remain) << 1) / remain;
            if (take == 0)
                continue;
            // ::printf("** OVER %lu to %lu ITERATE from %lu to %lu **\n", start, stop, cursor, cursor + take);
            jobs.emplace_back(::std::move(func), cursor + 0ul, cursor + take, ::std::forward<Args>(args)...);
            pool.submit(jobs.back());
            cursor += take;
        }
        // Wait jobs
        for (auto&& job: jobs)
            job.wait();
    }
};

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
    auto& operator*() const noexcept {
        return *cursor;
    }
    auto& operator[](size_t n) const noexcept {
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
    auto const& dims() const noexcept {
        return dim;
    }
    /** Get the array strides.
     * @return Array strides
    **/
    auto const& strides() const noexcept {
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

// -------------------------------------------------------------------------- //

/** Quickly compute the squared l2-distance between two vectors.
 * @param dim Gradient dimension
 * @param va  Vector A
 * @param vb  Vector B
 * @return (A - B)²
**/
template<class Float> static Float squared_distance(size_t dim, Float const* va, Float const* vb) {
    /** Aliased classes.
    **/
    using Mutex = ::std::mutex;
    using Guard = ::std::lock_guard<Mutex>;
    // Shared state
    Mutex lock;
    Float res = 0;
    // Loop
    For<Float const*, Float const*, Mutex*, Float*>::run(pool, 0, dim, [](size_t start, size_t stop, Float const* va, Float const* vb, Mutex* lock, Float* res) {
        // Run
        Float sum = 0;
        for (size_t x = start; x < stop; ++x) {
            auto delta = va[x] - vb[x];
            sum += delta * delta;
        }
        { // Update 'res'
            Guard guard{*lock};
            *res += sum;
        }
    }, ::std::move(va), ::std::move(vb), &lock, &res);
    // Return result
    return res;
}

#define declare_squared_distance(type) \
    extern "C" type squared_distance_##type(size_t dim, type const* va, type const* vb) { \
        return squared_distance<type>(dim, va, vb); \
    }
declare_squared_distance(float)
declare_squared_distance(double)
#undef declare_squared_distance

// -------------------------------------------------------------------------- //

/** Quickly compute the median coordinate by coordinate.
 * @param dim    Gradient dimension
 * @param n      Number of input gradients
 * @param inputs Input gradients (will be mangled)
 * @param output Output gradient
**/
template<class Float> static void median(size_t dim, size_t n, Float* inputs, Float* output) {
    For<size_t, size_t, Float*, Float*>::run(pool, 0, dim, [](size_t start, size_t stop, size_t dim, size_t n, Float* raw_inputs, Float* raw_output) {
        Array<Array<Float>> inputs{raw_inputs, {n, dim}};
        Array<Float> output{raw_output, {dim}};
        for (size_t x = start; x < stop; ++x) { // Coordinates to work on
            typename decltype(inputs)::Iter axis;
            typename decltype(inputs)::Iter aend;
            ::std::tie(axis, aend) = inputs.axis(0, x);
            auto median = axis + (aend - axis) / 2ul;
            ::std::nth_element(axis, median, aend, [](Float a, Float b) {
                if (!::std::isfinite(a))
                    return false;
                if (!::std::isfinite(b))
                    return true;
                return a < b;
            });
            output[x] = *median;
        }
    }, ::std::move(dim), ::std::move(n), ::std::move(inputs), ::std::move(output));
}

#define declare_median(type) \
    extern "C" void median_##type(size_t dim, size_t n, type* inputs, type* output) { \
        median<type>(dim, n, inputs, output); \
    }
declare_median(float)
declare_median(double)
#undef declare_median

/** Quickly compute the averaged median coordinate by coordinate.
 * @param dim    Gradient dimension
 * @param n      Number of input gradients
 * @param beta   Number of averaged elements
 * @param inputs Input gradients (will be mangled)
 * @param output Output gradient
**/
template<class Float> static void averaged_median(size_t dim, size_t n, size_t beta, Float* inputs, Float* output) {
    For<size_t, size_t, size_t, Float*, Float*>::run(pool, 0, dim, [](size_t start, size_t stop, size_t dim, size_t n, size_t beta, Float* raw_inputs, Float* raw_output) {
        Array<Array<Float>> inputs{raw_inputs, {n, dim}};
        Array<Float> output{raw_output, {dim}};
        for (size_t x = start; x < stop; ++x) { // Coordinates to work on
            typename decltype(inputs)::Iter axis, aend;
            ::std::tie(axis, aend) = inputs.axis(0, x);
            auto length = aend - axis;
            auto median = axis + length / 2ul;
            ::std::nth_element(axis, median, aend);
            auto zero = *median;
            ::std::nth_element(axis, axis + (beta - 1), aend, [zero](Float x, Float y) {
                auto dx = x - zero;
                if (dx < 0)
                    dx = -dx;
                auto dy = y - zero;
                if (dy < 0)
                    dy = -dy;
                return dx < dy;
            });
            auto average = axis[0];
            for (size_t i = 1; i < beta; ++i)
                average += axis[i];
            output[x] = average / static_cast<Float>(beta);
        }
    }, ::std::move(dim), ::std::move(n), ::std::move(beta), ::std::move(inputs), ::std::move(output));
}

#define declare_averaged_median(type) \
    extern "C" void averaged_median_##type(size_t dim, size_t n, size_t beta, type* inputs, type* output) { \
        averaged_median<type>(dim, n, beta, inputs, output); \
    }
declare_averaged_median(float)
declare_averaged_median(double)
#undef declare_averaged_median

/** Quickly compute the average with NaN support coordinate by coordinate.
 * @param dim    Gradient dimension
 * @param n      Number of input gradients
 * @param inputs Input gradients
 * @param output Output gradient
**/
template<class Float> static void average_nan(size_t dim, size_t n, Float* inputs, Float* output) {
    For<size_t, size_t, Float*, Float*>::run(pool, 0, dim, [](size_t start, size_t stop, size_t dim, size_t n, Float* raw_inputs, Float* raw_output) {
        Array<Array<Float>> inputs{raw_inputs, {n, dim}};
        Array<Float> output{raw_output, {dim}};
        for (size_t x = start; x < stop; ++x) { // Coordinates to work on
            typename decltype(inputs)::Iter axis;
            ::std::tie(axis, ::std::ignore) = inputs.axis(0, x);
            size_t count = 0;
            Float  sum = 0;
            for (size_t i = 0; i < n; ++i) {
                auto v = axis[i];
                if (!::std::isfinite(v))
                    continue;
                ++count;
                sum += v;
            }
            output[x] = sum / static_cast<Float>(count);
        }
    }, ::std::move(dim), ::std::move(n), ::std::move(inputs), ::std::move(output));
}

#define declare_average_nan(type) \
    extern "C" void average_nan_##type(size_t dim, size_t n, type* inputs, type* output) { \
        average_nan<type>(dim, n, inputs, output); \
    }
declare_average_nan(float)
declare_average_nan(double)
#undef declare_average_nan

// -------------------------------------------------------------------------- //

/** "Asymmetric" distances manager.
 * @param Float Floating-point type to use
**/
template<class Float> class DistanceManager final {
private:
    /** Aliased classes.
    **/
    using This   = DistanceManager<Float>;
    using Inputs = Array<Array<Float const>>;
    using Axis   = typename Inputs::Iter;
private:
    ::std::unique_ptr<Float[]> distances; // Distances between gradients
    size_t nbworkers; // Number of workers
public:
    /** Deleted copy constructor/assignment.
    **/
    DistanceManager(This const&) = delete;
    This& operator=(This const&) = delete;
    /** Full distances constructor.
     * @param inputs Stacked input gradients
    **/
    DistanceManager(Inputs const& inputs): distances{new Float[inputs.dims()[0] * (inputs.dims()[0] - 1)]}, nbworkers{inputs.dims()[0]} {
        auto n = inputs.dims()[0];
        auto d = inputs.dims()[1];
        for (size_t i = 0; i < n - 1; ++i) {
            for (size_t j = i + 1; j < n; ++j) {
                Axis ai, aj;
                ::std::tie(ai, ::std::ignore) = inputs.axis(1, i);
                ::std::tie(aj, ::std::ignore) = inputs.axis(1, j);
                auto dist = squared_distance<Float>(d, ::std::addressof(*ai), ::std::addressof(*aj)); // NOTE: Small hack since 'squared_distance' does not support axis iterators
                if (!::std::isfinite(dist)) // Exclude gradient with at least one non-finite coordinate
                    dist = std::numeric_limits<Float>::max();
                get(i, j) = dist;
                get(j, i) = dist;
            }
        }
    }
private:
    /** Get the current array offset for two gradients.
     * @param i First gradient uid
     * @param j Second gradient uid, must be != j
     * @return Array offset
    **/
    auto pos(size_t i, size_t j) const noexcept {
        return i * (nbworkers - 1) + (j < i ? j : j - 1);
    }
public:
    /** Get the remaining number of workers.
     * @return Remaining number of workers
    **/
    auto size() const noexcept {
        return nbworkers;
    }
    /** Get the current distance reference between two gradients.
     * @param i First gradient uid
     * @param j Second gradient uid, must be != j
     * @return Distance from i to j
    **/
    auto& get(size_t i, size_t j) noexcept {
        return distances[pos(i, j)];
    }
    auto& get(size_t i, size_t j) const noexcept {
        return distances[pos(i, j)];
    }
};

/** Score manager.
 * @param Float Floating-point type to use
**/
template<class Float> class ScoreManager final {
private:
    /** Aliased classes.
    **/
    using This     = ScoreManager<Float>;
    using Distance = DistanceManager<Float>;
    using Tuple    = ::std::tuple<size_t, Float>;
    using Scores   = ::std::vector<Tuple>;
    using Inputs   = Array<Array<Float const>>;
    using Axis     = Iterator<Float>;
private:
    Distance const& distances; // Bound distance manager
    Scores scores; // (uid, score) for each (remaining) gradients
    size_t f; // Number of byzantine workers
private:
    /** Read the map, taking care that map[...] != uid.
     * @param uid Value of uid
     * @param val Value of map[...]
     * @return (Corrected) value of map[...]
    **/
    static size_t shift(size_t uid, size_t val) noexcept {
        if (val >= uid)
            return val + 1;
        return val;
    }
public:
    /** Deleted copy constructor/assignment.
    **/
    ScoreManager(This const&) = delete;
    This& operator=(This const&) = delete;
    /** Initial scores constructor and distance pruning.
     * @param distances Distances between gradients
     * @param f         Number of byzantine workers
    **/
    ScoreManager(Distance& distances, size_t f): distances{distances}, scores(distances.size()), f{f} {
        For<Distance*, Scores*, size_t>::run(pool, 0, scores.size(), [](size_t start, size_t stop, Distance* distances, Scores* scores, size_t f) {
            auto const n  = scores->size();
            auto const in = n - f - 2;
            size_t map[n - 1]; // Separate selected 'uid' from non-selected ones
            for (size_t i = 0; i < sizeof(map) / sizeof(*map); ++i)
                map[i] = i;
            for (size_t uid = start; uid < stop; ++uid) {
                ::std::nth_element(&map[0], &map[in], &map[n - 1], [=](size_t i, size_t j) {
                    return distances->get(shift(uid, i), uid) < distances->get(shift(uid, j), uid);
                });
                Float score = 0;
                { // Compute score + distance pruning
                    size_t i = 0;
                    for (; i < in; ++i) // Score = sum of 'in' closest gradients
                        score += distances->get(shift(uid, map[i]), uid);
                    for (; i < n - 1; ++i) // For 'uid', reset distance of gradients further away
                        distances->get(shift(uid, map[i]), uid) = 0;
                }
                { // Write uid + score
                    auto& tuple = (*scores)[uid];
                    ::std::get<0>(tuple) = uid;
                    ::std::get<1>(tuple) = score;
                }
            }
        }, &distances, &scores, ::std::move(f));
/* { // Debug (print distances)
    ::std::cout << ::std::endl;
    auto const n = scores.size();
    for (size_t i = 0; i < n; ++i) {
        ::std::cout << "distances[" << i << "]: ";
        for (size_t j = 0; j < n; ++j) {
            if (j == i) {
                ::std::cout << "---\t";
                continue;
            }
            ::std::cout << distances.get(i, j) << "\t";
        }
        ::std::cout << ::std::endl;
    }
} */
    }
public:
    /** Select the m = n - f - 2 smallest scoring gradient uids.
     * @param axis Output axis receiving the average of the selected gradients
    **/
    void select(Inputs const& inputs, Axis const& axis) noexcept {
        auto const n = scores.size();
        auto const m = n - f - 2;
        { // Selection and averaging
/* { // NOTE: Debug (print scores)
    ::std::cout << "scores   = {";
    auto i = 0;
    for (auto& score: scores) {
        if ((i++) > 0)
            ::std::cout << ", ";
        ::std::cout << ::std::get<0>(score) << ": " << ::std::get<1>(score);
    }
    ::std::cout << "}" << ::std::endl;
} */
            // Selection
            ::std::nth_element(scores.begin(), scores.begin() + m, scores.end(), [](Tuple const& i, Tuple const& j) {
                return ::std::get<1>(i) < ::std::get<1>(j);
            });
/* { // NOTE: Debug (print "ranks")
    ::std::cout << "ranks    = [";
    typename ::std::remove_cv<decltype(m)>::type i = 0;
    for (auto& score: scores) {
        if (i > 0)
            ::std::cout << (i == m ? " | " : ", ");
        ++i;
        ::std::cout << ::std::get<0>(score);
    }
    ::std::cout << "]" << ::std::endl;
} */
            // Averaging
            For<Inputs const*, Scores const*, Axis const*, size_t>::run(pool, 0, inputs.dims()[1], [](size_t start, size_t stop, Inputs const* inputs, Scores const* scores, Axis const* axis, size_t m) {
                for (size_t i = start; i < stop; ++i) {
                    Float val = 0;
                    for (size_t j = 0; j < m; ++j)
                        val += (*inputs)[::std::get<0>((*scores)[j])][i];
                    (*axis)[i] = val / static_cast<Float>(m);
                }
            }, ::std::addressof(inputs), ::std::addressof(scores), ::std::addressof(axis), m + 0);
/* { // NOTE: Debug (print selected gradient)
    ::std::cout << "selected = [";
    for (size_t i = 0; i < inputs.dims()[1]; ++i) {
        if (i > 0)
            ::std::cout << ", ";
        ::std::cout << axis[i];
    }
    ::std::cout << "]" << ::std::endl;
} */
        }
        { // Remove smallest scoring gradient (scores' vector is partially sorted)
            // Lookup smallest scoring uid
            size_t pos = 0;
            auto   val = ::std::get<1>(scores[pos]);
            for (size_t i = 1; i < m; ++i) { // After 'm', only higher scores
                auto prop = ::std::get<1>(scores[i]);
                if (prop < val) {
                    pos = i;
                    val = prop;
                }
            }
            // Remove selected gradient
            auto uid = ::std::get<0>(scores[pos]);
            if (pos < n - 1) // Swap with last element first
                ::std::swap(scores[pos], scores[n - 1]);
            scores.pop_back();
            // Update remaining gradient scores
            for (auto& score: scores)
                ::std::get<1>(score) -= distances.get(uid, ::std::get<0>(score));
        }
    }
};

/** Quickly compute Bulyan.
 * @param d       Gradient dimension
 * @param n       Number of input gradients
 * @param f       Number of byzantine gradients
 * @param s       Number of selected gradients
 * @param raw_in  Input gradients
 * @param raw_sel Selected gradients (buffer)
 * @param raw_out Output gradient
**/
template<class Float> static void bulyan(size_t d, size_t n, size_t f, size_t s, Float const* raw_in, Float* raw_sel, Float* raw_out) {
    // NOTE: Possible bug in distance computation/pruning (don't fix, use ops version instead)
    // Wrap arrays
    Array<Array<Float const>> inputs{raw_in, {n, d}};
    Array<Array<Float>>       selects{raw_sel, {s, d}};
    Array<Float>              output{raw_out, {d}};
    // Compute distances
    DistanceManager<Float> distances{inputs};
    // Compute initial scores and distance pruning
    ScoreManager<Float> scores{distances, f};
    // Selection loop
    for (size_t k = 0; k < s; ++k) {
        typename Array<Array<Float>>::Iter axis;
        ::std::tie(axis, ::std::ignore) = selects.axis(1, k);
        scores.select(inputs, axis);
    }
    // Averaged-median coordinate-by-coordinate
    averaged_median<Float>(d, s, s - 2 * f, raw_sel, raw_out);
}

#define declare_bulyan(type) \
    extern "C" void bulyan_##type(size_t d, size_t n, size_t f, size_t s, type const* raw_in, type* raw_sel, type* raw_out) { \
        bulyan<type>(d, n, f, s, raw_in, raw_sel, raw_out); \
    }
declare_bulyan(float)
declare_bulyan(double)
#undef declare_bulyan
