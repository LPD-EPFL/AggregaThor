/**
 * @file   op.cpp
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
 * Bulyan over Multi-Krum GAR, TensorFlow custom operation.
 *
 * Based on the algorithm introduced in the following paper:
 *   El Mhamdi El Mahdi, Guerraoui Rachid, and Rouault Sébastien.
 *   The Hidden Vulnerability of Distributed Learning in Byzantium.
 *   In Dy, J. and Krause, A. (eds.), Proceedings of the 35th International
 *   Conference on Machine Learning, volume 80 of Proceedings of Machine
 *   Learning  Research, pp. 3521-3530, Stockholmsmässan, Stockholm Sweden,
 *   10-15 Jul 2018. PMLR. URL http://proceedings.mlr.press/v80/mhamdi18a.html.
**/

#include <common.hpp>
#include "decl.hpp"

// -------------------------------------------------------------------------- //
// Op declaration and shape inference

REGISTER_OP(OP_TEXT)
    .Input("gradients: T")
    .Output("aggregated: T")
    .Attr("T: {float, double}")
    .Attr("f: int")
    .Attr("m: int")
    .SetShapeFn([](InferenceContext* c) {
        auto&& input_tn = c->input(0);
        ShapeHandle dummy;
        TF_RETURN_IF_ERROR(c->WithRank(input_tn, 2, &dummy));
        c->set_output(0, c->MakeShape(::std::vector<DimensionHandle>{c->Dim(input_tn, 1)}));
        return Status::OK();
    });

// -------------------------------------------------------------------------- //
// Interface implementation
namespace OP_NAME {

template<class Device, class T> class Interface: public OpKernel {
protected:
    int f; // Number of byzantine workers
    int m; // Number of averaged gradients for Krum
public:
    explicit Interface(OpKernelConstruction* context): OpKernel{context} {
        OP_REQUIRES_OK(context, context->GetAttr("f", &f));
        OP_REQUIRES(context, f >= 0, errors::InvalidArgument("Need f >= 0, got ", f));
        OP_REQUIRES_OK(context, context->GetAttr("m", &m));
        OP_REQUIRES(context, m >= 1, errors::InvalidArgument("Need m >= 1, got ", m));
    }
public:
    void Compute(OpKernelContext* context) override {
        Tensor const& input_tn = context->input(0);
        OP_REQUIRES(context, input_tn.NumElements() <= tensorflow::kint32max, errors::InvalidArgument("Too many elements in tensor"));
        auto n = input_tn.dim_size(0);
        auto d = input_tn.dim_size(1);
        Tensor* output_tn = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape{d}, &output_tn));
        Kernel<Device, T>::process(*context, n, f, d, m, input_tn, *output_tn);
    }
};

}
// -------------------------------------------------------------------------- //
// Interface-kernel registrations

// CPU kernel
#define REGISTER_CPU(T) \
    extern template class OP_NAME::Kernel<CPUDevice, T>; \
    REGISTER_KERNEL_BUILDER(Name(OP_TEXT).Device(DEVICE_CPU).TypeConstraint<T>("T"), OP_NAME::Interface<CPUDevice, T>)
REGISTER_CPU(float);
REGISTER_CPU(double);
#undef REGISTER_CPU

// GPU kernel
#ifdef GOOGLE_CUDA

#define REGISTER_GPU(T) \
    extern template class OP_NAME::Kernel<GPUDevice, T>; \
    REGISTER_KERNEL_BUILDER(Name(OP_TEXT).Device(DEVICE_GPU).TypeConstraint<T>("T"), OP_NAME::Interface<GPUDevice, T>)
REGISTER_GPU(float);
REGISTER_GPU(double);
#undef REGISTER_GPU

#endif
