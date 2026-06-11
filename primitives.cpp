//
// Created by Aditya Dudeja on 06/06/26.
//

#include <vector>
#include <iostream>
#include "matrix.h"
// int main() {}



class Primitive {
public:
    bool evaluated = false;
    bool traced = false;
    uint8_t* out_buffer = nullptr;
    std::atomic<uint32_t>* out_refcount = nullptr;
    id<MTLBuffer> out_metal_buffer = nullptr;
    virtual ~Primitive() = default;
    virtual void eval_cpu(matrix& out) = 0;
    virtual void eval_metal(matrix& out) { eval_cpu(out); }
    virtual void build_trace_cpu(matrix& out) = 0;
    virtual void build_trace_metal(matrix& out) = 0;
    virtual void execute_trace_cpu(matrix& out) = 0;
    virtual void execute_trace_metal(matrix& out) = 0;
    virtual void clear_trace_checks() = 0;
    virtual std::vector<matrix> vjp(const matrix& grad_out) = 0;
    virtual matrix jvp(const std::vector<matrix>& tangents) = 0;
};


class AdditionPrimitive : public Primitive {
public:
    matrix a;
    matrix b;
    BroadcastDescriptor* desc_a = nullptr;
    BroadcastDescriptor* desc_b = nullptr;

    AdditionPrimitive(const matrix& a, const matrix& b) : a(a), b(b) {}
    void eval_cpu(matrix &out) override {
        if (a.tape) { a.tape->eval_cpu(a); }
        if (b.tape) { b.tape->eval_cpu(b); }

        if (!out.buffer) {
            if (out.tape->out_buffer) {
                out.buffer = out.tape->out_buffer;
                out.metalBuffer = out.tape->out_metal_buffer;
                out.refCount = out.tape->out_refcount;
                out.refCount->fetch_add(1);
            } else {
                out.buffer = new uint8_t[out.effectiveBufferSize() * dtype_size(out.type)];
                out.begin_refcount();
                out.buildMetalBuffer();

                out.tape->out_buffer = (uint8_t*)out.buffer;
                out.tape->out_metal_buffer = out.metalBuffer;
                out.tape->out_refcount = out.refCount;
            }
        }
        if (evaluated) {return;} else {evaluated = true;}
        if (compare_shapes(a, b) && !(a.flags & NON_CONTIGUOUS_FLAG) && !(b.flags & NON_CONTIGUOUS_FLAG)) {
            a.add_cpu(b, out, EvalType::EVAL_CPU);
        } else {
            a.add_cpu_brodcasted(b, out, EvalType::EVAL_CPU);
        }
    };
    void eval_metal(matrix &out) override {
        if (a.tape) { a.tape->eval_metal(a); }
        if (b.tape) { b.tape->eval_metal(b); }
        if (!out.buffer) {
            if (out.tape->out_buffer) {
                out.buffer = out.tape->out_buffer;
                out.metalBuffer = out.tape->out_metal_buffer;
                out.refCount = out.tape->out_refcount;
                out.refCount->fetch_add(1);
            } else {
                out.buffer = new uint8_t[out.effectiveBufferSize() * dtype_size(out.type)];
                out.begin_refcount();
                out.buildMetalBuffer();

                out.tape->out_buffer = (uint8_t*)out.buffer;
                out.tape->out_metal_buffer = out.metalBuffer;
                out.tape->out_refcount = out.refCount;
            }
        }
        if (evaluated) {return;} else {evaluated = true;}
        if (compare_shapes(a, b) && !(a.flags & NON_CONTIGUOUS_FLAG) && !(b.flags & NON_CONTIGUOUS_FLAG)) {
            a.add_cpu(b, out, EvalType::EVAL_METAL);
        } else {
            a.add_cpu_brodcasted(b, out, EvalType::EVAL_METAL);
        }
    }
    void build_trace_cpu(matrix& out) override {
        if (a.tape) { a.tape->build_trace_cpu(a); }
        if (b.tape) { b.tape->build_trace_cpu(b); }

        if (!out.buffer) {
            if (out.tape->out_buffer) {
                out.buffer = out.tape->out_buffer;
                out.metalBuffer = out.tape->out_metal_buffer;
                out.refCount = out.tape->out_refcount;
                out.refCount->fetch_add(1);
                return;
            } else {
                out.buffer = new uint8_t[out.effectiveBufferSize() * dtype_size(out.type)];
                out.begin_refcount();
                out.buildMetalBuffer();

                out.tape->out_buffer = (uint8_t*)out.buffer;
                out.tape->out_metal_buffer = out.metalBuffer;
                out.tape->out_refcount = out.refCount;
            }
        }


        if (compare_shapes(a, b) && !(a.flags & NON_CONTIGUOUS_FLAG) && !(b.flags & NON_CONTIGUOUS_FLAG)) {
            a.add_cpu(b, out, EvalType::BUILD_TRACE);
        } else {
            a.add_cpu_brodcasted(b, out, EvalType::BUILD_TRACE);
        }
    }
    void build_trace_metal(matrix& out) override {
        if (a.tape) { a.tape->build_trace_metal(a);}
        if (b.tape) { b.tape->build_trace_metal(b);}
        if (!out.buffer) {
            if (out.tape->out_buffer) {
                out.buffer = out.tape->out_buffer;
                out.metalBuffer = out.tape->out_metal_buffer;
                out.refCount = out.tape->out_refcount;
                out.refCount->fetch_add(1);
                return;
            } else {
                out.buffer = new uint8_t[out.effectiveBufferSize() * dtype_size(out.type)];
                out.begin_refcount();
                out.buildMetalBuffer();

                out.tape->out_buffer = (uint8_t*)out.buffer;
                out.tape->out_metal_buffer = out.metalBuffer;
                out.tape->out_refcount = out.refCount;
            }
        }

        if (compare_shapes(a, b) && !(a.flags & NON_CONTIGUOUS_FLAG) && !(b.flags & NON_CONTIGUOUS_FLAG)) {
            a.add_gpu(b, out, EvalType::BUILD_TRACE);
        } else {
            a.add_gpu_brodcasted(b, out, EvalType::BUILD_TRACE);
        }
    }
    void execute_trace_cpu(matrix &out) override {
        if (a.tape) { a.tape->execute_trace_cpu(a); }
        if (b.tape) { b.tape->execute_trace_cpu(b); }
        if (!out.buffer) {
            if (out.tape->out_buffer) {
                out.buffer = out.tape->out_buffer;
                out.metalBuffer = out.tape->out_metal_buffer;
                out.refCount = out.tape->out_refcount;
                out.refCount->fetch_add(1);
            } else {
                out.buffer = new uint8_t[out.effectiveBufferSize() * dtype_size(out.type)];
                out.begin_refcount();
                out.buildMetalBuffer();

                out.tape->out_buffer = (uint8_t*)out.buffer;
                out.tape->out_metal_buffer = out.metalBuffer;
                out.tape->out_refcount = out.refCount;
            }
        }


        if (evaluated) {return;} else { evaluated =true; }
        if (compare_shapes(a, b) && !(a.flags & NON_CONTIGUOUS_FLAG) && !(b.flags & NON_CONTIGUOUS_FLAG)) {
            a.add_cpu(b, out, EvalType::EXEC_TRACE_CPU);
        } else {
            a.add_cpu_brodcasted(b, out, EvalType::EXEC_TRACE_CPU);
        }
    };
    
    void execute_trace_metal(matrix& out) override {
        if (a.tape) { a.tape->execute_trace_metal(a);}
        if (b.tape) { b.tape->execute_trace_metal(b);}

        if (!out.buffer) {
            if (out.tape->out_buffer) {
                out.buffer = out.tape->out_buffer;
                out.metalBuffer = out.tape->out_metal_buffer;
                out.refCount = out.tape->out_refcount;
                out.refCount->fetch_add(1);
            } else {
                out.buffer = new uint8_t[out.effectiveBufferSize() * dtype_size(out.type)];
                out.begin_refcount();
                out.buildMetalBuffer();

                out.tape->out_buffer = (uint8_t*)out.buffer;
                out.tape->out_metal_buffer = out.metalBuffer;
                out.tape->out_refcount = out.refCount;
            }
        }
        if (evaluated) {return;} else { evaluated =true; }
        if (compare_shapes(a, b) && !(a.flags & NON_CONTIGUOUS_FLAG) && !(b.flags & NON_CONTIGUOUS_FLAG)) {
            a.add_gpu(b, out, EvalType::EXEC_TRACE_METAL);
        } else {
            a.add_gpu_brodcasted(b, out, EvalType::EXEC_TRACE_METAL);
        }
    }

    std::vector<matrix> vjp(const matrix& grad_out) override {return {};}
    matrix jvp(const std::vector<matrix>& tangents) override {return matrix(0, dtype::UInt8);}
    void clear_trace_checks() override {
        evaluated = false;
        if (a.tape) { a.tape->clear_trace_checks(); }
        if (b.tape) { b.tape->clear_trace_checks(); }
    };
};

class SubtractionPrimitive : public Primitive {
    public:
    matrix a;
    matrix b;

    SubtractionPrimitive(const matrix& a, const matrix& b) : a(a), b(b) {}
    void eval_cpu(matrix &out) override {};
    void eval_metal(matrix &out) { eval_cpu(out); }
    void build_trace_cpu(matrix& out) override {}
    void build_trace_metal(matrix& out) override {}
    void execute_trace_cpu(matrix& out) override {}
    void execute_trace_metal(matrix& out) override {}
    std::vector<matrix> vjp(const matrix& grad_out) override {return {};}
    matrix jvp(const std::vector<matrix>& tangents) override {return matrix(0, dtype::UInt8);}
    void clear_trace_checks() override {
        evaluated = false;
        if (a.tape) { a.tape->clear_trace_checks(); }
        if (b.tape) { b.tape->clear_trace_checks(); }
    };
};

class MultiplicationPrimitive : public Primitive {
    public:
    matrix a;
    matrix b;
    uint8_t* out_buffer = nullptr;
    MultiplicationPrimitive(const matrix& a, const matrix& b) : a(a), b(b) {}
    void eval_cpu(matrix &out) override {};
    void eval_metal(matrix &out) { eval_cpu(out); }
    void build_trace_cpu(matrix& out) override {}
    void build_trace_metal(matrix& out) override {}
    void execute_trace_cpu(matrix& out) override {}
    void execute_trace_metal(matrix& out) override {}
    std::vector<matrix> vjp(const matrix& grad_out) override {return {};}
    matrix jvp(const std::vector<matrix>& tangents) override {return matrix(0, dtype::UInt8);}
    void clear_trace_checks() override {
        evaluated = false;
        if (a.tape) { a.tape->clear_trace_checks(); }
        if (b.tape) { b.tape->clear_trace_checks(); }
    };
};

class DivisionPrimitive : public Primitive {
    public:
    matrix a;
    matrix b;
    uint8_t* out_buffer = nullptr;
    DivisionPrimitive(const matrix& a, const matrix& b) : a(a), b(b) {}
    void eval_cpu(matrix &out) override {};
    void eval_metal(matrix &out) { eval_cpu(out); }
    void build_trace_cpu(matrix& out) override {}
    void build_trace_metal(matrix& out) override {}
    void execute_trace_cpu(matrix& out) override {}
    void execute_trace_metal(matrix& out) override {}
    std::vector<matrix> vjp(const matrix& grad_out) override {return {};}
    matrix jvp(const std::vector<matrix>& tangents) override {return matrix(0, dtype::UInt8);}
    void clear_trace_checks() override {
        evaluated = false;
        if (a.tape) { a.tape->clear_trace_checks(); }
        if (b.tape) { b.tape->clear_trace_checks(); }
    };
};

//using size_m = uint32;
//constexpr int SBO_MAX_DIMS = 3;
//union array_descriptor {
//    size_m inline_buffer[SBO_MAX_DIMS * 2]; // For dims <= 3
//    SharedArrayDescriptor* shared_arr_desc;  // For dims > 3
//};

