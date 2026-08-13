//
// Created by Aditya Dudeja on 06/06/26.
//

#include <vector>
#include <iostream>
#include <atomic>
#include "matrix.h"




static matrix ensure_graph_ready(const matrix& m) {
    if (!m.tape) {
        if (!m.buffer) {
            throw std::runtime_error("Cannot add an empty, tapeless matrix to the compute graph!");
        }
        if (!m.refCount) {
            const_cast<matrix&>(m).begin_refcount();
        }
        if (!m.metalBuffer) {
            const_cast<matrix&>(m).buildMetalBuffer();
        }
    }
    return m;
}

static const std::vector<matrix>& ensure_graph_ready(const std::vector<matrix>& v) {
    for (const auto& m : v) ensure_graph_ready(m);
    return v;
}


#include <functional>
#include <typeinfo>
// int main() {}
#pragma once


class Primitive {
public:
    std::atomic<uint32_t> primitive_refCount{1};
    bool evaluated = false;
    bool traced = false;
    uint64_t version = 1;
    int64_t last_visited_pass_id = 0;
    uint8_t* out_buffer = nullptr;
    std::atomic<uint32_t>* out_refcount = nullptr;
    id<MTLBuffer> out_metal_buffer = nil;

    
    virtual matrix* get_borrowed_input() { return nullptr; }

    virtual ~Primitive() {
        if (out_refcount) {
            if (out_refcount->fetch_sub(1, std::memory_order_acq_rel) == 1) {
                if (out_buffer) delete[] static_cast<uint8_t*>(out_buffer);
                delete out_refcount;
            }
        }
    }

    void update_cache(uint8_t* new_buf, id<MTLBuffer> new_metal, std::atomic<uint32_t>* new_refcount) {
        if (out_refcount == new_refcount) return;
        if (out_refcount) {
            if (out_refcount->fetch_sub(1, std::memory_order_acq_rel) == 1) {
                if (out_buffer) delete[] static_cast<uint8_t*>(out_buffer);
                delete out_refcount;
            }
        }
        out_buffer = new_buf;
        out_metal_buffer = new_metal;
        out_refcount = new_refcount;
        out_refcount->fetch_add(1, std::memory_order_relaxed);
        
    }
    
    virtual void eval_cpu(matrix& out, EvalType eval_type) = 0;
    virtual void eval_metal(matrix& out, EvalType eval_type) =0;

    virtual void clear_trace_checks() = 0;
    virtual std::vector<matrix> vjp(matrix& grad_out) = 0;
    virtual matrix jvp(std::vector<matrix>& tangents) = 0;
    virtual matrix vmap(std::function<matrix(const matrix&)> func, std::vector<int> in_axis) = 0;
    virtual std::vector<matrix> get_inputs() = 0;
    virtual bool invalidate_pass(uint64_t current_pass_id) = 0;
};

class SwapLeafPrimitive : public Primitive {
    
public:
    
    SwapLeafPrimitive(matrix& output) {
        out_buffer = (uint8_t*)output.buffer;
        out_metal_buffer = output.metalBuffer;
        out_refcount = output.refCount;
        if (out_refcount) out_refcount->fetch_add(1, std::memory_order_relaxed);
    }

    void eval_cpu(matrix &out, EvalType eval_type) override { evaluated = true; }
    void eval_metal(matrix &out, EvalType eval_type) override { evaluated = true; }


    std::vector<matrix> vjp(matrix& grad_out) override {
        // a is vector a.reshape=>c soo J = (dc/da) and for reshape we didnt change anything so its 1 so J = 1 reshaped
        return { grad_out };
    }
    matrix jvp(std::vector<matrix>& tangents) override {
        // a is vector a.reshape=>c soo J = (dc/da) and for reshape we didnt change anything so its 1 so J = 1 reshaped
        return matrix::scalar(1.0f);

    }
// Converts dirty=true into evaluated=false along every path leading to a dirty node,
    // then resets dirty. This is needed because nodes only know their parents, not their
    // children, so the only way to propagate "this subtree needs re-eval" is to walk down
    // from parents and pull the signal up via return value.
    //
    // Memoization (last_visited_pass_id): a shared node (diamond dependency) gets visited
    // once per pass by its first parent, which does the real work and returns the result.
    // Every subsequent parent visiting it in the *same* pass hits the early-return and reads
    // !evaluated instead of recomputing. This is safe specifically because this function
    // never sets evaluated = true — it only ever leaves it untouched or forces it to false.
    // So by the time a second parent visits, whatever the first parent already wrote to
    // `evaluated` is final for this pass, and !evaluated is guaranteed to match what a
    // recompute would have returned.
    //
    // The base-case `return false` (when !dirty && !inv) looks inconsistent with the memo
    // branch's `return !evaluated`, but it can't actually diverge in practice: if evaluated
    // were already false here, invariant (parent can't be evaluated=true if child is
    // evaluated=false) guarantees every ancestor above this node already has evaluated=false
    // too, so this function was never going to flip anything back to true regardless of what
    // it returns. A stale re-invalidation with no exec in between is therefore a true no-op,
    // not a correctness gap.
    bool invalidate_pass(uint64_t current_pass_id) override {
        if (this->last_visited_pass_id == current_pass_id) return !this->evaluated;
        this->last_visited_pass_id = current_pass_id;

        bool inv = false;
        uint64_t new_version = this->version;

        if (inv) {
            this->evaluated = false;
            this->version = new_version;
            return true;
        }
        return false;
    }
    void clear_trace_checks() override {
        evaluated = false;

    };
    matrix vmap(std::function<matrix(const matrix &)> func, std::vector<int> in_axis) override { return matrix(0, dtype::UInt8); }
    std::vector<matrix> get_inputs() override {
        return {};
    }
    
};

class ConvolvePrimitive : public Primitive {
public:
    matrix input;
    matrix kernel;
    std::vector<int> padding;
    std::vector<int> stride;
    std::vector<int> dilation;
    int groups;

    ConvolvePrimitive(const matrix& input, const matrix& kernel, const std::vector<int>& padding, const std::vector<int>& stride, const std::vector<int>& dilation, int groups)
        : input(ensure_graph_ready(input)), kernel(ensure_graph_ready(kernel)), padding(padding), stride(stride), dilation(dilation), groups(groups) {
    }

    void eval_cpu(matrix& out, EvalType eval_type) override {
        if (input.tape && !input.tape->evaluated) { input.tape->eval_cpu(input, eval_type); }
        if (kernel.tape && !kernel.tape->evaluated) { kernel.tape->eval_cpu(kernel, eval_type); }
        input.update_from_trace();
        kernel.update_from_trace();

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
                out.tape->out_refcount->fetch_add(1, std::memory_order_relaxed);
            }
        }
        if (eval_type == EvalType::COMPILE_TRACE) {
            return;
        }
        if (evaluated) {return;} else {evaluated = true;}

        // input.conv_cpu(kernel, out, eval_type);
    }

    void eval_metal(matrix& out, EvalType eval_type) override {
        if (input.tape && !input.tape->evaluated) { input.tape->eval_metal(input, eval_type); }
        if (kernel.tape && !kernel.tape->evaluated) { kernel.tape->eval_metal(kernel, eval_type); }
        input.update_from_trace();
        kernel.update_from_trace();

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
                out.tape->out_refcount->fetch_add(1, std::memory_order_relaxed);
            }
        }
        if (eval_type == EvalType::COMPILE_TRACE) {
            return;
        }
        if (evaluated) {return;} else {evaluated = true;}
        
        if (input.dims == 3) {
            input.conv1d_gpu(kernel, out);
        } else if (input.dims == 4) {
            input.conv2d_gpu(kernel, out);
        } else if (input.dims == 5) {
            input.conv3d_gpu(kernel, out);
        } else {
            input.conv_gpu(kernel, out);
        }
    }

    matrix jvp(std::vector<matrix>& tangents) override {
        throw std::runtime_error("JVP not implemented for ConvolvePrimitive");
    }

    std::vector<matrix> vjp(matrix& grad_out) override {
        throw std::runtime_error("VJP not implemented for ConvolvePrimitive");
    }

    matrix* get_borrowed_input() override {
        return nullptr;
    }

    // Converts dirty=true into evaluated=false along every path leading to a dirty node,
    // then resets dirty. This is needed because nodes only know their parents, not their
    // children, so the only way to propagate "this subtree needs re-eval" is to walk down
    // from parents and pull the signal up via return value.
    //
    // Memoization (last_visited_pass_id): a shared node (diamond dependency) gets visited
    // once per pass by its first parent, which does the real work and returns the result.
    // Every subsequent parent visiting it in the *same* pass hits the early-return and reads
    // !evaluated instead of recomputing. This is safe specifically because this function
    // never sets evaluated = true — it only ever leaves it untouched or forces it to false.
    // So by the time a second parent visits, whatever the first parent already wrote to
    // `evaluated` is final for this pass, and !evaluated is guaranteed to match what a
    // recompute would have returned.
    //
    // The base-case `return false` (when !dirty && !inv) looks inconsistent with the memo
    // branch's `return !evaluated`, but it can't actually diverge in practice: if evaluated
    // were already false here, invariant (parent can't be evaluated=true if child is
    // evaluated=false) guarantees every ancestor above this node already has evaluated=false
    // too, so this function was never going to flip anything back to true regardless of what
    // it returns. A stale re-invalidation with no exec in between is therefore a true no-op,
    // not a correctness gap.
// Converts dirty=true into evaluated=false along every path leading to a dirty node,
    // then resets dirty. This is needed because nodes only know their parents, not their
    // children, so the only way to propagate "this subtree needs re-eval" is to walk down
    // from parents and pull the signal up via return value.
    //
    // Memoization (last_visited_pass_id): a shared node (diamond dependency) gets visited
    // once per pass by its first parent, which does the real work and returns the result.
    // Every subsequent parent visiting it in the *same* pass hits the early-return and reads
    // !evaluated instead of recomputing. This is safe specifically because this function
    // never sets evaluated = true — it only ever leaves it untouched or forces it to false.
    // So by the time a second parent visits, whatever the first parent already wrote to
    // `evaluated` is final for this pass, and !evaluated is guaranteed to match what a
    // recompute would have returned.
    //
    // The base-case `return false` (when !dirty && !inv) looks inconsistent with the memo
    // branch's `return !evaluated`, but it can't actually diverge in practice: if evaluated
    // were already false here, invariant (parent can't be evaluated=true if child is
    // evaluated=false) guarantees every ancestor above this node already has evaluated=false
    // too, so this function was never going to flip anything back to true regardless of what
    // it returns. A stale re-invalidation with no exec in between is therefore a true no-op,
    // not a correctness gap.
    bool invalidate_pass(uint64_t current_pass_id) override {
        if (this->last_visited_pass_id == current_pass_id) return !this->evaluated;
        this->last_visited_pass_id = current_pass_id;

        bool inv = false;
        uint64_t new_version = this->version;
        if (input.tape) {
            if (input.tape->invalidate_pass(current_pass_id)) { inv = true; }
            if (input.tape->version > this->version) inv = true;
            new_version = std::max(input.tape->version, new_version);
        }
        if (kernel.tape) {
            if (kernel.tape->invalidate_pass(current_pass_id)) { inv = true; }
            if (kernel.tape->version > this->version) inv = true;
            new_version = std::max(kernel.tape->version, new_version);
        }

        if (inv) {
            this->evaluated = false;
            this->version = new_version;
            return true;
        }
        return false;
    }
    void clear_trace_checks() override {
        evaluated = false;
        if (input.tape && input.tape->evaluated) input.tape->clear_trace_checks();
        if (kernel.tape && kernel.tape->evaluated) kernel.tape->clear_trace_checks();
    }
    
    matrix vmap(std::function<matrix(const matrix &)> func, std::vector<int> in_axis) override { return matrix(0, dtype::UInt8); }

    std::vector<matrix> get_inputs() override {
        return {input, kernel};
    }
};

class AdditionPrimitive : public Primitive {
public:
    matrix a;
    matrix b;
    BroadcastDescriptor* desc_a = nullptr;
    BroadcastDescriptor* desc_b = nullptr;
    ~AdditionPrimitive() {
        if (desc_a) BroadcastDescriptor::destroy(desc_a);
        if (desc_b) BroadcastDescriptor::destroy(desc_b);
    }
    CollapsedDims_3 collapsed_dims_3;
    bool dims_collapsed = false;

    AdditionPrimitive(const matrix& a, const matrix& b) : a(ensure_graph_ready(a)), b(ensure_graph_ready(b)) {
    }
    AdditionPrimitive(const matrix& a, const matrix& b, CollapsedDims_3 collapsed_dims_3) : a(ensure_graph_ready(a)), b(ensure_graph_ready(b)) , collapsed_dims_3(collapsed_dims_3) {
        dims_collapsed = true;
    }
    void eval_cpu(matrix &out, EvalType eval_type) override {
        if (a.tape && !a.tape->evaluated) { a.tape->eval_cpu(a, eval_type); }
        if (b.tape && !b.tape->evaluated) { b.tape->eval_cpu(b, eval_type); }
        a.update_from_trace();
        b.update_from_trace();

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
                out.tape->out_refcount->fetch_add(1, std::memory_order_relaxed);
            }
        }
        if (eval_type == EvalType::COMPILE_TRACE) {
            return;
        }
        if (evaluated) {return;} else {evaluated = true;}

        a.add_cpu_brodcasted(b, out, eval_type);

    };
    void eval_metal(matrix &out, EvalType eval_type) override {
        if (a.tape && !a.tape->evaluated) { a.tape->eval_metal(a, eval_type); }
        if (b.tape && !b.tape->evaluated) { b.tape->eval_metal(b, eval_type); }
        a.update_from_trace();
        b.update_from_trace();
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
                out.tape->out_refcount->fetch_add(1, std::memory_order_relaxed);
            }
        }
        if (eval_type == EvalType::COMPILE_TRACE) {
            return;
        }
        if (evaluated) {return;} else {
            evaluated = true;
        }
        a.add_gpu_brodcasted(b, out, eval_type);
    }


    std::vector<matrix> vjp(matrix& grad_out) override {
        // a, b, c are vectors (a+b)=>c soo J = (dc/da, dc/db) and for addition its 1 so J = (1, 1)
        return {grad_out.unbroadcast_shape(a.shape(), a.dims), grad_out.unbroadcast_shape(b.shape(), b.dims)};
    }

    matrix jvp(std::vector<matrix>& tangents) override {
        // a, b, c are vectors (a+b)=>c soo J = (dc/da, dc/db) and for addition its 1 so J = (1, 1)
        // tangents are coming from the inputs so tangents are {da/.., db/...}
        // we need to find the total derivative of c so da/.. + db/...
        matrix total_derivative = tangents[0] + tangents[1];
        return total_derivative;
    }
// Converts dirty=true into evaluated=false along every path leading to a dirty node,
    // then resets dirty. This is needed because nodes only know their parents, not their
    // children, so the only way to propagate "this subtree needs re-eval" is to walk down
    // from parents and pull the signal up via return value.
    //
    // Memoization (last_visited_pass_id): a shared node (diamond dependency) gets visited
    // once per pass by its first parent, which does the real work and returns the result.
    // Every subsequent parent visiting it in the *same* pass hits the early-return and reads
    // !evaluated instead of recomputing. This is safe specifically because this function
    // never sets evaluated = true — it only ever leaves it untouched or forces it to false.
    // So by the time a second parent visits, whatever the first parent already wrote to
    // `evaluated` is final for this pass, and !evaluated is guaranteed to match what a
    // recompute would have returned.
    //
    // The base-case `return false` (when !dirty && !inv) looks inconsistent with the memo
    // branch's `return !evaluated`, but it can't actually diverge in practice: if evaluated
    // were already false here, invariant (parent can't be evaluated=true if child is
    // evaluated=false) guarantees every ancestor above this node already has evaluated=false
    // too, so this function was never going to flip anything back to true regardless of what
    // it returns. A stale re-invalidation with no exec in between is therefore a true no-op,
    // not a correctness gap.
    bool invalidate_pass(uint64_t current_pass_id) override {
        if (this->last_visited_pass_id == current_pass_id) return !this->evaluated;
        this->last_visited_pass_id = current_pass_id;

        bool inv = false;
        uint64_t new_version = this->version;
        if (a.tape) {
            if (a.tape->invalidate_pass(current_pass_id)) { inv = true; }
            if (a.tape->version > this->version) inv = true;
            new_version = std::max(a.tape->version, new_version);
        }
        if (b.tape) {
            if (b.tape->invalidate_pass(current_pass_id)) { inv = true; }
            if (b.tape->version > this->version) inv = true;
            new_version = std::max(b.tape->version, new_version);
        }

        if (inv) {
            this->evaluated = false;
            this->version = new_version;
            return true;
        }
        return false;
    }
    void clear_trace_checks() override {
        evaluated = false;
        if (a.tape) { a.tape->clear_trace_checks(); }
        if (b.tape) { b.tape->clear_trace_checks(); }
    };
    matrix vmap(std::function<matrix(const matrix &)> func, std::vector<int> in_axis) override { return matrix(0, dtype::UInt8); }
    std::vector<matrix> get_inputs() override {
        return {a, b};
    }
};

class SubtractionPrimitive : public Primitive {
    public:
    matrix a;
    matrix b;
    BroadcastDescriptor* desc_a = nullptr;
    BroadcastDescriptor* desc_b = nullptr;
    ~SubtractionPrimitive() {
        if (desc_a) BroadcastDescriptor::destroy(desc_a);
        if (desc_b) BroadcastDescriptor::destroy(desc_b);
    }
    CollapsedDims_3 collapsed_dims_3;
    bool dims_collapsed = false;

    SubtractionPrimitive(const matrix& a, const matrix& b) : a(ensure_graph_ready(a)), b(ensure_graph_ready(b)) {
    }
    void eval_cpu(matrix &out, EvalType eval_type) override {
        if (a.tape && !a.tape->evaluated) { a.tape->eval_cpu(a, eval_type); }
        if (b.tape && !b.tape->evaluated) { b.tape->eval_cpu(b, eval_type); }
        a.update_from_trace();
        b.update_from_trace();

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
                out.tape->out_refcount->fetch_add(1, std::memory_order_relaxed);
            }
        }
        if (eval_type == EvalType::COMPILE_TRACE) {
            return;
        }
        if (evaluated) {return;} else {evaluated = true;}
        a.subtract_cpu_brodcasted(b, out, eval_type);
    };
    void eval_metal(matrix &out, EvalType eval_type) override {
        if (a.tape && !a.tape->evaluated) { a.tape->eval_metal(a, eval_type); }
        if (b.tape && !b.tape->evaluated) { b.tape->eval_metal(b, eval_type); }
        a.update_from_trace();
        b.update_from_trace();
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
                out.tape->out_refcount->fetch_add(1, std::memory_order_relaxed);
            }
        }
        if (eval_type == EvalType::COMPILE_TRACE) {
            return;
        }
        if (evaluated) {return;} else {evaluated = true;}
        a.subtract_gpu_brodcasted(b, out, eval_type);
    }


    std::vector<matrix> vjp(matrix& grad_out) override {
        // a, b, c are vectors (a-b)=>c soo J = (dc/da, dc/db) and for subtraction its 1, -1 so J = (1, -1)
        return {grad_out.unbroadcast_shape(a.shape(), a.dims), grad_out.unbroadcast_shape(b.shape(), b.dims) * -1.0f};
    }

    matrix jvp(std::vector<matrix>& tangents) override {
        // a, b, c are vectors (a+b)=>c soo J = (dc/da, dc/db) and for subtraction its 1 so J = (1, 1)
        // tangents are coming from the inputs so tangents are {da/.., db/...}
        // we need to find the total derivative of c so da/.. - db/...
        matrix total_derivative = tangents[0] - tangents[1];
        return total_derivative;
    }
// Converts dirty=true into evaluated=false along every path leading to a dirty node,
    // then resets dirty. This is needed because nodes only know their parents, not their
    // children, so the only way to propagate "this subtree needs re-eval" is to walk down
    // from parents and pull the signal up via return value.
    //
    // Memoization (last_visited_pass_id): a shared node (diamond dependency) gets visited
    // once per pass by its first parent, which does the real work and returns the result.
    // Every subsequent parent visiting it in the *same* pass hits the early-return and reads
    // !evaluated instead of recomputing. This is safe specifically because this function
    // never sets evaluated = true — it only ever leaves it untouched or forces it to false.
    // So by the time a second parent visits, whatever the first parent already wrote to
    // `evaluated` is final for this pass, and !evaluated is guaranteed to match what a
    // recompute would have returned.
    //
    // The base-case `return false` (when !dirty && !inv) looks inconsistent with the memo
    // branch's `return !evaluated`, but it can't actually diverge in practice: if evaluated
    // were already false here, invariant (parent can't be evaluated=true if child is
    // evaluated=false) guarantees every ancestor above this node already has evaluated=false
    // too, so this function was never going to flip anything back to true regardless of what
    // it returns. A stale re-invalidation with no exec in between is therefore a true no-op,
    // not a correctness gap.
    bool invalidate_pass(uint64_t current_pass_id) override {
        if (this->last_visited_pass_id == current_pass_id) return !this->evaluated;
        this->last_visited_pass_id = current_pass_id;

        bool inv = false;
        uint64_t new_version = this->version;
        if (a.tape) {
            if (a.tape->invalidate_pass(current_pass_id)) { inv = true; }
            if (a.tape->version > this->version) inv = true;
            new_version = std::max(a.tape->version, new_version);
        }
        if (b.tape) {
            if (b.tape->invalidate_pass(current_pass_id)) { inv = true; }
            if (b.tape->version > this->version) inv = true;
            new_version = std::max(b.tape->version, new_version);
        }

        if (inv) {
            this->evaluated = false;
            this->version = new_version;
            return true;
        }
        return false;
    }
    void clear_trace_checks() override {
        evaluated = false;
        if (a.tape) { a.tape->clear_trace_checks(); }
        if (b.tape) { b.tape->clear_trace_checks(); }
    };
    matrix vmap(std::function<matrix(const matrix &)> func, std::vector<int> in_axis) override { return matrix(0, dtype::UInt8); }
    std::vector<matrix> get_inputs() override {
        return {a, b};
    }
};

class MultiplicationPrimitive : public Primitive {
    public:
    matrix a;
    matrix b;
    BroadcastDescriptor* desc_a = nullptr;
    BroadcastDescriptor* desc_b = nullptr;
    ~MultiplicationPrimitive() {
        if (desc_a) BroadcastDescriptor::destroy(desc_a);
        if (desc_b) BroadcastDescriptor::destroy(desc_b);
    }
    CollapsedDims_3 collapsed_dims_3;
    bool dims_collapsed = false;
    MultiplicationPrimitive(const matrix& a, const matrix& b) : a(ensure_graph_ready(a)), b(ensure_graph_ready(b)) {
    }
    void eval_cpu(matrix &out, EvalType eval_type) override {
        if (a.tape && !a.tape->evaluated) { a.tape->eval_cpu(a, eval_type); }
        if (b.tape && !b.tape->evaluated) { b.tape->eval_cpu(b, eval_type); }
        a.update_from_trace();
        b.update_from_trace();
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
                out.tape->out_refcount->fetch_add(1, std::memory_order_relaxed);
            }
        }
        if (eval_type == EvalType::COMPILE_TRACE) {
            return;
        }
        if (evaluated) {return;} else {evaluated = true;}
        a.multiply_cpu_brodcasted(b, out, eval_type);
    };
    void eval_metal(matrix &out, EvalType eval_type) override {
        if (a.tape && !a.tape->evaluated) { a.tape->eval_metal(a, eval_type); }
        if (b.tape && !b.tape->evaluated) { b.tape->eval_metal(b, eval_type); }
        a.update_from_trace();
        b.update_from_trace();
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
                out.tape->out_refcount->fetch_add(1, std::memory_order_relaxed);
            }
        }
        if (eval_type == EvalType::COMPILE_TRACE) {
            return;
        }

        if (evaluated) {return;} else {
            evaluated = true;
        }

        a.multiply_gpu_brodcasted(b, out, eval_type);
    }


    std::vector<matrix> vjp(matrix& grad_out) override {
        // a, b, c are vectors (a*b)=>c soo J = (b * dc/da, a * dc/db)
        matrix grad_a_full = grad_out * b;
        matrix grad_b_full = grad_out * a;

        return {grad_a_full.unbroadcast_shape(a.shape(), a.dims), grad_b_full.unbroadcast_shape(b.shape(), b.dims)};
    }

    matrix jvp(std::vector<matrix>& tangents) override {
        // a, b, c are vectors (a*b)=>c soo J = (b * dc/da, a * dc/db)
        // tangents are coming from the inputs so tangents are {da/.., db/...}
        // we need to find the total derivative of c so b*  da/.. + a * db/...
        matrix total_derivative = b * tangents[0] + a * tangents[1];
        return total_derivative;
    }
// Converts dirty=true into evaluated=false along every path leading to a dirty node,
    // then resets dirty. This is needed because nodes only know their parents, not their
    // children, so the only way to propagate "this subtree needs re-eval" is to walk down
    // from parents and pull the signal up via return value.
    //
    // Memoization (last_visited_pass_id): a shared node (diamond dependency) gets visited
    // once per pass by its first parent, which does the real work and returns the result.
    // Every subsequent parent visiting it in the *same* pass hits the early-return and reads
    // !evaluated instead of recomputing. This is safe specifically because this function
    // never sets evaluated = true — it only ever leaves it untouched or forces it to false.
    // So by the time a second parent visits, whatever the first parent already wrote to
    // `evaluated` is final for this pass, and !evaluated is guaranteed to match what a
    // recompute would have returned.
    //
    // The base-case `return false` (when !dirty && !inv) looks inconsistent with the memo
    // branch's `return !evaluated`, but it can't actually diverge in practice: if evaluated
    // were already false here, invariant (parent can't be evaluated=true if child is
    // evaluated=false) guarantees every ancestor above this node already has evaluated=false
    // too, so this function was never going to flip anything back to true regardless of what
    // it returns. A stale re-invalidation with no exec in between is therefore a true no-op,
    // not a correctness gap.
    bool invalidate_pass(uint64_t current_pass_id) override {
        if (this->last_visited_pass_id == current_pass_id) return !this->evaluated;
        this->last_visited_pass_id = current_pass_id;

        bool inv = false;
        uint64_t new_version = this->version;
        if (a.tape) {
            if (a.tape->invalidate_pass(current_pass_id)) { inv = true; }
            if (a.tape->version > this->version) inv = true;
            new_version = std::max(a.tape->version, new_version);
        }
        if (b.tape) {
            if (b.tape->invalidate_pass(current_pass_id)) { inv = true; }
            if (b.tape->version > this->version) inv = true;
            new_version = std::max(b.tape->version, new_version);
        }

        if (inv) {
            this->evaluated = false;
            this->version = new_version;
            return true;
        }
        return false;
    }
    void clear_trace_checks() override {
        evaluated = false;
        if (a.tape) { a.tape->clear_trace_checks(); }
        if (b.tape) { b.tape->clear_trace_checks(); }
    };
    matrix vmap(std::function<matrix(const matrix &)> func, std::vector<int> in_axis) override { return matrix(0, dtype::UInt8); }
    std::vector<matrix> get_inputs() override {
        return {a, b};
    }
};

class DivisionPrimitive : public Primitive {
    public:
    matrix a;
    matrix b;
    BroadcastDescriptor* desc_a = nullptr;
    BroadcastDescriptor* desc_b = nullptr;
    ~DivisionPrimitive() {
        if (desc_a) BroadcastDescriptor::destroy(desc_a);
        if (desc_b) BroadcastDescriptor::destroy(desc_b);
    }
    CollapsedDims_3 collapsed_dims_3;
    bool dims_collapsed = false;

    DivisionPrimitive(const matrix& a, const matrix& b) : a(ensure_graph_ready(a)), b(ensure_graph_ready(b)) {
    }
    void eval_cpu(matrix &out, EvalType eval_type) override {
        if (a.tape && !a.tape->evaluated) { a.tape->eval_cpu(a, eval_type); }
        if (b.tape && !b.tape->evaluated) { b.tape->eval_cpu(b, eval_type); }
        a.update_from_trace();
        b.update_from_trace();

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
                out.tape->out_refcount->fetch_add(1, std::memory_order_relaxed);
            }
        }
        if (eval_type == EvalType::COMPILE_TRACE) {
            return;
        }
        if (evaluated) {return;} else {evaluated = true;}
        a.divide_cpu_brodcasted(b, out, eval_type);
    };
    void eval_metal(matrix &out, EvalType eval_type) override {
        if (a.tape && !a.tape->evaluated) { a.tape->eval_metal(a, eval_type); }
        if (b.tape && !b.tape->evaluated) { b.tape->eval_metal(b, eval_type); }
        a.update_from_trace();
        b.update_from_trace();
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
                out.tape->out_refcount->fetch_add(1, std::memory_order_relaxed);
            }
        }
        if (eval_type == EvalType::COMPILE_TRACE) {
            return;
        }
        if (evaluated) {return;} else {evaluated = true;}
        a.divide_gpu_brodcasted(b, out, eval_type);
    }


    std::vector<matrix> vjp(matrix& grad_out) override {
        // a, b, c are vectors (a/b)=>c soo J = (dc/da * 1/b, dc/db * a * 1/b^2)
        matrix grad_a_full = grad_out / b;
        matrix grad_b_full = grad_out * matrix::scalar(-1, a.type) * a / (b * b);

        return {grad_a_full.unbroadcast_shape(a.shape(), a.dims), grad_b_full.unbroadcast_shape(b.shape(), b.dims)};
    }

    matrix jvp(std::vector<matrix>& tangents) override {
        // a, b, c are vectors (a/b)=>c soo J = (b * dc/da - a * dc/db)/b^2
        // tangents are coming from the inputs so tangents are {da/.., db/...}
        // we need to find the total derivative of c so (b * dc/da - a * dc/db)/b^2
        matrix total_derivative = (b * tangents[0] - a * tangents[1]) / (b * b);
        return total_derivative;
    }
// Converts dirty=true into evaluated=false along every path leading to a dirty node,
    // then resets dirty. This is needed because nodes only know their parents, not their
    // children, so the only way to propagate "this subtree needs re-eval" is to walk down
    // from parents and pull the signal up via return value.
    //
    // Memoization (last_visited_pass_id): a shared node (diamond dependency) gets visited
    // once per pass by its first parent, which does the real work and returns the result.
    // Every subsequent parent visiting it in the *same* pass hits the early-return and reads
    // !evaluated instead of recomputing. This is safe specifically because this function
    // never sets evaluated = true — it only ever leaves it untouched or forces it to false.
    // So by the time a second parent visits, whatever the first parent already wrote to
    // `evaluated` is final for this pass, and !evaluated is guaranteed to match what a
    // recompute would have returned.
    //
    // The base-case `return false` (when !dirty && !inv) looks inconsistent with the memo
    // branch's `return !evaluated`, but it can't actually diverge in practice: if evaluated
    // were already false here, invariant (parent can't be evaluated=true if child is
    // evaluated=false) guarantees every ancestor above this node already has evaluated=false
    // too, so this function was never going to flip anything back to true regardless of what
    // it returns. A stale re-invalidation with no exec in between is therefore a true no-op,
    // not a correctness gap.
    bool invalidate_pass(uint64_t current_pass_id) override {
        if (this->last_visited_pass_id == current_pass_id) return !this->evaluated;
        this->last_visited_pass_id = current_pass_id;

        bool inv = false;
        uint64_t new_version = this->version;
        if (a.tape) {
            if (a.tape->invalidate_pass(current_pass_id)) { inv = true; }
            if (a.tape->version > this->version) inv = true;
            new_version = std::max(a.tape->version, new_version);
        }
        if (b.tape) {
            if (b.tape->invalidate_pass(current_pass_id)) { inv = true; }
            if (b.tape->version > this->version) inv = true;
            new_version = std::max(b.tape->version, new_version);
        }

        if (inv) {
            this->evaluated = false;
            this->version = new_version;
            return true;
        }
        return false;
    }
    void clear_trace_checks() override {
        evaluated = false;
        if (a.tape) { a.tape->clear_trace_checks(); }
        if (b.tape) { b.tape->clear_trace_checks(); }
    };
    matrix vmap(std::function<matrix(const matrix &)> func, std::vector<int> in_axis) override { return matrix(0, dtype::UInt8); }
    std::vector<matrix> get_inputs() override {
        return {a, b};
    }
};

//using size_m = uint32;
//constexpr int SBO_MAX_DIMS = 3;
//union array_descriptor {
//    size_m inline_buffer[SBO_MAX_DIMS * 2]; // For dims <= 3
//    SharedArrayDescriptor* shared_arr_desc;  // For dims > 3
//};

class StackPrimitive : public Primitive {
public:
    std::vector<matrix> inputs;
    int axis;
    StackPrimitive(const std::vector<matrix>& stack, const int axis) : inputs(ensure_graph_ready(stack)), axis(axis) {
        for (matrix& a : inputs) {
        }
    }
    void eval_cpu(matrix &out, EvalType eval_type) override {
        for (size_t i = 0; i < inputs.size(); i++) {
            if (inputs[i].tape && !inputs[i].tape->evaluated) { inputs[i].tape->eval_cpu(inputs[i], eval_type); }
            inputs[i].update_from_trace();
        }

        if (!out.buffer) {
            if (out.tape->out_buffer) {
                out.buffer = out.tape->out_buffer;
                out.metalBuffer = out.tape->out_metal_buffer;
                out.refCount = out.tape->out_refcount;
                out.refCount->fetch_add(1);
                if (eval_type == EvalType::COMPILE_TRACE) {return;}
            } else {
                out.buffer = new uint8_t[out.effectiveBufferSize() * dtype_size(out.type)];
                out.begin_refcount();
                out.buildMetalBuffer();

                out.tape->out_buffer = (uint8_t*)out.buffer;
                out.tape->out_metal_buffer = out.metalBuffer;
                out.tape->out_refcount = out.refCount;
                out.tape->out_refcount->fetch_add(1, std::memory_order_relaxed);
            }
        }
        if (eval_type == EvalType::COMPILE_TRACE) {
            return;
        }
        if (evaluated) {return;} else {evaluated = true;}
        matrix::stack(inputs, out, axis, ExecutionDevice::CPU);
    };
    void eval_metal(matrix &out, EvalType eval_type) override {
        for (size_t i = 0; i < inputs.size(); i++) {
            if (inputs[i].tape && !inputs[i].tape->evaluated) { inputs[i].tape->eval_metal(inputs[i], eval_type); }
            inputs[i].update_from_trace();
        }
        if (!out.buffer) {
            if (out.tape->out_buffer) {
                out.buffer = out.tape->out_buffer;
                out.metalBuffer = out.tape->out_metal_buffer;
                out.refCount = out.tape->out_refcount;
                out.refCount->fetch_add(1);
                if (eval_type == EvalType::COMPILE_TRACE) {return;}
            } else {
                out.buffer = new uint8_t[out.effectiveBufferSize() * dtype_size(out.type)];
                out.begin_refcount();
                out.buildMetalBuffer();

                out.tape->out_buffer = (uint8_t*)out.buffer;
                out.tape->out_metal_buffer = out.metalBuffer;
                out.tape->out_refcount = out.refCount;
                out.tape->out_refcount->fetch_add(1, std::memory_order_relaxed);
            }
        }
        if (eval_type == EvalType::COMPILE_TRACE) {
            return;
        }
        if (evaluated) {return;} else {evaluated = true;}
        matrix::stack(inputs, out, axis, ExecutionDevice::METAL);
    }


    std::vector<matrix> vjp(matrix& grad_out) override {
        // a, b, c are vectors (a,b)=>c soo J = (dc/da, dc/db) and for addition its 1 so J = (1, 1)
        std::vector<matrix> grad_vec;
        grad_vec.reserve(inputs.size());
        for (int i = 0; i < inputs.size(); i++) {
            grad_vec.push_back(grad_out.slice(i, axis));
        }
        return grad_vec;
    }
    matrix jvp(std::vector<matrix>& tangents) override {
        // a, b, c are vectors (a,b)=>c soo J = (dc/da, dc/db) and for stack we just need to stack the derivative
        // tangents are coming from the inputs so tangents are {da/.., db/...}
        // we need to find the total derivative of c so (da/.. , db/...)
        matrix total_derivative(tangents[0].dims + 1, tangents[0].type);
        matrix::stack(tangents,total_derivative, axis);
        return total_derivative;
    }
// Converts dirty=true into evaluated=false along every path leading to a dirty node,
    // then resets dirty. This is needed because nodes only know their parents, not their
    // children, so the only way to propagate "this subtree needs re-eval" is to walk down
    // from parents and pull the signal up via return value.
    //
    // Memoization (last_visited_pass_id): a shared node (diamond dependency) gets visited
    // once per pass by its first parent, which does the real work and returns the result.
    // Every subsequent parent visiting it in the *same* pass hits the early-return and reads
    // !evaluated instead of recomputing. This is safe specifically because this function
    // never sets evaluated = true — it only ever leaves it untouched or forces it to false.
    // So by the time a second parent visits, whatever the first parent already wrote to
    // `evaluated` is final for this pass, and !evaluated is guaranteed to match what a
    // recompute would have returned.
    //
    // The base-case `return false` (when !dirty && !inv) looks inconsistent with the memo
    // branch's `return !evaluated`, but it can't actually diverge in practice: if evaluated
    // were already false here, invariant (parent can't be evaluated=true if child is
    // evaluated=false) guarantees every ancestor above this node already has evaluated=false
    // too, so this function was never going to flip anything back to true regardless of what
    // it returns. A stale re-invalidation with no exec in between is therefore a true no-op,
    // not a correctness gap.
    bool invalidate_pass(uint64_t current_pass_id) override {
        if (this->last_visited_pass_id == current_pass_id) return !this->evaluated;
        this->last_visited_pass_id = current_pass_id;

        bool inv = false;
        uint64_t new_version = this->version;
        for (size_t i = 0; i < inputs.size(); i++) {
            if (inputs[i].tape) {
                if (inputs[i].tape->invalidate_pass(current_pass_id)) { inv = true; }
                if (inputs[i].tape->version > this->version) inv = true;
                new_version = std::max(inputs[i].tape->version, new_version);
            }
        }

        if (inv) {
            this->evaluated = false;
            this->version = new_version;
            return true;
        }
        return false;
    }
    void clear_trace_checks() override {
        evaluated = false;
        for (size_t i = 0; i < inputs.size(); i++) {
            if (inputs[i].tape) {inputs[i].tape->clear_trace_checks(); }
        }
    };
    matrix vmap(std::function<matrix(const matrix &)> func, std::vector<int> in_axis) override { return matrix(0, dtype::UInt8); }
    std::vector<matrix> get_inputs() override {
        return inputs;
    }
};

class ConcatPrimitive : public Primitive {
public:
    std::vector<matrix> inputs;
    int axis;
    ConcatPrimitive(const std::vector<matrix>& stack, const int axis) : inputs(ensure_graph_ready(stack)), axis(axis) {
        for (matrix& a : inputs) {
        }
    }
    void eval_cpu(matrix &out, EvalType eval_type) override {
        for (size_t i = 0; i < inputs.size(); i++) {
            if (inputs[i].tape && !inputs[i].tape->evaluated) { inputs[i].tape->eval_cpu(inputs[i], eval_type); }
            inputs[i].update_from_trace();
        }

        if (!out.buffer) {
            if (out.tape->out_buffer) {
                out.buffer = out.tape->out_buffer;
                out.metalBuffer = out.tape->out_metal_buffer;
                out.refCount = out.tape->out_refcount;
                out.refCount->fetch_add(1);
                if (eval_type == EvalType::COMPILE_TRACE) {return;}
            } else {
                out.buffer = new uint8_t[out.effectiveBufferSize() * dtype_size(out.type)];
                out.begin_refcount();
                out.buildMetalBuffer();

                out.tape->out_buffer = (uint8_t*)out.buffer;
                out.tape->out_metal_buffer = out.metalBuffer;
                out.tape->out_refcount = out.refCount;
                out.tape->out_refcount->fetch_add(1, std::memory_order_relaxed);
            }
        }
        if (eval_type == EvalType::COMPILE_TRACE) {
            return;
        }
        if (evaluated) {return;} else {evaluated = true;}
        matrix::concat(inputs, out, axis, ExecutionDevice::CPU);
    };
    void eval_metal(matrix &out, EvalType eval_type) override {
        for (size_t i = 0; i < inputs.size(); i++) {
            if (inputs[i].tape && !inputs[i].tape->evaluated) { inputs[i].tape->eval_metal(inputs[i], eval_type); }
            inputs[i].update_from_trace();
        }
        if (!out.buffer) {
            if (out.tape->out_buffer) {
                out.buffer = out.tape->out_buffer;
                out.metalBuffer = out.tape->out_metal_buffer;
                out.refCount = out.tape->out_refcount;
                out.refCount->fetch_add(1);
                if (eval_type == EvalType::COMPILE_TRACE) {return;}
            } else {
                out.buffer = new uint8_t[out.effectiveBufferSize() * dtype_size(out.type)];
                out.begin_refcount();
                out.buildMetalBuffer();

                out.tape->out_buffer = (uint8_t*)out.buffer;
                out.tape->out_metal_buffer = out.metalBuffer;
                out.tape->out_refcount = out.refCount;
                out.tape->out_refcount->fetch_add(1, std::memory_order_relaxed);
            }
        }
        if (eval_type == EvalType::COMPILE_TRACE) {
            return;
        }
        if (evaluated) {return;} else {evaluated = true;}
        matrix::concat(inputs, out, axis, ExecutionDevice::METAL);
    }


    std::vector<matrix> vjp(matrix& grad_out) override {
        // a, b, c are vectors (a,b)=>c soo J = (dc/da, dc/db) and for addition its 1 so J = (1, 1)
        std::vector<matrix> grad_vec;
        grad_vec.reserve(inputs.size());
        size_m index = 0;
        for (int i = 0; i < inputs.size(); i++) {
            grad_vec.push_back(grad_out.slice(R(index, index + inputs[i].shape()[axis]), axis));
            index += inputs[i].shape()[axis];
        }
        return grad_vec;
    }
    matrix jvp(std::vector<matrix>& tangents) override {
        // a, b, c are vectors (a,b)=>c soo J = (dc/da, dc/db) and for stack we just need to stack the derivative
        // tangents are coming from the inputs so tangents are {da/.., db/...}
        // we need to find the total derivative of c so (da/.. , db/...)
        return matrix::concat(tangents, axis);;
    }
// Converts dirty=true into evaluated=false along every path leading to a dirty node,
    // then resets dirty. This is needed because nodes only know their parents, not their
    // children, so the only way to propagate "this subtree needs re-eval" is to walk down
    // from parents and pull the signal up via return value.
    //
    // Memoization (last_visited_pass_id): a shared node (diamond dependency) gets visited
    // once per pass by its first parent, which does the real work and returns the result.
    // Every subsequent parent visiting it in the *same* pass hits the early-return and reads
    // !evaluated instead of recomputing. This is safe specifically because this function
    // never sets evaluated = true — it only ever leaves it untouched or forces it to false.
    // So by the time a second parent visits, whatever the first parent already wrote to
    // `evaluated` is final for this pass, and !evaluated is guaranteed to match what a
    // recompute would have returned.
    //
    // The base-case `return false` (when !dirty && !inv) looks inconsistent with the memo
    // branch's `return !evaluated`, but it can't actually diverge in practice: if evaluated
    // were already false here, invariant (parent can't be evaluated=true if child is
    // evaluated=false) guarantees every ancestor above this node already has evaluated=false
    // too, so this function was never going to flip anything back to true regardless of what
    // it returns. A stale re-invalidation with no exec in between is therefore a true no-op,
    // not a correctness gap.
    bool invalidate_pass(uint64_t current_pass_id) override {
        if (this->last_visited_pass_id == current_pass_id) return !this->evaluated;
        this->last_visited_pass_id = current_pass_id;

        bool inv = false;
        uint64_t new_version = this->version;
        for (size_t i = 0; i < inputs.size(); i++) {
            if (inputs[i].tape) {
                if (inputs[i].tape->invalidate_pass(current_pass_id)) { inv = true; }
                if (inputs[i].tape->version > this->version) inv = true;
                new_version = std::max(inputs[i].tape->version, new_version);
            }
        }

        if (inv) {
            this->evaluated = false;
            this->version = new_version;
            return true;
        }
        return false;
    }
    void clear_trace_checks() override {
        evaluated = false;
        for (size_t i = 0; i < inputs.size(); i++) {
            if (inputs[i].tape) {inputs[i].tape->clear_trace_checks(); }
        }
    };
    matrix vmap(std::function<matrix(const matrix &)> func, std::vector<int> in_axis) override { return matrix(0, dtype::UInt8); }
    std::vector<matrix> get_inputs() override {
        return inputs;
    }
};


class SlicePrimitive : public Primitive {
public:
    matrix* get_borrowed_input() override { return &input; }
    matrix input;
    array_descriptor slice_desc;
    std::vector<size_m> slice_indices;
    std::vector<size_m> unsqeeze_mask;
    size_t slice_size;
    size_t offset;
    SlicePrimitive(const matrix& input_mat, array_descriptor& input_slice_desc, const std::vector<size_m>& input_slice_indices, const std::vector<size_m>& input_unsqueeze_mask, size_t input_slice_size, size_t input_offset ):
    input(ensure_graph_ready(input_mat)),
    slice_desc(input_slice_desc),
    slice_indices(input_slice_indices),
    unsqeeze_mask(input_unsqueeze_mask),
    slice_size(input_slice_size),
    offset(input_offset) {}
    
    ~SlicePrimitive() override {
        if (out_refcount) {
            out_refcount->fetch_sub(1, std::memory_order_relaxed);
            out_refcount = nullptr; // Let input.~matrix() do the final safe deletion
        }
    }
    
    void eval_cpu(matrix &out, EvalType eval_type) override {
        if (input.tape && !input.tape->evaluated) { input.tape->eval_cpu(input, eval_type); };
        input.update_from_trace();
        if (!out.buffer) {
            if (out.tape->out_buffer) {
                out.buffer = out.tape->out_buffer;
                out.metalBuffer = out.tape->out_metal_buffer;
                out.refCount = out.tape->out_refcount;
                out.refCount->fetch_add(1);
                if (eval_type == EvalType::COMPILE_TRACE) {return;}
            } else if (input.buffer) {
                out.buffer = (uint8_t*)input.buffer + offset * dtype_size(out.type);
                // out.metalBuffer = input.metalBuffer;
                out.buildMetalBuffer();
                out.refCount = input.refCount;
                out.refCount->fetch_add(1);

                out.tape->out_buffer = (uint8_t*)out.buffer;
                out.tape->out_metal_buffer = out.metalBuffer;
                out.tape->out_refcount = out.refCount;
                out.tape->out_refcount->fetch_add(1, std::memory_order_relaxed);
            }
        }
        if (out.buffer != (uint8_t*)input.buffer) {
            out.releaseBuffer();
            out.buffer = (uint8_t*)input.buffer + offset * dtype_size(out.type);
            out.metalBuffer = input.metalBuffer;
            out.refCount = input.refCount;
            out.refCount->fetch_add(1);

            out.tape->out_buffer = (uint8_t*)out.buffer;
            out.tape->out_metal_buffer = out.metalBuffer;
            out.tape->out_refcount = out.refCount;
            out.tape->out_refcount->fetch_add(1, std::memory_order_relaxed);
        }
        if (eval_type == EvalType::COMPILE_TRACE) {
            return;
        }
        if (evaluated) {return;} else {evaluated = true;}
    };
    void eval_metal(matrix &out, EvalType eval_type) override {
        if (input.tape && !input.tape->evaluated) { input.tape->eval_metal(input, eval_type); };
        input.update_from_trace();
        if (!out.buffer) {
            if (out.tape->out_buffer) {
                out.buffer = out.tape->out_buffer;
                out.metalBuffer = out.tape->out_metal_buffer;
                out.refCount = out.tape->out_refcount;
                out.refCount->fetch_add(1);
                if (eval_type == EvalType::COMPILE_TRACE) {return;}
            } else if (input.buffer) {
                out.buffer = (uint8_t*)input.buffer + offset * dtype_size(out.type);
                out.metalBuffer = input.metalBuffer;
                out.refCount = input.refCount;
                out.refCount->fetch_add(1);

                out.tape->out_buffer = (uint8_t*)out.buffer;
                out.tape->out_metal_buffer = out.metalBuffer;
                out.tape->out_refcount = out.refCount;
                out.tape->out_refcount->fetch_add(1, std::memory_order_relaxed);
            }
        }
        if (out.buffer != (uint8_t*)input.buffer) {
            out.releaseBuffer();
            out.buffer = (uint8_t*)input.buffer + offset * dtype_size(out.type);
            out.metalBuffer = input.metalBuffer; // BUG HERE SLICE BUFFER SHOULD BE AT AN OFFSET
            out.refCount = input.refCount;
            out.refCount->fetch_add(1);

            out.tape->out_buffer = (uint8_t*)out.buffer;
            out.tape->out_metal_buffer = out.metalBuffer;
            out.tape->out_refcount = out.refCount;
            out.tape->out_refcount->fetch_add(1, std::memory_order_relaxed);
        }
        if (eval_type == EvalType::COMPILE_TRACE) {
            return;
        }
        if (evaluated) {return;} else {evaluated = true;}
    }


    std::vector<matrix> vjp(matrix& grad_out) override {
        // a is vectors a[.., .., ...]=>c soo J = (dc/da, dc/db) = Identity padded with zeros for the orignal matrix values that were'nt in the slice
        // offset = ind[0] * strides[0] + ind[1] * strides[1] ... + ind[n-1] * strides[n-1];
        matrix padded_grad_out(input.dims, input.type);
        grad_out.unsqueeze(unsqeeze_mask.data(), unsqeeze_mask.size()).pad(slice_indices, padded_grad_out, {0});
        return {padded_grad_out};
    }
    matrix jvp(std::vector<matrix>& tangents) override {
        // a are vectors a[.., .., ...]=>c soo J = (dc/da, dc/db)=>J=(1, 1) and only return the slice of the incoming derivative
        return tangents[0].slice(slice_desc, slice_indices, unsqeeze_mask, offset);
    }
// Converts dirty=true into evaluated=false along every path leading to a dirty node,
    // then resets dirty. This is needed because nodes only know their parents, not their
    // children, so the only way to propagate "this subtree needs re-eval" is to walk down
    // from parents and pull the signal up via return value.
    //
    // Memoization (last_visited_pass_id): a shared node (diamond dependency) gets visited
    // once per pass by its first parent, which does the real work and returns the result.
    // Every subsequent parent visiting it in the *same* pass hits the early-return and reads
    // !evaluated instead of recomputing. This is safe specifically because this function
    // never sets evaluated = true — it only ever leaves it untouched or forces it to false.
    // So by the time a second parent visits, whatever the first parent already wrote to
    // `evaluated` is final for this pass, and !evaluated is guaranteed to match what a
    // recompute would have returned.
    //
    // The base-case `return false` (when !dirty && !inv) looks inconsistent with the memo
    // branch's `return !evaluated`, but it can't actually diverge in practice: if evaluated
    // were already false here, invariant (parent can't be evaluated=true if child is
    // evaluated=false) guarantees every ancestor above this node already has evaluated=false
    // too, so this function was never going to flip anything back to true regardless of what
    // it returns. A stale re-invalidation with no exec in between is therefore a true no-op,
    // not a correctness gap.
    bool invalidate_pass(uint64_t current_pass_id) override {
        if (this->last_visited_pass_id == current_pass_id) return !this->evaluated;
        this->last_visited_pass_id = current_pass_id;

        bool inv = false;
        uint64_t new_version = this->version;
        if (input.tape) {
            if (input.tape->invalidate_pass(current_pass_id)) { inv = true; }
            if (input.tape->version > this->version) inv = true;
            new_version = std::max(input.tape->version, new_version);
        }

        if (inv) {
            this->evaluated = false;
            this->version = new_version;
            return true;
        }
        return false;
    }
    void clear_trace_checks() override {
        evaluated = false;
        if (input.tape) {input.tape->clear_trace_checks(); }
        
    };
    matrix vmap(std::function<matrix(const matrix &)> func, std::vector<int> in_axis) override { return matrix(0, dtype::UInt8); }
    std::vector<matrix> get_inputs() override {
        return {input};
    }
};

class SliceAssignPrimitive : public Primitive {
public:
    matrix input;
    matrix rhs;
    array_descriptor slice_desc;
    std::vector<size_m> slice_indices;
    size_t slice_size;
    size_t offset;

    SliceAssignPrimitive(matrix& input_mat, const matrix& rhs_mat, array_descriptor& input_slice_desc, std::vector<size_m>& input_slice_indices, size_t input_slice_size, size_t input_offset )
        : input(ensure_graph_ready(input_mat)), rhs(ensure_graph_ready(rhs_mat)), slice_desc(input_slice_desc), slice_indices(input_slice_indices), slice_size(input_slice_size), offset(input_offset) {
        if (input.buffer) input.begin_refcount();
        if (rhs.buffer) rhs.begin_refcount();
    }

    void eval_cpu(matrix &out, EvalType eval_type) override {
        if (input.tape && !input.tape->evaluated) { input.tape->eval_cpu(input, eval_type); }
        if (rhs.tape && !rhs.tape->evaluated) { rhs.tape->eval_cpu(rhs, eval_type); }
        input.update_from_trace();
        rhs.update_from_trace();

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
                out.tape->out_refcount->fetch_add(1, std::memory_order_relaxed);
            }
        }
        if (eval_type == EvalType::COMPILE_TRACE) return;
        if (evaluated) return; else evaluated = true;

        matrix::copyCPUinplace(out, input, 0);
        
        matrix out_sliced = matrix(out.dims, out.type);
        out_sliced.set_array_desc(slice_desc);
        out_sliced.total_size = out_sliced.accumul(0, out_sliced.dims);
        out_sliced.flags |= NON_OWNERSHIP_FLAG | NON_CONTIGUOUS_FLAG;
        out_sliced.buffer = out.buffer;
        
        matrix::copyCPUinplace(out_sliced, rhs, offset);
    }

    void eval_metal(matrix &out, EvalType eval_type) override {
        if (input.tape && !input.tape->evaluated) { input.tape->eval_metal(input, eval_type); }
        if (rhs.tape && !rhs.tape->evaluated) { rhs.tape->eval_metal(rhs, eval_type); }
        input.update_from_trace();
        rhs.update_from_trace();

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
                out.tape->out_refcount->fetch_add(1, std::memory_order_relaxed);
            }
        }
        if (eval_type == EvalType::COMPILE_TRACE) return;
        if (evaluated) return; else evaluated = true;

        matrix::copyGPUinplace(out, input, 0, Execution::Encode);
        
        matrix out_sliced = matrix(out.dims, out.type);
        out_sliced.set_array_desc(slice_desc);
        out_sliced.total_size = out_sliced.accumul(0, out_sliced.dims);
        out_sliced.flags |= NON_OWNERSHIP_FLAG | NON_CONTIGUOUS_FLAG;
        out_sliced.buffer = out.buffer;
        out_sliced.metalBuffer = out.metalBuffer;
        
        matrix::copyGPUinplace(out_sliced, rhs, offset, Execution::Encode);
    }

    std::vector<matrix> vjp(matrix& grad_out) override {
        matrix dc_dlhs = grad_out.slice_assign(slice_desc, slice_indices, offset, matrix::scalar(0.0f, input.type));
        matrix dc_drhs = grad_out.slice(slice_desc, slice_indices, {}, offset);
        return {dc_dlhs, dc_drhs};
    }
    
    matrix jvp(std::vector<matrix>& tangents) override {
        return tangents[0].slice_assign(slice_desc, slice_indices, offset, tangents[1]);
    }

// Converts dirty=true into evaluated=false along every path leading to a dirty node,
    // then resets dirty. This is needed because nodes only know their parents, not their
    // children, so the only way to propagate "this subtree needs re-eval" is to walk down
    // from parents and pull the signal up via return value.
    //
    // Memoization (last_visited_pass_id): a shared node (diamond dependency) gets visited
    // once per pass by its first parent, which does the real work and returns the result.
    // Every subsequent parent visiting it in the *same* pass hits the early-return and reads
    // !evaluated instead of recomputing. This is safe specifically because this function
    // never sets evaluated = true — it only ever leaves it untouched or forces it to false.
    // So by the time a second parent visits, whatever the first parent already wrote to
    // `evaluated` is final for this pass, and !evaluated is guaranteed to match what a
    // recompute would have returned.
    //
    // The base-case `return false` (when !dirty && !inv) looks inconsistent with the memo
    // branch's `return !evaluated`, but it can't actually diverge in practice: if evaluated
    // were already false here, invariant (parent can't be evaluated=true if child is
    // evaluated=false) guarantees every ancestor above this node already has evaluated=false
    // too, so this function was never going to flip anything back to true regardless of what
    // it returns. A stale re-invalidation with no exec in between is therefore a true no-op,
    // not a correctness gap.
    bool invalidate_pass(uint64_t current_pass_id) override {
        if (this->last_visited_pass_id == current_pass_id) return !this->evaluated;
        this->last_visited_pass_id = current_pass_id;

        bool inv = false;
        uint64_t new_version = this->version;
        if (input.tape) {
            if (input.tape->invalidate_pass(current_pass_id)) { inv = true; }
            if (input.tape->version > this->version) inv = true;
            new_version = std::max(input.tape->version, new_version);
        }
        if (rhs.tape) {
            if (rhs.tape->invalidate_pass(current_pass_id)) { inv = true; }
            if (rhs.tape->version > this->version) inv = true;
            new_version = std::max(rhs.tape->version, new_version);
        }

        if (inv) {
            this->evaluated = false;
            this->version = new_version;
            return true;
        }
        return false;
    }

    void clear_trace_checks() override {
        evaluated = false;
        if (input.tape) input.tape->clear_trace_checks();
        if (rhs.tape) rhs.tape->clear_trace_checks();
    }

    matrix* get_borrowed_input() override { return nullptr; }
    
    matrix vmap(std::function<matrix(const matrix &)> func, std::vector<int> in_axis) override { return matrix(0, dtype::UInt8); }
    std::vector<matrix> get_inputs() override { return {input, rhs}; }
};

class MaxReductionPrimitive : public Primitive {
public:
    matrix input;
    int axis;
    bool keepdims;
    
    CollapsedDims_2 collapsed_dims;
    bool has_collapsed_dims = false;

    MaxReductionPrimitive(matrix& input_mat, int input_axis, bool input_keepdims) 
        : input(input_mat), axis(input_axis), keepdims(input_keepdims) {}

    void eval_cpu(matrix &out, EvalType eval_type) override {
        if (input.tape && !input.tape->evaluated) { input.tape->eval_cpu(input, eval_type); };
        input.update_from_trace();
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
                out.tape->out_refcount->fetch_add(1, std::memory_order_relaxed);
            }
        }
        // ONLY Allocate, don't execute or mark as evaluated!
        if (eval_type == EvalType::COMPILE_TRACE) {return;}

        if (evaluated) {return;} else {evaluated = true;}
        input.max(out, axis, keepdims, ExecutionDevice::CPU);
    }

    void eval_metal(matrix &out, EvalType eval_type) override {
        if (input.tape && !input.tape->evaluated) { input.tape->eval_metal(input, eval_type); };
        input.update_from_trace();
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
                out.tape->out_refcount->fetch_add(1, std::memory_order_relaxed);
            }
        }
        // ONLY Allocate, don't execute or mark as evaluated!
        if (eval_type == EvalType::COMPILE_TRACE) {return;}

        if (evaluated) {return;} else {evaluated = true;}
        input.max(out, axis, keepdims, ExecutionDevice::METAL);
    }
    std::vector<matrix> vjp(matrix& grad_out) override {
        // a, b, c are vectors (a)=>c soo J = (dc/da) and for sum its 1 so J = (1, 1)
        return {};
    }
    matrix jvp(std::vector<matrix>& tangents) override {
        // a, b, c are vectors (a,b)=>c soo J = (dc/da, dc/db) and for stack we just need to stack the derivative
        // tangents are coming from the inputs so tangents are {da/.., db/...}
        // we need to find the total derivative of c so (da/.. , db/...)

        return matrix(0, dtype::UInt8);
    }
// Converts dirty=true into evaluated=false along every path leading to a dirty node,
    // then resets dirty. This is needed because nodes only know their parents, not their
    // children, so the only way to propagate "this subtree needs re-eval" is to walk down
    // from parents and pull the signal up via return value.
    //
    // Memoization (last_visited_pass_id): a shared node (diamond dependency) gets visited
    // once per pass by its first parent, which does the real work and returns the result.
    // Every subsequent parent visiting it in the *same* pass hits the early-return and reads
    // !evaluated instead of recomputing. This is safe specifically because this function
    // never sets evaluated = true — it only ever leaves it untouched or forces it to false.
    // So by the time a second parent visits, whatever the first parent already wrote to
    // `evaluated` is final for this pass, and !evaluated is guaranteed to match what a
    // recompute would have returned.
    //
    // The base-case `return false` (when !dirty && !inv) looks inconsistent with the memo
    // branch's `return !evaluated`, but it can't actually diverge in practice: if evaluated
    // were already false here, invariant (parent can't be evaluated=true if child is
    // evaluated=false) guarantees every ancestor above this node already has evaluated=false
    // too, so this function was never going to flip anything back to true regardless of what
    // it returns. A stale re-invalidation with no exec in between is therefore a true no-op,
    // not a correctness gap.
    bool invalidate_pass(uint64_t current_pass_id) override {
        if (this->last_visited_pass_id == current_pass_id) return !this->evaluated;
        this->last_visited_pass_id = current_pass_id;

        bool inv = false;
        uint64_t new_version = this->version;
        if (input.tape) {
            if (input.tape->invalidate_pass(current_pass_id)) { inv = true; }
            if (input.tape->version > this->version) inv = true;
            new_version = std::max(input.tape->version, new_version);
        }

        if (inv) {
            this->evaluated = false;
            this->version = new_version;
            return true;
        }
        return false;
    }
    void clear_trace_checks() override {
        evaluated = false;

        if (input.tape) {input.tape->clear_trace_checks(); }

    };
    matrix vmap(std::function<matrix(const matrix &)> func, std::vector<int> in_axis) override { return matrix(0, dtype::UInt8); }
    std::vector<matrix> get_inputs() override {
        return { input };
    }
};

class MinReductionPrimitive : public Primitive {
public:
    matrix input;
    int axis;
    bool keepdims;
    
    CollapsedDims_2 collapsed_dims;
    bool has_collapsed_dims = false;

    MinReductionPrimitive(matrix& input_mat, int input_axis, bool input_keepdims) 
        : input(input_mat), axis(input_axis), keepdims(input_keepdims) {}

    void eval_cpu(matrix &out, EvalType eval_type) override {
        if (input.tape && !input.tape->evaluated) { input.tape->eval_cpu(input, eval_type); };
        input.update_from_trace();
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
                out.tape->out_refcount->fetch_add(1, std::memory_order_relaxed);
            }
        }
        // ONLY Allocate, don't execute or mark as evaluated!
        if (eval_type == EvalType::COMPILE_TRACE) {return;}

        if (evaluated) {return;} else {evaluated = true;}
        input.min(out, axis, keepdims, ExecutionDevice::CPU);
    }

    void eval_metal(matrix &out, EvalType eval_type) override {
        if (input.tape && !input.tape->evaluated) { input.tape->eval_metal(input, eval_type); };
        input.update_from_trace();
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
                out.tape->out_refcount->fetch_add(1, std::memory_order_relaxed);
            }
        }
        // ONLY Allocate, don't execute or mark as evaluated!
        if (eval_type == EvalType::COMPILE_TRACE) {return;}

        if (evaluated) {return;} else {evaluated = true;}
        input.min(out, axis, keepdims, ExecutionDevice::METAL);
    }
    std::vector<matrix> vjp(matrix& grad_out) override {
        // a, b, c are vectors (a)=>c soo J = (dc/da) and for sum its 1 so J = (1, 1)
        return {};
    }
    matrix jvp(std::vector<matrix>& tangents) override {
        // a, b, c are vectors (a,b)=>c soo J = (dc/da, dc/db) and for stack we just need to stack the derivative
        // tangents are coming from the inputs so tangents are {da/.., db/...}
        // we need to find the total derivative of c so (da/.. , db/...)

        return matrix(0, dtype::UInt8);
    }
// Converts dirty=true into evaluated=false along every path leading to a dirty node,
    // then resets dirty. This is needed because nodes only know their parents, not their
    // children, so the only way to propagate "this subtree needs re-eval" is to walk down
    // from parents and pull the signal up via return value.
    //
    // Memoization (last_visited_pass_id): a shared node (diamond dependency) gets visited
    // once per pass by its first parent, which does the real work and returns the result.
    // Every subsequent parent visiting it in the *same* pass hits the early-return and reads
    // !evaluated instead of recomputing. This is safe specifically because this function
    // never sets evaluated = true — it only ever leaves it untouched or forces it to false.
    // So by the time a second parent visits, whatever the first parent already wrote to
    // `evaluated` is final for this pass, and !evaluated is guaranteed to match what a
    // recompute would have returned.
    //
    // The base-case `return false` (when !dirty && !inv) looks inconsistent with the memo
    // branch's `return !evaluated`, but it can't actually diverge in practice: if evaluated
    // were already false here, invariant (parent can't be evaluated=true if child is
    // evaluated=false) guarantees every ancestor above this node already has evaluated=false
    // too, so this function was never going to flip anything back to true regardless of what
    // it returns. A stale re-invalidation with no exec in between is therefore a true no-op,
    // not a correctness gap.
    bool invalidate_pass(uint64_t current_pass_id) override {
        if (this->last_visited_pass_id == current_pass_id) return !this->evaluated;
        this->last_visited_pass_id = current_pass_id;

        bool inv = false;
        uint64_t new_version = this->version;
        if (input.tape) {
            if (input.tape->invalidate_pass(current_pass_id)) { inv = true; }
            if (input.tape->version > this->version) inv = true;
            new_version = std::max(input.tape->version, new_version);
        }

        if (inv) {
            this->evaluated = false;
            this->version = new_version;
            return true;
        }
        return false;
    }
    void clear_trace_checks() override {
        evaluated = false;

        if (input.tape) {input.tape->clear_trace_checks(); }

    };
    matrix vmap(std::function<matrix(const matrix &)> func, std::vector<int> in_axis) override { return matrix(0, dtype::UInt8); }
    std::vector<matrix> get_inputs() override {
        return { input };
    }

};

class MaxPrimitive : public Primitive {
public:
    matrix a;
    matrix b;
    BroadcastDescriptor* desc_a = nullptr;
    BroadcastDescriptor* desc_b = nullptr;
    ~MaxPrimitive() {
        if (desc_a) BroadcastDescriptor::destroy(desc_a);
        if (desc_b) BroadcastDescriptor::destroy(desc_b);
    }
    CollapsedDims_3 collapsed_dims_3;
    bool dims_collapsed = false;

    MaxPrimitive(const matrix& a, const matrix& b) : a(ensure_graph_ready(a)), b(ensure_graph_ready(b)) {
    }
    MaxPrimitive(const matrix& a, const matrix& b, CollapsedDims_3 collapsed_dims_3) : a(ensure_graph_ready(a)), b(ensure_graph_ready(b)), collapsed_dims_3(collapsed_dims_3) {
        dims_collapsed = true;
    }
    void eval_cpu(matrix &out, EvalType eval_type) override {
        if (a.tape && !a.tape->evaluated) { a.tape->eval_cpu(a, eval_type); }
        if (b.tape && !b.tape->evaluated) { b.tape->eval_cpu(b, eval_type); }
        a.update_from_trace();
        b.update_from_trace();

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
                out.tape->out_refcount->fetch_add(1, std::memory_order_relaxed);
            }
        }
        if (eval_type == EvalType::COMPILE_TRACE) { return; }
        if (evaluated) {return;} else {evaluated = true;}

        a.max(b, out, ExecutionDevice::CPU);
    }
    
    void eval_metal(matrix &out, EvalType eval_type) override {
        if (a.tape && !a.tape->evaluated) { a.tape->eval_metal(a, eval_type); }
        if (b.tape && !b.tape->evaluated) { b.tape->eval_metal(b, eval_type); }
        a.update_from_trace();
        b.update_from_trace();
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
                out.tape->out_refcount->fetch_add(1, std::memory_order_relaxed);
            }
        }
        if (eval_type == EvalType::COMPILE_TRACE) { return; }
        if (evaluated) {return;} else { evaluated = true; }
        a.max(b, out, ExecutionDevice::METAL);
    }

    std::vector<matrix> vjp(matrix& grad_out) override {
        return {matrix(0, dtype::UInt8), matrix(0, dtype::UInt8)};
    }
    matrix jvp(std::vector<matrix>& tangents) override {
        return matrix(0, dtype::UInt8);
    }
// Converts dirty=true into evaluated=false along every path leading to a dirty node,
    // then resets dirty. This is needed because nodes only know their parents, not their
    // children, so the only way to propagate "this subtree needs re-eval" is to walk down
    // from parents and pull the signal up via return value.
    //
    // Memoization (last_visited_pass_id): a shared node (diamond dependency) gets visited
    // once per pass by its first parent, which does the real work and returns the result.
    // Every subsequent parent visiting it in the *same* pass hits the early-return and reads
    // !evaluated instead of recomputing. This is safe specifically because this function
    // never sets evaluated = true — it only ever leaves it untouched or forces it to false.
    // So by the time a second parent visits, whatever the first parent already wrote to
    // `evaluated` is final for this pass, and !evaluated is guaranteed to match what a
    // recompute would have returned.
    //
    // The base-case `return false` (when !dirty && !inv) looks inconsistent with the memo
    // branch's `return !evaluated`, but it can't actually diverge in practice: if evaluated
    // were already false here, invariant (parent can't be evaluated=true if child is
    // evaluated=false) guarantees every ancestor above this node already has evaluated=false
    // too, so this function was never going to flip anything back to true regardless of what
    // it returns. A stale re-invalidation with no exec in between is therefore a true no-op,
    // not a correctness gap.
    bool invalidate_pass(uint64_t current_pass_id) override {
        if (this->last_visited_pass_id == current_pass_id) return !this->evaluated;
        this->last_visited_pass_id = current_pass_id;

        bool inv = false;
        uint64_t new_version = this->version;
        if (a.tape) {
            if (a.tape->invalidate_pass(current_pass_id)) { inv = true; }
            if (a.tape->version > this->version) inv = true;
            new_version = std::max(a.tape->version, new_version);
        }
        if (b.tape) {
            if (b.tape->invalidate_pass(current_pass_id)) { inv = true; }
            if (b.tape->version > this->version) inv = true;
            new_version = std::max(b.tape->version, new_version);
        }

        if (inv) {
            this->evaluated = false;
            this->version = new_version;
            return true;
        }
        return false;
    }
    void clear_trace_checks() override {
        evaluated = false;
        if (a.tape) { a.tape->clear_trace_checks(); }
        if (b.tape) { b.tape->clear_trace_checks(); }
    }
    matrix vmap(std::function<matrix(const matrix &)> func, std::vector<int> in_axis) override { return matrix(0, dtype::UInt8); }
    std::vector<matrix> get_inputs() override { return {a, b}; }
};

class MinPrimitive : public Primitive {
public:
    matrix a;
    matrix b;
    BroadcastDescriptor* desc_a = nullptr;
    BroadcastDescriptor* desc_b = nullptr;
    ~MinPrimitive() {
        if (desc_a) BroadcastDescriptor::destroy(desc_a);
        if (desc_b) BroadcastDescriptor::destroy(desc_b);
    }
    CollapsedDims_3 collapsed_dims_3;
    bool dims_collapsed = false;

    MinPrimitive(const matrix& a, const matrix& b) : a(ensure_graph_ready(a)), b(ensure_graph_ready(b)) {
    }
    MinPrimitive(const matrix& a, const matrix& b, CollapsedDims_3 collapsed_dims_3) : a(ensure_graph_ready(a)), b(ensure_graph_ready(b)), collapsed_dims_3(collapsed_dims_3) {
        dims_collapsed = true;
    }
    void eval_cpu(matrix &out, EvalType eval_type) override {
        if (a.tape && !a.tape->evaluated) { a.tape->eval_cpu(a, eval_type); }
        if (b.tape && !b.tape->evaluated) { b.tape->eval_cpu(b, eval_type); }
        a.update_from_trace();
        b.update_from_trace();

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
                out.tape->out_refcount->fetch_add(1, std::memory_order_relaxed);
            }
        }
        if (eval_type == EvalType::COMPILE_TRACE) { return; }
        if (evaluated) {return;} else {evaluated = true;}

        a.min(b, out, ExecutionDevice::CPU);
    }
    
    void eval_metal(matrix &out, EvalType eval_type) override {
        if (a.tape && !a.tape->evaluated) { a.tape->eval_metal(a, eval_type); }
        if (b.tape && !b.tape->evaluated) { b.tape->eval_metal(b, eval_type); }
        a.update_from_trace();
        b.update_from_trace();
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
                out.tape->out_refcount->fetch_add(1, std::memory_order_relaxed);
            }
        }
        if (eval_type == EvalType::COMPILE_TRACE) { return; }
        if (evaluated) {return;} else { evaluated = true; }
        a.min(b, out, ExecutionDevice::METAL);
    }

    std::vector<matrix> vjp(matrix& grad_out) override {
        return {matrix(0, dtype::UInt8), matrix(0, dtype::UInt8)};
    }
    matrix jvp(std::vector<matrix>& tangents) override {
        return matrix(0, dtype::UInt8);
    }
// Converts dirty=true into evaluated=false along every path leading to a dirty node,
    // then resets dirty. This is needed because nodes only know their parents, not their
    // children, so the only way to propagate "this subtree needs re-eval" is to walk down
    // from parents and pull the signal up via return value.
    //
    // Memoization (last_visited_pass_id): a shared node (diamond dependency) gets visited
    // once per pass by its first parent, which does the real work and returns the result.
    // Every subsequent parent visiting it in the *same* pass hits the early-return and reads
    // !evaluated instead of recomputing. This is safe specifically because this function
    // never sets evaluated = true — it only ever leaves it untouched or forces it to false.
    // So by the time a second parent visits, whatever the first parent already wrote to
    // `evaluated` is final for this pass, and !evaluated is guaranteed to match what a
    // recompute would have returned.
    //
    // The base-case `return false` (when !dirty && !inv) looks inconsistent with the memo
    // branch's `return !evaluated`, but it can't actually diverge in practice: if evaluated
    // were already false here, invariant (parent can't be evaluated=true if child is
    // evaluated=false) guarantees every ancestor above this node already has evaluated=false
    // too, so this function was never going to flip anything back to true regardless of what
    // it returns. A stale re-invalidation with no exec in between is therefore a true no-op,
    // not a correctness gap.
    bool invalidate_pass(uint64_t current_pass_id) override {
        if (this->last_visited_pass_id == current_pass_id) return !this->evaluated;
        this->last_visited_pass_id = current_pass_id;

        bool inv = false;
        uint64_t new_version = this->version;
        if (a.tape) {
            if (a.tape->invalidate_pass(current_pass_id)) { inv = true; }
            if (a.tape->version > this->version) inv = true;
            new_version = std::max(a.tape->version, new_version);
        }
        if (b.tape) {
            if (b.tape->invalidate_pass(current_pass_id)) { inv = true; }
            if (b.tape->version > this->version) inv = true;
            new_version = std::max(b.tape->version, new_version);
        }

        if (inv) {
            this->evaluated = false;
            this->version = new_version;
            return true;
        }
        return false;
    }
    void clear_trace_checks() override {
        evaluated = false;
        if (a.tape) { a.tape->clear_trace_checks(); }
        if (b.tape) { b.tape->clear_trace_checks(); }
    }
    matrix vmap(std::function<matrix(const matrix &)> func, std::vector<int> in_axis) override { return matrix(0, dtype::UInt8); }
    std::vector<matrix> get_inputs() override { return {a, b}; }
};

class SumPrimitive : public Primitive {
public:
    matrix input;
    int axis;
    bool NoRed;
    
    CollapsedDims_2 collapsed_dims;
    bool has_collapsed_dims = false;

    SumPrimitive(matrix& input_mat, int input_axis, bool input_NoRed): input(input_mat), axis(input_axis), NoRed (input_NoRed) {
    }
    void eval_cpu(matrix &out, EvalType eval_type) override {
        if (input.tape && !input.tape->evaluated) { input.tape->eval_cpu(input, eval_type); };
        input.update_from_trace();
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
                out.tape->out_refcount->fetch_add(1, std::memory_order_relaxed);
            }
        }
        if (eval_type == EvalType::COMPILE_TRACE) {
            return;
        }
        if (evaluated) {return;} else {evaluated = true;}

        input.sum(out, axis, NoRed, ExecutionDevice::CPU);
        
    };
    void eval_metal(matrix &out, EvalType eval_type) override {
        if (input.tape && !input.tape->evaluated) { input.tape->eval_metal(input, eval_type); };
        input.update_from_trace();
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
                out.tape->out_refcount->fetch_add(1, std::memory_order_relaxed);
            }
        }
        if (eval_type == EvalType::COMPILE_TRACE) {
            return;
        }
        if (evaluated) {return;} else {evaluated = true;}

        input.sum(out, axis, NoRed, ExecutionDevice::METAL);
        
    }


    std::vector<matrix> vjp(matrix& grad_out) override {
        // a, b, c are vectors (a)=>c soo J = (dc/da) and for sum its 1 so J = (1, 1)
        return {grad_out.broadcast_toV2((input.dims < SBO_MAX_DIMS ? input.array_desc.inline_buffer : input.array_desc.shared_arr_desc->shape()), input.dims)};
    }
    matrix jvp(std::vector<matrix>& tangents) override {
        // a, b, c are vectors (a,b)=>c soo J = (dc/da, dc/db) and for stack we just need to stack the derivative
        // tangents are coming from the inputs so tangents are {da/.., db/...}
        // we need to find the total derivative of c so (da/.. , db/...)

        return tangents[0].sum(axis, NoRed);
    }
// Converts dirty=true into evaluated=false along every path leading to a dirty node,
    // then resets dirty. This is needed because nodes only know their parents, not their
    // children, so the only way to propagate "this subtree needs re-eval" is to walk down
    // from parents and pull the signal up via return value.
    //
    // Memoization (last_visited_pass_id): a shared node (diamond dependency) gets visited
    // once per pass by its first parent, which does the real work and returns the result.
    // Every subsequent parent visiting it in the *same* pass hits the early-return and reads
    // !evaluated instead of recomputing. This is safe specifically because this function
    // never sets evaluated = true — it only ever leaves it untouched or forces it to false.
    // So by the time a second parent visits, whatever the first parent already wrote to
    // `evaluated` is final for this pass, and !evaluated is guaranteed to match what a
    // recompute would have returned.
    //
    // The base-case `return false` (when !dirty && !inv) looks inconsistent with the memo
    // branch's `return !evaluated`, but it can't actually diverge in practice: if evaluated
    // were already false here, invariant (parent can't be evaluated=true if child is
    // evaluated=false) guarantees every ancestor above this node already has evaluated=false
    // too, so this function was never going to flip anything back to true regardless of what
    // it returns. A stale re-invalidation with no exec in between is therefore a true no-op,
    // not a correctness gap.
    bool invalidate_pass(uint64_t current_pass_id) override {
        if (this->last_visited_pass_id == current_pass_id) return !this->evaluated;
        this->last_visited_pass_id = current_pass_id;

        bool inv = false;
        uint64_t new_version = this->version;
        if (input.tape) {
            if (input.tape->invalidate_pass(current_pass_id)) { inv = true; }
            if (input.tape->version > this->version) inv = true;
            new_version = std::max(input.tape->version, new_version);
        }

        if (inv) {
            this->evaluated = false;
            this->version = new_version;
            return true;
        }
        return false;
    }
    void clear_trace_checks() override {
        evaluated = false;

        if (input.tape) {input.tape->clear_trace_checks(); }

    };
    matrix vmap(std::function<matrix(const matrix &)> func, std::vector<int> in_axis) override { return matrix(0, dtype::UInt8); }
    std::vector<matrix> get_inputs() override {
        return { input };
    }
};

class PaddingPrimitive : public Primitive {
public:
    matrix input;
    matrix value;
    size_t offset;
    std::vector<size_m> padding_range;
    std::vector<size_m> out_shape;
    PaddingPrimitive(const matrix& input_mat, const matrix& input_value, size_t input_offset, const std::vector<size_m>& input_padding_range, const const std::vector<size_m>& output_shape): input(input_mat), value(input_value), offset(input_offset), padding_range(input_padding_range), out_shape(output_shape) {
    }
    void eval_cpu(matrix &out, EvalType eval_type) override {
        if (input.tape && !input.tape->evaluated) { input.tape->eval_cpu(input, eval_type); };
        input.update_from_trace();
        if (!out.buffer) {
            if (out.tape->out_buffer) {
                out.buffer = out.tape->out_buffer;
                out.metalBuffer = out.tape->out_metal_buffer;
                out.refCount = out.tape->out_refcount;
                out.refCount->fetch_add(1);
                if (eval_type == EvalType::COMPILE_TRACE) {return;}
            } else {
                out.buffer = new uint8_t[out.effectiveBufferSize() * dtype_size(out.type)];
                out.begin_refcount();
                out.buildMetalBuffer();
                out.tape->out_buffer = (uint8_t*)out.buffer;
                out.tape->out_metal_buffer = out.metalBuffer;
                out.tape->out_refcount = out.refCount;
                out.tape->out_refcount->fetch_add(1, std::memory_order_relaxed);
            }
        }
        if (eval_type == EvalType::COMPILE_TRACE) {
            return;
        }
        if (evaluated) {return;} else {evaluated = true;}

        input.pad( padding_range, out, value, ExecutionDevice::CPU);

    };
    void eval_metal(matrix &out, EvalType eval_type) override {
        if (input.tape && !input.tape->evaluated) { input.tape->eval_metal(input, eval_type); };
        input.update_from_trace();
        if (!out.buffer) {
            if (out.tape->out_buffer) {
                out.buffer = out.tape->out_buffer;
                out.metalBuffer = out.tape->out_metal_buffer;
                out.refCount = out.tape->out_refcount;
                out.refCount->fetch_add(1);
                if (eval_type == EvalType::COMPILE_TRACE) {return;}
            } else {
                out.buffer = new uint8_t[out.effectiveBufferSize() * dtype_size(out.type)];
                out.begin_refcount();
                out.buildMetalBuffer();

                out.tape->out_buffer = (uint8_t*)out.buffer;
                out.tape->out_metal_buffer = out.metalBuffer;
                out.tape->out_refcount = out.refCount;
                out.tape->out_refcount->fetch_add(1, std::memory_order_relaxed);
            }
        }
        if (eval_type == EvalType::COMPILE_TRACE) {
            return;
        }
        if (evaluated) {return;} else {evaluated = true;}

        input.pad( padding_range, out, value, ExecutionDevice::METAL);

    }


    std::vector<matrix> vjp(matrix& grad_out) override {
        // a, c are vectors (padding ... a .. padding)=>c soo J = (dc/da, dc/db) and for addition its 1 so J = (1, 1)
        matrix grad_sliced = grad_out.slice(input.array_desc, padding_range, {}, offset);
        return { grad_sliced };
    }
    matrix jvp(std::vector<matrix>& tangents) override {
        // a, b, c are vectors (a,b)=>c soo J = (dc/da, dc/db) and for stack we just need to stack the derivative
        // tangents are coming from the inputs so tangents are {da/.., db/...}
        // we need to find the total derivative of c so (da/.. , db/...)

        return tangents[0].pad(padding_range, value);
    }
// Converts dirty=true into evaluated=false along every path leading to a dirty node,
    // then resets dirty. This is needed because nodes only know their parents, not their
    // children, so the only way to propagate "this subtree needs re-eval" is to walk down
    // from parents and pull the signal up via return value.
    //
    // Memoization (last_visited_pass_id): a shared node (diamond dependency) gets visited
    // once per pass by its first parent, which does the real work and returns the result.
    // Every subsequent parent visiting it in the *same* pass hits the early-return and reads
    // !evaluated instead of recomputing. This is safe specifically because this function
    // never sets evaluated = true — it only ever leaves it untouched or forces it to false.
    // So by the time a second parent visits, whatever the first parent already wrote to
    // `evaluated` is final for this pass, and !evaluated is guaranteed to match what a
    // recompute would have returned.
    //
    // The base-case `return false` (when !dirty && !inv) looks inconsistent with the memo
    // branch's `return !evaluated`, but it can't actually diverge in practice: if evaluated
    // were already false here, invariant (parent can't be evaluated=true if child is
    // evaluated=false) guarantees every ancestor above this node already has evaluated=false
    // too, so this function was never going to flip anything back to true regardless of what
    // it returns. A stale re-invalidation with no exec in between is therefore a true no-op,
    // not a correctness gap.
    bool invalidate_pass(uint64_t current_pass_id) override {
        if (this->last_visited_pass_id == current_pass_id) return !this->evaluated;
        this->last_visited_pass_id = current_pass_id;

        bool inv = false;
        uint64_t new_version = this->version;
        if (input.tape) {
            if (input.tape->invalidate_pass(current_pass_id)) { inv = true; }
            if (input.tape->version > this->version) inv = true;
            new_version = std::max(input.tape->version, new_version);
        }

        if (inv) {
            this->evaluated = false;
            this->version = new_version;
            return true;
        }
        return false;
    }
    void clear_trace_checks() override {
        evaluated = false;

        if (input.tape) {input.tape->clear_trace_checks(); }

    };
    matrix vmap(std::function<matrix(const matrix &)> func, std::vector<int> in_axis) override { return matrix(0, dtype::UInt8); }
    std::vector<matrix> get_inputs() override {
        return {input, value};
    }
};

class BrodcastPrimitive : public Primitive {
public:
    matrix* get_borrowed_input() override { return &input; }
    matrix input;
    array_descriptor brodcast_desc;
    size_t broadcasted_dims;
    BrodcastPrimitive(matrix& input_mat, array_descriptor input_brodcast_desc, size_t input_broadcasted_dims): input(ensure_graph_ready(input_mat)), brodcast_desc(input_brodcast_desc), broadcasted_dims(input_broadcasted_dims) {
        brodcast_desc.retain(broadcasted_dims);
    }
    void eval_cpu(matrix &out, EvalType eval_type) override {
        if (input.tape && !input.tape->evaluated) { input.tape->eval_cpu(input, eval_type); };
        input.update_from_trace();
        if (!out.buffer) {
            if (out.tape->out_buffer) {
                out.buffer = out.tape->out_buffer;
                out.metalBuffer = out.tape->out_metal_buffer;
                out.refCount = out.tape->out_refcount;
                out.refCount->fetch_add(1);
                if (eval_type == EvalType::COMPILE_TRACE) {return;}
            } else {
                out.buffer = (uint8_t*)input.buffer;
                out.metalBuffer = input.metalBuffer;
                out.refCount = input.refCount;
                out.refCount->fetch_add(1);

                out.tape->out_buffer = (uint8_t*)out.buffer;
                out.tape->out_metal_buffer = out.metalBuffer;
                out.tape->out_refcount = out.refCount;
                out.tape->out_refcount->fetch_add(1, std::memory_order_relaxed);
            }
        }
        if (out.buffer != (uint8_t*)input.buffer) {
            out.releaseBuffer();
            out.buffer = (uint8_t*)input.buffer;
            out.metalBuffer = input.metalBuffer;
            out.refCount = input.refCount;
            if (out.refCount) out.refCount->fetch_add(1);
            out.tape->update_cache((uint8_t*)out.buffer, out.metalBuffer, out.refCount);
            // out.tape->out_buffer = (uint8_t*)out.buffer;
            // out.tape->out_metal_buffer = out.metalBuffer;
            // out.tape->out_refcount = out.refCount;
            // out.tape->out_refcount->fetch_add(1, std::memory_order_relaxed);
        }
        if (eval_type == EvalType::COMPILE_TRACE) {
            return;
        }
        if (evaluated) {return;} else {evaluated = true;}
    };
    void eval_metal(matrix &out, EvalType eval_type) override {
        if (input.tape && !input.tape->evaluated) { input.tape->eval_metal(input, eval_type); };
        input.update_from_trace();
        if (!out.buffer) {
            if (out.tape->out_buffer) {
                out.buffer = out.tape->out_buffer;
                out.metalBuffer = out.tape->out_metal_buffer;
                out.refCount = out.tape->out_refcount;
                out.refCount->fetch_add(1);
                if (eval_type == EvalType::COMPILE_TRACE) {return;}
            } else {
                out.buffer = (uint8_t*)input.buffer;
                out.metalBuffer = input.metalBuffer;
                out.refCount = input.refCount;
                out.refCount->fetch_add(1);

                out.tape->out_buffer = (uint8_t*)out.buffer;
                out.tape->out_metal_buffer = out.metalBuffer;
                out.tape->out_refcount = out.refCount;
                out.tape->out_refcount->fetch_add(1, std::memory_order_relaxed);
            }
        }
        if (out.buffer != (uint8_t*)input.buffer) {
            out.releaseBuffer();
            out.buffer = (uint8_t*)input.buffer;
            out.metalBuffer = input.metalBuffer;
            out.refCount = input.refCount;
            if (out.refCount) out.refCount->fetch_add(1);
            
            out.tape->update_cache((uint8_t*)out.buffer, out.metalBuffer, out.refCount);
            // out.tape->out_buffer = (uint8_t*)out.buffer;
            // out.tape->out_metal_buffer = out.metalBuffer;
            // out.tape->out_refcount = out.refCount;
            // out.tape->out_refcount->fetch_add(1, std::memory_order_relaxed);
        }
        if (eval_type == EvalType::COMPILE_TRACE) {
            return;
        }
        if (evaluated) {return;} else {evaluated = true;}
    }


    std::vector<matrix> vjp(matrix& grad_out) override {
        // a, c are vectors a=>c,  soo J = (dc/da) and as we didnt modify the data we just duplicated it so we need to sum the derivative of the duplications
        return {grad_out.unbroadcast_shape(input.shape(), input.dims)};
    }
    matrix jvp(std::vector<matrix>& tangents) override {
        // a, b, c are vectors (a,b)=>c soo J = (dc/da) and for broadcast we didnt modify the data we just duplicated it so we need to duplicate the derivative so it can flow down / forward;
        // tangents are coming from the inputs so tangents are {da/..,} so we need to broadcast them.
        return tangents[0].broadcast_toV2((broadcasted_dims < SBO_MAX_DIMS ? brodcast_desc.inline_buffer : brodcast_desc.shared_arr_desc->shape()), input.dims);
    }
// Converts dirty=true into evaluated=false along every path leading to a dirty node,
    // then resets dirty. This is needed because nodes only know their parents, not their
    // children, so the only way to propagate "this subtree needs re-eval" is to walk down
    // from parents and pull the signal up via return value.
    //
    // Memoization (last_visited_pass_id): a shared node (diamond dependency) gets visited
    // once per pass by its first parent, which does the real work and returns the result.
    // Every subsequent parent visiting it in the *same* pass hits the early-return and reads
    // !evaluated instead of recomputing. This is safe specifically because this function
    // never sets evaluated = true — it only ever leaves it untouched or forces it to false.
    // So by the time a second parent visits, whatever the first parent already wrote to
    // `evaluated` is final for this pass, and !evaluated is guaranteed to match what a
    // recompute would have returned.
    //
    // The base-case `return false` (when !dirty && !inv) looks inconsistent with the memo
    // branch's `return !evaluated`, but it can't actually diverge in practice: if evaluated
    // were already false here, invariant (parent can't be evaluated=true if child is
    // evaluated=false) guarantees every ancestor above this node already has evaluated=false
    // too, so this function was never going to flip anything back to true regardless of what
    // it returns. A stale re-invalidation with no exec in between is therefore a true no-op,
    // not a correctness gap.
    bool invalidate_pass(uint64_t current_pass_id) override {
        if (this->last_visited_pass_id == current_pass_id) return !this->evaluated;
        this->last_visited_pass_id = current_pass_id;

        bool inv = false;
        uint64_t new_version = this->version;
        if (input.tape) {
            if (input.tape->invalidate_pass(current_pass_id)) { inv = true; }
            if (input.tape->version > this->version) inv = true;
            new_version = std::max(input.tape->version, new_version);
        }

        if (inv) {
            this->evaluated = false;
            this->version = new_version;
            return true;
        }
        return false;
    }
    void clear_trace_checks() override {
        evaluated = false;

        if (input.tape) {input.tape->clear_trace_checks(); }

    };
    matrix vmap(std::function<matrix(const matrix &)> func, std::vector<int> in_axis) override { return matrix(0, dtype::UInt8); }
    std::vector<matrix> get_inputs() override {
        return {input};
    }
};

class ReshapePrimitive : public Primitive {
public:
    matrix* get_borrowed_input() override { return REQUIRES_NEW_BUFFER ? nullptr : &input; }
    matrix input;
    array_descriptor reshape_desc;
    size_t reshape_dim;
    bool REQUIRES_NEW_BUFFER;
    ReshapePrimitive(matrix& input_mat, array_descriptor input_reshape_desc, size_t input_reshape_dims, bool REQUIRES_NEW_BUFFER = true): input(ensure_graph_ready(input_mat)), reshape_desc(input_reshape_desc) , reshape_dim(input_reshape_dims), REQUIRES_NEW_BUFFER(REQUIRES_NEW_BUFFER) {
        reshape_desc.retain(reshape_dim);
    }
    void eval_cpu(matrix &out, EvalType eval_type) override {
        if (input.tape && !input.tape->evaluated) { input.tape->eval_cpu(input, eval_type); };
        input.update_from_trace();
        if (!out.buffer) {
            if (out.tape->out_buffer) {
                out.buffer = out.tape->out_buffer;
                out.metalBuffer = out.tape->out_metal_buffer;
                out.refCount = out.tape->out_refcount;
                out.refCount->fetch_add(1);
            } else if (REQUIRES_NEW_BUFFER) {
                out.buffer = new uint8_t[out.effectiveBufferSize() * dtype_size(out.type)];
                out.begin_refcount();
                out.buildMetalBuffer();
                out.tape->out_buffer = (uint8_t*)out.buffer;
                out.tape->out_metal_buffer = out.metalBuffer;
                out.tape->out_refcount = out.refCount;
                out.tape->out_refcount->fetch_add(1, std::memory_order_relaxed);
            } else {
                out.buffer = (uint8_t*)input.buffer;
                out.metalBuffer = input.metalBuffer;
                out.refCount = input.refCount;
                out.refCount->fetch_add(1);

                out.tape->out_buffer = (uint8_t*)out.buffer;
                out.tape->out_metal_buffer = out.metalBuffer;
                out.tape->out_refcount = out.refCount;
                out.tape->out_refcount->fetch_add(1, std::memory_order_relaxed);
            }
        }
        if (out.buffer != (uint8_t*)input.buffer && !REQUIRES_NEW_BUFFER) {
            out.releaseBuffer();
            out.buffer = (uint8_t*)input.buffer;
            out.metalBuffer = input.metalBuffer;
            out.refCount = input.refCount;
            if (out.refCount) out.refCount->fetch_add(1);
            
            out.tape->update_cache((uint8_t*)out.buffer, out.metalBuffer, out.refCount);
//            out.tape->out_buffer = (uint8_t*)out.buffer;
//            out.tape->out_metal_buffer = out.metalBuffer;
//            out.tape->out_refcount = out.refCount;
//            out.tape->out_refcount->fetch_add(1, std::memory_order_relaxed);
        }
        if (eval_type == EvalType::COMPILE_TRACE) {
            return;
        }
        if (evaluated) {return;} else {evaluated = true;}
        
        
        if (REQUIRES_NEW_BUFFER) {
            input.reshape_eval(out, reshape_desc, reshape_dim, ExecutionDevice::CPU);
        }
    };
    void eval_metal(matrix &out, EvalType eval_type) override {
        if (input.tape && !input.tape->evaluated) { input.tape->eval_metal(input, eval_type); };
        input.update_from_trace();

        if (!out.buffer) {
            if (out.tape->out_buffer) {
                out.buffer = out.tape->out_buffer;
                out.metalBuffer = out.tape->out_metal_buffer;
                out.refCount = out.tape->out_refcount;
                out.refCount->fetch_add(1);
            } else if (REQUIRES_NEW_BUFFER) {
                // If input buffer is non contigious we create a new buffer and first make it contiguous and change the shape and strides
                out.buffer = new uint8_t[out.effectiveBufferSize() * dtype_size(out.type)];
                out.begin_refcount();
                out.buildMetalBuffer();

                out.tape->out_buffer = (uint8_t*)out.buffer;
                out.tape->out_metal_buffer = out.metalBuffer;
                out.tape->out_refcount = out.refCount;
                out.tape->out_refcount->fetch_add(1, std::memory_order_relaxed);
            } else {
                out.buffer = (uint8_t*)input.buffer;
                out.metalBuffer = input.metalBuffer;
                out.refCount = input.refCount;
                out.refCount->fetch_add(1);

                out.tape->out_buffer = (uint8_t*)out.buffer;
                out.tape->out_metal_buffer = out.metalBuffer;
                out.tape->out_refcount = out.refCount;
                out.tape->out_refcount->fetch_add(1, std::memory_order_relaxed);
            }
        }
        if (out.buffer != (uint8_t*)input.buffer && !REQUIRES_NEW_BUFFER) {
            out.releaseBuffer();
            out.buffer = (uint8_t*)input.buffer;
            out.metalBuffer = input.metalBuffer;
            out.refCount = input.refCount;
            if (out.refCount) out.refCount->fetch_add(1);
            
            out.tape->update_cache((uint8_t*)out.buffer, out.metalBuffer, out.refCount);
//            out.tape->out_buffer = (uint8_t*)out.buffer;
//            out.tape->out_metal_buffer = out.metalBuffer;
//            out.tape->out_refcount = out.refCount;
//            out.tape->out_refcount->fetch_add(1, std::memory_order_relaxed);
        }

        if (eval_type == EvalType::COMPILE_TRACE) {
            return;
        }
        if (evaluated) {return;} else {evaluated = true;}
        
        if (REQUIRES_NEW_BUFFER) {
            input.reshape_eval(out, reshape_desc, reshape_dim, ExecutionDevice::METAL);
        }
    }


    std::vector<matrix> vjp(matrix& grad_out) override {
        // a is vector a.reshape=>c soo J = (dc/da) and for reshape we didnt change anything so its 1 so J = 1 reshaped
        return { grad_out.reshape(input.array_desc, input.dims) };
    }
    matrix jvp(std::vector<matrix>& tangents) override {
        // a is vector a.reshape=>c soo J = (dc/da) and for reshape we didnt change anything so its 1 so J = 1 reshaped
        return tangents[0].reshape(reshape_desc, reshape_dim);

    }
// Converts dirty=true into evaluated=false along every path leading to a dirty node,
    // then resets dirty. This is needed because nodes only know their parents, not their
    // children, so the only way to propagate "this subtree needs re-eval" is to walk down
    // from parents and pull the signal up via return value.
    //
    // Memoization (last_visited_pass_id): a shared node (diamond dependency) gets visited
    // once per pass by its first parent, which does the real work and returns the result.
    // Every subsequent parent visiting it in the *same* pass hits the early-return and reads
    // !evaluated instead of recomputing. This is safe specifically because this function
    // never sets evaluated = true — it only ever leaves it untouched or forces it to false.
    // So by the time a second parent visits, whatever the first parent already wrote to
    // `evaluated` is final for this pass, and !evaluated is guaranteed to match what a
    // recompute would have returned.
    //
    // The base-case `return false` (when !dirty && !inv) looks inconsistent with the memo
    // branch's `return !evaluated`, but it can't actually diverge in practice: if evaluated
    // were already false here, invariant (parent can't be evaluated=true if child is
    // evaluated=false) guarantees every ancestor above this node already has evaluated=false
    // too, so this function was never going to flip anything back to true regardless of what
    // it returns. A stale re-invalidation with no exec in between is therefore a true no-op,
    // not a correctness gap.
    bool invalidate_pass(uint64_t current_pass_id) override {
        if (this->last_visited_pass_id == current_pass_id) return !this->evaluated;
        this->last_visited_pass_id = current_pass_id;

        bool inv = false;
        uint64_t new_version = this->version;
        if (input.tape) {
            if (input.tape->invalidate_pass(current_pass_id)) { inv = true; }
            if (input.tape->version > this->version) inv = true;
            new_version = std::max(input.tape->version, new_version);
        }

        if (inv) {
            this->evaluated = false;
            this->version = new_version;
            return true;
        }
        return false;
    }
    void clear_trace_checks() override {
        evaluated = false;

        if (input.tape) {input.tape->clear_trace_checks(); }

    };
    matrix vmap(std::function<matrix(const matrix &)> func, std::vector<int> in_axis) override { return matrix(0, dtype::UInt8); }
    std::vector<matrix> get_inputs() override {
        return { input };
    }
};

class AsTypePrimitive : public Primitive {
public:
    matrix input;
    dtype new_type;
    explicit AsTypePrimitive(const matrix& input, dtype new_type) : input(ensure_graph_ready(input)), new_type(new_type) {
    }

    void eval_cpu(matrix &out, EvalType eval_type) override {
        if (input.tape && !input.tape->evaluated) { input.tape->eval_cpu(input, eval_type); };
        input.update_from_trace();
        if (!out.buffer) {
            if (out.tape->out_buffer) {
                out.buffer = out.tape->out_buffer;
                out.metalBuffer = out.tape->out_metal_buffer;
                out.refCount = out.tape->out_refcount;
                out.refCount->fetch_add(1);
                if (eval_type == EvalType::COMPILE_TRACE) {return;}
            } else {
                out.buffer = new uint8_t[out.effectiveBufferSize() * dtype_size(out.type)];
                out.begin_refcount();
                out.buildMetalBuffer();

                out.tape->out_buffer = (uint8_t*)out.buffer;
                out.tape->out_metal_buffer = out.metalBuffer;
                out.tape->out_refcount = out.refCount;
                out.tape->out_refcount->fetch_add(1, std::memory_order_relaxed);
            }
        }
        if (eval_type == EvalType::COMPILE_TRACE) {
            return;
        }
        if (evaluated) {return;} else {evaluated = true;}
        input.astype(out, new_type, eval_type, ExecutionDevice::CPU);
    };
    void eval_metal(matrix &out, EvalType eval_type) override {
        if (input.tape && !input.tape->evaluated) { input.tape->eval_metal(input, eval_type); };
        input.update_from_trace();
        if (!out.buffer) {
            if (out.tape->out_buffer) {
                out.buffer = out.tape->out_buffer;
                out.metalBuffer = out.tape->out_metal_buffer;
                out.refCount = out.tape->out_refcount;
                out.refCount->fetch_add(1);
                if (eval_type == EvalType::COMPILE_TRACE) {return;}
            } else {
                out.buffer = new uint8_t[out.effectiveBufferSize() * dtype_size(out.type)];
                out.begin_refcount();
                out.buildMetalBuffer();

                out.tape->out_buffer = (uint8_t*)out.buffer;
                out.tape->out_metal_buffer = out.metalBuffer;
                out.tape->out_refcount = out.refCount;
                out.tape->out_refcount->fetch_add(1, std::memory_order_relaxed);

            }
        }
        if (eval_type == EvalType::COMPILE_TRACE) {
            return;
        }
        if (evaluated) {return;} else {evaluated = true;}
        input.astype(out, new_type, eval_type, ExecutionDevice::METAL);
    }


    std::vector<matrix> vjp(matrix& grad_out) override {
        // a is vector a.reshape=>c soo J = (dc/da) and for reshape we didnt change anything so its 1 so J = 1 reshaped
        return { grad_out.astype(input.type) };
    }
    matrix jvp(std::vector<matrix>& tangents) override {
        // a is vector a.reshape=>c soo J = (dc/da) and for reshape we didnt change anything so its 1 so J = 1 reshaped
        return tangents[0].astype(new_type);

    }
// Converts dirty=true into evaluated=false along every path leading to a dirty node,
    // then resets dirty. This is needed because nodes only know their parents, not their
    // children, so the only way to propagate "this subtree needs re-eval" is to walk down
    // from parents and pull the signal up via return value.
    //
    // Memoization (last_visited_pass_id): a shared node (diamond dependency) gets visited
    // once per pass by its first parent, which does the real work and returns the result.
    // Every subsequent parent visiting it in the *same* pass hits the early-return and reads
    // !evaluated instead of recomputing. This is safe specifically because this function
    // never sets evaluated = true — it only ever leaves it untouched or forces it to false.
    // So by the time a second parent visits, whatever the first parent already wrote to
    // `evaluated` is final for this pass, and !evaluated is guaranteed to match what a
    // recompute would have returned.
    //
    // The base-case `return false` (when !dirty && !inv) looks inconsistent with the memo
    // branch's `return !evaluated`, but it can't actually diverge in practice: if evaluated
    // were already false here, invariant (parent can't be evaluated=true if child is
    // evaluated=false) guarantees every ancestor above this node already has evaluated=false
    // too, so this function was never going to flip anything back to true regardless of what
    // it returns. A stale re-invalidation with no exec in between is therefore a true no-op,
    // not a correctness gap.
    bool invalidate_pass(uint64_t current_pass_id) override {
        if (this->last_visited_pass_id == current_pass_id) return !this->evaluated;
        this->last_visited_pass_id = current_pass_id;

        bool inv = false;
        uint64_t new_version = this->version;
        if (input.tape) {
            if (input.tape->invalidate_pass(current_pass_id)) { inv = true; }
            if (input.tape->version > this->version) inv = true;
            new_version = std::max(input.tape->version, new_version);
        }

        if (inv) {
            this->evaluated = false;
            this->version = new_version;
            return true;
        }
        return false;
    }
    void clear_trace_checks() override {
        evaluated = false;

        if (input.tape) {input.tape->clear_trace_checks(); }

    };
    matrix vmap(std::function<matrix(const matrix &)> func, std::vector<int> in_axis) override { return matrix(0, dtype::UInt8); }
    std::vector<matrix> get_inputs() override {
        return { input };
    }
};

class TransposePrimitive : public Primitive {
public:
    matrix* get_borrowed_input() override { return &input; }
    matrix input;
    array_descriptor transpose_desc;
    TransposePrimitive(matrix& input_mat, array_descriptor input_transpose_desc): input(ensure_graph_ready(input_mat)), transpose_desc(input_transpose_desc) {
        transpose_desc.retain(input.dims);
    }
    void eval_cpu(matrix &out, EvalType eval_type) override {
        if (input.tape && !input.tape->evaluated) { input.tape->eval_cpu(input, eval_type); };
        input.update_from_trace();
        if (!out.buffer) {
            if (out.tape->out_buffer) {
                out.buffer = out.tape->out_buffer;
                out.metalBuffer = out.tape->out_metal_buffer;
                out.refCount = out.tape->out_refcount;
                out.refCount->fetch_add(1);
                if (eval_type == EvalType::COMPILE_TRACE) {return;}
            } else {
                out.buffer = (uint8_t*)input.buffer;
                out.metalBuffer = input.metalBuffer;
                out.refCount = input.refCount;
                out.refCount->fetch_add(1);

                out.tape->out_buffer = (uint8_t*)out.buffer;
                out.tape->out_metal_buffer = out.metalBuffer;
                out.tape->out_refcount = out.refCount;
                out.tape->out_refcount->fetch_add(1, std::memory_order_relaxed);
            }
        }
        if (out.buffer != (uint8_t*)input.buffer) {
            out.releaseBuffer();
            out.buffer = (uint8_t*)input.buffer;
            out.metalBuffer = input.metalBuffer;
            out.refCount = input.refCount;
            if (out.refCount) out.refCount->fetch_add(1);

            out.tape->out_buffer = (uint8_t*)out.buffer;
            out.tape->out_metal_buffer = out.metalBuffer;
            out.tape->out_refcount = out.refCount;
            out.tape->out_refcount->fetch_add(1, std::memory_order_relaxed);
        }
        if (out.buffer != (uint8_t*)input.buffer) {
            out.releaseBuffer();
            out.buffer = (uint8_t*)input.buffer;
            out.metalBuffer = input.metalBuffer;
            out.refCount = input.refCount;
            out.refCount->fetch_add(1);

            out.tape->out_buffer = (uint8_t*)out.buffer;
            out.tape->out_metal_buffer = out.metalBuffer;
            out.tape->out_refcount = out.refCount;
            out.tape->out_refcount->fetch_add(1, std::memory_order_relaxed);
        }
        if (eval_type == EvalType::COMPILE_TRACE) {
            return;
        }
        if (evaluated) {return;} else {evaluated = true;}
    };
    void eval_metal(matrix &out, EvalType eval_type) override {
        if (input.tape && !input.tape->evaluated) { input.tape->eval_metal(input, eval_type); };
        input.update_from_trace();
        if (!out.buffer) {
            if (out.tape->out_buffer) {
                out.buffer = out.tape->out_buffer;
                out.metalBuffer = out.tape->out_metal_buffer;
                out.refCount = out.tape->out_refcount;
                out.refCount->fetch_add(1);
                if (eval_type == EvalType::COMPILE_TRACE) {return;}
            } else {
                out.buffer = (uint8_t*)input.buffer;
                out.metalBuffer = input.metalBuffer;
                out.refCount = input.refCount;
                out.refCount->fetch_add(1);

                out.tape->out_buffer = (uint8_t*)out.buffer;
                out.tape->out_metal_buffer = out.metalBuffer;
                out.tape->out_refcount = out.refCount;
                out.tape->out_refcount->fetch_add(1, std::memory_order_relaxed);
            }
        }
        if (out.buffer != (uint8_t*)input.buffer) {
            out.releaseBuffer();
            out.buffer = (uint8_t*)input.buffer;
            out.metalBuffer = input.metalBuffer;
            out.refCount = input.refCount;
            if (out.refCount) out.refCount->fetch_add(1);

            out.tape->out_buffer = (uint8_t*)out.buffer;
            out.tape->out_metal_buffer = out.metalBuffer;
            out.tape->out_refcount = out.refCount;
            out.tape->out_refcount->fetch_add(1, std::memory_order_relaxed);
        }
        if (eval_type == EvalType::COMPILE_TRACE) {
            return;
        }
        if (evaluated) {return;} else {evaluated = true;}
    }


    std::vector<matrix> vjp(matrix& grad_out) override {
        // a is vector a.reshape=>c soo J = (dc/da) and for reshape we didnt change anything so its 1 so J = 1 reshaped
        return { grad_out.transpose(input.array_desc) };
    }
    matrix jvp(std::vector<matrix>& tangents) override {
        // a is vector a.reshape=>c soo J = (dc/da) and for reshape we didnt change anything so its 1 so J = 1 reshaped
        return tangents[0].transpose(transpose_desc);

    }
// Converts dirty=true into evaluated=false along every path leading to a dirty node,
    // then resets dirty. This is needed because nodes only know their parents, not their
    // children, so the only way to propagate "this subtree needs re-eval" is to walk down
    // from parents and pull the signal up via return value.
    //
    // Memoization (last_visited_pass_id): a shared node (diamond dependency) gets visited
    // once per pass by its first parent, which does the real work and returns the result.
    // Every subsequent parent visiting it in the *same* pass hits the early-return and reads
    // !evaluated instead of recomputing. This is safe specifically because this function
    // never sets evaluated = true — it only ever leaves it untouched or forces it to false.
    // So by the time a second parent visits, whatever the first parent already wrote to
    // `evaluated` is final for this pass, and !evaluated is guaranteed to match what a
    // recompute would have returned.
    //
    // The base-case `return false` (when !dirty && !inv) looks inconsistent with the memo
    // branch's `return !evaluated`, but it can't actually diverge in practice: if evaluated
    // were already false here, invariant (parent can't be evaluated=true if child is
    // evaluated=false) guarantees every ancestor above this node already has evaluated=false
    // too, so this function was never going to flip anything back to true regardless of what
    // it returns. A stale re-invalidation with no exec in between is therefore a true no-op,
    // not a correctness gap.
    bool invalidate_pass(uint64_t current_pass_id) override {
        if (this->last_visited_pass_id == current_pass_id) return !this->evaluated;
        this->last_visited_pass_id = current_pass_id;

        bool inv = false;
        uint64_t new_version = this->version;
        if (input.tape) {
            if (input.tape->invalidate_pass(current_pass_id)) { inv = true; }
            if (input.tape->version > this->version) inv = true;
            new_version = std::max(input.tape->version, new_version);
        }

        if (inv) {
            this->evaluated = false;
            this->version = new_version;
            return true;
        }
        return false;
    }
    void clear_trace_checks() override {
        evaluated = false;

        if (input.tape) {input.tape->clear_trace_checks(); }

    };
    matrix vmap(std::function<matrix(const matrix &)> func, std::vector<int> in_axis) override { return matrix(0, dtype::UInt8); }
    std::vector<matrix> get_inputs() override {
        return {input};
    }
};


class CompiledNodePrimitive : public Primitive {
public:
    matrix outer_input;
    matrix sample_parameter;
    matrix output_graph;

    CompiledNodePrimitive(matrix& outer_input, matrix& sample_parameter, matrix& output_graph) : outer_input(ensure_graph_ready(outer_input)), sample_parameter(sample_parameter), output_graph(output_graph) {
    }
    
    

    void eval_cpu(matrix &out, EvalType eval_type) override {
        if (outer_input.tape && !outer_input.tape->evaluated) { outer_input.tape->eval_cpu(outer_input, eval_type); }
        outer_input.update_from_trace();
        
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
                out.tape->out_refcount->fetch_add(1, std::memory_order_relaxed);
            }
        }
        if (eval_type == EvalType::COMPILE_TRACE) {return;}
        if (evaluated) {return;} else {evaluated = true;}
        // outer input gives its buffer to leading node
        outer_input.shareBuffer(sample_parameter);
        // out gives its buffer to the last node of comiled graph
        
        matrix* owning_node = &output_graph;
        while (owning_node->tape) {
            matrix* borrowed = owning_node->tape->get_borrowed_input();
            if (borrowed) {
                owning_node = borrowed;
            } else {
                break;
            }
        }
        
        if (owning_node->tape && typeid(*owning_node->tape) != typeid(SwapLeafPrimitive) && owning_node->buffer != sample_parameter.buffer) {
            out.shareBuffer(*owning_node);
        } else {
            std::cerr << "CompiledNodePrimitive: Head Tail Collision cant chop" << std::endl;
        }

        output_graph.execute_cpu();
        output_graph.clear_trace_checks();

        
    }
    void eval_metal(matrix &out, EvalType eval_type) override {
        if (outer_input.tape && !outer_input.tape->evaluated) { outer_input.tape->eval_metal(outer_input, eval_type); }
        outer_input.update_from_trace();

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
                out.tape->out_refcount->fetch_add(1, std::memory_order_relaxed);
            }
        }
        if (eval_type == EvalType::COMPILE_TRACE) {return;}
        if (evaluated) {return;} else {evaluated = true;}
        if (!outer_input.metalBuffer && outer_input.buffer) {
            outer_input.buildMetalBuffer();
        }
        
        // outer input gives its buffer to leading node
        outer_input.shareBuffer(sample_parameter);
        // out gives its buffer to the last node of comiled graph
        
        matrix* owning_node = &output_graph;
        while (owning_node->tape) {
            matrix* borrowed = owning_node->tape->get_borrowed_input();
            if (borrowed) {
                owning_node = borrowed;
            } else {
                break;
            }
        }
        
        if (owning_node->tape && typeid(*owning_node->tape) != typeid(SwapLeafPrimitive) && owning_node->buffer != sample_parameter.buffer) {
            out.shareBuffer(*owning_node);
        } else {
            std::cerr << "CompiledNodePrimitive: Head Tail Collision cant chop" << std::endl;
            throw std::runtime_error("CompiledNodePrimitive: Head Tail Collision - cannot chop");
        }

        output_graph.execute_metal();
        output_graph.clear_trace_checks();
    }


    std::vector<matrix> vjp(matrix& grad_out) override {
        // a is vector a.reshape=>c soo J = (dc/da) and for reshape we didnt change anything so its 1 so J = 1 reshaped
        return { grad_out };
    }
    matrix jvp(std::vector<matrix>& tangents) override {
        // a is vector a.reshape=>c soo J = (dc/da) and for reshape we didnt change anything so its 1 so J = 1 reshaped
        return matrix::scalar(1.0f);

    }
// Converts dirty=true into evaluated=false along every path leading to a dirty node,
    // then resets dirty. This is needed because nodes only know their parents, not their
    // children, so the only way to propagate "this subtree needs re-eval" is to walk down
    // from parents and pull the signal up via return value.
    //
    // Memoization (last_visited_pass_id): a shared node (diamond dependency) gets visited
    // once per pass by its first parent, which does the real work and returns the result.
    // Every subsequent parent visiting it in the *same* pass hits the early-return and reads
    // !evaluated instead of recomputing. This is safe specifically because this function
    // never sets evaluated = true — it only ever leaves it untouched or forces it to false.
    // So by the time a second parent visits, whatever the first parent already wrote to
    // `evaluated` is final for this pass, and !evaluated is guaranteed to match what a
    // recompute would have returned.
    //
    // The base-case `return false` (when !dirty && !inv) looks inconsistent with the memo
    // branch's `return !evaluated`, but it can't actually diverge in practice: if evaluated
    // were already false here, invariant (parent can't be evaluated=true if child is
    // evaluated=false) guarantees every ancestor above this node already has evaluated=false
    // too, so this function was never going to flip anything back to true regardless of what
    // it returns. A stale re-invalidation with no exec in between is therefore a true no-op,
    // not a correctness gap.
    bool invalidate_pass(uint64_t current_pass_id) override {
        if (this->last_visited_pass_id == current_pass_id) return !this->evaluated;
        this->last_visited_pass_id = current_pass_id;

        bool inv = false;
        uint64_t new_version = this->version;
        if (outer_input.tape) {
            if (outer_input.tape->invalidate_pass(current_pass_id)) { inv = true; }
            if (outer_input.tape->version > this->version) inv = true;
            new_version = std::max(outer_input.tape->version, new_version);
        }

        if (inv) {
            this->evaluated = false;
            this->version = new_version;
            return true;
        }
        return false;
    }
    void clear_trace_checks() override {
        evaluated = false;
        if (outer_input.tape) {outer_input.tape->clear_trace_checks(); }
    };
    matrix vmap(std::function<matrix(const matrix &)> func, std::vector<int> in_axis) override { return matrix(0, dtype::UInt8); }
    std::vector<matrix> get_inputs() override {
        return {};
    }
};




class MultiInputCompilePrimitive : public Primitive {
public:
    std::vector<matrix> outer_inputs;
    std::vector<matrix> sample_inputs;
    std::vector<matrix> inner_outputs;
    std::vector<matrix> outer_outputs;
    int output_index;
    std::vector<MultiInputCompilePrimitive*> siblings;

    MultiInputCompilePrimitive(std::vector<matrix>& outer_inputs, std::vector<matrix>& sample_inputs, std::vector<matrix>& inner_outputs, std::vector<matrix>& outer_outputs, int output_index)
        : outer_inputs(ensure_graph_ready(outer_inputs)), sample_inputs(sample_inputs), inner_outputs(inner_outputs), outer_outputs(outer_outputs), output_index(output_index) {
    }

    ~MultiInputCompilePrimitive() {
        for (auto sib : siblings) {
            if (sib && sib != this) {
                // Clear the tape pointer in the sibling's outer_outputs so it doesn't dereference this deleted instance
                if (output_index < sib->outer_outputs.size()) {
                    sib->outer_outputs[output_index].tape = nullptr;
                }
                
                // Clear this from sibling's vector
                auto it = std::find(sib->siblings.begin(), sib->siblings.end(), this);
                if (it != sib->siblings.end()) {
                    *it = nullptr;
                }
            }
        }
        
        // Clear all tape pointers in our own outer_outputs so their destructors don't attempt to releaseTape() on our siblings
        for (auto& out_i : outer_outputs) {
            out_i.tape = nullptr;
        }
    }

    void eval_cpu(matrix &out, EvalType eval_type) override {
        for (auto& o_in : outer_inputs) {
            if (o_in.tape && !o_in.tape->evaluated) { o_in.tape->eval_cpu(o_in, eval_type); }
            o_in.update_from_trace();
        }
        
        for (auto& out_i : outer_outputs) {
            if (!out_i.buffer) {
                if (out_i.tape && out_i.tape->out_buffer) {
                    out_i.buffer = out_i.tape->out_buffer;
                    out_i.metalBuffer = out_i.tape->out_metal_buffer;
                    out_i.refCount = out_i.tape->out_refcount;
                    out_i.refCount->fetch_add(1);
                } else if (out_i.tape) {
                    out_i.buffer = new uint8_t[out_i.effectiveBufferSize() * dtype_size(out_i.type)];
                    out_i.begin_refcount();
                    out_i.buildMetalBuffer();
                    out_i.tape->out_buffer = (uint8_t*)out_i.buffer;
                    out_i.tape->out_metal_buffer = out_i.metalBuffer;
                    out_i.tape->out_refcount = out_i.refCount;
                    out_i.tape->out_refcount->fetch_add(1, std::memory_order_relaxed);
                }
            }
        }
        
        if (eval_type == EvalType::COMPILE_TRACE) {return;}

        if (evaluated) {return;} else {
            evaluated = true;
            for (auto sib : siblings) {
                if (sib) sib->evaluated = true;
            }
        }

        for (size_t i = 0; i < outer_inputs.size(); i++) {
            if (!outer_inputs[i].metalBuffer && outer_inputs[i].buffer) {
                outer_inputs[i].buildMetalBuffer();
            }
            outer_inputs[i].shareBuffer(sample_inputs[i]);
        }

        for (size_t i = 0; i < outer_outputs.size(); i++) {
            matrix& inner_out = inner_outputs[i];
            matrix* owning = &inner_out;
            while (owning->tape) {
                matrix* borrowed = owning->tape->get_borrowed_input();
                if (borrowed) owning = borrowed;
                else break;
            }

            if (owning->tape && typeid(*owning->tape) != typeid(SwapLeafPrimitive)) {
                owning->releaseBuffer();
                owning->buffer = outer_outputs[i].buffer;
                owning->metalBuffer = outer_outputs[i].metalBuffer;
                owning->refCount = outer_outputs[i].refCount;
                if (owning->refCount) owning->refCount->fetch_add(1);
            }
        }

        for (auto& inner_out : inner_outputs) {
            inner_out.execute_cpu();
            inner_out.clear_trace_checks();
        }

    }

    void eval_metal(matrix &out, EvalType eval_type) override {
        for (auto& o_in : outer_inputs) {
            if (o_in.tape && !o_in.tape->evaluated) { o_in.tape->eval_metal(o_in, eval_type); }
            o_in.update_from_trace();
        }
        
        for (auto& out_i : outer_outputs) {
            if (!out_i.buffer) {
                if (out_i.tape && out_i.tape->out_buffer) {
                    out_i.buffer = out_i.tape->out_buffer;
                    out_i.metalBuffer = out_i.tape->out_metal_buffer;
                    out_i.refCount = out_i.tape->out_refcount;
                    out_i.refCount->fetch_add(1);
                } else if (out_i.tape) {
                    out_i.buffer = new uint8_t[out_i.effectiveBufferSize() * dtype_size(out_i.type)];
                    out_i.begin_refcount();
                    out_i.buildMetalBuffer();
                    out_i.tape->out_buffer = (uint8_t*)out_i.buffer;
                    out_i.tape->out_metal_buffer = out_i.metalBuffer;
                    out_i.tape->out_refcount = out_i.refCount;
                    out_i.tape->out_refcount->fetch_add(1, std::memory_order_relaxed);
                }
            }
        }
        
        if (eval_type == EvalType::COMPILE_TRACE) {return;}
        
        if (evaluated) {return;} else {
            for (auto sib : siblings) {
                sib->evaluated = true;
            }
        }
        for (size_t i = 0; i < outer_inputs.size(); i++) {
            outer_inputs[i].shareBuffer(sample_inputs[i]);
        }

        for (size_t i = 0; i < outer_outputs.size(); i++) {
            matrix& inner_out = inner_outputs[i];
            matrix* owning = &inner_out;
            while (owning->tape) {
                matrix* borrowed = owning->tape->get_borrowed_input();
                if (borrowed) owning = borrowed;
                else break;
            }

            if (owning->tape && typeid(*owning->tape) != typeid(SwapLeafPrimitive)) {
                owning->releaseBuffer();
                owning->buffer = outer_outputs[i].buffer;
                owning->metalBuffer = outer_outputs[i].metalBuffer;
                owning->refCount = outer_outputs[i].refCount;
                if (owning->refCount) owning->refCount->fetch_add(1);
            }
        }

        for (auto& inner_out : inner_outputs) {
            inner_out.execute_metal();
            inner_out.clear_trace_checks();
        }

    }

    std::vector<matrix> vjp(matrix& grad_out) override { return {}; }
    matrix jvp(std::vector<matrix>& tangents) override { return matrix(0, dtype::UInt8); }
// Converts dirty=true into evaluated=false along every path leading to a dirty node,
    // then resets dirty. This is needed because nodes only know their parents, not their
    // children, so the only way to propagate "this subtree needs re-eval" is to walk down
    // from parents and pull the signal up via return value.
    //
    // Memoization (last_visited_pass_id): a shared node (diamond dependency) gets visited
    // once per pass by its first parent, which does the real work and returns the result.
    // Every subsequent parent visiting it in the *same* pass hits the early-return and reads
    // !evaluated instead of recomputing. This is safe specifically because this function
    // never sets evaluated = true — it only ever leaves it untouched or forces it to false.
    // So by the time a second parent visits, whatever the first parent already wrote to
    // `evaluated` is final for this pass, and !evaluated is guaranteed to match what a
    // recompute would have returned.
    //
    // The base-case `return false` (when !dirty && !inv) looks inconsistent with the memo
    // branch's `return !evaluated`, but it can't actually diverge in practice: if evaluated
    // were already false here, invariant (parent can't be evaluated=true if child is
    // evaluated=false) guarantees every ancestor above this node already has evaluated=false
    // too, so this function was never going to flip anything back to true regardless of what
    // it returns. A stale re-invalidation with no exec in between is therefore a true no-op,
    // not a correctness gap.
    bool invalidate_pass(uint64_t current_pass_id) override {
        if (this->last_visited_pass_id == current_pass_id) return !this->evaluated;
        this->last_visited_pass_id = current_pass_id;

        bool inv = false;
        uint64_t new_version = this->version;
        for (size_t i = 0; i < outer_inputs.size(); i++) {
            if (outer_inputs[i].tape) {
                if (outer_inputs[i].tape->invalidate_pass(current_pass_id)) { inv = true; }
                if (outer_inputs[i].tape->version > this->version) inv = true;
                new_version = std::max(outer_inputs[i].tape->version, new_version);
            }
        }

        if (inv) {
            this->evaluated = false;
            this->version = new_version;
            return true;
        }
        return false;
    }
    void clear_trace_checks() override {
        evaluated = false;
        for (auto& o_in : outer_inputs) {
            if (o_in.tape) { o_in.tape->clear_trace_checks(); }
        }
    }
    matrix vmap(std::function<matrix(const matrix &)> func, std::vector<int> in_axis) override { return matrix(0, dtype::UInt8); }
    std::vector<matrix> get_inputs() override { return outer_inputs; }
};



// ==================== CLAMP PRIMITIVE ====================
class ClampPrimitive : public Primitive {
public:
    matrix input;
    double min_val;
    double max_val;
    CollapsedDims_2 collapsed_dims;
    
    explicit ClampPrimitive(const matrix& input, double min_val, double max_val) : input(ensure_graph_ready(input)), min_val(min_val), max_val(max_val) {
    }

    void eval_cpu(matrix &out, EvalType eval_type) override {
        if (input.tape && !input.tape->evaluated) { input.tape->eval_cpu(input, eval_type); };
        input.update_from_trace();
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
                out.tape->out_refcount->fetch_add(1, std::memory_order_relaxed);
            }
        }
        // ONLY Allocate, don't execute or mark as evaluated!
        if (eval_type == EvalType::COMPILE_TRACE) {return;}

        if (evaluated) {return;} else {evaluated = true;}
        
        input.clamp(out, min_val, max_val, ExecutionDevice::CPU);
    }

    void eval_metal(matrix &out, EvalType eval_type) override {
        if (input.tape && !input.tape->evaluated) { input.tape->eval_metal(input, eval_type); };
        input.update_from_trace();
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
                out.tape->out_refcount->fetch_add(1, std::memory_order_relaxed);
            }
        }
        // ONLY Allocate, don't execute or mark as evaluated!
        if (eval_type == EvalType::COMPILE_TRACE) {return;}

        if (evaluated) {return;} else {evaluated = true;}
        input.clamp(out, min_val, max_val, ExecutionDevice::METAL);
    }

    std::vector<matrix> vjp(matrix& grad_out) override {
        // TODO: User to implement
        return {};
    }

    matrix jvp(std::vector<matrix>& tangents) override {
        // TODO: User to implement
        return matrix(0, dtype::Float);
    }

// Converts dirty=true into evaluated=false along every path leading to a dirty node,
    // then resets dirty. This is needed because nodes only know their parents, not their
    // children, so the only way to propagate "this subtree needs re-eval" is to walk down
    // from parents and pull the signal up via return value.
    //
    // Memoization (last_visited_pass_id): a shared node (diamond dependency) gets visited
    // once per pass by its first parent, which does the real work and returns the result.
    // Every subsequent parent visiting it in the *same* pass hits the early-return and reads
    // !evaluated instead of recomputing. This is safe specifically because this function
    // never sets evaluated = true — it only ever leaves it untouched or forces it to false.
    // So by the time a second parent visits, whatever the first parent already wrote to
    // `evaluated` is final for this pass, and !evaluated is guaranteed to match what a
    // recompute would have returned.
    //
    // The base-case `return false` (when !dirty && !inv) looks inconsistent with the memo
    // branch's `return !evaluated`, but it can't actually diverge in practice: if evaluated
    // were already false here, invariant (parent can't be evaluated=true if child is
    // evaluated=false) guarantees every ancestor above this node already has evaluated=false
    // too, so this function was never going to flip anything back to true regardless of what
    // it returns. A stale re-invalidation with no exec in between is therefore a true no-op,
    // not a correctness gap.
    bool invalidate_pass(uint64_t current_pass_id) override {
        if (this->last_visited_pass_id == current_pass_id) return !this->evaluated;
        this->last_visited_pass_id = current_pass_id;

        bool inv = false;
        uint64_t new_version = this->version;
        if (input.tape) {
            if (input.tape->invalidate_pass(current_pass_id)) { inv = true; }
            if (input.tape->version > this->version) inv = true;
            new_version = std::max(input.tape->version, new_version);
        }

        if (inv) {
            this->evaluated = false;
            this->version = new_version;
            return true;
        }
        return false;
    }
    void clear_trace_checks() override {
        if (!evaluated) return;
        evaluated = false;
        if (input.tape && input.tape->evaluated) { input.tape->clear_trace_checks(); }
    }
    matrix vmap(std::function<matrix(const matrix &)> func, std::vector<int> in_axis) override { return matrix(0, dtype::UInt8); }
    std::vector<matrix> get_inputs() override {
        return {input};
    }
};

// ==================== ABS PRIMITIVE ====================
class AbsPrimitive : public Primitive {
public:
    matrix input;
    CollapsedDims_2 collapsed_dims;
    
    explicit AbsPrimitive(const matrix& input) : input(ensure_graph_ready(input)) {
    }

    void eval_cpu(matrix &out, EvalType eval_type) override {
        if (input.tape && !input.tape->evaluated) { input.tape->eval_cpu(input, eval_type); };
        input.update_from_trace();
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
                out.tape->out_refcount->fetch_add(1, std::memory_order_relaxed);
            }
        }
        // ONLY Allocate, don't execute or mark as evaluated!
        if (eval_type == EvalType::COMPILE_TRACE) {return;}

        if (evaluated) {return;} else {evaluated = true;}
        
        input.abs(out, ExecutionDevice::CPU);
    }

    void eval_metal(matrix &out, EvalType eval_type) override {
        if (input.tape && !input.tape->evaluated) { input.tape->eval_metal(input, eval_type); };
        input.update_from_trace();
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
                out.tape->out_refcount->fetch_add(1, std::memory_order_relaxed);
            }
        }
        // ONLY Allocate, don't execute or mark as evaluated!
        if (eval_type == EvalType::COMPILE_TRACE) {return;}

        if (evaluated) {return;} else {evaluated = true;}
        input.abs(out, ExecutionDevice::METAL);
    }

    std::vector<matrix> vjp(matrix& grad_out) override {
        return { grad_out * (input / matrix::abs(input)) };
    }

    matrix jvp(std::vector<matrix>& tangents) override {
        return tangents[0] * (input / matrix::abs(input));
    }

// Converts dirty=true into evaluated=false along every path leading to a dirty node,
    // then resets dirty. This is needed because nodes only know their parents, not their
    // children, so the only way to propagate "this subtree needs re-eval" is to walk down
    // from parents and pull the signal up via return value.
    //
    // Memoization (last_visited_pass_id): a shared node (diamond dependency) gets visited
    // once per pass by its first parent, which does the real work and returns the result.
    // Every subsequent parent visiting it in the *same* pass hits the early-return and reads
    // !evaluated instead of recomputing. This is safe specifically because this function
    // never sets evaluated = true — it only ever leaves it untouched or forces it to false.
    // So by the time a second parent visits, whatever the first parent already wrote to
    // `evaluated` is final for this pass, and !evaluated is guaranteed to match what a
    // recompute would have returned.
    //
    // The base-case `return false` (when !dirty && !inv) looks inconsistent with the memo
    // branch's `return !evaluated`, but it can't actually diverge in practice: if evaluated
    // were already false here, invariant (parent can't be evaluated=true if child is
    // evaluated=false) guarantees every ancestor above this node already has evaluated=false
    // too, so this function was never going to flip anything back to true regardless of what
    // it returns. A stale re-invalidation with no exec in between is therefore a true no-op,
    // not a correctness gap.
    bool invalidate_pass(uint64_t current_pass_id) override {
        if (this->last_visited_pass_id == current_pass_id) return !this->evaluated;
        this->last_visited_pass_id = current_pass_id;

        bool inv = false;
        uint64_t new_version = this->version;
        if (input.tape) {
            if (input.tape->invalidate_pass(current_pass_id)) { inv = true; }
            if (input.tape->version > this->version) inv = true;
            new_version = std::max(input.tape->version, new_version);
        }

        if (inv) {
            this->evaluated = false;
            this->version = new_version;
            return true;
        }
        return false;
    }
    void clear_trace_checks() override {
        if (!evaluated) return;
        evaluated = false;
        if (input.tape && input.tape->evaluated) { input.tape->clear_trace_checks(); }
    }
    matrix vmap(std::function<matrix(const matrix &)> func, std::vector<int> in_axis) override { return matrix(0, dtype::UInt8); }
    std::vector<matrix> get_inputs() override {
        return {input};
    }
};

// ==================== LOG PRIMITIVE ====================
class LogPrimitive : public Primitive {
public:
    matrix input;
    CollapsedDims_2 collapsed_dims;
    
    explicit LogPrimitive(const matrix& input) : input(ensure_graph_ready(input)) {
    }

    void eval_cpu(matrix &out, EvalType eval_type) override {
        if (input.tape && !input.tape->evaluated) { input.tape->eval_cpu(input, eval_type); };
        input.update_from_trace();
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
                out.tape->out_refcount->fetch_add(1, std::memory_order_relaxed);
            }
        }
        if (eval_type == EvalType::COMPILE_TRACE) {return;}
        if (evaluated) {return;} else {evaluated = true;}
        input.log(out, ExecutionDevice::CPU);
    }

    void eval_metal(matrix &out, EvalType eval_type) override {
        if (input.tape && !input.tape->evaluated) { input.tape->eval_metal(input, eval_type); };
        input.update_from_trace();
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
                out.tape->out_refcount->fetch_add(1, std::memory_order_relaxed);
            }
        }
        if (eval_type == EvalType::COMPILE_TRACE) {return;}
        if (evaluated) {return;} else {evaluated = true;}
        input.log(out, ExecutionDevice::METAL);
    }
    
    matrix vmap(std::function<matrix(const matrix &)> func, std::vector<int> in_axis) override { return matrix(0, dtype::UInt8); }
    std::vector<matrix> get_inputs() override {
        return {input};
    }
    
// Converts dirty=true into evaluated=false along every path leading to a dirty node,
    // then resets dirty. This is needed because nodes only know their parents, not their
    // children, so the only way to propagate "this subtree needs re-eval" is to walk down
    // from parents and pull the signal up via return value.
    //
    // Memoization (last_visited_pass_id): a shared node (diamond dependency) gets visited
    // once per pass by its first parent, which does the real work and returns the result.
    // Every subsequent parent visiting it in the *same* pass hits the early-return and reads
    // !evaluated instead of recomputing. This is safe specifically because this function
    // never sets evaluated = true — it only ever leaves it untouched or forces it to false.
    // So by the time a second parent visits, whatever the first parent already wrote to
    // `evaluated` is final for this pass, and !evaluated is guaranteed to match what a
    // recompute would have returned.
    //
    // The base-case `return false` (when !dirty && !inv) looks inconsistent with the memo
    // branch's `return !evaluated`, but it can't actually diverge in practice: if evaluated
    // were already false here, invariant (parent can't be evaluated=true if child is
    // evaluated=false) guarantees every ancestor above this node already has evaluated=false
    // too, so this function was never going to flip anything back to true regardless of what
    // it returns. A stale re-invalidation with no exec in between is therefore a true no-op,
    // not a correctness gap.
    bool invalidate_pass(uint64_t current_pass_id) override {
        if (this->last_visited_pass_id == current_pass_id) return !this->evaluated;
        this->last_visited_pass_id = current_pass_id;

        bool inv = false;
        uint64_t new_version = this->version;
        if (input.tape) {
            if (input.tape->invalidate_pass(current_pass_id)) { inv = true; }
            if (input.tape->version > this->version) inv = true;
            new_version = std::max(input.tape->version, new_version);
        }

        if (inv) {
            this->evaluated = false;
            this->version = new_version;
            return true;
        }
        return false;
    }
    void clear_trace_checks() override {
        if (!evaluated) return;
        evaluated = false;
        if (input.tape && input.tape->evaluated) { input.tape->clear_trace_checks(); }
    }
    std::vector<matrix> vjp(matrix& grad_out) override {
        // a is vector a.reshape=>c soo J = (dc/da) and for reshape we didnt change anything so its 1 so J = 1 reshaped
        return { grad_out };
    }
    matrix jvp(std::vector<matrix>& tangents) override {
        // a is vector a.reshape=>c soo J = (dc/da) and for reshape we didnt change anything so its 1 so J = 1 reshaped
        return matrix::scalar(1.0f);

    }
};

// ==================== SIN PRIMITIVE ====================
class SinPrimitive : public Primitive {
public:
    matrix input;
    CollapsedDims_2 collapsed_dims;
    explicit SinPrimitive(const matrix& input) : input(ensure_graph_ready(input)) {
    }

    void eval_cpu(matrix &out, EvalType eval_type) override {
        if (input.tape && !input.tape->evaluated) { input.tape->eval_cpu(input, eval_type); };
        input.update_from_trace();
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
                out.tape->out_refcount->fetch_add(1, std::memory_order_relaxed);
            }
        }
        // ONLY Allocate, don't execute or mark as evaluated!
        if (eval_type == EvalType::COMPILE_TRACE) {return;}

        if (evaluated) {return;} else {evaluated = true;}
        input.sin(out, ExecutionDevice::CPU);
    }

    void eval_metal(matrix &out, EvalType eval_type) override {
        if (input.tape && !input.tape->evaluated) { input.tape->eval_metal(input, eval_type); };
        input.update_from_trace();
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
                out.tape->out_refcount->fetch_add(1, std::memory_order_relaxed);
            }
        }
        // ONLY Allocate, don't execute or mark as evaluated!
        if (eval_type == EvalType::COMPILE_TRACE) {return;}

        if (evaluated) {return;} else {evaluated = true;}
        input.sin(out, ExecutionDevice::METAL);
    }

    std::vector<matrix> vjp(matrix& grad_out) override {
        return { grad_out * matrix::cos(input) };
    }

    matrix jvp(std::vector<matrix>& tangents) override {
        return tangents[0] * matrix::cos(input);
    }

// Converts dirty=true into evaluated=false along every path leading to a dirty node,
    // then resets dirty. This is needed because nodes only know their parents, not their
    // children, so the only way to propagate "this subtree needs re-eval" is to walk down
    // from parents and pull the signal up via return value.
    //
    // Memoization (last_visited_pass_id): a shared node (diamond dependency) gets visited
    // once per pass by its first parent, which does the real work and returns the result.
    // Every subsequent parent visiting it in the *same* pass hits the early-return and reads
    // !evaluated instead of recomputing. This is safe specifically because this function
    // never sets evaluated = true — it only ever leaves it untouched or forces it to false.
    // So by the time a second parent visits, whatever the first parent already wrote to
    // `evaluated` is final for this pass, and !evaluated is guaranteed to match what a
    // recompute would have returned.
    //
    // The base-case `return false` (when !dirty && !inv) looks inconsistent with the memo
    // branch's `return !evaluated`, but it can't actually diverge in practice: if evaluated
    // were already false here, invariant (parent can't be evaluated=true if child is
    // evaluated=false) guarantees every ancestor above this node already has evaluated=false
    // too, so this function was never going to flip anything back to true regardless of what
    // it returns. A stale re-invalidation with no exec in between is therefore a true no-op,
    // not a correctness gap.
    bool invalidate_pass(uint64_t current_pass_id) override {
        if (this->last_visited_pass_id == current_pass_id) return !this->evaluated;
        this->last_visited_pass_id = current_pass_id;

        bool inv = false;
        uint64_t new_version = this->version;
        if (input.tape) {
            if (input.tape->invalidate_pass(current_pass_id)) { inv = true; }
            if (input.tape->version > this->version) inv = true;
            new_version = std::max(input.tape->version, new_version);
        }

        if (inv) {
            this->evaluated = false;
            this->version = new_version;
            return true;
        }
        return false;
    }
    void clear_trace_checks() override {
        if (!evaluated) return;
        evaluated = false;
        if (input.tape && input.tape->evaluated) { input.tape->clear_trace_checks(); }
    }
    matrix vmap(std::function<matrix(const matrix &)> func, std::vector<int> in_axis) override { return matrix(0, dtype::UInt8); }
    std::vector<matrix> get_inputs() override {
        return {input};
    }
};

// ==================== COS PRIMITIVE ====================
class CosPrimitive : public Primitive {
public:
    matrix input;
    CollapsedDims_2 collapsed_dims;
    explicit CosPrimitive(const matrix& input) : input(ensure_graph_ready(input)) {
    }

    void eval_cpu(matrix &out, EvalType eval_type) override {
        if (input.tape && !input.tape->evaluated) { input.tape->eval_cpu(input, eval_type); };
        input.update_from_trace();
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
                out.tape->out_refcount->fetch_add(1, std::memory_order_relaxed);
            }
        }
        if (eval_type == EvalType::COMPILE_TRACE) {return;}

        if (evaluated) {return;} else {evaluated = true;}
        input.cos(out, ExecutionDevice::CPU);
    }

    void eval_metal(matrix &out, EvalType eval_type) override {
        if (input.tape && !input.tape->evaluated) { input.tape->eval_metal(input, eval_type); };
        input.update_from_trace();
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
                out.tape->out_refcount->fetch_add(1, std::memory_order_relaxed);
            }
        }
        if (eval_type == EvalType::COMPILE_TRACE) {return;}

        if (evaluated) {return;} else {evaluated = true;}
        input.cos(out, ExecutionDevice::METAL);
    }

    std::vector<matrix> vjp(matrix& grad_out) override {
        return { grad_out * (matrix::sin(input) * -1.0f) };
    }

    matrix jvp(std::vector<matrix>& tangents) override {
        return tangents[0] * (matrix::sin(input) * -1.0f);
    }

// Converts dirty=true into evaluated=false along every path leading to a dirty node,
    // then resets dirty. This is needed because nodes only know their parents, not their
    // children, so the only way to propagate "this subtree needs re-eval" is to walk down
    // from parents and pull the signal up via return value.
    //
    // Memoization (last_visited_pass_id): a shared node (diamond dependency) gets visited
    // once per pass by its first parent, which does the real work and returns the result.
    // Every subsequent parent visiting it in the *same* pass hits the early-return and reads
    // !evaluated instead of recomputing. This is safe specifically because this function
    // never sets evaluated = true — it only ever leaves it untouched or forces it to false.
    // So by the time a second parent visits, whatever the first parent already wrote to
    // `evaluated` is final for this pass, and !evaluated is guaranteed to match what a
    // recompute would have returned.
    //
    // The base-case `return false` (when !dirty && !inv) looks inconsistent with the memo
    // branch's `return !evaluated`, but it can't actually diverge in practice: if evaluated
    // were already false here, invariant (parent can't be evaluated=true if child is
    // evaluated=false) guarantees every ancestor above this node already has evaluated=false
    // too, so this function was never going to flip anything back to true regardless of what
    // it returns. A stale re-invalidation with no exec in between is therefore a true no-op,
    // not a correctness gap.
    bool invalidate_pass(uint64_t current_pass_id) override {
        if (this->last_visited_pass_id == current_pass_id) return !this->evaluated;
        this->last_visited_pass_id = current_pass_id;

        bool inv = false;
        uint64_t new_version = this->version;
        if (input.tape) {
            if (input.tape->invalidate_pass(current_pass_id)) { inv = true; }
            if (input.tape->version > this->version) inv = true;
            new_version = std::max(input.tape->version, new_version);
        }

        if (inv) {
            this->evaluated = false;
            this->version = new_version;
            return true;
        }
        return false;
    }
    void clear_trace_checks() override {
        if (!evaluated) return;
        evaluated = false;
        if (input.tape && input.tape->evaluated) { input.tape->clear_trace_checks(); }
    }
    matrix vmap(std::function<matrix(const matrix &)> func, std::vector<int> in_axis) override { return matrix(0, dtype::UInt8); }
    std::vector<matrix> get_inputs() override {
        return {input};
    }
};

// ==================== TAN PRIMITIVE ====================
class TanPrimitive : public Primitive {
public:
    matrix input;
    CollapsedDims_2 collapsed_dims;
    explicit TanPrimitive(const matrix& input) : input(ensure_graph_ready(input)) {
    }

    void eval_cpu(matrix &out, EvalType eval_type) override {
        if (input.tape && !input.tape->evaluated) { input.tape->eval_cpu(input, eval_type); };
        input.update_from_trace();
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
                out.tape->out_refcount->fetch_add(1, std::memory_order_relaxed);
            }
        }
        if (eval_type == EvalType::COMPILE_TRACE) {return;}

        if (evaluated) {return;} else {evaluated = true;}
        input.tan(out, ExecutionDevice::CPU);
    }

    void eval_metal(matrix &out, EvalType eval_type) override {
        if (input.tape && !input.tape->evaluated) { input.tape->eval_metal(input, eval_type); };
        input.update_from_trace();
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
                out.tape->out_refcount->fetch_add(1, std::memory_order_relaxed);
            }
        }
        if (eval_type == EvalType::COMPILE_TRACE) {return;}

        if (evaluated) {return;} else {evaluated = true;}
        input.tan(out, ExecutionDevice::METAL);
    }

    std::vector<matrix> vjp(matrix& grad_out) override {
        matrix cos_val = matrix::cos(input);
        return { grad_out / (cos_val * cos_val) };
    }

    matrix jvp(std::vector<matrix>& tangents) override {
        matrix cos_val = matrix::cos(input);
        return tangents[0] / (cos_val * cos_val);
    }

// Converts dirty=true into evaluated=false along every path leading to a dirty node,
    // then resets dirty. This is needed because nodes only know their parents, not their
    // children, so the only way to propagate "this subtree needs re-eval" is to walk down
    // from parents and pull the signal up via return value.
    //
    // Memoization (last_visited_pass_id): a shared node (diamond dependency) gets visited
    // once per pass by its first parent, which does the real work and returns the result.
    // Every subsequent parent visiting it in the *same* pass hits the early-return and reads
    // !evaluated instead of recomputing. This is safe specifically because this function
    // never sets evaluated = true — it only ever leaves it untouched or forces it to false.
    // So by the time a second parent visits, whatever the first parent already wrote to
    // `evaluated` is final for this pass, and !evaluated is guaranteed to match what a
    // recompute would have returned.
    //
    // The base-case `return false` (when !dirty && !inv) looks inconsistent with the memo
    // branch's `return !evaluated`, but it can't actually diverge in practice: if evaluated
    // were already false here, invariant (parent can't be evaluated=true if child is
    // evaluated=false) guarantees every ancestor above this node already has evaluated=false
    // too, so this function was never going to flip anything back to true regardless of what
    // it returns. A stale re-invalidation with no exec in between is therefore a true no-op,
    // not a correctness gap.
    bool invalidate_pass(uint64_t current_pass_id) override {
        if (this->last_visited_pass_id == current_pass_id) return !this->evaluated;
        this->last_visited_pass_id = current_pass_id;

        bool inv = false;
        uint64_t new_version = this->version;
        if (input.tape) {
            if (input.tape->invalidate_pass(current_pass_id)) { inv = true; }
            if (input.tape->version > this->version) inv = true;
            new_version = std::max(input.tape->version, new_version);
        }

        if (inv) {
            this->evaluated = false;
            this->version = new_version;
            return true;
        }
        return false;
    }
    void clear_trace_checks() override {
        evaluated = false;
        if (input.tape && input.tape->evaluated) { input.tape->clear_trace_checks(); }
    }
    matrix vmap(std::function<matrix(const matrix &)> func, std::vector<int> in_axis) override { return matrix(0, dtype::UInt8); }
    std::vector<matrix> get_inputs() override {
        return {input};
    }
};

class SqrtPrimitive : public Primitive {
public:
    matrix input;
    CollapsedDims_2 collapsed_dims;
    explicit SqrtPrimitive(const matrix& input) : input(ensure_graph_ready(input)) {
    }

    void eval_cpu(matrix &out, EvalType eval_type) override {
        if (input.tape && !input.tape->evaluated) { input.tape->eval_cpu(input, eval_type); };
        input.update_from_trace();
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
                out.tape->out_refcount->fetch_add(1, std::memory_order_relaxed);
            }
        }
        // ONLY Allocate, don't execute or mark as evaluated!
        if (eval_type == EvalType::COMPILE_TRACE) {return;}

        if (evaluated) {return;} else {evaluated = true;}
        input.sqrt(out, ExecutionDevice::CPU);
    }

    void eval_metal(matrix &out, EvalType eval_type) override {
        if (input.tape && !input.tape->evaluated) { input.tape->eval_metal(input, eval_type); };
        input.update_from_trace();
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
                out.tape->out_refcount->fetch_add(1, std::memory_order_relaxed);
            }
        }
        // ONLY Allocate, don't execute or mark as evaluated!
        if (eval_type == EvalType::COMPILE_TRACE) {return;}

        if (evaluated) {return;} else {evaluated = true;}
        input.sqrt(out, ExecutionDevice::METAL);
    }

    std::vector<matrix> vjp(matrix& grad_out) override {
        return { (1/2) * (1/matrix::sqrt(input)) * grad_out };
    }

    matrix jvp(std::vector<matrix>& tangents) override {
        return tangents[0] * (1/2) * (1/matrix::sqrt(input));
    }

// Converts dirty=true into evaluated=false along every path leading to a dirty node,
    // then resets dirty. This is needed because nodes only know their parents, not their
    // children, so the only way to propagate "this subtree needs re-eval" is to walk down
    // from parents and pull the signal up via return value.
    //
    // Memoization (last_visited_pass_id): a shared node (diamond dependency) gets visited
    // once per pass by its first parent, which does the real work and returns the result.
    // Every subsequent parent visiting it in the *same* pass hits the early-return and reads
    // !evaluated instead of recomputing. This is safe specifically because this function
    // never sets evaluated = true — it only ever leaves it untouched or forces it to false.
    // So by the time a second parent visits, whatever the first parent already wrote to
    // `evaluated` is final for this pass, and !evaluated is guaranteed to match what a
    // recompute would have returned.
    //
    // The base-case `return false` (when !dirty && !inv) looks inconsistent with the memo
    // branch's `return !evaluated`, but it can't actually diverge in practice: if evaluated
    // were already false here, invariant (parent can't be evaluated=true if child is
    // evaluated=false) guarantees every ancestor above this node already has evaluated=false
    // too, so this function was never going to flip anything back to true regardless of what
    // it returns. A stale re-invalidation with no exec in between is therefore a true no-op,
    // not a correctness gap.
    bool invalidate_pass(uint64_t current_pass_id) override {
        if (this->last_visited_pass_id == current_pass_id) return !this->evaluated;
        this->last_visited_pass_id = current_pass_id;

        bool inv = false;
        uint64_t new_version = this->version;
        if (input.tape) {
            if (input.tape->invalidate_pass(current_pass_id)) { inv = true; }
            if (input.tape->version > this->version) inv = true;
            new_version = std::max(input.tape->version, new_version);
        }

        if (inv) {
            this->evaluated = false;
            this->version = new_version;
            return true;
        }
        return false;
    }
    void clear_trace_checks() override {
        evaluated = false;
        if (input.tape && input.tape->evaluated) { input.tape->clear_trace_checks(); }
    }
    matrix vmap(std::function<matrix(const matrix &)> func, std::vector<int> in_axis) override { return matrix(0, dtype::UInt8); }
    std::vector<matrix> get_inputs() override {
        return {input};
    }
};

class ExpPrimitive : public Primitive {
public:
    matrix input;
    CollapsedDims_2 collapsed_dims;
    explicit ExpPrimitive(const matrix& input) : input(ensure_graph_ready(input)) {
    }

    void eval_cpu(matrix &out, EvalType eval_type) override {
        if (input.tape && !input.tape->evaluated) { input.tape->eval_cpu(input, eval_type); };
        input.update_from_trace();
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
                out.tape->out_refcount->fetch_add(1, std::memory_order_relaxed);
            }
        }
        // ONLY Allocate, don't execute or mark as evaluated!
        if (eval_type == EvalType::COMPILE_TRACE) {return;}

        if (evaluated) {return;} else {evaluated = true;}
        input.exp(out, ExecutionDevice::CPU);
    }

    void eval_metal(matrix &out, EvalType eval_type) override {
        if (input.tape && !input.tape->evaluated) { input.tape->eval_metal(input, eval_type); };
        input.update_from_trace();
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
                out.tape->out_refcount->fetch_add(1, std::memory_order_relaxed);
            }
        }
        // ONLY Allocate, don't execute or mark as evaluated!
        if (eval_type == EvalType::COMPILE_TRACE) {return;}

        if (evaluated) {return;} else {evaluated = true;}
        input.exp(out, ExecutionDevice::METAL);
    }

    std::vector<matrix> vjp(matrix& grad_out) override {
        return { grad_out * matrix::exp(input) };
    }

    matrix jvp(std::vector<matrix>& tangents) override {
        return tangents[0] * matrix::exp(input);
    }

// Converts dirty=true into evaluated=false along every path leading to a dirty node,
    // then resets dirty. This is needed because nodes only know their parents, not their
    // children, so the only way to propagate "this subtree needs re-eval" is to walk down
    // from parents and pull the signal up via return value.
    //
    // Memoization (last_visited_pass_id): a shared node (diamond dependency) gets visited
    // once per pass by its first parent, which does the real work and returns the result.
    // Every subsequent parent visiting it in the *same* pass hits the early-return and reads
    // !evaluated instead of recomputing. This is safe specifically because this function
    // never sets evaluated = true — it only ever leaves it untouched or forces it to false.
    // So by the time a second parent visits, whatever the first parent already wrote to
    // `evaluated` is final for this pass, and !evaluated is guaranteed to match what a
    // recompute would have returned.
    //
    // The base-case `return false` (when !dirty && !inv) looks inconsistent with the memo
    // branch's `return !evaluated`, but it can't actually diverge in practice: if evaluated
    // were already false here, invariant (parent can't be evaluated=true if child is
    // evaluated=false) guarantees every ancestor above this node already has evaluated=false
    // too, so this function was never going to flip anything back to true regardless of what
    // it returns. A stale re-invalidation with no exec in between is therefore a true no-op,
    // not a correctness gap.
    bool invalidate_pass(uint64_t current_pass_id) override {
        if (this->last_visited_pass_id == current_pass_id) return !this->evaluated;
        this->last_visited_pass_id = current_pass_id;

        bool inv = false;
        uint64_t new_version = this->version;
        if (input.tape) {
            if (input.tape->invalidate_pass(current_pass_id)) { inv = true; }
            if (input.tape->version > this->version) inv = true;
            new_version = std::max(input.tape->version, new_version);
        }

        if (inv) {
            this->evaluated = false;
            this->version = new_version;
            return true;
        }
        return false;
    }
    void clear_trace_checks() override {
        evaluated = false;
        if (input.tape && input.tape->evaluated) { input.tape->clear_trace_checks(); }
    }
    matrix vmap(std::function<matrix(const matrix &)> func, std::vector<int> in_axis) override { return matrix(0, dtype::UInt8); }
    std::vector<matrix> get_inputs() override {
        return {input};
    }
};

class DotPrimitive : public Primitive {
public:
    matrix a;
    matrix b_transposed;
    BroadcastDescriptor* desc_a = nullptr;
    BroadcastDescriptor* desc_b = nullptr;
    ~DotPrimitive() {
        if (desc_a) BroadcastDescriptor::destroy(desc_a);
        if (desc_b) BroadcastDescriptor::destroy(desc_b);
    }
    CollapsedDims_3 collapsed_dims_3;

    DotPrimitive(const matrix& a, const matrix& b_transposed)
        : a(ensure_graph_ready(a)), b_transposed(ensure_graph_ready(b_transposed)) {
    }

    void eval_cpu(matrix &out, EvalType eval_type) override {
        if (a.tape && !a.tape->evaluated) { a.tape->eval_cpu(a, eval_type); }
        if (b_transposed.tape && !b_transposed.tape->evaluated) { b_transposed.tape->eval_cpu(b_transposed, eval_type); }
        a.update_from_trace();
        b_transposed.update_from_trace();

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
                out.tape->out_refcount->fetch_add(1, std::memory_order_relaxed);
            }
        }
        if (eval_type == EvalType::COMPILE_TRACE) return;
        if (evaluated) return; else evaluated = true;

        a.dot_cpu(b_transposed, out);
    }

    void eval_metal(matrix &out, EvalType eval_type) override {
        if (a.tape && !a.tape->evaluated) { a.tape->eval_metal(a, eval_type); }
        if (b_transposed.tape && !b_transposed.tape->evaluated) { b_transposed.tape->eval_metal(b_transposed, eval_type); }
        a.update_from_trace();
        b_transposed.update_from_trace();

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
                out.tape->out_refcount->fetch_add(1, std::memory_order_relaxed);
            }
        }
        if (eval_type == EvalType::COMPILE_TRACE) return;
        if (evaluated) return; else evaluated = true;

        a.dot_gpu(b_transposed, out);
    }

    std::vector<matrix> vjp(matrix& grad_out) override {
        // Y = A @ B_transposed^T
        // grad_A = grad_out @ B_transposed   (this directly fits our dot signature!)
        matrix grad_a = grad_out.dot(b_transposed, false);

        // grad_B_transposed = grad_out^T @ A
        std::vector<size_m> axes(grad_out.dims);
        int stop_idx = grad_out.dims >= 2 ? grad_out.dims - 2 : grad_out.dims;
        for (int i = 0; i < stop_idx; ++i) axes[i] = i;
        if (grad_out.dims >= 2) {
            axes[grad_out.dims - 2] = grad_out.dims - 1;
            axes[grad_out.dims - 1] = grad_out.dims - 2;
        }
        matrix grad_b = grad_out.transpose(axes).dot(a, false);
        
        return {grad_a.unbroadcast_shape(a.shape(), a.dims), grad_b.unbroadcast_shape(b_transposed.shape(), b_transposed.dims)};
    }

    matrix jvp(std::vector<matrix>& tangents) override {
        return tangents[0].dot(b_transposed, true) + a.dot(tangents[1], true);
    }

// Converts dirty=true into evaluated=false along every path leading to a dirty node,
    // then resets dirty. This is needed because nodes only know their parents, not their
    // children, so the only way to propagate "this subtree needs re-eval" is to walk down
    // from parents and pull the signal up via return value.
    //
    // Memoization (last_visited_pass_id): a shared node (diamond dependency) gets visited
    // once per pass by its first parent, which does the real work and returns the result.
    // Every subsequent parent visiting it in the *same* pass hits the early-return and reads
    // !evaluated instead of recomputing. This is safe specifically because this function
    // never sets evaluated = true — it only ever leaves it untouched or forces it to false.
    // So by the time a second parent visits, whatever the first parent already wrote to
    // `evaluated` is final for this pass, and !evaluated is guaranteed to match what a
    // recompute would have returned.
    //
    // The base-case `return false` (when !dirty && !inv) looks inconsistent with the memo
    // branch's `return !evaluated`, but it can't actually diverge in practice: if evaluated
    // were already false here, invariant (parent can't be evaluated=true if child is
    // evaluated=false) guarantees every ancestor above this node already has evaluated=false
    // too, so this function was never going to flip anything back to true regardless of what
    // it returns. A stale re-invalidation with no exec in between is therefore a true no-op,
    // not a correctness gap.
    bool invalidate_pass(uint64_t current_pass_id) override {
        if (this->last_visited_pass_id == current_pass_id) return !this->evaluated;
        this->last_visited_pass_id = current_pass_id;

        bool inv = false;
        uint64_t new_version = this->version;
        if (a.tape) {
            if (a.tape->invalidate_pass(current_pass_id)) { inv = true; }
            if (a.tape->version > this->version) inv = true;
            new_version = std::max(a.tape->version, new_version);
        }
        if (b_transposed.tape) {
            if (b_transposed.tape->invalidate_pass(current_pass_id)) { inv = true; }
            if (b_transposed.tape->version > this->version) inv = true;
            new_version = std::max(b_transposed.tape->version, new_version);
        }

        if (inv) {
            this->evaluated = false;
            this->version = new_version;
            return true;
        }
        return false;
    }
    void clear_trace_checks() override {
        evaluated = false;
        if (a.tape) { a.tape->clear_trace_checks(); }
        if (b_transposed.tape) { b_transposed.tape->clear_trace_checks(); }
    }
    matrix vmap(std::function<matrix(const matrix &)> func, std::vector<int> in_axis) override { return matrix(0, dtype::UInt8); }
    std::vector<matrix> get_inputs() override { return {a, b_transposed}; }
};

class ExecutionBoundaryPrimitive : public Primitive {
public:
    matrix input;
    std::function<void(matrix&)> lambda;
    ExecutionDevice boundary_exec_device;

    ExecutionBoundaryPrimitive(matrix& input, std::function<void(matrix&)> lambda, ExecutionDevice exec_device)
        : input(input), lambda(lambda), boundary_exec_device(exec_device) {}

    void eval_cpu(matrix& out, EvalType eval_type) override {
        if (input.tape && !input.tape->evaluated) { 
            if (boundary_exec_device == ExecutionDevice::METAL) {
                input.tape->eval_metal(input, eval_type);
            } else {
                input.tape->eval_cpu(input, eval_type);
            }
        };
        input.update_from_trace();

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
                out.tape->out_refcount->fetch_add(1, std::memory_order_relaxed);
            }
        }

        if (eval_type == EvalType::COMPILE_TRACE) {return;}

        if (evaluated) {return;} else {evaluated = true;}
        
        matrix::copyCPUinplace(out, input, 0);

        // out.eval_cpu(); copyInplace cpu always evaluates instantly
        if (lambda) lambda(out);
    }

    void eval_metal(matrix& out, EvalType eval_type) override {
        if (input.tape && !input.tape->evaluated) { 
            if (boundary_exec_device == ExecutionDevice::METAL) {
                input.tape->eval_metal(input, eval_type);
            } else {
                input.tape->eval_cpu(input, eval_type);
            }
        };
        input.update_from_trace();

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
                out.tape->out_refcount->fetch_add(1, std::memory_order_relaxed);
            }
        }

        if (eval_type == EvalType::COMPILE_TRACE) {return;}

        if (evaluated) {return;} else {evaluated = true;}

        if (lambda) {
            matrix::copyGPUinplace(out, input, 0, Execution::EncodeAndExecute);
            lambda(out);
        } else {
            matrix::copyGPUinplace(out, input, 0);
        }
    }

    std::vector<matrix> vjp(matrix& grad_out) override {
        return {grad_out};
    }
    
    matrix jvp(std::vector<matrix>& tangents) override {
        return tangents[0];
    }
    
    std::vector<matrix> get_inputs() override {
        return {input};
    }
    
// Converts dirty=true into evaluated=false along every path leading to a dirty node,
    // then resets dirty. This is needed because nodes only know their parents, not their
    // children, so the only way to propagate "this subtree needs re-eval" is to walk down
    // from parents and pull the signal up via return value.
    //
    // Memoization (last_visited_pass_id): a shared node (diamond dependency) gets visited
    // once per pass by its first parent, which does the real work and returns the result.
    // Every subsequent parent visiting it in the *same* pass hits the early-return and reads
    // !evaluated instead of recomputing. This is safe specifically because this function
    // never sets evaluated = true — it only ever leaves it untouched or forces it to false.
    // So by the time a second parent visits, whatever the first parent already wrote to
    // `evaluated` is final for this pass, and !evaluated is guaranteed to match what a
    // recompute would have returned.
    //
    // The base-case `return false` (when !dirty && !inv) looks inconsistent with the memo
    // branch's `return !evaluated`, but it can't actually diverge in practice: if evaluated
    // were already false here, invariant (parent can't be evaluated=true if child is
    // evaluated=false) guarantees every ancestor above this node already has evaluated=false
    // too, so this function was never going to flip anything back to true regardless of what
    // it returns. A stale re-invalidation with no exec in between is therefore a true no-op,
    // not a correctness gap.
    bool invalidate_pass(uint64_t current_pass_id) override {
        if (this->last_visited_pass_id == current_pass_id) return !this->evaluated;
        this->last_visited_pass_id = current_pass_id;

        bool inv = false;
        uint64_t new_version = this->version;
        if (input.tape) {
            if (input.tape->invalidate_pass(current_pass_id)) { inv = true; }
            if (input.tape->version > this->version) inv = true;
            new_version = std::max(input.tape->version, new_version);
        }

        if (inv) {
            this->evaluated = false;
            this->version = new_version;
            return true;
        }
        return false;
    }
    
    void clear_trace_checks() override {
        evaluated = false;
        if (input.tape) input.tape->clear_trace_checks();
    }
    matrix vmap(std::function<matrix(const matrix &)> func, std::vector<int> in_axis) override { return matrix(0, dtype::UInt8); }
};

class TakePrimitive : public Primitive {
public:
    matrix source;
    matrix indices;
    int axis;

    TakePrimitive(const matrix& source_in, const matrix& indices_in, int axis_in)
        : source(source_in), indices(indices_in), axis(axis_in) {}

    matrix* get_borrowed_input() override { return nullptr; } // Allocates new memory
    
    void eval_cpu(matrix &out, EvalType eval_type) override {
        if (source.tape && !source.tape->evaluated) { source.tape->eval_cpu(source, eval_type); }
        if (indices.tape && !indices.tape->evaluated) { indices.tape->eval_cpu(indices, eval_type); }
        source.update_from_trace();
        indices.update_from_trace();

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
                out.tape->out_refcount->fetch_add(1, std::memory_order_relaxed);
            }
        }
        if (eval_type == EvalType::COMPILE_TRACE) return;
        if (evaluated) return; else evaluated = true;

        source.take_backend(indices, out, axis, ExecutionDevice::CPU);
    }

    void eval_metal(matrix &out, EvalType eval_type) override {
        if (source.tape && !source.tape->evaluated) { source.tape->eval_metal(source, eval_type); }
        if (indices.tape && !indices.tape->evaluated) { indices.tape->eval_metal(indices, eval_type); }
        source.update_from_trace();
        indices.update_from_trace();

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
                out.tape->out_refcount->fetch_add(1, std::memory_order_relaxed);
            }
        }
        if (eval_type == EvalType::COMPILE_TRACE) return;
        if (evaluated) return; else evaluated = true;

        source.take_backend(indices, out, axis, ExecutionDevice::METAL);
    }
    
    std::vector<matrix> vjp(matrix& grad_out) override {
        return {grad_out};
    }
    
    matrix jvp(std::vector<matrix>& tangents) override {
        return tangents[0];
    }
    
// Converts dirty=true into evaluated=false along every path leading to a dirty node,
    // then resets dirty. This is needed because nodes only know their parents, not their
    // children, so the only way to propagate "this subtree needs re-eval" is to walk down
    // from parents and pull the signal up via return value.
    //
    // Memoization (last_visited_pass_id): a shared node (diamond dependency) gets visited
    // once per pass by its first parent, which does the real work and returns the result.
    // Every subsequent parent visiting it in the *same* pass hits the early-return and reads
    // !evaluated instead of recomputing. This is safe specifically because this function
    // never sets evaluated = true — it only ever leaves it untouched or forces it to false.
    // So by the time a second parent visits, whatever the first parent already wrote to
    // `evaluated` is final for this pass, and !evaluated is guaranteed to match what a
    // recompute would have returned.
    //
    // The base-case `return false` (when !dirty && !inv) looks inconsistent with the memo
    // branch's `return !evaluated`, but it can't actually diverge in practice: if evaluated
    // were already false here, invariant (parent can't be evaluated=true if child is
    // evaluated=false) guarantees every ancestor above this node already has evaluated=false
    // too, so this function was never going to flip anything back to true regardless of what
    // it returns. A stale re-invalidation with no exec in between is therefore a true no-op,
    // not a correctness gap.
    bool invalidate_pass(uint64_t current_pass_id) override {
        if (this->last_visited_pass_id == current_pass_id) return !this->evaluated;
        this->last_visited_pass_id = current_pass_id;

        bool inv = false;
        uint64_t new_version = this->version;
        if (source.tape) {
            if (source.tape->invalidate_pass(current_pass_id)) { inv = true; }
            if (source.tape->version > this->version) inv = true;
            new_version = std::max(source.tape->version, new_version);
        }
        if (indices.tape) {
            if (indices.tape->invalidate_pass(current_pass_id)) { inv = true; }
            if (indices.tape->version > this->version) inv = true;
            new_version = std::max(indices.tape->version, new_version);
        }

        if (inv) {
            this->evaluated = false;
            this->version = new_version;
            return true;
        }
        return false;
    }
    
    void clear_trace_checks() override {
        evaluated = false;
        if (source.tape) source.tape->clear_trace_checks();
        if (indices.tape) indices.tape->clear_trace_checks();
    }
    
    matrix vmap(std::function<matrix(const matrix &)> func, std::vector<int> in_axis) override {
        return matrix(0, dtype::UInt8);
    }
    
    std::vector<matrix> get_inputs() override {
        return {source, indices};
    }
};

class CrossPrimitive : public Primitive {
public:
    matrix a;
    matrix b;
    BroadcastDescriptor* desc_a = nullptr;
    BroadcastDescriptor* desc_b = nullptr;
    CollapsedDims_3 collapsed_dims_3;
    
    ~CrossPrimitive() {
        if (desc_a) BroadcastDescriptor::destroy(desc_a);
        if (desc_b) BroadcastDescriptor::destroy(desc_b);
    }
    
    CrossPrimitive(const matrix& a, const matrix& b) : a(ensure_graph_ready(a)), b(ensure_graph_ready(b)) {
    }
    
    void eval_cpu(matrix &out, EvalType eval_type) override {
        if (a.tape && !a.tape->evaluated) { a.tape->eval_cpu(a, eval_type); }
        if (b.tape && !b.tape->evaluated) { b.tape->eval_cpu(b, eval_type); }
        a.update_from_trace();
        b.update_from_trace();
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
                out.tape->out_refcount->fetch_add(1, std::memory_order_relaxed);
            }
        }
        if (eval_type == EvalType::COMPILE_TRACE) return;
        if (evaluated) return; else evaluated = true;
        a.cross_cpu_brodcasted(b, out);
    }
    void eval_metal(matrix &out, EvalType eval_type) override {
        if (a.tape && !a.tape->evaluated) { a.tape->eval_metal(a, eval_type); }
        if (b.tape && !b.tape->evaluated) { b.tape->eval_metal(b, eval_type); }
        a.update_from_trace();
        b.update_from_trace();
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
                out.tape->out_refcount->fetch_add(1, std::memory_order_relaxed);
            }
        }
        if (eval_type == EvalType::COMPILE_TRACE) return;
        if (evaluated) return; else evaluated = true;
        a.cross_gpu_brodcasted(b, out);
    }
    std::vector<matrix> vjp(matrix& grad_out) override {
        // grad_a = b x grad_out
        // grad_b = grad_out x a
        return {matrix::cross(b, grad_out, -1), matrix::cross(grad_out, a, -1)};
    }
    matrix jvp(std::vector<matrix>& tangents) override {
        // d(a x b) = da x b + a x db
        return matrix::cross(tangents[0], b, -1) + matrix::cross(a, tangents[1], -1);
    }
// Converts dirty=true into evaluated=false along every path leading to a dirty node,
    // then resets dirty. This is needed because nodes only know their parents, not their
    // children, so the only way to propagate "this subtree needs re-eval" is to walk down
    // from parents and pull the signal up via return value.
    //
    // Memoization (last_visited_pass_id): a shared node (diamond dependency) gets visited
    // once per pass by its first parent, which does the real work and returns the result.
    // Every subsequent parent visiting it in the *same* pass hits the early-return and reads
    // !evaluated instead of recomputing. This is safe specifically because this function
    // never sets evaluated = true — it only ever leaves it untouched or forces it to false.
    // So by the time a second parent visits, whatever the first parent already wrote to
    // `evaluated` is final for this pass, and !evaluated is guaranteed to match what a
    // recompute would have returned.
    //
    // The base-case `return false` (when !dirty && !inv) looks inconsistent with the memo
    // branch's `return !evaluated`, but it can't actually diverge in practice: if evaluated
    // were already false here, invariant (parent can't be evaluated=true if child is
    // evaluated=false) guarantees every ancestor above this node already has evaluated=false
    // too, so this function was never going to flip anything back to true regardless of what
    // it returns. A stale re-invalidation with no exec in between is therefore a true no-op,
    // not a correctness gap.
    bool invalidate_pass(uint64_t current_pass_id) override {
        if (this->last_visited_pass_id == current_pass_id) return !this->evaluated;
        this->last_visited_pass_id = current_pass_id;

        bool inv = false;
        uint64_t new_version = this->version;
        if (a.tape) {
            if (a.tape->invalidate_pass(current_pass_id)) { inv = true; }
            if (a.tape->version > this->version) inv = true;
            new_version = std::max(a.tape->version, new_version);
        }
        if (b.tape) {
            if (b.tape->invalidate_pass(current_pass_id)) { inv = true; }
            if (b.tape->version > this->version) inv = true;
            new_version = std::max(b.tape->version, new_version);
        }

        if (inv) {
            this->evaluated = false;
            this->version = new_version;
            return true;
        }
        return false;
    }
    void clear_trace_checks() override {
        evaluated = false;
        if (a.tape) { a.tape->clear_trace_checks(); }
        if (b.tape) { b.tape->clear_trace_checks(); }
    }
    matrix vmap(std::function<matrix(const matrix &)> func, std::vector<int> in_axis) override { return matrix(0, dtype::UInt8); }
    
    std::vector<matrix> get_inputs() override {
        return {a, b};
    }
};

class LeafPrimitive : public Primitive {
public:
    LeafPrimitive(matrix& output) {
        out_buffer = (uint8_t*)output.buffer;
        out_metal_buffer = output.metalBuffer;
        out_refcount = output.refCount;
        if (out_refcount) out_refcount->fetch_add(1, std::memory_order_relaxed);
    }
    void eval_cpu(matrix &out, EvalType eval_type) override { evaluated = true; }
    void eval_metal(matrix &out, EvalType eval_type) override { evaluated = true; }
    std::vector<matrix> vjp(matrix& grad_out) override {
        return { grad_out };
    }
    
    matrix jvp(std::vector<matrix>& tangents) override {
        return matrix::scalar(1.0f);
    }
    
// Converts dirty=true into evaluated=false along every path leading to a dirty node,
    // then resets dirty. This is needed because nodes only know their parents, not their
    // children, so the only way to propagate "this subtree needs re-eval" is to walk down
    // from parents and pull the signal up via return value.
    //
    // Memoization (last_visited_pass_id): a shared node (diamond dependency) gets visited
    // once per pass by its first parent, which does the real work and returns the result.
    // Every subsequent parent visiting it in the *same* pass hits the early-return and reads
    // !evaluated instead of recomputing. This is safe specifically because this function
    // never sets evaluated = true — it only ever leaves it untouched or forces it to false.
    // So by the time a second parent visits, whatever the first parent already wrote to
    // `evaluated` is final for this pass, and !evaluated is guaranteed to match what a
    // recompute would have returned.
    //
    // The base-case `return false` (when !dirty && !inv) looks inconsistent with the memo
    // branch's `return !evaluated`, but it can't actually diverge in practice: if evaluated
    // were already false here, invariant (parent can't be evaluated=true if child is
    // evaluated=false) guarantees every ancestor above this node already has evaluated=false
    // too, so this function was never going to flip anything back to true regardless of what
    // it returns. A stale re-invalidation with no exec in between is therefore a true no-op,
    // not a correctness gap.
    bool invalidate_pass(uint64_t current_pass_id) override {
        if (this->last_visited_pass_id == current_pass_id) return !this->evaluated;
        this->last_visited_pass_id = current_pass_id;

        bool inv = false;
        uint64_t new_version = this->version;

        if (inv) {
            this->evaluated = false;
            this->version = new_version;
            return true;
        }
        return false;
    }
    
    void clear_trace_checks() override {
        evaluated = false;
    }
    
    matrix vmap(std::function<matrix(const matrix &)> func, std::vector<int> in_axis) override { return matrix(0, dtype::UInt8); }
    
    std::vector<matrix> get_inputs() override {
        return {};
    }
};
