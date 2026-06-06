//
// Created by Aditya Dudeja on 06/06/26.
//

#include <vector>
#include <iostream>
#include "matrix.mm"
int main() {}

class matrix;

class Primitive {
    public:
    virtual ~Primitive() = default;
    virtual void eval_cpu(matrix& out) = 0;
    virtual void eval_metal(matrix& out) { eval_cpu(out); }
    virtual void std::vector<matrix> vjp(const matrix& grad_out) = 0;
    virtual matrix jvp(const std::vector<matrix>& tangents) = 0;
};

