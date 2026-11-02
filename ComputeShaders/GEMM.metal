////
////  GEMM.metal
////  AdityaIntelligenceProMax
////
////  Created by Aditya Dudeja on 30/10/26.
////
//
//#include <metal_stdlib>
//#include "Utils.h"
//
//using namespace metal;
//
//constant constexpr ushort TileW = 64;
//constant constexpr ushort TileH = 32;
//
//constant constexpr ushort Sx = 2;
//constant constexpr ushort Sy = 2;
//
//struct GEMM_param {
//    int M;
//    int N;
//    int K;
//};
//
//[[kernel]] gemm_(device float* C [[buffer(0)]], constant float* A [[buffer(1)]], constant float* B [[buffer(2)]], constant GEMM_param& param [[buffer(3)]]
//                 ushort2 tid [[]])
