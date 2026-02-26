//
//  custom_kernels.metal
//  AdityaIntelligenceProMax
//
//  Created by Aditya Dudeja on 20/11/25.
//

#include <metal_stdlib>
#include "Utils.h"

using namespace metal;


kernel void custom_kernel1(device uint8_t* outMat [[buffer(0)]], device const uint8_t* inMat [[buffer(1)]], constant size_m* shapeOut [[buffer(2)]], constant size_m* stridesOut [[buffer(3)]], constant size_m* shapeIn [[buffer(4)]], constant size_m* stridesIn [[buffer(5)]], constant float& t [[buffer(6)]], uint gid [[thread_position_in_grid]]) {
    int yIndex = gid / shapeIn[1];
    int xIndex = gid % shapeIn[1];
    float x = (float)xIndex / (float)shapeIn[1];
    float y = (float)yIndex / (float)shapeIn[0];
    
    float2 FC = float2((float)xIndex, (float)yIndex);
    float2 r  = float2((float)shapeIn[1], (float)shapeIn[0]);

    float4 out = float4(0.0);
    float2 l = float2(0.0);

    // p = (FC*2 - r) / r.y
    float2 p = (FC * 2.0 - r) / r.y;

    // l += abs(.7 - dot(p,p))
    float scalar = fabs(0.7 - dot(p, p));
    l += scalar;

    // v = p * (1 - l) / .2
    float2 v = p * (1.0 - l) / 0.2;


    // ---- Loop matches GLSL: i starts at 0, but first iteration uses i=1 ----
    for (float i = 0.0; ; ) {

        i += 1.0;
        if (i > 8.0) break;

        // out += (sin(v.xyyx)+1.) * abs(v.x-v.y) * .2
        float4 v_xyyx = float4(v.x, v.y, v.y, v.x);
        out += (sin(v_xyyx) + 1.0) * fabs(v.x - v.y) * 0.2;

        // v += cos(v.yx*i + float2(0,i) + t)/i + .7
        v += cos(v.yx * i + float2(0.0, i) + t) / i + 0.7;
    }


    // ---- FINAL COLOR (the GLSL does this AFTER loop) ----
    float4 denom = out + 1e-6;   // avoid NaN
    out = tanh(exp(p.y * float4(1, -1, -2, 0)) * exp(-4.0 * l.x) / denom);


    // clamp to [0,1]
    out = saturate(out);

    // write final 8-bit RGBA
    outMat[yIndex * stridesOut[0] + xIndex * stridesOut[1] + 0] = uint8_t(out.z * 255.0f);
    outMat[yIndex * stridesOut[0] + xIndex * stridesOut[1] + 1] = uint8_t(out.y * 255.0f);
    outMat[yIndex * stridesOut[0] + xIndex * stridesOut[1] + 2] = uint8_t(out.x * 255.0f);
    outMat[yIndex * stridesOut[0] + xIndex * stridesOut[1] + 3] = uint8_t(255);
//
//    outMat[4 * gid + 0] = 255;
//    outMat[4 * gid + 1] = 0;
//    outMat[4 * gid + 2] = 0;
//    outMat[4 * gid + 3] = 255;
}

kernel void custom_kernel2(device uint8_t* outMat [[buffer(0)]], device const uint8_t* inMat [[buffer(1)]], constant size_m* shapeOut [[buffer(2)]], constant size_m* stridesOut [[buffer(3)]], constant size_m* shapeIn [[buffer(4)]], constant size_m* stridesIn [[buffer(5)]], constant float& t [[buffer(6)]], uint gid [[thread_position_in_grid]]) {
    int yIndex = gid / shapeIn[1];
    int xIndex = gid % shapeIn[1];
    float x = (float)xIndex / (float)shapeIn[1];
    float y = (float)yIndex / (float)shapeIn[0];
    
    float3 FC = float3((float)xIndex, (float)yIndex, 0);
    float2 r  = float2((float)shapeIn[1], (float)shapeIn[0]);

    float4 o = float4(0.0);

    // GLSL had: for(float z,d,i; i++<1e2; ...)
    float z = 0.0;
    float d = 0.0;
    float i = 0.0;

    for (;;) {

        i += 1.0;
        if (i > 100.0) break;

        // vec3 p = z * normalize(FC.rgb*2.-r.xyx);
        float3 p = z * normalize(FC * 2.0 - float3(r.x, r.y, r.x));

        // p = vec3(atan(p.y,p.x)*2., p.z/3., length(p.xy)-6.);
        p = float3(
            atan2(p.y, p.x) * 2.0,
            p.z / 3.0,
            length(p.xy) - 6.0
        );

        // inner fractal-like loop
        d = 1.0;
        for (; d < 9.0; d += 1.0) {
            p += sin(p.yzx * d - t + 0.2 * i) / d;
        }

        // z += d = .2 * length(vec4(.1*cos(p*3.)-.1, p.z));
        {
            float4 V = float4(0.1 * cos(p * 3.0) - 0.1, p.z);
            float d_new = 0.2 * length(V);
            d = d_new;
            z += d_new;
        }

        // o += (1. + cos(i*.7+t+vec4(6,1,2,0))) / d / i
        float4 addTerm = (1.0 + cos(i * 0.7 + t + float4(6, 1, 2, 0))) / (d * i);
        o += addTerm;
    }

    // final output
    o = tanh(o * o / 900.0);   // 9e2 == 900

    o = saturate(o);

    // write final 8-bit RGBA
    outMat[yIndex * stridesOut[0] + xIndex * stridesOut[1] + 0] = uint8_t(o.z * 255.0f);
    outMat[yIndex * stridesOut[0] + xIndex * stridesOut[1] + 1] = uint8_t(o.y * 255.0f);
    outMat[yIndex * stridesOut[0] + xIndex * stridesOut[1] + 2] = uint8_t(o.x * 255.0f);
    outMat[yIndex * stridesOut[0] + xIndex * stridesOut[1] + 3] = uint8_t(255);
}

float3x3 rotate3D(float angle, float3 axis)
{
    float3 a = normalize(axis);
    float s = sin(angle);
    float c = cos(angle);
    float ic = 1.0 - c;

    return float3x3(
        ic * a.x * a.x + c,         ic * a.x * a.y - a.z * s,   ic * a.x * a.z + a.y * s,
        ic * a.y * a.x + a.z * s,   ic * a.y * a.y + c,         ic * a.y * a.z - a.x * s,
        ic * a.z * a.x - a.y * s,   ic * a.z * a.y + a.x * s,   ic * a.z * a.z + c
    );
}


kernel void custom_kernel3(device uint8_t* outMat [[buffer(0)]], device const uint8_t* inMat [[buffer(1)]], constant size_m* shapeOut [[buffer(2)]], constant size_m* stridesOut [[buffer(3)]], constant size_m* shapeIn [[buffer(4)]], constant size_m* stridesIn [[buffer(5)]], constant float& t [[buffer(6)]], uint gid [[thread_position_in_grid]]) {
    int yIndex = gid / shapeIn[1];
    int xIndex = gid % shapeIn[1];
    float x = (float)xIndex / (float)shapeIn[1];
    float y = (float)yIndex / (float)shapeIn[0];
    
    float3 FC = float3((float)xIndex, (float)yIndex, 0);
    float2 r  = float2((float)shapeIn[1], (float)shapeIn[0]);

    float2 FC2 = float2((float)xIndex, (float)yIndex);
    float3 FC3 = float3(FC2, 0.0);      // <- REQUIRED fix
    float3 r3  = float3(r.x, r.y, r.x); // because GLSL used r.xyx

    float4 o = float4(0.0);

    // Original GLSL: for(float i,s,d,l,K,P; d+i++ < 1e2; )
    float i = 0.0;
    float s = 0.0;
    float d = 0.0;
    float l = 0.0;
    float K = 0.0;
    float P = 0.0;

    for (;;) {

        if (d + i >= 100.0) break;
        i += 1.0;

        // vec3 p = vec3((FC.xy*2.-r)/r.y*d, d-2.) * rotate3D(...)
        float3 baseP = float3(((FC2 * 2.0 - r.xy) / r.y) * d, d - 2.0);
        float3 p = baseP * rotate3D(t * 0.2, float3(0,1,0));

        // k = p * .02
        float3 k = p * 0.02;

        // v = p / (k.y -= .02)
        k.y -= 0.02;
        float3 v = p / k.y;

        // j = min(cos(v)-.9, min(v = 6e1 - abs(v), v.zyx)) * k.y
        float3 j1 = cos(v) - 0.9;
        float3 v2 = 60.0 - fabs(v);
        float3 j2 = min(v2, v2.zyx);
        float3 j = min(j1, j2) * k.y;

        // j = min(j, j.z)
        j = min(j, j.z);

        // d += s = min(
        //      max(abs(l = length(p) - .99) - .01,
        //          min(j.x, K = length(k))),
        //      p.y + 1.)
        float lp = length(p) - 0.99;
        l = lp;
        float absTerm = fabs(lp) - 0.01;

        K = length(k);
        float innerMin = min(j.x, K);

        float m = min(max(absTerm, innerMin), p.y + 1.0);

        s = m;
        d += m;

        // o += vec4(3,5,9,9) * (max(-l, j.x/(P=dot(p,p))) / P / 6e4 / (1e-4+s) + s/4e5/K/K)
        P = max(dot(p,p), 1e-6); // avoid NaN

        float termA = max(-l, j.x / P) / P / 60000.0 / (1e-4 + s);
        float termB = s / 400000.0 / (K*K + 1e-6);

        o += float4(3,5,9,9) * (termA + termB);
    }

    // final color
//    o = tanh(o);
//    o = saturate(o);

    // write final 8-bit RGBA
    outMat[yIndex * stridesOut[0] + xIndex * stridesOut[1] + 0] = uint8_t(o.z * 255.0f);
    outMat[yIndex * stridesOut[0] + xIndex * stridesOut[1] + 1] = uint8_t(o.y * 255.0f);
    outMat[yIndex * stridesOut[0] + xIndex * stridesOut[1] + 2] = uint8_t(o.x * 255.0f);
    outMat[yIndex * stridesOut[0] + xIndex * stridesOut[1] + 3] = uint8_t(255);
}

kernel void custom_kernel4(device uint8_t* outMat [[buffer(0)]], device const uint8_t* inMat [[buffer(1)]], constant size_m* shapeOut [[buffer(2)]], constant size_m* stridesOut [[buffer(3)]], constant size_m* shapeIn [[buffer(4)]], constant size_m* stridesIn [[buffer(5)]], constant float& t [[buffer(6)]], uint gid [[thread_position_in_grid]]) {
    int yIndex = gid / shapeIn[1];
    int xIndex = gid % shapeIn[1];
    float x = (float)xIndex / (float)shapeIn[1];
    float y = (float)yIndex / (float)shapeIn[0];
    
    float3 FC = float3((float)xIndex, (float)yIndex, 0);
    float2 r  = float2((float)shapeIn[1], (float)shapeIn[0]);

    // Promote FC.xy → float3 (GLSL used FC.rgb)
    float2 FC2 = float2((float)xIndex, (float)yIndex);
    float3 FC3 = float3(FC2, 0.0);

    // r.xyy used in GLSL → build explicit float3
    float3 r3 = float3(r.x, r.y, r.y);

    float4 o = float4(0.0);

    // Declare c, p outside loops like GLSL
    float3 c = float3(0.0);
    float3 p = float3(0.0);

    // GLSL: for(float i,z,f,l; i++ < 6e1; ...)
    float i = 0.0;
    float z = 0.0;
    float f = 0.0;
    float l = 0.0;

    for (;;) {

        i += 1.0;
        if (i > 60.0) break;

        // ---------- inner loop initialization ----------
        // c = p = z * normalize(FC.rgb*2 - r.xyy);
        float3 dir = normalize(FC3 * 2.0 - r3);
        p = z * dir;
        c = p;

        // p.z -= t/.2   →   p.z -= t * 5
        p.z -= t * 5.0;

        // c.z += 9.
        c.z += 9.0;

        // f = 0.0
        f = 0.0;

        // ---------- inner loop ----------
        for (;;) {
            f += 1.0;
            if (f > 7.0) break;

            // p += sin(p*f + z*.2 + 1./l).yzx / f
            float invl = (fabs(l) < 1e-6 ? 0.0 : 1.0 / l);
            float3 tmp = sin(p * f + float3(z * 0.2) + float3(invl));
            p += tmp.yzx / f;
        }

        // ---------- outer-loop body ----------

        // f = 8 - length((c+p).xy)
        f = 8.0 - length((c + p).xy);

        // clamp f: f = min(max(f, -f*.2), l=length(c+4*sin(...)))
        float ls = length(c + 4.0 * sin(t + float3(0,8,4)));
        l = ls;

        float fclamped = min(max(f, -f * 0.2), l);

        // z += f = .01 + fclamped / 7.
        float f_new = 0.01 + fclamped / 7.0;
        f = f_new;
        z += f_new;

        // o += vec4(5,1,l,1) / l / l / f / z
        float lsafe = max(l, 1e-6);
        float denom = lsafe * lsafe * f * max(z, 1e-6);
        o += float4(5, 1, l, 1) / denom;
    }

    // final tone map like GLSL
    o = tanh(o / 300.0);

    // clamp for safety
    o = saturate(o);



//    // write final 8-bit RGBA
    outMat[yIndex * stridesOut[0] + xIndex * stridesOut[1] + 0] = uint8_t(o.z * 255.0f);
    outMat[yIndex * stridesOut[0] + xIndex * stridesOut[1] + 1] = uint8_t(o.y * 255.0f);
    outMat[yIndex * stridesOut[0] + xIndex * stridesOut[1] + 2] = uint8_t(o.x * 255.0f);
    outMat[yIndex * stridesOut[0] + xIndex * stridesOut[1] + 3] = uint8_t(255);
}
