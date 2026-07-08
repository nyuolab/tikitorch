// hls_math.h — minimal local shim so the forge_pc_xformer_* kernels csim with a plain g++ (no Vitis).
// Maps the hls:: transcendental/round API the kernels use to <cmath>. The EC2 Vitis run uses the real header.
#ifndef __HLS_MATH_SHIM_H__
#define __HLS_MATH_SHIM_H__
#include <cmath>
namespace hls {
    static inline float round(float x){ return ::roundf(x); }
    static inline float floor(float x){ return ::floorf(x); }
    static inline float log2(float x){ return ::log2f(x); }
    static inline float exp(float x){ return ::expf(x); }
    static inline float sqrt(float x){ return ::sqrtf(x); }
    static inline float ldexp(float x,int e){ return ::ldexpf(x,e); }
    static inline double round(double x){ return ::round(x); }
    static inline double floor(double x){ return ::floor(x); }
    static inline double log2(double x){ return ::log2(x); }
    static inline double exp(double x){ return ::exp(x); }
    static inline double sqrt(double x){ return ::sqrt(x); }
    static inline double ldexp(double x,int e){ return ::ldexp(x,e); }
}
#endif
