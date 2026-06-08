// forge_hls_types.hpp — portability shim for Vitis HLS vs. native g++.
//
// When compiled under Vitis HLS (synthesis or csim), the toolchain defines
// __VITIS_HLS__ (or in some flows __SYNTHESIS__). In that case we use
// ap_int<N> for bit-precise arithmetic.
//
// Outside Vitis (e.g. standalone testbench compiled with g++ on the dev
// host), ap_int.h is unavailable; we fall back to fixed-width stdint types.
// The HLS PRAGMAs are ignored by g++ (with a warning), which is fine for
// the testbench whose job is functional verification, not synthesis.

#ifndef FORGE_HLS_TYPES_HPP
#define FORGE_HLS_TYPES_HPP

#if defined(__VITIS_HLS__) || defined(__SDSVHLS__) || defined(__SYNTHESIS__)
    #include "ap_int.h"
    typedef ap_int<16> data_t;
    typedef ap_int<32> acc_t;
#else
    #include <cstdint>
    typedef int16_t data_t;
    typedef int32_t acc_t;
#endif

#endif  // FORGE_HLS_TYPES_HPP
