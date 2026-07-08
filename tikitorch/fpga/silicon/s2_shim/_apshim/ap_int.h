// Minimal ap_int / ap_uint shim — JUST enough of the Vitis HLS arbitrary-precision API to compile + CSIM the
// forge_pc_xformer_block kernel + testbench with a plain g++ (no Vitis). Widths used: 6/16/32/48/64 (native
// int64-backed, masked + sign/zero-extended to width) and the wide ap_uint<512> burst word (uint32[16] with a
// 32-bit-aligned .range()). NOT a general ap_int — only the ops this kernel uses. The point is to run the
// int64-golden bit-exact CSIM locally and fast; the EC2 Vitis run remains the authoritative check.
#ifndef __AP_INT_SHIM_H__
#define __AP_INT_SHIM_H__
#include <stdint.h>

template <int W> struct ap_int {
    long long raw;
    static long long norm(long long x) {
        if (W >= 64) return x;
        unsigned long long mask = ((1ULL << W) - 1);
        unsigned long long m = (unsigned long long)x & mask;
        unsigned long long sb = 1ULL << (W - 1);
        if (m & sb) m |= ~mask;                 // sign-extend
        return (long long)m;
    }
    ap_int() : raw(0) {}
    ap_int(long long x) : raw(norm(x)) {}
    ap_int(int x) : raw(norm((long long)x)) {}
    template <int M> ap_int(const ap_int<M>& o) : raw(norm(o.raw)) {}
    operator long long() const { return raw; }
    ap_int& operator=(long long x) { raw = norm(x); return *this; }
    ap_int& operator+=(long long x) { raw = norm(raw + x); return *this; }
    ap_int& operator-=(long long x) { raw = norm(raw - x); return *this; }
    bool operator[](int b) const { return ((unsigned long long)raw >> b) & 1ULL; }
};
template <int W> struct ap_uint {
    unsigned long long raw;
    static unsigned long long norm(unsigned long long x) {
        if (W >= 64) return x;
        return x & ((1ULL << W) - 1);
    }
    ap_uint() : raw(0) {}
    ap_uint(long long x) : raw(norm((unsigned long long)x)) {}
    ap_uint(unsigned long long x) : raw(norm(x)) {}
    ap_uint(int x) : raw(norm((unsigned long long)(long long)x)) {}
    template <int M> ap_uint(const ap_int<M>& o) : raw(norm((unsigned long long)o.raw)) {}
    template <int M> ap_uint(const ap_uint<M>& o) : raw(norm(o.raw)) {}
    operator unsigned long long() const { return raw; }
    ap_uint& operator=(unsigned long long x) { raw = norm(x); return *this; }
    ap_uint& operator+=(unsigned long long x) { raw = norm(raw + x); return *this; }
    ap_uint& operator-=(unsigned long long x) { raw = norm(raw - x); return *this; }
    bool operator[](int b) const { return (raw >> b) & 1ULL; }
};

// ---- ap_uint<512> burst word: 16 x uint32 lanes, 32-bit-aligned .range() (all this kernel uses) ----
template <> struct ap_uint<512> {
    uint32_t w[16];
    ap_uint() { for (int i = 0; i < 16; i++) w[i] = 0; }
    ap_uint(int x) { for (int i = 0; i < 16; i++) w[i] = 0; w[0] = (uint32_t)x; }
    struct rangeproxy {
        ap_uint<512>* p; int hi, lo;
        // read: bits [lo,hi] (hi-lo+1<=32, 32-aligned in this kernel) -> the lane value as a (non-negative)
        // long long; a subsequent (ap_int<32>) cast does the int32 sign-extension. Single conversion -> no
        // ambiguity. (returns long long, not unsigned, so the (ap_int<32>) cast picks the long long ctor.)
        operator long long() const {
            int lane = lo >> 5; int width = hi - lo + 1;
            uint32_t v = p->w[lane];
            if (width >= 32) return (long long)(unsigned long long)v;
            return (long long)(v & ((1u << width) - 1));
        }
        template <int M> rangeproxy& operator=(const ap_int<M>& v) { int lane = lo >> 5; p->w[lane] = (uint32_t)(unsigned long long)v.raw; return *this; }
        template <int M> rangeproxy& operator=(const ap_uint<M>& v) { int lane = lo >> 5; p->w[lane] = (uint32_t)v.raw; return *this; }
        rangeproxy& operator=(unsigned long long v) { int lane = lo >> 5; p->w[lane] = (uint32_t)v; return *this; }
        rangeproxy& operator=(const rangeproxy& o) { int lane = lo >> 5; p->w[lane] = (uint32_t)(unsigned long long)o; return *this; }
    };
    rangeproxy range(int hi, int lo) { rangeproxy r; r.p = this; r.hi = hi; r.lo = lo; return r; }
    // const read
    unsigned long long rangec(int hi, int lo) const { int lane = lo >> 5; uint32_t v = w[lane]; int width = hi - lo + 1; return (width >= 32) ? v : (v & ((1u << width) - 1)); }
};
// allow (ap_int<32>)word.range(...) : construct ap_int<W> from the rangeproxy (via its unsigned long long)
template <int W> static inline ap_int<W> mk_ai(unsigned long long v) { return ap_int<W>((long long)(int32_t)v); }

#endif
