// forge_relu.cpp — multiplier-free ReLU FORWARD + BACKWARD for the FPGA transformer-training rig (S1).
//
// The trivial-but-load-bearing nonlinearity of the mult-free transformer block. ReLU is the chosen
// activation (charlm_mixer.py --mlp-act relu: GELU needs a multiply, ReLU does not). This kernel is
// pure compare/select — NO multiply, NO shift-add even — so DSP=0 is structural, not a closure trick.
//
// ALGORITHM (matches charlm_mixer.py nn.ReLU() exactly):
//   FWD: y[k] = max(x[k], 0)            and  mask[k] = (x[k] > 0) ? 1 : 0
//   BWD: gx[k] = mask[k] ? gO[k] : 0     (straight-through where active, zero where clamped)
// Note the boundary: x==0 is NOT active (mask=0), matching PyTorch ReLU' (derivative 0 at 0), so the
// forward y and the backward mask are consistent (y==0 <=> mask==0).
//
// One kernel, two modes — mode lives in the buffer at OFF_MODE (no second top, no two AFIs):
//   buffer[OFF_MODE] == 0  -> FWD : read OFF_X,  write OFF_Y and OFF_MASK
//   buffer[OFF_MODE] != 0  -> BWD : read OFF_GO and OFF_MASK, write OFF_GX
// The mask is its OWN int32 region (1 word per element). Packing it into the sign bit of y would alias
// a legitimately-zero activation against an inactive one and would force the BWD path to re-derive the
// mask from y (fragile if y were ever rescaled); a separate, explicit mask region is bit-exact and
// trivially testable. Each element is stored one-per-int32 (probe-self-contained; the real rig can
// bit-pack the mask 32:1, the on-chip logic is identical).
//
// SHAPE: operates over a FLAT vector of length RL_LEN = N * max(D, FFN). The largest S1 activation
// tensor through ReLU is the FFN hidden state N*FFN = 16*512 = 8192 (the d-model post-attn residual is
// N*D = 16*128 = 2048, smaller). 8192 % 16 == 0 so every region is 512-bit aligned. The kernel always
// processes RL_LEN elements; callers that need fewer simply zero-pad (max(0,0)=0, grad of pad = 0).
//
// WIDE ap_uint<512> load/store: 16 int32 / word, II=1. Only the 16-lane word tile is unrolled; the
// outer block loop is NOT unrolled (the throughput-sweep lesson). DSP=0, 4.0 ns close, tiny LUT.
//
// Part xcvu47p-fsvh2892-2L-e, target 4.0 ns. csim+csynth via synth_relu.tcl.

#include <stdint.h>
#include <ap_int.h>

// ---- S1 config (d768-general defines kept; S1 shrinks D/FFN/N) ----
#define RL_D    128      // d_model
#define RL_FFN  512      // FFN hidden width  (the largest dim through ReLU)
#define RL_N    16       // microbatch tokens
#define RL_LEN  (RL_N * RL_FFN)   // = 8192 elements (the largest S1 ReLU tensor); % 16 == 0
#define RL_WORDS (RL_LEN / 16)    // = 512 wide (512-bit) words

typedef ap_int<32>   data_t;     // x / y / gO / gx values (full int32; ReLU never grows magnitude)
typedef ap_uint<512> wide_t;     // 16 x int32 packed AXI word

// flat int32 DDR buffer layout (probe-local, self-contained):
//   [ MODE(1) pad to 16 | X (LEN) | GO (LEN) | MASK (LEN) | Y (LEN) | GX (LEN) ]
// MODE sits in its own 16-word header so every payload region stays 512-bit aligned.
static const int OFF_MODE = 0;                         // scalar: 0=FWD, !=0=BWD
static const int OFF_X    = 16;                         // FWD input  (also BWD unused)
static const int OFF_GO   = OFF_X   + RL_LEN;           // BWD upstream grad
static const int OFF_MASK = OFF_GO  + RL_LEN;           // FWD out / BWD in : 1 = active (x>0)
static const int OFF_Y    = OFF_MASK + RL_LEN;          // FWD output
static const int OFF_GX   = OFF_Y   + RL_LEN;           // BWD output

static const int OFF_X_W    = OFF_X    / 16;            // region bases in 512-bit (16-int32) words
static const int OFF_GO_W   = OFF_GO   / 16;
static const int OFF_MASK_W = OFF_MASK / 16;
static const int OFF_Y_W    = OFF_Y    / 16;
static const int OFF_GX_W   = OFF_GX   / 16;

void forge_relu(int32_t *buffer, wide_t *io_wide) {
#pragma HLS INTERFACE m_axi port=buffer  max_widen_bitwidth=512 offset=slave bundle=gmem
#pragma HLS INTERFACE m_axi port=io_wide max_widen_bitwidth=512 offset=slave bundle=gmemio
#pragma HLS INTERFACE s_axilite port=buffer
#pragma HLS INTERFACE s_axilite port=io_wide
#pragma HLS INTERFACE s_axilite port=return

    bool bwd = (buffer[OFF_MODE] != 0);

    if (!bwd) {
        // ---- FORWARD: y = max(x,0); mask = (x>0) ----
        // Read X wide, write Y and MASK wide. One wide word = 16 elements; II=1.
    FWD: for (int w = 0; w < RL_WORDS; w++) {
#pragma HLS PIPELINE II=1
            wide_t xin = io_wide[OFF_X_W + w];
            wide_t yout, mout;
            for (int k = 0; k < 16; k++) {
#pragma HLS UNROLL
                data_t x = (data_t)(ap_int<32>)xin.range(32 * k + 31, 32 * k);
                bool   act = (x > 0);                       // x==0 is INACTIVE (PyTorch ReLU' at 0 = 0)
                data_t y = act ? x : (data_t)0;
                yout.range(32 * k + 31, 32 * k) = (ap_uint<32>)y;
                mout.range(32 * k + 31, 32 * k) = act ? (ap_uint<32>)1 : (ap_uint<32>)0;
            }
            io_wide[OFF_Y_W + w]    = yout;
            io_wide[OFF_MASK_W + w] = mout;
        }
    } else {
        // ---- BACKWARD: gx = mask ? gO : 0 ----
        // Read GO and MASK wide, write GX wide. II=1.
    BWD: for (int w = 0; w < RL_WORDS; w++) {
#pragma HLS PIPELINE II=1
            wide_t gin = io_wide[OFF_GO_W + w];
            wide_t min = io_wide[OFF_MASK_W + w];
            wide_t gout;
            for (int k = 0; k < 16; k++) {
#pragma HLS UNROLL
                data_t g    = (data_t)(ap_int<32>)gin.range(32 * k + 31, 32 * k);
                bool   act  = (min.range(32 * k + 31, 32 * k) != 0);
                gout.range(32 * k + 31, 32 * k) = act ? (ap_uint<32>)g : (ap_uint<32>)0;
            }
            io_wide[OFF_GX_W + w] = gout;
        }
    }
}
