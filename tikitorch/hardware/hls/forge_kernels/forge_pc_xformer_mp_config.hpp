// forge_pc_xformer_mp_config.hpp — DDR layout for the BF16 MIXED-PRECISION distributed transformer-block shard
// kernel (forge_pc_xformer_mp_block.cpp). DISTINCT from the int kernel's layout (forge_pc_xformer_config.hpp,
// which forge_pc_xformer_block.cpp owns and which this header includes UNMODIFIED).
//
// The mixed-precision shard keeps TWO views of the model, exactly like the de-risked bf16 golden
// (forge_pc_xformer_bf16_golden_block.hpp BF16_BWD=2):
//   * the INT16 master + dyadic-Adam int state at the base XO_* offsets — drives the deployed mult-free FORWARD
//     (g_forward, byte-identical to the int kernel). dyad(int16 master)*2^-WFRAC == dyad(float master).
//   * a FLOAT master + bf16 dyadic-Adam float state in the appended MP_* region — the trainable quantity the
//     BF16 STE BACKWARD differentiates (bf_forward / bf_backward / bf_adam_one operate on this).
// The forward also caches the WIDE (float) pre-quantization activations the STE backward needs.
//
// Every float lives in ONE int32 lane (bit-reinterpret), so the int kernel's 512-bit burst I/O is reused
// lane-for-lane. The whole MP region is appended AFTER the base X_BUFLEN so the int layout is untouched.
#ifndef FORGE_PC_XFORMER_MP_CONFIG_HPP
#define FORGE_PC_XFORMER_MP_CONFIG_HPP

#include "forge_pc_xformer_config.hpp"   // base dims (X_D/X_H/...) + the int kernel's DDR layout, UNMODIFIED

// ---- FLOAT-master model (bf16 golden BFM): the trainable weights, char-LM natural scale ----
// Order matches BFM / the int masters: WQ WK WV WO WUP WDN | LN1G LN1B LN2G LN2B | EMB POS | LNFG LNFB WHEAD.
static const int MP_BASE  = X_AL(X_BUFLEN);                 // append after the int layout (16-word aligned)
static const int MP_FWQ   = MP_BASE;                        // [D*D] float query proj master
static const int MP_FWK   = MP_FWQ   + X_D*X_D;             // [D*D]
static const int MP_FWV   = MP_FWK   + X_D*X_D;             // [D*D]
static const int MP_FWO   = MP_FWV   + X_D*X_D;             // [D*D]
static const int MP_FWUP  = MP_FWO   + X_D*X_D;             // [FFN*D]
static const int MP_FWDN  = MP_FWUP  + X_FFN*X_D;           // [D*FFN]
static const int MP_FLN1G = MP_FWDN  + X_D*X_FFN;           // [D] (frozen)
static const int MP_FLN1B = MP_FLN1G + X_D;                 // [D] (frozen)
static const int MP_FLN2G = MP_FLN1B + X_D;                 // [D]
static const int MP_FLN2B = MP_FLN2G + X_D;                 // [D]
static const int MP_FEMB  = MP_FLN2B + X_D;                 // [VOCAB*D] (INPUT role)
static const int MP_FPOS  = MP_FEMB  + X_VOCAB*X_D;         // [SEQ*D]   (INPUT role)
static const int MP_FLNFG = MP_FPOS  + X_SEQ*X_D;           // [D] (OUTPUT role)
static const int MP_FLNFB = MP_FLNFG + X_D;                 // [D]
static const int MP_FWHEAD= MP_FLNFB + X_D;                 // [VOCAB*D] (OUTPUT role) float head master

// ---- bf16/float dyadic-Adam m/v for every trainable float matrix (same flat order, two arrays) ----
// reuse X_NW (= all matmul + head weights) for the matmul/head adam; embed/pos get their own m/v.
#define MP_OFF_FWQ    0
#define MP_OFF_FWK    (X_D*X_D)
#define MP_OFF_FWV    (X_D*X_D*2)
#define MP_OFF_FWO    (X_D*X_D*3)
#define MP_OFF_FWUP   (X_D*X_D*4)
#define MP_OFF_FWDN   (X_D*X_D*4 + X_FFN*X_D)
#define MP_OFF_FHEAD  (X_D*X_D*4 + X_FFN*X_D + X_D*X_FFN)
static const int MP_MOM   = X_AL(MP_FWHEAD + X_VOCAB*X_D);  // [X_NW] float first-moment EMA
static const int MP_VAR   = MP_MOM   + X_NW;                // [X_NW] float second-moment EMA
static const int MP_MEMB  = MP_VAR   + X_NW;                // [VOCAB*D] embed adam m   (INPUT role)
static const int MP_VEMB  = MP_MEMB  + X_VOCAB*X_D;         // [VOCAB*D] embed adam v
static const int MP_MPOS  = MP_VEMB  + X_VOCAB*X_D;         // [SEQ*D]   pos adam m
static const int MP_VPOS  = MP_MPOS  + X_SEQ*X_D;           // [SEQ*D]   pos adam v

// ---- WIDE (float) forward-for-backward cache (the STE operands; == bf16 golden BFA) ----
static const int MP_WR    = X_AL(MP_VPOS + X_SEQ*X_D);      // [SEQ*D]  R = embed+pos (block input, float)
static const int MP_WX1   = MP_WR    + X_SEQ*X_D;           // [SEQ*D]  LN1 out (post fdyad LN)
static const int MP_WQ    = MP_WX1   + X_SEQ*X_D;           // [SEQ*D]  post-qact Q
static const int MP_WK    = MP_WQ    + X_SEQ*X_D;           // [SEQ*D]  post-qact K
static const int MP_WV    = MP_WK    + X_SEQ*X_D;           // [SEQ*D]  post-qact V
static const int MP_WATTN = MP_WV    + X_SEQ*X_D;           // [SEQ*D]  attention out (post qact)
static const int MP_WM    = MP_WATTN + X_SEQ*X_D;           // [SEQ*D]  residual mid M
static const int MP_WX2   = MP_WM    + X_SEQ*X_D;           // [SEQ*D]  LN2 out
static const int MP_WHD   = MP_WX2   + X_SEQ*X_D;           // [SEQ*FFN] FFN hidden (post relu+qact)
static const int MP_WRP   = MP_WHD   + X_SEQ*X_FFN;         // [SEQ*D]  final residual R'
static const int MP_WXF   = MP_WRP   + X_SEQ*X_D;           // [SEQ*D]  LNF out (OUTPUT)
static const int MP_WLG   = MP_WXF   + X_SEQ*X_D;           // [SEQ*VOCAB] wide logits (OUTPUT)
static const int MP_WPRAW = MP_WLG   + X_SEQ*X_VOCAB;       // [H*SEQ*SEQ] softmax probs (sum=1 per causal row)
static const int MP_WRMASK= MP_WPRAW + X_H*X_SEQ*X_SEQ;     // [SEQ*FFN] relu 0/1 mask (float)
// per-token LN mean + normalizer s (the analytic LN backward needs s; mean is unused but kept for parity)
static const int MP_WLN1M = MP_WRMASK+ X_SEQ*X_FFN;         // [SEQ]
static const int MP_WLN1S = MP_WLN1M + X_SEQ;               // [SEQ]
static const int MP_WLN2M = MP_WLN1S + X_SEQ;               // [SEQ]
static const int MP_WLN2S = MP_WLN2M + X_SEQ;               // [SEQ]
static const int MP_WLNFM = MP_WLN2S + X_SEQ;               // [SEQ]
static const int MP_WLNFS = MP_WLNFM + X_SEQ;               // [SEQ]

// ---- wide inter-chip error message (the up-message is bf16-wide, one float per lane) ----
static const int MP_GRIN  = X_AL(MP_WLNFS + X_SEQ);         // [SEQ*D]  incoming wide error (bf16) from downstream
static const int MP_GROUT = MP_GRIN  + X_SEQ*X_D;           // [SEQ*D]  outgoing wide error (bf16) to upstream

// ---- DEBUG-ONLY gradient dump region (gradcheck tb): kernel writes its raw bf16 gradients here so the tb can
//      compare them DIRECTLY to the golden bf_backward (bg.*), not just the post-optimizer weights. Always
//      reserved (keeps MP_BUFLEN stable); only written when the kernel is built -DMP_DEBUG_GRADS. ----
static const int MP_DBG    = X_AL(MP_GROUT + X_SEQ*X_D);
static const int MP_DG_WQ  = MP_DBG;                       // [D*D]
static const int MP_DG_WK  = MP_DG_WQ  + X_D*X_D;          // [D*D]
static const int MP_DG_WV  = MP_DG_WK  + X_D*X_D;          // [D*D]
static const int MP_DG_WO  = MP_DG_WV  + X_D*X_D;          // [D*D]
static const int MP_DG_WUP = MP_DG_WO  + X_D*X_D;          // [FFN*D]
static const int MP_DG_WDN = MP_DG_WUP + X_FFN*X_D;        // [D*FFN]
static const int MP_DG_WH  = MP_DG_WDN + X_D*X_FFN;        // [VOCAB*D]
static const int MP_DG_GR  = MP_DG_WH  + X_VOCAB*X_D;      // [SEQ*D]  gR (== gRout source)
static const int MP_DG_GQ  = MP_DG_GR  + X_SEQ*X_D;        // [SEQ*D]
static const int MP_DG_GK  = MP_DG_GQ  + X_SEQ*X_D;        // [SEQ*D]
static const int MP_DG_GV  = MP_DG_GK  + X_SEQ*X_D;        // [SEQ*D]
static const int MP_DG_END = MP_DG_GV  + X_SEQ*X_D;

// ---- gradient ACCUMULATORS (effective-batch / ACCUM>1): the kernel sums the per-sub-batch weight grads here
//      across an accumulation window (zero on the first sub-batch, add each, dyadic-Adam on the last over the
//      sum). Persisted in DDR so multiple BWD kicks accumulate -> effective batch = ACCUM. ----
static const int MP_GACC     = X_AL(MP_DG_END);
static const int MP_GACC_WQ  = MP_GACC;                    // [D*D]
static const int MP_GACC_WK  = MP_GACC_WQ  + X_D*X_D;      // [D*D]
static const int MP_GACC_WV  = MP_GACC_WK  + X_D*X_D;      // [D*D]
static const int MP_GACC_WO  = MP_GACC_WV  + X_D*X_D;      // [D*D]
static const int MP_GACC_WUP = MP_GACC_WO  + X_D*X_D;      // [FFN*D]
static const int MP_GACC_WDN = MP_GACC_WUP + X_FFN*X_D;    // [D*FFN]
static const int MP_GACC_WH  = MP_GACC_WDN + X_D*X_FFN;    // [VOCAB*D]  (OUTPUT chip head)
static const int MP_GACC_EMB = MP_GACC_WH  + X_VOCAB*X_D;  // [VOCAB*D]  (INPUT chip embed)
static const int MP_GACC_POS = MP_GACC_EMB + X_VOCAB*X_D;  // [SEQ*D]    (INPUT chip pos)
static const int MP_GACC_END = MP_GACC_POS + X_SEQ*X_D;

// ---- CROSS-SUB-BATCH PIPELINING: MP_SLOTS-deep copies of the PER-SUB-BATCH volatile state ------------------
// REBUILD #4: pipeline slot depth S=8 (was 2). SLOT = cfg lane 6 = 0..MP_SLOTS-1, indexed by the GLOBAL
// sub-batch index (g % MP_SLOTS). MP_SLOTS in-flight sub-batches can overlap without clobbering each other's
// wide activation cache / neighbour messages -> up to ~MP_SLOTS-deep overlap (was 2). ONLY the per-sub-batch
// volatile state is replicated: the WHOLE wide cache MP_W* + MP_GRIN + MP_GROUT (one span per slot), plus the
// slot-indexed token/target lanes. The GRAD ACCUMULATORS (MP_GACC_*), the float/int MASTERS (MP_F*/XO_*), and
// the dyadic-Adam moments (MP_MOM/MP_VAR/MP_MEMB..) are NOT per-sub-batch -> single-slot, never replicated.
//
// [MP_WR .. MP_GROUT_END) is ONE CONTIGUOUS span (verified: every region is a 16-multiple and MP_GRIN's X_AL is
// a no-op), so a single additive offset maps any slot-0 wide/msg region to slot k's mirror. Slot 0 IS the base
// region (offset 0); slots 1..MP_SLOTS-1 are MP_SLOTS-1 mirror spans appended contiguously with stride
// MP_PAR_SPAN. MP_PAR (== old slot-1 mirror start) stays put so the offset for slot 1 is unchanged:
//   slot_off(k) = (k==0) ? 0 : (MP_PAR_OFF + (k-1)*MP_PAR_SPAN)     [== MP_PAR + (k-1)*MP_PAR_SPAN - MP_WR]
static const int MP_SLOTS    = 8;                               // pipeline slot depth (rebuild #4; was 2)
static const int MP_PAR_SPAN = (MP_GROUT + X_SEQ*X_D) - MP_WR;   // contiguous per-slot wide/msg span (words)
static const int MP_PAR      = X_AL(MP_GACC_END);                // start of the appended slot>=1 mirror block
static const int MP_PAR_OFF  = MP_PAR - MP_WR;                   // add to a slot-0 wide/msg offset -> slot-1 copy
static const int MP_PAR_END  = MP_PAR + (MP_SLOTS-1)*MP_PAR_SPAN;// end of ALL MP_SLOTS-1 mirrors (stride SPAN)
// slot-indexed token / target lanes (int32): the WIDE forward embed (chip0), the CE (OUTPUT bwd), and the INPUT
// bwd embed-scatter read tok/tgt at MP_TOK/MP_TGT + SLOT*X_SEQ. The deploy-int forward keeps reading XO_TOK.
static const int MP_TOK      = MP_PAR_END;                       // [MP_SLOTS*SEQ] tokens (slot k at MP_TOK+k*SEQ)
static const int MP_TGT      = MP_TOK + MP_SLOTS*X_SEQ;          // [MP_SLOTS*SEQ] targets (slot k at MP_TGT+k*SEQ)

static const int MP_BUFLEN= X_AL(MP_TGT + MP_SLOTS*X_SEQ);

// per-slot span must be one contiguous block for the single additive slot offset to be valid
static_assert(MP_GRIN == MP_WLNFS + X_SEQ, "parity span not contiguous: MP_GRIN has an alignment gap");
static_assert(MP_PAR_SPAN % X_WL == 0,     "parity span must be 16-word aligned");
static_assert(MP_PAR_OFF % X_WL == 0,      "MP_PAR_OFF must be 16-word aligned (burst I/O)");
static_assert(MP_PAR % X_WL == 0,          "mirror block base must be 16-word aligned (per-slot burst I/O)");
static_assert(MP_PAR_END == MP_TOK,        "slot mirrors must end exactly at MP_TOK (no collision / no gap)");
static_assert((MP_SLOTS*X_SEQ) % X_WL == 0,"tok/tgt slot lanes must be 16-word aligned");

#endif
