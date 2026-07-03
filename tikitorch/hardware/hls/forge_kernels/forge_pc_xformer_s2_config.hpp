// forge_pc_xformer_s2_config.hpp — dims + DDR layout for the SCALE-2 (30M, D=512) DDR-streaming + K-wide-MAC
// transformer-block training shard (forge_pc_xformer_s2_block.cpp). This is the PHASE-1 kernel of the
// SCALE_30M_124M_DESIGN: same model-parallel role/phase/CFG contract as forge_pc_xformer_mp_block, but the
// weights / optimizer state / gradient accumulators are NO LONGER on-chip resident — they are STREAMED from
// gmem in double-buffered tiles, and the contraction runs on a K-wide MAC engine (S2_K products/cycle).
//
// SELF-CONTAINED (the base forge_pc_xformer_config.hpp was removed from this worktree): all X_*-equivalent
// dims/roles/phases/alignment are (re)defined here under the S2_ prefix. Every quantity is #ifndef-overridable
// so a D=64-class csim (S2_D=64 S2_H=2 S2_DH=32 S2_FFN=256 S2_SEQ=64 S2_VOCAB=64) can validate the arithmetic.
//
// Every float lives in ONE int32 lane (bit-reinterpret), reusing the 512-bit burst path lane-for-lane.
//
// TRAIN-ONLY: the deploy-int forward of the mp kernel is OMITTED (§ "the deploy-int forward path can be OMITTED
// entirely in s2 — deploy comes later"). Only the WIDE float-master forward + BF16 STE backward + dyadic-Adam
// live here. The softmax-prob tensor (H*SEQ*SEQ) is NOT cached — the backward RECOMPUTES it (PRAW recompute).
#ifndef FORGE_PC_XFORMER_S2_CONFIG_HPP
#define FORGE_PC_XFORMER_S2_CONFIG_HPP

// ------------------------------- dims (all overridable for csim) -------------------------------
#ifndef S2_D
#define S2_D     512        // d_model
#endif
#ifndef S2_H
#define S2_H     8          // attention heads
#endif
#ifndef S2_DH
#define S2_DH    64         // head dim (H*DH == D)
#endif
#ifndef S2_FFN
#define S2_FFN   2048       // FFN inner width (4*D)
#endif
#ifndef S2_SEQ
#define S2_SEQ   256        // sequence length (per-kick "batch")
#endif
#ifndef S2_VOCAB
#define S2_VOCAB 256        // vocab (byte-level for the Phase-0 30M de-risk; BPE at the rebuild)
#endif
#ifndef S2_MBITS
#define S2_MBITS 2          // fine-dyadic LayerNorm mantissa bits
#endif
#ifndef S2_ABITS
#define S2_ABITS 9          // qact activation band
#endif

// ------------------------------- K-wide MAC engine + streaming tiles -------------------------------
#ifndef S2_K
#define S2_K     32         // contraction lanes: K products/cycle into a balanced adder tree (compile-time; sweep me)
#endif
#ifndef S2_TILE_O
#define S2_TILE_O 128       // weight-tile output rows  (max cap; runtime tile from CFG lane 7, 0 => this default)
#endif
#ifndef S2_TILE_I
#define S2_TILE_I 128       // weight-tile contraction cols (max cap; runtime from CFG lane 8; MUST be a multiple of S2_K)
#endif
#ifndef S2_SEQ_TILE
#define S2_SEQ_TILE 64      // activation-strip row block over SEQ (max cap; runtime from CFG lane 9)
#endif

// ------------------------------- roles / phases (CFG word lanes) -------------------------------
#define S2_PH_FWD   0
#define S2_PH_BWD   1
#define S2_RL_INPUT 0       // chip 0: token+pos embed, then block 0
#define S2_RL_HID   1       // chips 1..L-2: block only
#define S2_RL_OUT   2       // chip L-1: block, final-LN, LM head, CE

// ------------------------------- burst / alignment -------------------------------
#define S2_WL 16                                        // int32 lanes per 512-bit word
#define S2_AL(n) ((((n)+S2_WL-1)/S2_WL)*S2_WL)          // round up to 16 words

// gmem CFG word: lane 0 phase,1 role,2 do_up,3 lrbits,4 gzero,5 train,6 slot,7 tile_o,8 tile_i,9 seq_tile
static const int S2O_CFG = 0;

// ------------------------------- FLOAT master (streamed tile-by-tile; adam writes back here) -------------------------------
static const int S2_FWQ   = S2O_CFG + S2_WL;                 // [D*D]
static const int S2_FWK   = S2_FWQ   + S2_D*S2_D;            // [D*D]
static const int S2_FWV   = S2_FWK   + S2_D*S2_D;            // [D*D]
static const int S2_FWO   = S2_FWV   + S2_D*S2_D;            // [D*D]
static const int S2_FWUP  = S2_FWO   + S2_D*S2_D;            // [FFN*D]
static const int S2_FWDN  = S2_FWUP  + S2_FFN*S2_D;          // [D*FFN]
static const int S2_FLN1G = S2_FWDN  + S2_D*S2_FFN;          // [D] (frozen)
static const int S2_FLN1B = S2_FLN1G + S2_D;                 // [D] (frozen)
static const int S2_FLN2G = S2_FLN1B + S2_D;                 // [D]
static const int S2_FLN2B = S2_FLN2G + S2_D;                 // [D]
static const int S2_FEMB  = S2_FLN2B + S2_D;                 // [VOCAB*D] (INPUT)
static const int S2_FPOS  = S2_FEMB  + S2_VOCAB*S2_D;        // [SEQ*D]   (INPUT)
static const int S2_FLNFG = S2_FPOS  + S2_SEQ*S2_D;          // [D] (OUTPUT)
static const int S2_FLNFB = S2_FLNFG + S2_D;                 // [D]
static const int S2_FWHEAD= S2_FLNFB + S2_D;                 // [VOCAB*D] (OUTPUT)

// ------------------------------- dyadic-Adam m/v (fp32 in DDR this version) -------------------------------
// flat over the matmul+head weights (same order), streamed RMW; embed/pos get their own m/v.
#define S2_NW (S2_D*S2_D*4 + S2_FFN*S2_D + S2_D*S2_FFN + S2_VOCAB*S2_D)
#define S2_OFF_FWQ    0
#define S2_OFF_FWK    (S2_D*S2_D)
#define S2_OFF_FWV    (S2_D*S2_D*2)
#define S2_OFF_FWO    (S2_D*S2_D*3)
#define S2_OFF_FWUP   (S2_D*S2_D*4)
#define S2_OFF_FWDN   (S2_D*S2_D*4 + S2_FFN*S2_D)
#define S2_OFF_FHEAD  (S2_D*S2_D*4 + S2_FFN*S2_D + S2_D*S2_FFN)
static const int S2_MOM   = S2_AL(S2_FWHEAD + S2_VOCAB*S2_D);// [NW]
static const int S2_VAR   = S2_MOM   + S2_NW;               // [NW]
static const int S2_MEMB  = S2_VAR   + S2_NW;               // [VOCAB*D] (INPUT)
static const int S2_VEMB  = S2_MEMB  + S2_VOCAB*S2_D;       // [VOCAB*D]
static const int S2_MPOS  = S2_VEMB  + S2_VOCAB*S2_D;       // [SEQ*D]   (INPUT)
static const int S2_VPOS  = S2_MPOS  + S2_SEQ*S2_D;         // [SEQ*D]

// ------------------------------- WIDE fwd-for-bwd cache (NO Praw — recomputed) -------------------------------
static const int S2_WR    = S2_AL(S2_VPOS + S2_SEQ*S2_D);   // [SEQ*D]  block input R (embed+pos or down-msg)
static const int S2_WX1   = S2_WR    + S2_SEQ*S2_D;         // [SEQ*D]  LN1 out
static const int S2_WQ    = S2_WX1   + S2_SEQ*S2_D;         // [SEQ*D]  post-qact Q  (PRAW recompute reads this)
static const int S2_WK    = S2_WQ    + S2_SEQ*S2_D;         // [SEQ*D]  post-qact K  (PRAW recompute reads this)
static const int S2_WV    = S2_WK    + S2_SEQ*S2_D;         // [SEQ*D]  post-qact V
static const int S2_WATTN = S2_WV    + S2_SEQ*S2_D;         // [SEQ*D]  attention out (post qact)
static const int S2_WM    = S2_WATTN + S2_SEQ*S2_D;         // [SEQ*D]  residual mid M
static const int S2_WX2   = S2_WM    + S2_SEQ*S2_D;         // [SEQ*D]  LN2 out
static const int S2_WHD   = S2_WX2   + S2_SEQ*S2_D;         // [SEQ*FFN] FFN hidden (post relu+qact)
static const int S2_WRP   = S2_WHD   + S2_SEQ*S2_FFN;       // [SEQ*D]  final residual R'
static const int S2_WXF   = S2_WRP   + S2_SEQ*S2_D;         // [SEQ*D]  LNF out (OUTPUT)
static const int S2_WLG   = S2_WXF   + S2_SEQ*S2_D;         // [SEQ*VOCAB] wide logits (OUTPUT)
static const int S2_WRMASK= S2_WLG   + S2_SEQ*S2_VOCAB;     // [SEQ*FFN] relu 0/1 mask
static const int S2_WLN1S = S2_WRMASK+ S2_SEQ*S2_FFN;       // [SEQ] LN1 normalizer s
static const int S2_WLN2S = S2_WLN1S + S2_SEQ;              // [SEQ] LN2 normalizer s
static const int S2_WLNFS = S2_WLN2S + S2_SEQ;              // [SEQ] LNF normalizer s (OUTPUT)

// ------------------------------- inter-chip wide (bf16) error message -------------------------------
static const int S2_GRIN  = S2_AL(S2_WLNFS + S2_SEQ);       // [SEQ*D] incoming wide error (down-msg staged here on FWD)
static const int S2_GROUT = S2_GRIN  + S2_SEQ*S2_D;         // [SEQ*D] outgoing wide error / down-msg

// ------------------------------- gradient ACCUMULATORS (streamed RMW; effective batch = ACCUM) -------------------------------
static const int S2_GACC     = S2_AL(S2_GROUT + S2_SEQ*S2_D);
static const int S2_GACC_WQ  = S2_GACC;                     // [D*D]
static const int S2_GACC_WK  = S2_GACC_WQ  + S2_D*S2_D;     // [D*D]
static const int S2_GACC_WV  = S2_GACC_WK  + S2_D*S2_D;     // [D*D]
static const int S2_GACC_WO  = S2_GACC_WV  + S2_D*S2_D;     // [D*D]
static const int S2_GACC_WUP = S2_GACC_WO  + S2_D*S2_D;     // [FFN*D]
static const int S2_GACC_WDN = S2_GACC_WUP + S2_FFN*S2_D;   // [D*FFN]
static const int S2_GACC_WH  = S2_GACC_WDN + S2_D*S2_FFN;   // [VOCAB*D] (OUTPUT)
static const int S2_GACC_EMB = S2_GACC_WH  + S2_VOCAB*S2_D; // [VOCAB*D] (INPUT)
static const int S2_GACC_POS = S2_GACC_EMB + S2_VOCAB*S2_D; // [SEQ*D]   (INPUT)
static const int S2_GACC_END = S2_GACC_POS + S2_SEQ*S2_D;

// ------------------------------- DEBUG raw-gradient dump (gradcheck TB) -------------------------------
// reserved always (keeps S2_BUFLEN stable); only written when built -DS2_DEBUG_GRADS.
static const int S2_DBG    = S2_AL(S2_GACC_END);
static const int S2_DG_WQ  = S2_DBG;                        // [D*D]
static const int S2_DG_WK  = S2_DG_WQ  + S2_D*S2_D;         // [D*D]
static const int S2_DG_WV  = S2_DG_WK  + S2_D*S2_D;         // [D*D]
static const int S2_DG_WO  = S2_DG_WV  + S2_D*S2_D;         // [D*D]
static const int S2_DG_WUP = S2_DG_WO  + S2_D*S2_D;         // [FFN*D]
static const int S2_DG_WDN = S2_DG_WUP + S2_FFN*S2_D;       // [D*FFN]
static const int S2_DG_WH  = S2_DG_WDN + S2_D*S2_FFN;       // [VOCAB*D]
static const int S2_DG_GR  = S2_DG_WH  + S2_VOCAB*S2_D;     // [SEQ*D]  gR (== gRout source)
static const int S2_DG_GQ  = S2_DG_GR  + S2_SEQ*S2_D;       // [SEQ*D]
static const int S2_DG_GK  = S2_DG_GQ  + S2_SEQ*S2_D;       // [SEQ*D]
static const int S2_DG_GV  = S2_DG_GK  + S2_SEQ*S2_D;       // [SEQ*D]
static const int S2_DG_END = S2_DG_GV  + S2_SEQ*S2_D;

// ------------------------------- SLOT parity (cross-sub-batch pipelining), S=2 this version -------------------------------
// slot 0 = base [S2_WR..S2_GROUT_END); slot 1 = one appended mirror span. Masters/adam/gacc are single-slot.
static const int S2_SLOTS    = 2;
static const int S2_PAR_SPAN = (S2_GROUT + S2_SEQ*S2_D) - S2_WR;   // contiguous per-slot wide/msg span (words)
static const int S2_PAR      = S2_AL(S2_DG_END);                    // start of the slot>=1 mirror block
static const int S2_PAR_OFF  = S2_PAR - S2_WR;                      // add to a slot-0 wide/msg offset -> slot-1 copy
static const int S2_PAR_END  = S2_PAR + (S2_SLOTS-1)*S2_PAR_SPAN;
// slot-indexed token / target lanes (int32)
static const int S2_TOK      = S2_PAR_END;                          // [SLOTS*SEQ]
static const int S2_TGT      = S2_TOK + S2_SLOTS*S2_SEQ;            // [SLOTS*SEQ]

// ------------------------------- BACKWARD activation-gradient scratch (DDR; streamed row-blocked) -------------------------------
// The G0 footprint-wall fix (§2.3/§2.4): the bwd activation-gradient tensors were on-chip full-SEQ `static`
// arrays. They now live in DDR and are streamed as [SEQ_TILE x D] / [SEQ_TILE x FFN] strips. SINGLE bank
// (one slot in flight per bwd invocation; every grad folds into its GACC before the kernel returns, so no
// per-slot mirror is needed here). ONE shared temp (S2_BG_TMP) covers the never-simultaneously-live
// {gM2, dx-route tmp, gR1}. These are DDR only — they add ZERO on-chip residency.
static const int S2_BG      = S2_AL(S2_TGT + S2_SLOTS*S2_SEQ);
static const int S2_BG_LG   = S2_BG;                        // [SEQ*VOCAB] gLG
static const int S2_BG_XF   = S2_BG_LG   + S2_SEQ*S2_VOCAB; // [SEQ*D]    gXF
static const int S2_BG_RP   = S2_BG_XF   + S2_SEQ*S2_D;     // [SEQ*D]    gRP (incoming residual grad)
static const int S2_BG_M    = S2_BG_RP   + S2_SEQ*S2_D;     // [SEQ*D]    gM
static const int S2_BG_HD   = S2_BG_M    + S2_SEQ*S2_D;     // [SEQ*FFN]  gHD
static const int S2_BG_X2   = S2_BG_HD   + S2_SEQ*S2_FFN;   // [SEQ*D]    gX2
static const int S2_BG_R    = S2_BG_X2   + S2_SEQ*S2_D;     // [SEQ*D]    gR  (running residual grad / emit src)
static const int S2_BG_ATTN = S2_BG_R    + S2_SEQ*S2_D;     // [SEQ*D]    gATTN
static const int S2_BG_Q    = S2_BG_ATTN + S2_SEQ*S2_D;     // [SEQ*D]    gQ
static const int S2_BG_K    = S2_BG_Q    + S2_SEQ*S2_D;     // [SEQ*D]    gK
static const int S2_BG_V    = S2_BG_K    + S2_SEQ*S2_D;     // [SEQ*D]    gV
static const int S2_BG_X1   = S2_BG_V    + S2_SEQ*S2_D;     // [SEQ*D]    gX1
static const int S2_BG_TMP  = S2_BG_X1   + S2_SEQ*S2_D;     // [SEQ*D]    shared temp (gM2 / dx-route tmp / gR1)
static const int S2_BG_END  = S2_BG_TMP  + S2_SEQ*S2_D;

static const int S2_BUFLEN   = S2_AL(S2_BG_END);

// per-slot span contiguity (single additive slot offset must be valid)
static_assert(S2_GRIN == S2_WLNFS + S2_SEQ, "parity span not contiguous: S2_GRIN has an alignment gap");
static_assert(S2_PAR_SPAN % S2_WL == 0,     "parity span must be 16-word aligned");
static_assert(S2_PAR_OFF % S2_WL == 0,      "S2_PAR_OFF must be 16-word aligned (burst I/O)");
static_assert(S2_PAR % S2_WL == 0,          "mirror block base must be 16-word aligned");
static_assert(S2_PAR_END == S2_TOK,         "slot mirrors must end exactly at S2_TOK (no collision / no gap)");
static_assert((S2_SLOTS*S2_SEQ) % S2_WL == 0,"tok/tgt slot lanes must be 16-word aligned");
static_assert(S2_TILE_I % S2_K == 0,        "S2_TILE_I must be a multiple of S2_K (K-block tiling)");
static_assert(S2_H*S2_DH == S2_D,           "H*DH must equal D");
static_assert(S2_BG % S2_WL == 0,           "backward-grad scratch base must be 16-word aligned (burst I/O)");
static_assert((S2_SEQ*S2_D) % S2_WL == 0,   "SEQ*D must be 16-word aligned (strip burst I/O)");
static_assert((S2_SEQ*S2_FFN) % S2_WL == 0, "SEQ*FFN must be 16-word aligned (strip burst I/O)");
static_assert((S2_SEQ*S2_VOCAB) % S2_WL == 0,"SEQ*VOCAB must be 16-word aligned (strip burst I/O)");

#endif
