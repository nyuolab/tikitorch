// forge_pc_xformer_config.hpp — dims + DDR layout for the DISTRIBUTED PC transformer-block shard kernel
// (forge_pc_xformer_block.cpp). ONE transformer block per physical FPGA; a whole transformer trained
// MODEL-PARALLEL across L chips, each holding ONE block's weights + optimizer state in its own DDR, exchanging
// ONLY a neighbour hidden-state (downstream, forward) + a neighbour error (upstream, backward). Role
// (INPUT/HIDDEN/OUTPUT) + phase (FWD/BWD) read from a CFG word so ONE AFI loads on every slot.
//
// Same dims as the LLM instance's single-card trainer (forge_s1_config.hpp) so the per-block arithmetic is the
// validated one; the DISTRIBUTED sharding + neighbour-message I/O is the new part. Dims are intentionally
// small (D=64) so a full block + its backward + dyadic-Adam state fit one SLR.
#ifndef FORGE_PC_XFORMER_CONFIG_HPP
#define FORGE_PC_XFORMER_CONFIG_HPP

#define X_D        64        // d_model (log2=6 -> LayerNorm mean = sum >> 6)
#define X_LOG2_D   6
#define X_H        2         // attention heads
#define X_DH       32        // head dim (H*DH == D)
#define X_FFN      256       // FFN inner width (4*D)
#define X_SEQ      64        // sequence length (the per-kick "batch" = one sequence of tokens)
#define X_VOCAB    64        // vocab size
#define X_MBITS    2         // fine-dyadic LayerNorm mantissa bits (1.0..1.75 in 0.25 steps)
#define X_LUT_N    256       // exp2 LUT entries (host-staged ROM)

// activation max-shift targets (keep activations in a fixed int band, like forge_pc_block)
#define X_ABITS    9         // hidden-state message band (the cross-chip down message; fits int16)
#define X_EBITS    12        // error message band (the cross-chip up message; fits int16)
#define X_AT_DELTA 2         // attention softmax gap divisor (Delta = 4)
#define X_AT_PFRAC 16        // Q16 attention probabilities

// dyadic-Adam (multiplier-free): m,v EMA betas = 1 - 2^-3. NOTE: the master weights here are int16 (small
// range), so the per-step update must be O(1) integer — the dyadic-Adam step m>>se is already ~sign(m)*adapt
// (a few), so LR_SHIFT=0 (>0 rounds it to zero = frozen weights). Weight decay on int16 masters must be very
// gentle (W>>5 = W/32 per step swamps the ±few update); WD_SHIFT>=14 ~ off.
#define X_ADAM_BSH 3
#define X_LR_SHIFT 0
#define X_WD_SHIFT 14

// roles / phases (CFG word lanes)
#define X_PH_FWD   0
#define X_PH_BWD   1
#define X_RL_INPUT 0         // chip 0: token-embed + pos, then block 0
#define X_RL_HID   1         // chips 1..L-2: block only
#define X_RL_OUT   2         // chip L-1: block, then final-LN + LM head + CE loss/grad

#define X_WL 16              // int32 lanes per 512-bit word

// ---- DDR layout (int32 words; every region 16-word aligned; ONE layout for all roles -> one AFI) ----
// Persistent per chip (never crosses): CFG, block weights, dyadic-Adam m/v, INPUT embed/pos, OUTPUT lnf/head.
// Staged per kick by host: HIN (down msg in), TGT (OUTPUT targets), EIN (up msg in), TOK (INPUT token ids).
// Written by kernel: HOUT (down msg out), EOUT (up msg out), LOSS/LOGITS (OUTPUT). Cached fwd->bwd: the
// residual-stream + LN stats + attn probs/argmax + relu mask (so BWD recomputes nothing).
#define XA(name, prev, size) static const int name = (prev); static const int name##_END = (prev) + (size);
#define X_AL(n) ((((n)+X_WL-1)/X_WL)*X_WL)        // round up to 16

static const int XO_CFG   = 0;
static const int XO_WQ    = XO_CFG  + X_WL;                 // [D*D] query proj (int16 master)
static const int XO_WK    = XO_WQ   + X_D*X_D;              // [D*D]
static const int XO_WV    = XO_WK   + X_D*X_D;              // [D*D]
static const int XO_WO    = XO_WV   + X_D*X_D;              // [D*D] attn output proj
static const int XO_WUP   = XO_WO   + X_D*X_D;              // [FFN*D]
static const int XO_WDN   = XO_WUP  + X_FFN*X_D;            // [D*FFN]
static const int XO_LN1G  = XO_WDN  + X_D*X_FFN;            // [D] gamma (int16)
static const int XO_LN1B  = XO_LN1G + X_D;                  // [D] beta
static const int XO_LN2G  = XO_LN1B + X_D;                  // [D]
static const int XO_LN2B  = XO_LN2G + X_D;                  // [D]
static const int XO_EMBED = XO_LN2B + X_D;                  // [VOCAB*D] (INPUT role)
static const int XO_POS   = XO_EMBED+ X_VOCAB*X_D;          // [SEQ*D]   (INPUT role)
static const int XO_LNFG  = XO_POS  + X_SEQ*X_D;            // [D]       (OUTPUT role) final LN gamma
static const int XO_LNFB  = XO_LNFG + X_D;                  // [D]
static const int XO_WHEAD = XO_LNFB + X_D;                  // [VOCAB*D] (OUTPUT role) LM head
static const int XO_LUT   = XO_WHEAD+ X_VOCAB*X_D;          // [LUT_N] exp2 ROM (host-staged), Q16
// ---- dyadic-Adam state (m,v int32) for every trainable matrix, same order ----
#define X_NW (X_D*X_D*4 + X_FFN*X_D + X_D*X_FFN + X_VOCAB*X_D)   // all block + head weights (LN params updated too but small)
static const int XO_MOM   = X_AL(XO_LUT + X_LUT_N);        // [X_NW] first-moment EMA
static const int XO_VAR   = XO_MOM  + X_NW;                 // [X_NW] second-moment EMA
// ---- per-kick staged messages ----
static const int XO_TOK   = XO_VAR  + X_NW;                 // [SEQ]   token ids (INPUT role)
static const int XO_TGT   = XO_TOK  + X_AL(X_SEQ);          // [SEQ]   target ids (OUTPUT role)
static const int XO_HIN   = XO_TGT  + X_AL(X_SEQ);          // [SEQ*D] incoming hidden state (down msg)
static const int XO_EIN   = XO_HIN  + X_SEQ*X_D;            // [SEQ*D] incoming error (up msg)
static const int XO_HOUT  = XO_EIN  + X_SEQ*X_D;            // [SEQ*D] outgoing hidden state (down msg)
static const int XO_EOUT  = XO_HOUT + X_SEQ*X_D;            // [SEQ*D] outgoing error (up msg)
// ---- cached activations (fwd -> bwd), per token ----
static const int XO_RIN   = XO_EOUT + X_SEQ*X_D;            // [SEQ*D] block input residual (post-embed or =HIN)
static const int XO_X1    = XO_RIN  + X_SEQ*X_D;            // [SEQ*D] LN1 output
static const int XO_X2    = XO_X1   + X_SEQ*X_D;            // [SEQ*D] LN2 output
static const int XO_ATTNO = XO_X2   + X_SEQ*X_D;            // [SEQ*D] attention output (pre WO)
static const int XO_RMID  = XO_ATTNO+ X_SEQ*X_D;            // [SEQ*D] residual after attn (block midpoint)
static const int XO_HUP   = XO_RMID + X_SEQ*X_D;            // [SEQ*FFN] FFN hidden (post-relu)
static const int XO_P     = XO_HUP  + X_SEQ*X_FFN;          // [H*SEQ*SEQ] attn probs Q16 (cached for bwd)
static const int XO_LN1S  = XO_P    + X_H*X_SEQ*X_SEQ;      // [SEQ*2] LN1 (e,nq) fine-dyadic stats
static const int XO_LN2S  = XO_LN1S + X_SEQ*2;              // [SEQ*2]
static const int XO_LNFS  = XO_LN2S + X_SEQ*2;              // [SEQ*2] (OUTPUT)
static const int XO_RMASK = XO_LNFS + X_SEQ*2;              // [SEQ*FFN] relu mask
static const int XO_LOGIT = XO_RMASK+ X_SEQ*X_FFN;          // [SEQ*VOCAB] (OUTPUT) logits
static const int XO_LOSS  = XO_LOGIT+ X_SEQ*X_VOCAB;        // [16] (OUTPUT) loss accumulator
static const int XO_RP    = X_AL(XO_LOSS + 16);             // [SEQ*D] cached final residual stream R' (BWD LNF/EOUT seed)
static const int X_BUFLEN = X_AL(XO_RP + X_SEQ*X_D);

#endif
