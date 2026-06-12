// forge_ce_lut.hpp — exp2 ROM for the balanced LUT cross-entropy gradient.
//
// FT_CE_EXP2_Q16[i] = round(65536 * 2^(-i/16)): Q16 fixed point, 16 index
// steps per octave, 16-octave range. On fabric this is a 256-word BRAM ROM;
// the float-sim counterpart is tikitorch.functional.shift_softmax's
// exp2_lut_table. Index = (zmax - logit) >> CFG_CE_ZSHIFT (a comparator and
// an arithmetic shift -- no multiply, no DSP).
//
// The gradient consumer MUST scatter s = sum(p) (not 1) at the target so
// every gradient row sums to zero: an unbalanced row bias is integrated by
// sign-based weight updates into a constant-predictor collapse (measured on
// MorphSSM char-LM; see tikitorch/functional.py TropicalCrossEntropyFn and
// tests/test_functional.py::test_lut_shift_ce_grad_rows_sum_to_zero).

#ifndef FORGE_CE_LUT_HPP
#define FORGE_CE_LUT_HPP

#include <cstdint>

static const int32_t FT_CE_EXP2_Q16[256] = {
     65536,  62757,  60097,  57549,  55109,  52773,  50535,  48393,
     46341,  44376,  42495,  40693,  38968,  37316,  35734,  34219,
     32768,  31379,  30048,  28774,  27554,  26386,  25268,  24196,
     23170,  22188,  21247,  20347,  19484,  18658,  17867,  17109,
     16384,  15689,  15024,  14387,  13777,  13193,  12634,  12098,
     11585,  11094,  10624,  10173,   9742,   9329,   8933,   8555,
      8192,   7845,   7512,   7194,   6889,   6597,   6317,   6049,
      5793,   5547,   5312,   5087,   4871,   4664,   4467,   4277,
      4096,   3922,   3756,   3597,   3444,   3298,   3158,   3025,
      2896,   2774,   2656,   2543,   2435,   2332,   2233,   2139,
      2048,   1961,   1878,   1798,   1722,   1649,   1579,   1512,
      1448,   1387,   1328,   1272,   1218,   1166,   1117,   1069,
      1024,    981,    939,    899,    861,    825,    790,    756,
       724,    693,    664,    636,    609,    583,    558,    535,
       512,    490,    470,    450,    431,    412,    395,    378,
       362,    347,    332,    318,    304,    292,    279,    267,
       256,    245,    235,    225,    215,    206,    197,    189,
       181,    173,    166,    159,    152,    146,    140,    134,
       128,    123,    117,    112,    108,    103,     99,     95,
        91,     87,     83,     79,     76,     73,     70,     67,
        64,     61,     59,     56,     54,     52,     49,     47,
        45,     43,     41,     40,     38,     36,     35,     33,
        32,     31,     29,     28,     27,     26,     25,     24,
        23,     22,     21,     20,     19,     18,     17,     17,
        16,     15,     15,     14,     13,     13,     12,     12,
        11,     11,     10,     10,     10,      9,      9,      8,
         8,      8,      7,      7,      7,      6,      6,      6,
         6,      5,      5,      5,      5,      5,      4,      4,
         4,      4,      4,      4,      3,      3,      3,      3,
         3,      3,      3,      2,      2,      2,      2,      2,
         2,      2,      2,      2,      2,      2,      2,      1,
         1,      1,      1,      1,      1,      1,      1,      1
};

#endif  // FORGE_CE_LUT_HPP
