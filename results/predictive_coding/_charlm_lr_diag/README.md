# char-LM local-PC learning-rate sweep (supporting data for CHARLM_LOCAL_PC.md)

Same setup for all: ctx12, hidden 256, depth 4, 300k chars, wd 2e-4, drop 0.1, 14 epochs,
best-val. Shows the multiplier-free local-PC updates need a ~3× smaller lr than backprop;
the filename encodes the arm + lr (the jsonl predates the `lr` field):

| file | arm | lr | best val bpc | note |
|---|---|---|---|---|
| `lr_bpdyadic.jsonl` | bp_dyadic (backprop bar) | 1.5e-3 | 2.370 | smooth |
| `lr_relax5e4.jsonl` | relax (sync local PC) | 5e-4 | **2.327** | smooth — best; matches/beats bp |
| `lr_relax2e4.jsonl` | relax (sync local PC) | 2e-4 | 2.400 | smooth but still descending @ep13 (lr too low) |
| `lr_async5e4.jsonl` | relax_async (async/fabric local PC) | 5e-4 | **2.341** | smooth — fabric form matches |

At lr 1.5e-3 (the backprop lr) the local-PC arms instead peak at ~ep2 (2.68) then degrade
— the early-peak-then-degrade that masqueraded as a fundamental gap until the lr was fixed.
