# Connect4 MuZero scaling — dyadic vs maxplus vs dense (tropicalsgd)

_Two-player negamax-MCTS self-play; score = win + 0.5*draw in a gauntlet vs depth-limited negamax of increasing strength. Board value is the OBLIQUE win/draw/loss outcome. This tractable-compute study compares dyadic, max-plus, and dense models. A software model-closure audit does not cover MCTS/loss arithmetic or establish FPGA DSP use._

| arm | n | best vs neg-d2 (train) | vs neg-d1 | vs neg-d2 (gauntlet, mean) |
|---|---|---|---|---|
| tropical-dyadic | 1 | 0.50±0.00 | 0.50 | 0.50 |
