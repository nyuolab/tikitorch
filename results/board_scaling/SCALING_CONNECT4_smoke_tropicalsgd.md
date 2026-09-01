# Connect4 MuZero scaling — dyadic vs maxplus vs dense (tropicalsgd)

_Two-player negamax-MCTS self-play; score = win + 0.5*draw in a gauntlet vs depth-limited negamax of increasing strength. Board value is the OBLIQUE win/draw/loss outcome. Reported MuZero/AlphaZero SOLVE Connect4 (perfect play); this is the tractable-compute, multiplier-free (zero-DSP) analogue — the point is the substrate scales structurally, and that the dyadic oblique core holds up where max-plus underfits._

| arm | n | best vs neg-d2 (train) | vs neg-d1 | vs neg-d2 (gauntlet, mean) |
|---|---|---|---|---|
| tropical-dyadic | 1 | 0.50±0.00 | 0.50 | 0.50 |