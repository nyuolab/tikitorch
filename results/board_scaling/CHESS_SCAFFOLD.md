# Chess MuZero scaffold — dyadic tropical configuration

_Pipeline scaffold, NOT a strength or whole-system hardware claim. The Python loop (python-chess env,
4096-action from-to encoding, 769-d planes) runs end-to-end and its training loss falls. The dyadic
network components are shift/add-oriented, but MCTS, normalization, loss, and orchestration retain
explicit arithmetic boundaries. In this 14-game evaluation the trained agent loses to random and
draws every game against the 1-ply material agent._

Config: dyadic tropical, latent 64, 12 sims, 8 iters x 10 self-play games, TropicalSGD optimizer.

Training loss: 27.546 -> 11.999 over 8 iters (curve [27.5462, 23.1972, 20.2098, 18.7021, 16.6047, 14.6859, 13.1631, 11.9995]).

| opponent | win | draw | loss | score |
|---|---|---|---|---|
| random | 0% | 14% | 86% | 0.07 |
| material(1-ply) | 0% | 100% | 0% | 0.50 |
| random (untrained control) | 0% | 79% | 21% | 0.39 |
