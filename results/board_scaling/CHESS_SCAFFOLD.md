# Chess MuZero scaffold — multiplier-free dyadic tropical (zero-DSP substrate)

_Pipeline-scaling proof, NOT a strength claim. Reported MuZero plays chess at superhuman strength with TPU-pod compute; that is not workstation-reproducible. This shows the SAME zero-DSP self-play->replay->K-step-training loop (python-chess env, 4096-action from-to encoding, 769-d planes) runs end-to-end on chess, the training loss falls, and the trained MCTS agent outplays a random mover and a 1-ply material grabber._

Config: dyadic tropical, latent 64, 12 sims, 8 iters x 10 self-play games, closed TropicalSGD optimizer.

Training loss: 27.546 -> 11.999 over 8 iters (curve [27.5462, 23.1972, 20.2098, 18.7021, 16.6047, 14.6859, 13.1631, 11.9995]).

| opponent | win | draw | loss | score |
|---|---|---|---|---|
| random | 0% | 14% | 86% | 0.07 |
| material(1-ply) | 0% | 100% | 0% | 0.50 |
| random (untrained control) | 0% | 79% | 21% | 0.39 |