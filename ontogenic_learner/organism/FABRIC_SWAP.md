# Warm-start swap protocol: fabric phylogeny → organism ontogeny

When the fabric program's 30M (later 124M) checkpoint lands, it becomes this organism's new
phylogeny. **A swap is a new birth from an older phylogeny** — exactly the paradigm's
division of labor (GPU/fabric = phylogeny; organism = ontogeny).

## Formats
- **Fabric masters**: flat float32 arrays per chip (`master<l>.f32` / `xfmp_master<l>.f32`),
  per-layer `(W, b)` in layer order (see `forge_pc_xformer_mp_shard.cpp` dump_master_file).
- **Organism**: torch `state_dict` of `DyadicPCNet` (`layers.{l}.weight/bias`, plus skip
  projections where dims change).

## Swap procedure
1. Converter: reshape each chip's flat arrays to `(out, in)` per the dims header; assemble a
   `state_dict`; load with `strict=False` and verify every tensor was consumed.
2. **Vocab/dims must match or be re-frozen**: a new phylogeny with different D/V/ctx means a
   NEW vocab.json and a NEW life (the input dim `ctx·V` is fixed per life).
3. Adam moments are **not portable** across substrates — both optimizers initialize fresh.
4. Surprise EMA re-warmed on 20 batches of the base domain; reservoir starts empty (or
   seeded with base-domain memories, the G1 recipe).
5. Ledger records `BIRTH` with the phylogeny's provenance (fabric run tag + AGFI + commit).

## Status
- v0 births from an on-box wiki phylogeny (`cfg.birth_steps` cascade steps).
- Blocked/waiting on the fabric side: their 30M S2 kernel training run; the L=2 backward
  defect fix (dossier: `../results/g1b_cert/G1B_RESULT.md`) for organism-on-fabric.
