"""tikitorch.research — the shared PC research engine, promoted from
scripts/ (repo-split 2026-07-29) so both the text-paper drivers and the
ontogenic-agent repo consume ONE engine via the package.

scripts/pc_convrnn_charlm.py and scripts/nanogpt_shakespeare.py remain as
identity-preserving shims (same module object via sys.modules aliasing) so
every existing flat import keeps working unchanged."""
