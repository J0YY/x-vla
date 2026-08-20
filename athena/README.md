# Athena experiments

This directory runs the χ-VLA LIBERO evaluation natively on Athena through Slurm. It does not
depend on Modal. The runner preserves the paper protocol:

- official packaged LIBERO initial states, with no random-reset fallback
- all ten LIBERO-Object tasks
- 50 canonical episodes per task for the matched evaluation
- 280 simulator steps, 10 settle steps, and receding-horizon execution
- the corrected 180-degree camera transform, `[::-1, ::-1]`
- the vocabulary and normalization statistics reconstructed from the exact training cache

The conventional control uses the same χ-VLA skeleton, dimensions, training data, and linear
action head. Only the internal block family changes to softmax attention, SwiGLU, and RMSNorm.
The SwiGLU ranks are chosen to parameter-match the χ model at each block width.

Submit a smoke test from Athena with:

```bash
mkdir -p /work/joy/x-vla-workshop/logs /work/joy/x-vla-workshop/results
sbatch --job-name=xvla-smoke athena/slurm_xvla.sbatch \
  --mode smoke \
  --architecture conventional \
  --checkpoint artifacts/ckpt_linear_conventional_vit_s0.pt \
  --cache artifacts/libero_frames_100000_64.pkl \
  --output results/conventional_s0_smoke.json
```

After the smoke test passes, change `--mode smoke` to `--mode capability` for the full evaluation.
Use `--mode profile` when only latency, memory, and parameter counts are needed.
