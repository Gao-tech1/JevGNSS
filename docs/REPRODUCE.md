# Reproduction

## Check the bundle

```bash
python scripts/reproduce.py --mode smoke --assets /path/to/processed-assets
```

This verifies the registered file hashes, the four-route whitelist, the row counts, 30 student checkpoints, 48 positioning checkpoints, and 12 TDL checkpoints.

## Recompute the frozen results

```bash
python scripts/reproduce.py --mode evaluate --assets /path/to/processed-assets
```

The program aggregates the per-route results, recomputes the frozen-student risk AUC, and runs nonlinear WLS on every positioning candidate using the published frozen weights and corrections. The maximum coordinate difference must be below 0.1 mm. Every metric must match `expected/metrics.json` within `1e-6`.

`--skip-numerical-replay` is only for a fast automatic test. It is not a full reproduction of the paper numbers.

## Retraining

`train-and-evaluate` keeps the training entry point in the source tree. It is not the published reproduction path. The published check is the frozen evaluation above.
