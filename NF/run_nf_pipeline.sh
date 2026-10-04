#!/usr/bin/env bash
set -euo pipefail

N_SIMS="${N_SIMS:-1000}"
JULIA_THREADS="${JULIA_THREADS:-8}"

echo "=== 1/4 Generate simulator dataset ==="
julia -t "${JULIA_THREADS}" 01_generate_nf_dataset.jl \
  --n "${N_SIMS}" \
  --seed 123 \
  --outdir nf_results

echo "=== 2/4 Train NPE + NSF ==="
python 02_train_npe.py train \
  --data nf_results/simulations.csv \
  --prior nf_results/prior_bounds.csv \
  --outdir nf_results/npe \
  --device cuda \
  --noise-fraction 0.01

echo "=== 3/4 Sample target posterior ==="
python 02_train_npe.py sample \
  --model nf_results/npe/posterior.pkl \
  --target nf_results/target_landscape.csv \
  --outdir nf_results/posterior \
  --n-samples 5000

echo "=== 4/4 Posterior predictive validation ==="
julia -t "${JULIA_THREADS}" 03_validate_posterior.jl \
  --samples nf_results/posterior/posterior_samples.csv \
  --target nf_results/target_landscape.csv \
  --n 200 \
  --outdir nf_results/validation

echo "Done."
