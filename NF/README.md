# NPE + Neural Spline Flow pipeline for the Approach-A phage ODE

This folder is a complete first-pass pipeline for learning

`q_psi(theta_env | F)`

from the deterministic ODE in `approach_a(1).jl`.

## What is reused exactly from Approach A

The Julia simulator keeps:

- the 7 states `[B, I1_1, I1_2, I2_1, I2_2, V1, V2]`
- the exact RHS
- `ALPHA = 1e-6`
- the per-strain `lambda`
- the burst/latent-period trade-off `beta(lambda) = RHO * (2/lambda - ECLIPSE)`
- `LAM_REF = 0.02`
- `T_END = 100`
- the 11-point phenotype grid from `0.010` to `0.050`
- fitness `log(V1(T)/V2(T))/T`
- `Tsit5()` with `reltol=abstol=1e-9`
- environment `theta = [B0, M0, delta, r, K]`

The NPE is learned in `z = log(theta)` space, matching the conditioning choice in Approach A.

## Important prior choice

`01_generate_nf_dataset.jl` currently uses a **factor-5 log-uniform prior around THETA_TRUE** purely as a computational starting point.

That is not a claim that these are the final biologically defensible prior bounds. Edit `PRIOR_LOW` and `PRIOR_HIGH` once you have agreed the biologically admissible environment.

The script optionally enforces `K > B0`.

## Why there is a small fitness-noise option

The ODE is deterministic. Conditioning on an exact landscape can therefore produce an extremely thin or even lower-dimensional inverse set.

The Python training script defaults to `--noise-fraction 0.01`, meaning each fitness coordinate receives Gaussian noise with SD equal to 1% of that coordinate's training-set SD.

Interpretation: the flow learns environments compatible with the target **within a small landscape tolerance**. Set `--noise-fraction 0` to test exact deterministic conditioning.

## 1. Julia environment on Aura

Use the same Julia environment that already runs Approach A. If needed:

```julia
using Pkg
Pkg.add("OrdinaryDiffEq")
Pkg.add("OrdinaryDiffEqTsit5")
```

Dataset generation is CPU work. The ODE has only seven states, so many CPU threads are more useful than the GPU here.

Smoke test:

```bash
julia -t 8 01_generate_nf_dataset.jl \
    --n 1000 \
    --seed 123 \
    --outdir nf_results
```

Production example:

```bash
julia -t 32 01_generate_nf_dataset.jl \
    --n 50000 \
    --seed 123 \
    --outdir nf_results
```

Outputs:

- `nf_results/simulations.csv`
- `nf_results/target_landscape.csv`
- `nf_results/theta_true.csv`
- `nf_results/prior_bounds.csv`
- `nf_results/simulation_config.txt`

## 2. Python environment on Aura

Create an environment, then install PyTorch appropriate for Aura's CUDA setup.

```bash
python -m venv nf_env
source nf_env/bin/activate
pip install --upgrade pip
```

Install the PyTorch build appropriate to the CUDA module available on Aura, then:

```bash
pip install sbi numpy pandas
```

Check the GPU:

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

## 3. Train NPE + Neural Spline Flow

```bash
python 02_train_npe.py train \
    --data nf_results/simulations.csv \
    --prior nf_results/prior_bounds.csv \
    --outdir nf_results/npe \
    --device cuda \
    --noise-fraction 0.01 \
    --hidden-features 128 \
    --num-transforms 5 \
    --batch-size 512
```

The flow learns `q_psi(z | F)`, where `z = log(theta_env)` and the complete 11-point fitness vector is the conditioning input.

Outputs:

- `nf_results/npe/posterior.pkl`
- `nf_results/npe/metadata.json`
- `nf_results/npe/holdout_simulations.csv`

## 4. Condition on the Approach-A target landscape

```bash
python 02_train_npe.py sample \
    --model nf_results/npe/posterior.pkl \
    --target nf_results/target_landscape.csv \
    --outdir nf_results/posterior \
    --n-samples 5000
```

Outputs:

- `nf_results/posterior/posterior_samples.csv`
- `nf_results/posterior/posterior_summary.csv`
- `nf_results/posterior/conditioning_target.csv`

Each sample is a candidate environment from the learned inverse distribution.

## 5. Posterior-predictive validation through the original ODE

This is essential. Do not judge the flow only from its training likelihood.

```bash
julia -t 32 03_validate_posterior.jl \
    --samples nf_results/posterior/posterior_samples.csv \
    --target nf_results/target_landscape.csv \
    --n 200 \
    --outdir nf_results/validation
```

Outputs:

- `posterior_predictive_summary.csv`
- `posterior_predictive_landscapes.csv`

The key quantity is landscape RMSE after each posterior environment is fed back through the exact ODE.

## Recommended order on Aura

1. `N=1,000` simulations — make sure the pipeline runs end-to-end.
2. `N=10,000` — inspect posterior predictive RMSE and posterior diversity.
3. `N=50,000` or `100,000` — only if performance continues to improve.
4. Repeat for several held-out synthetic targets, not only `THETA_TRUE`.
5. Compare posterior modes with the multi-start degeneracy already observed in Approach A.
6. Only after the ODE pipeline is validated, send posterior environments through the stochastic ABM.

## What is computationally expensive?

The expensive part is generating `N x K` ODE simulations:

`N_environments x 11 phenotypes`.

This is why the Julia script parallelises environments across CPU threads.

NSF training is the GPU stage and should usually be much cheaper than generating a very large simulator dataset.

## Scientific outputs to inspect

A successful first result should show:

1. Posterior samples are diverse in `theta_env`.
2. Several distinct parameter combinations appear if the inverse problem is non-identifiable.
3. When those samples are rerun through the original ODE, their generated landscapes remain close to `F_target`.
4. The known `THETA_TRUE` is compatible with the inferred inverse distribution, but the posterior need not collapse onto it if the landscape is genuinely non-identifying.
