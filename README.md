# RELACE

**Retrospective Likelihood-based Action Credit Estimation for Long-Horizon Language Agents**

**Authors:** Sayak Chakrabarti ([sayaksc@gmail.com](mailto:sayaksc@gmail.com)) and Sathish Indurthi ([sathish.indurthi@gmail.com](mailto:sathish.indurthi@gmail.com)).

RELACE is a critic-free reinforcement-learning method for multi-turn language agents. It uses the observed outcome of a trajectory to reassess its executed actions, then combines this retrospective signal with comparisons between actions taken from matching environment states.

**This repository is modified from [Lang Feng and collaborators' verl-agent](https://github.com/langfengQ/verl-agent), which is built on [veRL](https://github.com/volcengine/verl).** We retain its agent-environment rollout infrastructure, PPO training components, and GiGPO grouping utilities, and add RELACE advantage estimation and hindsight scoring. Original copyright and license notices are preserved.

## Method

For each task, RELACE samples a group of trajectories. After termination:

1. The frozen behavior policy scores each executed action under its original context and an outcome-augmented context using teacher forcing.
2. The ratio of the two length-normalized likelihood scores is normalized within its trajectory and clipped.
3. This retrospective factor reweights the discounted task return. Optional adjacent-step smoothing stabilizes the resulting targets.
4. Weighted targets are compared within state-matched groups to form local advantages. A success-protecting mask suppresses negative local corrections on successful trajectories.
5. The local signal is added to the episode-level relative advantage for the policy update.

RELACE changes advantage-estimation targets; it does not require an auxiliary critic, a learned reward model, or extra autoregressive rollouts for credit scoring. Terminal outcomes are used only for training-time credit estimation.

The implementation is in [relace/core_relace.py](relace/core_relace.py), with hindsight scoring integrated into [ray_trainer.py](verl/trainer/ppo/ray_trainer.py).

## Results reported in the manuscript

The accompanying manuscript is titled *RELACE: Retrospective Likelihood-based Action Credit Estimation for Long-Horizon Language Agents*. Its Table 1 reports:

| Initial policy | ALFWorld success (%) | WebShop success (%) | WebShop score |
|---|---:|---:|---:|
| Qwen2.5-1.5B-Instruct | 96.35 ± 0.45 | 79.43 ± 2.74 | 90.90 ± 1.04 |
| Qwen2.5-7B-Instruct | 97.66 ± 0.78 | 82.29 ± 1.80 | 90.34 ± 0.80 |

These are manuscript results, not results produced by the repository's smoke tests. The release launchers use Appendix C's listed algorithm settings; reproducing the reported aggregates also requires the manuscript's evaluation and seed protocol.

## Installation

Training requires Linux, NVIDIA GPUs, a CUDA-compatible driver/toolkit, and Python 3.10 or newer. The launchers default to one node with eight GPUs. The manuscript's timing experiment used eight H100 80 GB GPUs; memory requirements depend on model size, sequence length, and batching.

From the repository root, create and activate a dedicated environment:

```bash
conda create -n relace python=3.10 -y
conda activate relace
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
python -m pip install flash-attn==2.7.4.post1 --no-build-isolation
python -m pip check
```

The requirements select vLLM 0.11.0 with PyTorch 2.8.0 and torchvision 0.23.0, following [vLLM's versioned CUDA requirements](https://github.com/vllm-project/vllm/blob/v0.11.0/requirements/cuda.txt), and Transformers 4.57.1. Build FlashAttention against this installed PyTorch/CUDA stack. The package and Python namespace remain `verl` for compatibility with the inherited framework.

### ALFWorld

```bash
python -m pip install alfworld==0.4.2
alfworld-download -f
```

ALFWorld uses its default cache directory unless `ALFWORLD_DATA` is set. If you choose another location, set it before both downloading and training. The launcher uses the text-based `alfworld/AlfredTWEnv` environment.

### WebShop

Install OpenJDK 11, then run the environment setup helper:

```bash
conda install -c conda-forge openjdk=11 -y
bash scripts/setup_webshop.sh
```

The helper installs WebShop-specific runtime dependencies, downloads the 1,000-product catalog and goal annotations, downloads the spaCy English model, and builds the Lucene search index. It preserves the training PyTorch/Transformers stack instead of installing the old vendored WebShop training requirements.

The supplied launcher uses `env.webshop.use_small=True`, matching the catalog prepared by this helper. Downloaded datasets and indexes are not included in the repository. Set `JAVA_HOME` to your Java 11 installation if Pyserini cannot locate the JVM.

## Train RELACE

Run from the repository root after preparing the chosen environment:

```bash
# Qwen2.5-1.5B-Instruct
bash examples/relace_trainer/run_alfworld.sh
bash examples/relace_trainer/run_webshop.sh

# Qwen2.5-7B-Instruct
MODEL_PATH=Qwen/Qwen2.5-7B-Instruct bash examples/relace_trainer/run_alfworld.sh
MODEL_PATH=Qwen/Qwen2.5-7B-Instruct bash examples/relace_trainer/run_webshop.sh
```

Each launcher creates local placeholder Parquet rows under `data/relace/text/`. The environments provide the actual tasks at reset; no external dataset is downloaded to create these rows. Training logs and checkpoints are written to `outputs/<experiment-name>/`.

Useful environment-variable overrides:

```bash
MODEL_PATH=Qwen/Qwen2.5-1.5B-Instruct \
NUM_GPUS=8 SEED=1 \
OUTPUT_DIR="$PWD/outputs/relace_webshop_seed1" \
bash examples/relace_trainer/run_webshop.sh
```

Use `DATA_DIR`, `EXPERIMENT_NAME`, `TRAIN_BATCH_SIZE`, `VAL_BATCH_SIZE`, `GROUP_SIZE`, `PPO_MINI_BATCH_SIZE`, `PPO_MICRO_BATCH_SIZE`, and `GPU_MEMORY_UTILIZATION` to adjust storage and resource settings. If reducing GPU count, adjust batching and memory settings together; the default profile targets eight GPUs.

Additional arguments are Hydra overrides and take precedence over the launch defaults:

```bash
bash examples/relace_trainer/run_webshop.sh \
  trainer.total_training_steps=200 \
  trainer.save_freq=25
```

To inspect the command without preparing data or starting training:

```bash
DRY_RUN=1 bash examples/relace_trainer/run_alfworld.sh
```

Logging defaults to the console. To enable Weights & Biases, supply `WANDB_API_KEY` through your environment and set `LOGGER=wandb`. No API keys are stored in the repository or required for console logging.

## Configuration

Select RELACE with `algorithm.adv_estimator=relace`. Its settings live under `algorithm.relace` in [ppo_trainer.yaml](verl/trainer/config/ppo_trainer.yaml). The launchers set the following defaults from Appendix C:

| Setting | Default |
|---|---|
| Training tasks / rollouts per task | 16 / 8 |
| Validation batch size | 128 |
| Actor learning rate | `1e-6` |
| Actor KL coefficient | `0.01` |
| KL shaping of environment reward | Disabled |
| Discount `algorithm.relace.gamma` | `0.6` |
| Likelihood-score temperature `temp` | `1.0` |
| Retrospective clipping `clip_min`, `clip_max` | `0.8`, `1.2` |
| Local-advantage weight `omega` | `1.0` |
| Temporal smoothing `smooth`, `alpha` | Enabled, `0.5` |
| Success protection `no_harm` | Enabled |
| State similarity threshold | `0.95` |
| PPO mini-batch size | 64; 128 for WebShop 7B |
| Rollout / validation temperature | `1.0` / `0.4` |
| Episode limit | ALFWorld: 50; WebShop: 15 |
| Validation frequency | Every 5 training steps |

The score temperature is distinct from the generation temperature. State grouping is configured through `algorithm.gigpo.enable_similarity`, `algorithm.gigpo.similarity_thresh`, and `algorithm.gigpo.mode` because RELACE reuses GiGPO's grouping utilities.

The launcher chooses the `alfworld` or `webshop` hindsight template automatically. `RELACE_TEMP`, `RELACE_CLIP_MIN`, `RELACE_CLIP_MAX`, `RELACE_OMEGA`, `RELACE_GAMMA`, `RELACE_SMOOTH`, `RELACE_ALPHA`, `RELACE_NO_HARM`, and `RELACE_HINDSIGHT_TEMPLATE` override the corresponding settings. `RELACE_GAMMA` sets both the algorithm discount and the RELACE discount.

Metrics use the `relace/` prefix, including retrospective-ratio statistics, local-advantage magnitude, success-mask statistics, and hindsight scoring time.

### GiGPO with RELACE-style smoothing and success protection

The GiGPO baseline (`algorithm.adv_estimator=gigpo`) can use the same two refinements RELACE applies to its local advantage. Both are off by default, which is the original GiGPO, and act only on the step-level term:

| Setting | Effect |
|---|---|
| `algorithm.gigpo.smooth`, `algorithm.gigpo.alpha` | Before the anchor-state-group normalization, smooth each step's discounted return along its trajectory: `q'_t = alpha * q_t + (1 - alpha) * q_{t+1}` (the last step keeps its own value). `alpha=1.0` is a no-op. |
| `algorithm.gigpo.no_harm` | After the anchor-state-group normalization, zero the *negative* step advantages of rows whose episode succeeded (`episode_success > 0`). The episode-level term is unchanged. |

The settings live under `algorithm.gigpo` in [ppo_trainer.yaml](verl/trainer/config/ppo_trainer.yaml). The GiGPO launchers use the same training profile as the RELACE ones, so the two are directly comparable, and expose the options as environment variables:

```bash
bash examples/gigpo_trainer/run_webshop.sh                  # smoothing (alpha 0.5) + no-harm on
GIGPO_ALPHA=0.7 bash examples/gigpo_trainer/run_alfworld.sh # change the smoothing coefficient
GIGPO_SMOOTH=False GIGPO_NO_HARM=False \
  bash examples/gigpo_trainer/run_webshop.sh                # original GiGPO
```

| Variable | Setting | Launcher default |
|---|---|---|
| `GIGPO_SMOOTH` | `algorithm.gigpo.smooth` | `True` |
| `GIGPO_ALPHA` | `algorithm.gigpo.alpha` | `0.5` |
| `GIGPO_NO_HARM` | `algorithm.gigpo.no_harm` | `True` |
| `GIGPO_OMEGA` | `algorithm.gigpo.step_advantage_w` | `1.0` |
| `GIGPO_GAMMA` | `algorithm.gamma` | `0.6` (as RELACE) |

The launcher defaults enable both refinements, while the YAML defaults keep them off so `adv_estimator=gigpo` stays the original GiGPO when launched any other way. Resource and data variables (`MODEL_PATH`, `SEED`, `NUM_GPUS`, `GROUP_SIZE`, ...) and trailing Hydra overrides work as in `run.sh`; `DRY_RUN=1` prints the command.

With these enabled, `gigpo/do_no_harm_success_frac` and `gigpo/do_no_harm_suppressed_frac` are logged. The implementation is in [core_gigpo.py](gigpo/core_gigpo.py); `tests/test_gigpo.py` checks it against hand-computed values and against RELACE's own helpers.

## Evaluate or resume

To evaluate a saved training checkpoint, use the same initial model architecture and launch settings:

```bash
MODEL_PATH=Qwen/Qwen2.5-1.5B-Instruct \
bash examples/relace_trainer/run_webshop.sh \
  trainer.val_only=True \
  trainer.resume_mode=resume_path \
  trainer.resume_from_path="$PWD/outputs/relace_webshop_seed1/global_step_200"
```

Use an existing `global_step_<step>` directory from your run. Omit `trainer.val_only=True` to resume training. The inherited [model merger](scripts/model_merger.py) is available for exporting distributed checkpoints; run it with `--help` to see its supported options.

## Trajectory analysis (diagnostic)

To inspect how RELACE assigns credit inside one rollout group without training:

```bash
CHECKPOINT="$PWD/outputs/relace_alfworld_seed0/global_step_150" \
GROUP_SIZE=8 MAX_ATTEMPTS=20 \
bash examples/relace_trainer/analyze_trajectories.sh alfworld
```

From the fixed checkpoint, the script samples one task group per attempt until a group has at least one successful trajectory, one failed trajectory, and a matched state visited by two or more trajectories. It then scores that group with the training code path: reward, `log pi(a|X_t)`, hindsight teacher forcing, and `compute_advantage`. No parameters are updated. `CHECKPOINT` is an FSDP `global_step_<N>` directory and must be loaded with the `NUM_GPUS` and `MODEL_PATH` used in training. Leave it unset to analyse the `MODEL_PATH` weights, or point `MODEL_PATH` to a merged Hugging Face export. Other `run.sh` variables and Hydra overrides apply. The settings live under `trajectory_analysis` in [ppo_trainer.yaml](verl/trainer/config/ppo_trainer.yaml).

Results are written to `outputs/relace_trajectory_analysis_<env>/trajectory_analysis/<timestamp>/`:

| File | Contents |
|---|---|
| `group.json` | The whole group: every action with observation, action, matched-state group, terminal reward, success flag, `p`, `pH`, `u`, `rho`, discounted/weighted/smoothed returns, macro advantage, local advantage before/after the success mask, final advantage, the `rho=1` recomputation, and both prompts |
| `summary.md` | Trajectories, shared states, settings, metrics, and an all-actions RELACE vs `rho=1` table |
| `trajectories/T<k>.md`, `.csv` | Per-trajectory tables, including the side-by-side `rho=1` comparison |
| `prompts/T<k>.md` | Original and hindsight prompts for each action |
| `relace_inputs.pt` | Saved inputs for CPU replay: `python -m relace.trajectory_analysis replay <file> --out <dir>` |

The `rho=1` variant reuses the identical batch and state grouping and changes only the retrospective weight. `group.json` also records a cross-check against the advantages produced by `compute_advantage`, whose difference should be 0.

## Credit-estimation diagnostics (online, optional)

During RELACE training, `credit_diagnostics` compares GiGPO-style and RELACE-style **unnormalized local credit** against an independent action-value proxy on the *same rollout batch and frozen behavior policy*. It is a proxy for the variance-reduction condition, not a measurement from two separately trained runs. It is read-only: it changes no reward, advantage, gradient, sampling, or update, and adds no rollouts, scoring passes, simulator snapshots, or critic. Metrics go through the existing training logger (W&B when `LOGGER=wandb`), keyed by the global training step.

**Enable it** (off by default). It is read once at start-up, so it takes effect on your next launch or checkpoint resume. `trainer.resume_mode` defaults to `auto`, so relaunching with the same `EXPERIMENT_NAME` and `OUTPUT_DIR` resumes from the latest `global_step_<N>`:

```bash
LOGGER=wandb WANDB_API_KEY=... \
bash examples/relace_trainer/run_webshop.sh \
  credit_diagnostics.enable=True \
  'credit_diagnostics.extra_gammas=[0.6,0.95]'
```

To resume from a specific checkpoint instead, add `trainer.resume_mode=resume_path trainer.resume_from_path="$PWD/outputs/<experiment>/global_step_<N>"`. A relaunch starts a new process, so `Tracking` opens a W&B run as usual; to continue the old run, export W&B's standard `WANDB_RUN_ID=<id>` and `WANDB_RESUME=allow` first (the id is printed as `wandb run id:` at start-up; this is W&B behavior, not exercised here). The diagnostics add no state to checkpoints.

| `credit_diagnostics.` | Default | Meaning |
|---|---|---|
| `enable` | `False` | Turn the diagnostics on (only with `algorithm.adv_estimator=relace`). |
| `log_interval` | `5` | Run on updates that are multiples of this (so a resumed run keeps its cadence). |
| `min_reference_trajectories` | `2` | Distinct other-fold trajectories required to form `Q_hat(o, a)`. |
| `extra_gammas` | `[]` | Extra discounts, evaluated on the same trajectories and `rho`. The training `algorithm.gamma` is always evaluated. |
| `fold_seed` | `0` | Salt of the trajectory-to-fold rule. |

### What is computed

For each valid rollout step `i`, with the implementation's indexing `G_i = gamma^(T_i - t_i) * R_i` (one-based `t`, episode length `T`, episode return `R`), `rho_i` the trajectory-normalized, clipped RELACE weight, and `D_i = (rho_i - 1) * G_i`. These are taken *before* smoothing, local standardization, the success mask, and the episode-level term, with the same `gamma` for both estimators.

* **Cross-fitting.** Within each task (`uid`), whole trajectories are split into two folds by a hash of `(fold_seed, task id, trajectory id)`, alternating by rank. No reward, outcome, or weight is consulted. `Q_hat(o_i, a_i)` for a row is built only from the **other** fold: rows with the same task, the existing RELACE anchor group (its similarity matching is reused), and the **exact executed response token sequence** (padding excluded; different reasoning is never collapsed onto the same parsed command). It is the *trajectory-balanced* mean of `G`: average within each reference trajectory, then across reference trajectories, so a looping trajectory cannot dominate. Rows with fewer than `min_reference_trajectories` reference trajectories get no `Q_hat` and are dropped, never replaced by an observation mean or a stale estimate.
* **Diagnostic groups.** One per `(task, anchor group, evaluation fold)`, restricted to rows with a valid `Q_hat`, kept if it has at least two such rows from at least two distinct evaluation trajectories. Folds are never recombined before group moments. Inside a group, with every mean over exactly the same rows and denominator `n_g`:

```
A_G = G - mean(G)      A_H = rho*G - mean(rho*G)      A_ref = Q_hat - mean(Q_hat)
mse_gigpo  = mean((A_G - A_ref)^2)       mse_relace = mean((A_H - A_ref)^2)       gap = mse_relace - mse_gigpo
N_hat = G - Q_hat      var_D = mean((D - mean D)^2)      cov_ND = mean((N_hat - mean N_hat)(D - mean D))
gap  ==  var_D + 2 * cov_ND
```

* **Aggregation.** Occurrence-weighted, `sum_g(n_g * m_g) / sum_g(n_g)`, in float64. All-zero-return groups stay in (they are legitimate zeros); singleton and unsupported groups are excluded and counted. The relative reduction is `100 * (mse_gigpo - mse_relace) / mse_gigpo` from the aggregate MSEs, and is omitted when `mse_gigpo < 1e-8` (`RELATIVE_DENOMINATOR_FLOOR`). Nothing is carried across updates.
* **Where it runs.** `compute_advantage` runs on the Ray driver over the fully gathered batch, before `balance_batch`, so every task group is complete; there are no rank-local fragments to merge. Cost is a few Python passes over the batch rows, every `log_interval` updates (`credit_diag/compute_seconds`). If it ever raises, the error is printed, `credit_diag/failed=1` is logged, and training continues unchanged.

### W&B keys

Error metrics are logged per discount under `credit_diag/gamma_<g>/` (for example `gamma_0.6`, `gamma_0.95`), only when at least one diagnostic group exists. A missing key means "not computable", never zero.

| Key (under `credit_diag/gamma_<g>/`) | Meaning |
|---|---|
| `proxy_mse_gigpo`, `proxy_mse_relace` | Occurrence-weighted proxy credit-estimation MSE |
| `proxy_mse_gap` | `proxy_mse_relace - proxy_mse_gigpo`; **negative means RELACE is closer to the proxy** |
| `correction_variance` | `var(D)`: the variance RELACE's correction adds |
| `noise_correction_covariance` | `cov(N_hat, D)` |
| `proxy_gap_from_covariance` | `var(D) + 2 cov(N_hat, D)`, the same gap computed the other way |
| `relative_mse_reduction_pct` | Reduction of the aggregate MSE in percent (omitted below the floor) |
| `group_mean_proxy_gap`, `fraction_groups_negative_gap` | Unweighted mean of `gap_g`; share of groups with `gap_g < 0` |
| `identity_max_abs_error` | Largest `|gap_g - (var_D + 2 cov)|` over groups (floating-point check, around 1e-14) |

Coverage, independent of the discount, under `credit_diag/coverage/`: `total_valid_steps`, `steps_with_valid_q_reference`, `steps_with_insufficient_reference`, `eligible_steps`, `eligible_step_fraction`, `eligible_groups`, `evaluated_group_folds`, `original_anchor_groups`, `singleton_groups` (fewer than two evaluation trajectories), `insufficient_reference_groups` (enough evaluation trajectories but missing references), `ref_trajectories_{mean,min,max}` (over rows with a valid reference), and `fraction_groups_fully_retained` (eligible groups that lost no evaluation row to missing references). Coverage is always logged, even when no error metric can be computed. `credit_diag/compute_seconds` is the overhead.

### Charts to create

1. `proxy_mse_gigpo` and `proxy_mse_relace` against training step.
2. `proxy_mse_gap` with a horizontal reference line at 0 (below 0 = lower RELACE proxy MSE).
3. `correction_variance` and `-2 * noise_correction_covariance` on one chart (a W&B expression, `-2 * noise_correction_covariance`). The condition holds when the second exceeds the first, which is the same statement as a negative gap.
4. `coverage/eligible_step_fraction` next to these, since they only describe the supported subset.

### How to read it

These are finite-group, cross-fitted **proxy** advantage-error metrics. The covariance expression and the MSE difference are two calculations of one sample identity, not independent evidence. Limitations: `Q_hat` is imperfect and exists only where the policy repeated an exact response; observation matching is the approximate anchor grouping; the metrics describe the **supported evaluation subset**, not the full production group (filtering and cross-fitting shrink it, and `fraction_groups_fully_retained` says by how much); groups are finite and centered on their own sample mean; every evaluation row of a (state, action) shares one reference estimate; and the trajectory-balanced weighting is a choice. They support an analysis of the proposed mechanism; they do not establish the population statement or that RELACE improves policy performance.

**Expect low coverage.** Responses are free-form `<think>...</think><action>...</action>` text sampled at temperature 1.0, and the action key is the exact token sequence, so two trajectories rarely repeat the same response at a matched state. With `GROUP_SIZE=8` each reference fold holds only about four trajectories per task, and `min_reference_trajectories=2` needs two of them to agree exactly. Watch `eligible_step_fraction` before trusting the error curves; `min_reference_trajectories=1` raises coverage at the cost of a noisier `Q_hat`.

## Repository layout

```text
relace/                         RELACE advantage estimation, trajectory analysis, online credit diagnostics
examples/relace_trainer/         ALFWorld and WebShop launchers, trajectory-analysis launcher
examples/gigpo_trainer/          GiGPO launchers (optional RELACE-style smoothing / no-harm)
examples/data_preprocess/        Local placeholder-data preparation
agent_system/                   Environment adapters, memory, and rollouts
verl/                           Training framework and workers
  trainer/dataset.py            Shared dataset/sampler construction
gigpo/                          Reused grouping and normalization utilities
scripts/                        WebShop setup and checkpoint export
tests/                          Focused RELACE and launcher checks
```

## Tests

The numerical and launcher tests run without GPUs or Ray:

```bash
python -m pip install pytest torch numpy pandas pyarrow
python -m pytest -q tests
```

These checks cover numerical invariants, success protection, padding, local data preparation, and launch configuration. They do not replace an end-to-end GPU training run.
