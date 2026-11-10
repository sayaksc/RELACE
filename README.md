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

```bash
bash examples/relace_trainer/run_webshop.sh \
  algorithm.adv_estimator=gigpo \
  algorithm.gigpo.smooth=True algorithm.gigpo.alpha=0.5 algorithm.gigpo.no_harm=True
```

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

## Repository layout

```text
relace/                         RELACE advantage estimation and trajectory analysis
examples/relace_trainer/         ALFWorld and WebShop launchers, trajectory-analysis launcher
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
