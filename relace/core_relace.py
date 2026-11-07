# Copyright 2025 Nanyang Technological University (NTU), Singapore
# and the verl-agent (GiGPO) team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import defaultdict
from typing import Any

import numpy as np
import torch
from gigpo.core_gigpo import build_step_group

"""
Core functions for RELACE.

RELACE keeps the GiGPO batch protocol: each row is one environment step/action,
and row-level advantages are broadcast onto response-token positions with the
existing response mask. The macro term is the same GRPO/GiGPO episode-level
normalization used in this repo; the micro term uses hindsight log-probs over
the same action tokens.
"""


def _as_numpy(values: Any, dtype=None) -> np.ndarray:
    if isinstance(values, torch.Tensor):
        values = values.detach().cpu().numpy()
    values = np.asarray(values)
    if dtype is not None:
        values = values.astype(dtype)
    return values


def _as_row_tensor(values: Any, like: torch.Tensor) -> torch.Tensor:
    if isinstance(values, torch.Tensor):
        return values.to(device=like.device, dtype=like.dtype).reshape(-1).clone()
    return torch.as_tensor(_as_numpy(values, np.float32), device=like.device, dtype=like.dtype).reshape(-1).clone()


def _stack_stats(values: list[torch.Tensor], fallback_mean: torch.Tensor | None = None):
    if len(values) == 1:
        mean = values[0] if fallback_mean is None else fallback_mean
        std = values[0].new_tensor(1.0)
        return mean, std
    stacked = torch.stack(values)
    return stacked.mean(), stacked.std(unbiased=True)


def _build_timesteps(traj_index: np.ndarray, timestep: np.ndarray | None) -> np.ndarray:
    if timestep is not None:
        return _as_numpy(timestep, np.int64).reshape(-1)

    counters = defaultdict(int)
    timesteps = np.zeros(len(traj_index), dtype=np.int64)
    for row, traj_uid in enumerate(traj_index):
        counters[traj_uid] += 1
        timesteps[row] = counters[traj_uid]
    return timesteps


def _build_episode_lengths(traj_index: np.ndarray, timesteps: np.ndarray, episode_lengths: np.ndarray | None) -> np.ndarray:
    if episode_lengths is not None:
        lengths = _as_numpy(episode_lengths, np.float32).reshape(-1).astype(np.int64)
    else:
        traj2length = {}
        for traj_uid in np.unique(traj_index):
            locs = np.where(traj_index == traj_uid)[0]
            traj2length[traj_uid] = int(np.max(timesteps[locs]))
        lengths = np.asarray([traj2length[traj_uid] for traj_uid in traj_index], dtype=np.int64)
    return lengths


def grpo_macro_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    traj_index: np.ndarray,
    row_scores: np.ndarray | torch.Tensor | None = None,
    epsilon: float = 1e-6,
    norm_adv_by_std: bool = True,
    compute_mean_std_cross_steps: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Reproduce this repo's GRPO/GiGPO episode-level advantage.

    With the default `compute_mean_std_cross_steps=True`, scores are collected
    across all rows in a task group, matching `core_algos.compute_grpo_outcome_advantage`
    and `core_gigpo.episode_norm_reward`. With `compute_mean_std_cross_steps=False`,
    each trajectory contributes once and every row in that trajectory receives
    the same trajectory-level normalized score.
    """
    response_length = token_level_rewards.shape[-1]
    if row_scores is None:
        scores = token_level_rewards.sum(dim=-1).clone()
    else:
        scores = _as_row_tensor(row_scores, token_level_rewards)
    if scores.shape[0] != token_level_rewards.shape[0]:
        raise ValueError(f"row_scores must have one value per row, got {scores.shape[0]} for batch size {token_level_rewards.shape[0]}")

    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}
    seen_pairs = set()
    pair2score = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            pair = (index[i], traj_index[i])
            if pair in seen_pairs:
                continue
            pair2score[pair] = scores[i]
            id2score[index[i]].append(scores[i])
            if not compute_mean_std_cross_steps:
                seen_pairs.add(pair)

        for idx, values in id2score.items():
            if len(values) == 1:
                id2mean[idx] = scores.new_tensor(0.0)
                id2std[idx] = scores.new_tensor(1.0)
            elif len(values) > 1:
                id2mean[idx], id2std[idx] = _stack_stats(values)
            else:
                raise ValueError(f"no score in prompt index: {idx}")

        for i in range(bsz):
            if not compute_mean_std_cross_steps:
                scores[i] = pair2score[(index[i], traj_index[i])]
            if norm_adv_by_std:
                scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
            else:
                scores[i] = scores[i] - id2mean[index[i]]

    return scores.unsqueeze(-1).tile([1, response_length]) * response_mask, scores




def _compute_hindsight_probability(
    hindsight_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    temp: float = 1.0,
) -> torch.Tensor:
    if hindsight_log_probs.dim() == 1:
        return hindsight_log_probs
    if temp <= 0:
        raise ValueError(f"relace.temp must be positive, got {temp}")
    mask = response_mask.to(dtype=hindsight_log_probs.dtype)
    raw_lengths = mask.sum(dim=-1)
    lengths = raw_lengths.clamp_min(1.0)
    log_sums = (hindsight_log_probs * mask).sum(dim=-1)
    probs = torch.exp(log_sums / (temp * lengths))
    return torch.where(raw_lengths > 0, probs, torch.zeros_like(probs))

def _normalize_by_group(
    qvalues: torch.Tensor,
    values: torch.Tensor,
    index: np.ndarray,
    epsilon: float,
    remove_std: bool = False,
) -> tuple[torch.Tensor, float]:
    
    qscores = qvalues.clone()
    scores = values.clone()
    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i].item())

        for idx in id2score:
            # Move list of scalars back to tensor on the correct device
            vals = torch.tensor(id2score[idx], dtype=scores.dtype, device=scores.device)
            if len(vals) == 1:
                id2mean[idx] = vals.mean()
                id2std[idx] = torch.tensor(1.0, device=scores.device)
            elif len(vals) > 1:
                id2mean[idx] = vals.mean()
                id2std[idx] = vals.std(unbiased=True)
            else:
                raise ValueError(f"no score in prompt index: {idx}")

        for i in range(bsz):
            if remove_std:
                qscores[i] = qscores[i] - id2mean[index[i]]
            else:
                qscores[i] = (qscores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
                
        # Calculate metric for logging
        abs_means = [torch.abs(m).item() for m in id2mean.values()]
        micro_group_mean_abs_max = max(abs_means) if abs_means else 0.0
    
    # Return 1D scores (not expanded token-level advantages) and the metric
    return qscores, micro_group_mean_abs_max


def _smooth_q_values(q_values: torch.Tensor, traj_index: np.ndarray, timesteps: np.ndarray, alpha: float) -> torch.Tensor:
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"relace.alpha must be in [0, 1], got {alpha}")

    smoothed = q_values.clone()
    for traj_uid in np.unique(traj_index):
        locs = np.where(traj_index == traj_uid)[0]
        locs = locs[np.argsort(timesteps[locs], kind="stable")]
        for pos, row in enumerate(locs):
            next_row = locs[pos + 1] if pos + 1 < len(locs) else row
            smoothed[row] = alpha * q_values[row] + (1.0 - alpha) * q_values[next_row]
    return smoothed


def _apply_do_no_harm_mask(
    micro_rows: torch.Tensor,
    success_mask: np.ndarray | torch.Tensor | None,
) -> tuple[torch.Tensor, float, float]:
    if success_mask is None:
        return micro_rows, 0.0, 0.0

    success_rows = _as_row_tensor(success_mask, micro_rows) > 0
    if success_rows.shape[0] != micro_rows.shape[0]:
        raise ValueError(f"success_mask must have one value per row, got {success_rows.shape[0]} for batch size {micro_rows.shape[0]}")

    suppressed = success_rows & (micro_rows < 0)
    masked_rows = torch.where(suppressed, torch.zeros_like(micro_rows), micro_rows)
    success_frac = success_rows.to(torch.float32).mean().detach().cpu().item()
    suppressed_frac = suppressed.to(torch.float32).mean().detach().cpu().item()
    return masked_rows, success_frac, suppressed_frac


def _metrics(
    rho: torch.Tensor,
    pi_hind: torch.Tensor,
    q_values: torch.Tensor,
    micro_rows: torch.Tensor,
    clip_min: float,
    clip_max: float,
    micro_group_mean_abs_max: float,
    do_no_harm_success_frac: float,
    do_no_harm_suppressed_frac: float,
) -> dict[str, float]:
    with torch.no_grad():
        lower_hits = (rho <= clip_min + 1e-6).to(torch.float32).mean()
        upper_hits = (rho >= clip_max - 1e-6).to(torch.float32).mean()
        return {
            "relace/rho_mean": rho.mean().detach().cpu().item(),
            "relace/rho_std": rho.std(unbiased=False).detach().cpu().item(),
            "relace/rho_min": rho.min().detach().cpu().item(),
            "relace/rho_max": rho.max().detach().cpu().item(),
            "relace/rho_clip_min_frac": lower_hits.detach().cpu().item(),
            "relace/rho_clip_max_frac": upper_hits.detach().cpu().item(),
            "relace/pi_hind_mean": pi_hind.mean().detach().cpu().item(),
            "relace/qh_abs_mean": q_values.abs().mean().detach().cpu().item(),
            "relace/micro_abs_mean": micro_rows.abs().mean().detach().cpu().item(),
            "relace/micro_group_mean_abs_max": micro_group_mean_abs_max,
            "relace/do_no_harm_success_frac": do_no_harm_success_frac,
            "relace/do_no_harm_suppressed_frac": do_no_harm_suppressed_frac,
        }


def compute_relace_components(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    hindsight_log_probs: torch.Tensor,
    action_log_probs: torch.Tensor,
    anchor_obs: np.array,
    index: np.ndarray,
    traj_index: np.ndarray,
    timestep: np.ndarray | None = None,
    episode_lengths: np.ndarray | None = None,
    episode_returns: np.ndarray | torch.Tensor | None = None,
    success_mask: np.ndarray | torch.Tensor | None = None,
    epsilon: float = 1e-6,
    temp: float = 1.0,
    clip_min: float = 0.1,
    clip_max: float = 10.0,
    omega: float = 1.0,
    gamma: float = 0.95,
    smooth: bool = False,
    alpha: float = 1.0,
    mode: str = "mean_std_norm",
    enable_similarity: bool = True,
    similarity_thresh: float = 0.95,
    no_harm: bool = True,
    step_group_uids: np.ndarray | None = None,
    unit_rho: bool = False,
) -> dict[str, Any]:
    """
    Run the RELACE advantage computation and return every intermediate quantity.

    `compute_relace_advantage` is a thin wrapper around this function, so training
    and diagnostics share one code path. Two keyword-only diagnostics hooks exist;
    their defaults reproduce training exactly:

    * `step_group_uids`: reuse a precomputed state grouping instead of calling
      `build_step_group` (whose cluster labels are random UUIDs).
    * `unit_rho`: replace the retrospective weight rho with 1 for every row while
      keeping every other input and setting fixed (ablation of the hindsight term).

    All per-row tensors have shape (bs,); token-level tensors have shape (bs, L).
    """
    if mode == "mean_std_norm":
        remove_std = False
    elif mode == "mean_norm":
        remove_std = True
    else:
        raise ValueError(f"Unknown mode: {mode}")

    if clip_min > clip_max:
        raise ValueError(f"relace.clip_min ({clip_min}) must be <= relace.clip_max ({clip_max})")

    index = _as_numpy(index, object).reshape(-1)
    traj_index = _as_numpy(traj_index, object).reshape(-1)
    timesteps = _build_timesteps(traj_index, timestep)
    lengths_np = _build_episode_lengths(traj_index, timesteps, episode_lengths)

    bsz = token_level_rewards.shape[0]
    if not (len(index) == len(traj_index) == len(timesteps) == len(lengths_np) == bsz):
        raise ValueError("RELACE row metadata must have one value per batch row")

    if np.any(lengths_np - timesteps < 0):
        bad = np.where(lengths_np - timesteps < 0)[0][:5]
        raise ValueError(f"RELACE found timestep greater than episode length at rows {bad.tolist()}")

    # pH: length-normalized likelihood score of the executed action under the
    # outcome-augmented (hindsight) prompt.
    pi_hind = _compute_hindsight_probability(hindsight_log_probs, response_mask, temp=temp)

    if action_log_probs is None:
        raise ValueError("RELACE requires action_log_probs (log pi(a | X_t)) to form rho.")
    # p: the same score under the original prompt X_t.
    pi_orig = _compute_hindsight_probability(action_log_probs, response_mask, temp=temp)

    # u = pH / p, self-normalized by its trajectory mean, then clipped -> rho.
    raw_ratio = (pi_hind) / (pi_orig + epsilon)
    ratio_traj_mean = torch.empty_like(raw_ratio)
    for traj_uid in np.unique(traj_index):
        locs = np.where(traj_index == traj_uid)[0]
        locs_t = torch.as_tensor(locs, device=pi_hind.device, dtype=torch.long)
        ratio_traj_mean[locs_t] = raw_ratio[locs_t].mean()

    rho = torch.clamp(raw_ratio / (ratio_traj_mean + epsilon), min=clip_min, max=clip_max)
    if unit_rho:
        rho = torch.ones_like(rho)

    if episode_returns is None:
        returns = token_level_rewards.sum(dim=-1).clone()
    else:
        returns = _as_row_tensor(episode_returns, token_level_rewards)
    if returns.shape[0] != bsz:
        raise ValueError(f"episode_returns must have one value per row, got {returns.shape[0]} for batch size {bsz}")

    macro_advantages, macro_rows = grpo_macro_advantage(
        token_level_rewards=token_level_rewards,
        response_mask=response_mask,
        index=index,
        traj_index=traj_index,
        row_scores=returns,
        epsilon=epsilon,
        norm_adv_by_std=True,
        compute_mean_std_cross_steps=(not remove_std),
    )

    exponents = torch.as_tensor(lengths_np - timesteps, device=returns.device, dtype=returns.dtype)
    discounts = torch.pow(returns.new_tensor(gamma), exponents)
    discounted_returns = discounts * returns  # reported only
    # Keep the original evaluation order, (rho * discounts) * returns, so results are bit-identical.
    weighted_returns = rho * discounts * returns
    q_values = weighted_returns

    if smooth:
        q_values = _smooth_q_values(q_values, traj_index=traj_index, timesteps=timesteps, alpha=alpha)

    # Anchor state grouping (Eq. 6 in the paper).
    if step_group_uids is None:
        step_group_uids = build_step_group(anchor_obs, index, enable_similarity, similarity_thresh)
    else:
        step_group_uids = _as_numpy(step_group_uids, object).reshape(-1)
        if len(step_group_uids) != bsz:
            raise ValueError(f"step_group_uids must have one value per row, got {len(step_group_uids)} for batch size {bsz}")

    micro_rows_unmasked, micro_group_mean_abs_max = _normalize_by_group(
        qvalues=q_values,
        values=q_values,
        index=step_group_uids,
        epsilon=epsilon,
        remove_std=remove_std
    )
    if no_harm:
        micro_rows, do_no_harm_success_frac, do_no_harm_suppressed_frac = _apply_do_no_harm_mask(
            micro_rows=micro_rows_unmasked,
            success_mask=success_mask,
        )
    else:
        micro_rows = micro_rows_unmasked
        do_no_harm_success_frac = 0.0
        do_no_harm_suppressed_frac = 0.0

    micro_advantages = micro_rows.unsqueeze(-1).tile([1, response_mask.shape[-1]]) * response_mask
    advantages = macro_advantages + omega * micro_advantages

    metrics = _metrics(
        rho=rho,
        pi_hind=pi_hind,
        q_values=q_values,
        micro_rows=micro_rows,
        clip_min=clip_min,
        clip_max=clip_max,
        micro_group_mean_abs_max=micro_group_mean_abs_max,
        do_no_harm_success_frac=do_no_harm_success_frac,
        do_no_harm_suppressed_frac=do_no_harm_suppressed_frac,
    )
    return {
        "advantages": advantages,
        "metrics": metrics,
        # row metadata
        "index": index,
        "traj_index": traj_index,
        "timesteps": timesteps,
        "episode_lengths": lengths_np,
        "step_group_uids": step_group_uids,
        # retrospective weight
        "p": pi_orig,
        "p_hindsight": pi_hind,
        "raw_ratio": raw_ratio,
        "raw_ratio_traj_mean": ratio_traj_mean,
        "rho": rho,
        # returns
        "terminal_returns": returns,
        "discounts": discounts,
        "discounted_returns": discounted_returns,
        "weighted_returns": weighted_returns,
        "smoothed_returns": q_values,
        # advantages (row scalars; token-level = row * response_mask)
        "macro_rows": macro_rows,
        "local_rows_unmasked": micro_rows_unmasked,
        "local_rows": micro_rows,
        "final_rows": macro_rows + omega * micro_rows,
        "macro_advantages": macro_advantages,
        "local_advantages": micro_advantages,
        "omega": omega,
    }


def compute_relace_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    hindsight_log_probs: torch.Tensor,
    action_log_probs: torch.Tensor,
    anchor_obs: np.array,
    index: np.ndarray,
    traj_index: np.ndarray,
    timestep: np.ndarray | None = None,
    episode_lengths: np.ndarray | None = None,
    episode_returns: np.ndarray | torch.Tensor | None = None,
    success_mask: np.ndarray | torch.Tensor | None = None,
    epsilon: float = 1e-6,
    temp: float = 1.0,
    clip_min: float = 0.1,
    clip_max: float = 10.0,
    omega: float = 1.0,
    gamma: float = 0.95,
    smooth: bool = False,
    alpha: float = 1.0,
    mode: str = "mean_std_norm",
    enable_similarity: bool = True,
    similarity_thresh: float = 0.95,
    no_harm: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """
    Compute RELACE token-level advantages.

    Args follow the GiGPO row protocol. `timestep` is one-based (`t=1..T`).
    `hindsight_log_probs` must be aligned to `response_mask` and contains the
    actor log-probabilities of the original action tokens under the
    final-state-augmented prompt. See `compute_relace_components` for the
    intermediate quantities.
    """
    components = compute_relace_components(
        token_level_rewards=token_level_rewards,
        response_mask=response_mask,
        hindsight_log_probs=hindsight_log_probs,
        action_log_probs=action_log_probs,
        anchor_obs=anchor_obs,
        index=index,
        traj_index=traj_index,
        timestep=timestep,
        episode_lengths=episode_lengths,
        episode_returns=episode_returns,
        success_mask=success_mask,
        epsilon=epsilon,
        temp=temp,
        clip_min=clip_min,
        clip_max=clip_max,
        omega=omega,
        gamma=gamma,
        smooth=smooth,
        alpha=alpha,
        mode=mode,
        enable_similarity=enable_similarity,
        similarity_thresh=similarity_thresh,
        no_harm=no_harm,
    )
    advantages = components["advantages"]
    return advantages, advantages, components["metrics"]
