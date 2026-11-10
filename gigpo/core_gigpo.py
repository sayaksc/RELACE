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

from __future__ import annotations

import numpy as np
import torch
from collections import defaultdict, Counter
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from verl import DataProto
import uuid
import warnings

from difflib import SequenceMatcher
from typing import Sequence, List, Dict, Any


"""
Core functions to implement the GiGPO algorithm (https://arxiv.org/abs/2505.10978).
The function implemented in this file should be used by trainer with different distributed strategies to implement GiGPO.
"""

# ---------------------------------------------------------- #
# --------------- General Functions of GiGPO --------------- #
# ---------------------------------------------------------- #
def to_hashable(x):
    """Convert an object into a hashable type (used for clustering/grouping)."""
    if isinstance(x, (int, float, str, bool)):
        return x
    elif isinstance(x, (np.integer, np.floating)):
        return x.item()
    elif isinstance(x, np.ndarray):
        return tuple(x.flatten())
    elif isinstance(x, (list, tuple)):
        return tuple(to_hashable(e) for e in x)
    elif isinstance(x, dict):
        return tuple(sorted((k, to_hashable(v)) for k, v in x.items()))
    else:
        raise TypeError(f"Unsupported type: {type(x)}")

def summarize_group_size(group_size: list):
    """
    Summarize the dynamics of step-level group.
    Args:
        group_size : List[int]
    """
    counts = Counter(group_size)
    total = sum(counts.values())
    max_size = max(counts)

    summary = {}
    for size in range(1, max_size + 1):
        cnt = counts.get(size, 0)
        prop = cnt / total if total > 0 else 0
        summary[size] = (cnt, prop)

    print("Summary of step-level group sizes:")
    print("Size | Count | Proportion")
    print("-------------------------")
    for size, (cnt, prop) in summary.items():
        if prop:
            print(f"{size:>4} | {cnt:>5} | {prop:>9.2%}")
            
def are_similar(a: str, b: str, threshold: float = 0.95) -> bool:
    """
    Check whether two text observations are similar enough.
    
    Args:
        a, b (str): Input strings to compare.
        threshold (float): Minimum similarity ratio.
    
    Returns:
        bool: True if similarity >= threshold.
    """
    if not isinstance(a, str) or not isinstance(b, str):
        raise ValueError("Only text-based observations are supported for similarity-based GiGPO in this version.")
    return SequenceMatcher(None, a, b).ratio() >= threshold

def compute_step_discounted_returns(batch: DataProto, gamma: float):
    """
    Compute discounted returns for each trajectory. (Eq. 5 in the paper)
    
    Args:
        batch (DataProto): Input batch.
        gamma (float): Discount factor.
    
    Returns:
        torch.Tensor: Discounted returns.
    """
    rewards = batch.non_tensor_batch['rewards'].astype(np.float32)
    traj_uids = batch.non_tensor_batch['traj_uid']
    active_masks = batch.non_tensor_batch['active_masks'].astype(np.float32)
    returns_by_traj = {}
    unique_traj_uids = np.unique(traj_uids)
    for uid in unique_traj_uids:
        # Get indices for this trajectory
        traj_indices = np.where(traj_uids == uid)[0]
        
        # Extract rewards and masks for this trajectory
        traj_rewards = rewards[traj_indices]
        traj_active_masks = active_masks[traj_indices]
        assert traj_active_masks.all(), "active_masks should be all 1s for the same trajectory"
        
        # Calculate returns
        traj_returns = np.zeros_like(traj_rewards)
        running_return = 0
        
        # Calculate returns from the end to the start
        for t in reversed(range(len(traj_rewards))):
            running_return = traj_rewards[t] + gamma * running_return
            traj_returns[t] = running_return
        
        # Store the results
        returns_by_traj[uid] = traj_returns
    
    # Recombine the returns into the original batch order
    all_returns = np.zeros_like(rewards)
    for i, uid in enumerate(traj_uids):
        traj_indices = np.where(traj_uids == uid)[0]
        idx_in_traj = np.where(traj_indices == i)[0][0]  # Find position of i in its trajectory
        all_returns[i] = returns_by_traj[uid][idx_in_traj]
    
    all_returns = torch.tensor(all_returns, dtype=torch.float32, device=batch.batch['input_ids'].device)
    return all_returns


def _build_timesteps(traj_index: np.ndarray, timestep: np.ndarray | None) -> np.ndarray:
    """One-based step number per row; falls back to the row order within each trajectory."""
    if timestep is not None:
        return np.asarray(timestep).astype(np.int64).reshape(-1)

    counters = defaultdict(int)
    timesteps = np.zeros(len(traj_index), dtype=np.int64)
    for row, traj_uid in enumerate(traj_index):
        counters[traj_uid] += 1
        timesteps[row] = counters[traj_uid]
    return timesteps


def smooth_step_values(values: torch.Tensor, traj_index: np.ndarray, timesteps: np.ndarray, alpha: float) -> torch.Tensor:
    """
    One-step temporal smoothing of per-step values along each trajectory:
    `v'_t = alpha * v_t + (1 - alpha) * v_{t+1}`, with the last step of a trajectory
    keeping its own value. `alpha=1.0` leaves `values` unchanged. This is the same
    smoothing RELACE applies to its Q-values (`relace.core_relace._smooth_q_values`).

    Args:
        values: `(torch.Tensor)` shape: (bs,)
        traj_index: `(np.array)` shape: (bs,)
        timesteps: `(np.array)` shape: (bs,), the order of rows within a trajectory.
        alpha: float in [0, 1].
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"gigpo.alpha must be in [0, 1], got {alpha}")

    smoothed = values.clone()
    for traj_uid in np.unique(traj_index):
        locs = np.where(traj_index == traj_uid)[0]
        locs = locs[np.argsort(timesteps[locs], kind="stable")]
        for pos, row in enumerate(locs):
            next_row = locs[pos + 1] if pos + 1 < len(locs) else row
            smoothed[row] = alpha * values[row] + (1.0 - alpha) * values[next_row]
    return smoothed


def apply_no_harm_mask(
    step_rows: torch.Tensor,
    success_mask: np.ndarray | torch.Tensor | None,
) -> tuple[torch.Tensor, float, float]:
    """
    Do-no-harm mask for the step-level advantage: on rows that belong to a successful
    episode, negative step advantages are zeroed so the step term can never push down
    an action of a trajectory that reached the goal. Positive step advantages and all
    rows of unsuccessful episodes are untouched. This is the same mask RELACE applies
    to its local advantage (`relace.core_relace._apply_do_no_harm_mask`).

    Args:
        step_rows: `(torch.Tensor)` shape: (bs,), normalized step advantages (one scalar per row).
        success_mask: `(np.array | torch.Tensor | None)` shape: (bs,), positive = successful
            episode. `None` disables the mask.

    Returns:
        masked rows, fraction of rows in a successful episode, fraction of rows suppressed.
    """
    if success_mask is None:
        return step_rows, 0.0, 0.0

    if isinstance(success_mask, torch.Tensor):
        success_values = success_mask.detach().to(device=step_rows.device, dtype=step_rows.dtype).reshape(-1)
    else:
        success_values = torch.as_tensor(
            np.asarray(success_mask).astype(np.float32), device=step_rows.device, dtype=step_rows.dtype
        ).reshape(-1)
    success_rows = success_values > 0
    if success_rows.shape[0] != step_rows.shape[0]:
        raise ValueError(f"success_mask must have one value per row, got {success_rows.shape[0]} for batch size {step_rows.shape[0]}")

    suppressed = success_rows & (step_rows < 0)
    masked_rows = torch.where(suppressed, torch.zeros_like(step_rows), step_rows)
    success_frac = success_rows.to(torch.float32).mean().detach().cpu().item()
    suppressed_frac = suppressed.to(torch.float32).mean().detach().cpu().item()
    return masked_rows, success_frac, suppressed_frac

# ---------------------------------------------------------- #
# ---------------- Core Functions of GiGPO ----------------- #
# ---------------------------------------------------------- #

def compute_gigpo_outcome_advantage(token_level_rewards: torch.Tensor,
                                   step_rewards: torch.Tensor,
                                   response_mask: torch.Tensor,
                                   anchor_obs: np.array,
                                   index: np.array,
                                   traj_index: np.array,
                                   epsilon: float = 1e-6,
                                   step_advantage_w: float = 1.0,
                                   mode: str = "mean_norm",
                                   enable_similarity: bool = False,
                                   similarity_thresh: float = 0.95,
                                   timestep: np.ndarray | None = None,
                                   success_mask: np.ndarray | torch.Tensor | None = None,
                                   smooth: bool = False,
                                   alpha: float = 1.0,
                                   no_harm: bool = False,
                                   return_metrics: bool = False,
                                   ):
    """
    Compute the advantages for GiGPO (https://arxiv.org/abs/2505.10978).

    Two optional RELACE-style refinements of the step-level term (both off by default,
    which reproduces the original GiGPO exactly):

    * `smooth`: smooth the step values (the discounted returns in `step_rewards`) along each
      trajectory, `q'_t = alpha * q_t + (1 - alpha) * q_{t+1}`, before the anchor-state-group
      normalization. `timestep` (one-based) orders the rows of a trajectory; if omitted, row
      order within each trajectory is used.
    * `no_harm`: after the anchor-state-group normalization, zero the *negative* step advantages
      of rows whose episode succeeded (`success_mask > 0`). The episode-level term is untouched.

    With `return_metrics=True` a third value is returned: a dict of `gigpo/do_no_harm_*` statistics.
    """
    if mode == "mean_std_norm":
        remove_std = False
    elif mode == "mean_norm":
        remove_std = True
    else:
        raise ValueError(f"Unknown mode: {mode}")

    if no_harm and success_mask is None:
        warnings.warn("gigpo no_harm is enabled but no success_mask was given; the mask is a no-op.", stacklevel=2)

    # Compute episode relative advantages (Eq. 3 in the paper).
    episode_advantages = episode_norm_reward(token_level_rewards, response_mask, index, traj_index, epsilon, remove_std)

    # Anchor state grouping (Eq. 6 in the paper).
    step_group_uids = build_step_group(anchor_obs, index, enable_similarity, similarity_thresh)

    # Optional temporal smoothing of the step values (RELACE-style).
    if smooth:
        traj_index_np = np.asarray(traj_index, dtype=object).reshape(-1)
        timesteps = _build_timesteps(traj_index_np, timestep)
        step_rewards = smooth_step_values(step_rewards, traj_index_np, timesteps, alpha)

    # Compute step relative advantages (Eq. 7 in the paper).
    step_rows = step_norm_rows(step_rewards, step_group_uids, epsilon, remove_std)

    # Optional do-no-harm mask on the step advantages (RELACE-style).
    do_no_harm_success_frac = 0.0
    do_no_harm_suppressed_frac = 0.0
    if no_harm:
        step_rows, do_no_harm_success_frac, do_no_harm_suppressed_frac = apply_no_harm_mask(step_rows, success_mask)
    step_advantages = step_rows.unsqueeze(-1).tile([1, response_mask.shape[-1]]) * response_mask

    # Compute joint advantages (Eq. 8 in the paper).
    scores = episode_advantages + step_advantage_w * step_advantages
    if return_metrics:
        metrics = {
            "gigpo/do_no_harm_success_frac": do_no_harm_success_frac,
            "gigpo/do_no_harm_suppressed_frac": do_no_harm_suppressed_frac,
        }
        return scores, scores, metrics
    return scores, scores


def episode_norm_reward(token_level_rewards: torch.Tensor,
                        response_mask: torch.Tensor,
                        index: np.array,
                        traj_index: np.array,
                        epsilon: float = 1e-6,
                        remove_std: bool = True,
                        compute_mean_std_cross_steps: bool = True,
                        ):
    """
    Compute episode-level advantage using mean-std normalization for GiGPO.
    (with only one scalar reward for each episode).
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        index: `(np.array)`
            shape: (bs,)
        traj_index: `(np.array)`
            shape: (bs,)
        epsilon: float
            A small value to avoid division by zero.
        remove_std: bool
            If True, the standard deviation is removed from the normalization.
        compute_mean_std_cross_steps: bool
            If True (more stable), the mean and std are computed across steps within one group. 
            If False (i.e., standard episode-level adv), the mean and std are computed across trajectories within one group.
    
    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    response_length = token_level_rewards.shape[-1]
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}
    seen_pairs = set()
    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            if (index[i], traj_index[i]) in seen_pairs:
                continue
            id2score[index[i]].append(scores[i])
            if not compute_mean_std_cross_steps:
                seen_pairs.add((index[i], traj_index[i]))

        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
                id2std[idx] = torch.std(torch.tensor([id2score[idx]]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            if remove_std:
                scores[i] = scores[i] - id2mean[index[i]]
            else:
                scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
        episode_advantages = scores.unsqueeze(-1).tile([1, response_length]) * response_mask

    return episode_advantages


def build_step_group(anchor_obs: np.array, index: np.array, enable_similarity: bool = False, similarity_thresh: float = 0.95, summarize: bool = False):
    """
    Group observations by index and then cluster identical observations within each index group.
    Assigns a unique step_group_uid (UUID) to each cluster.
    
    Parameters:
    -----------
    anchor_obs : np.array
        Array of observation strings
    index : np.array
        Array of episode_group_uid
    summarize : bool
        Whether to summarize the group sizes (default: True)
    enable_similarity : bool
        Whether to enable similarity-based step-level grouping (default: False)
    similarity_thresh : float
        Threshold for similarity to consider two observations as identical (default: 1.0, meaning exact match)
    
    Returns:
    --------
    np.array
        Array of step_group_uid values corresponding to the original anchor_obs array
    """
    if enable_similarity:
        assert similarity_thresh > 0.0 and similarity_thresh < 1.0, "When enabling similarity-based step-level group, similarity_thresh should be in (0, 1)"

    # Initialize the result array with placeholder values
    step_group_uids = np.empty(len(anchor_obs), dtype=object)
    
    # Get unique indices
    unique_indices = np.unique(index)

    group_size: List[int] = []
    # Process each unique index
    for idx in unique_indices:
        if not enable_similarity:
            # Get all observations for this index using np.where
            indices = np.where(index == idx)[0]
            obs_group = anchor_obs[indices]
            
            # Create clusters for identical observations
            clusters = defaultdict(list)
            for i, obs in enumerate(obs_group):
                clusters[to_hashable(obs)].append(indices[i])  # Store the original index position
            
            # Assign unique step_group_uid to each cluster
            for obs, original_indices in clusters.items():
                # Generate a UUID for this cluster
                uid = str(uuid.uuid4())
                
                # Assign the same step_group_uid to all elements in this cluster
                group_size.append(len(original_indices))
                for original_idx in original_indices:
                    step_group_uids[original_idx] = uid
        else:
            locs = np.where(index == idx)[0]
            obs_group = anchor_obs[locs]

            # Dynamically maintain clusters: [{rep: str, locs: List[int]} ...]
            clusters: List[Dict[str, Any]] = []

            for obs, loc in zip(obs_group, locs):
                 # Try to place into an existing cluster
                placed = False
                for cluster in clusters:
                    if are_similar(obs, cluster["rep"], similarity_thresh):
                        cluster["locs"].append(loc)
                        placed = True
                        break
                # If no matching cluster, create a new one
                if not placed:
                    clusters.append({"rep": obs, "locs": [loc]})

            # Assign a UUID to each cluster
            for cluster in clusters:
                uid = str(uuid.uuid4())
                group_size.append(len(cluster["locs"]))
                for loc in cluster["locs"]:
                    step_group_uids[loc] = uid

        # Validate that all elements have been assigned a uid
    if None in step_group_uids or np.any(step_group_uids == None):
        missing_indices = np.where(step_group_uids == None)[0]
        raise ValueError(f"Failed to assign UIDs to all observations. Missing at indices: {missing_indices}")

    if summarize:
        summarize_group_size(group_size)
    print(f"Avg size of step-level group: {np.mean(group_size)}")
    return step_group_uids


def step_norm_reward(step_rewards: torch.Tensor,
                      response_mask: torch.Tensor,
                      index: np.array,
                      epsilon: float = 1e-6,
                      remove_std: bool = True,
                      ):
    """
    Compute step-level advantage using mean-std normalization for GiGPO.
    Args:
        step_rewards: `(torch.Tensor)`
            shape: (bs,)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    response_length = response_mask.shape[-1]
    scores = step_norm_rows(step_rewards, index, epsilon, remove_std)
    return scores.unsqueeze(-1).tile([1, response_length]) * response_mask


def step_norm_rows(step_rewards: torch.Tensor,
                   index: np.array,
                   epsilon: float = 1e-6,
                   remove_std: bool = True,
                   ):
    """
    Row-level step advantages: normalize `step_rewards` within each step group (`index`).
    Returns one scalar per row, shape: (bs,). `step_norm_reward` broadcasts these onto tokens.
    """
    scores = step_rewards.clone()

    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])

        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
                id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
                id2std[idx] = torch.std(torch.tensor([id2score[idx]]))
            else:
                print(f"id2score: {id2score}")
                print(f"len(id2score[idx]): {len(id2score[idx])}")
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            if remove_std:
                scores[i] = scores[i] - id2mean[index[i]]
            else:
                scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)

    return scores

