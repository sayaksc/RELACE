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

"""
Online credit-estimation diagnostics for RELACE (proxy for the variance-reduction condition).

On ONE RELACE rollout batch (one frozen behaviour policy) this compares the unnormalized local credit of

    GiGPO-style :  G_i          = gamma ** (T_i - t_i) * R_i
    RELACE-style:  rho_i * G_i                         (rho_i = trajectory-normalized, clipped weight)

against an independent, cross-fitted action-value proxy Q_hat(o_i, a_i). With D_i = (rho_i - 1) * G_i and
N_hat_i = G_i - Q_hat_i, centring inside a diagnostic group g gives the sample identity

    mse_relace_g - mse_gigpo_g  ==  var_g(D) + 2 * cov_g(N_hat, D)         (denominator n_g)

so a negative gap (RELACE closer to Q_hat) means  -2 cov(N_hat, D) > var(D).

Everything here is a *diagnostic*. It is read-only with respect to training: it never writes to the batch,
the advantages, or any tensor it is handed, it draws no random numbers, and it adds no rollouts or forward passes.

Conventions (all documented because they are choices, not facts):

* Where it runs. `compute_advantage` runs on the Ray driver over the full gathered batch before `balance_batch`,
  so every task group is complete; there is no cross-rank aggregation to do. State is never carried across updates.
* Quantities are taken BEFORE smoothing, local standardization, the success mask, and the episode-level term.
  G uses the implementation's exponent `episode_length - timestep` (one-based timesteps) and the episode return.
* Tasks are the rollout `uid`s. Anchor groups are the `step_group_uids` already computed by RELACE (its similarity
  matching is reused, not re-implemented), so "same observation" means "same anchor group".
* Action key = the exact executed response token sequence selected by `response_mask` (padding excluded). Different
  reasoning/action text is never collapsed onto the same parsed environment command.
* Two-fold cross-fitting by COMPLETE trajectory inside each task. Folds are a pure function of (seed, task id,
  trajectory id): trajectories of a task are sorted by a hash of their ids and alternate between the two folds. No
  reward, outcome, or weight is consulted. A row in fold f is only ever referenced against fold 1-f.
* Q_hat(o, a) for an evaluation row = trajectory-balanced mean of the discounted returns G of the matching rows
  (same task, anchor group and action key) in the other fold: average within each reference trajectory first, then
  across reference trajectories, so a looping trajectory cannot dominate. It needs at least
  `min_reference_trajectories` distinct reference trajectories; otherwise the row has no reference and is dropped
  (never replaced by an observation mean, a parsed-command match, or a stale estimate).
* Diagnostic groups = (task, anchor group, evaluation fold), restricted to rows with a valid Q_hat, and kept only if
  they hold at least two such rows from at least two distinct evaluation trajectories. They describe the SUPPORTED
  SUBSET, not the full production group (which also contains unsupported rows and both folds).
* Aggregation is occurrence-weighted: sum_g(n_g * m_g) / sum_g(n_g), computed from float64 group moments.
  Folds are never merged before the per-group moments.

Limitations: Q_hat is noisy and only exists where the policy repeated an exact response; observation matching is the
(approximate) anchor grouping; the supported subset is selected by action coverage; groups are finite and centred on
their own sample mean; reference estimates are shared by all evaluation rows of a (state, action); and the reference
weighting is a choice. These are finite-sample proxy errors that support analysis of the mechanism. They do not
establish the population statement or that RELACE improves policy performance.
"""

from __future__ import annotations

import hashlib
import time
import traceback
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import torch

METRIC_PREFIX = "credit_diag"
NUM_FOLDS = 2

# The relative MSE reduction divides by the aggregate GiGPO proxy MSE, in units of return^2. Below this floor
# (e.g. all-zero-return groups) the ratio carries no information, so the metric is omitted, not logged as 0 or inf.
RELATIVE_DENOMINATOR_FLOOR = 1e-8


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class CreditDiagConfig:
    """`credit_diagnostics` block of ppo_trainer.yaml."""

    enable: bool = False
    log_interval: int = 5  # run every `log_interval` training updates
    min_reference_trajectories: int = 2  # distinct reference trajectories per (observation, action)
    extra_gammas: tuple[float, ...] = ()  # additional discount factors, e.g. (0.6, 0.95)
    fold_seed: int = 0  # salt of the reproducible trajectory-to-fold rule

    def __post_init__(self):
        if int(self.log_interval) < 1:
            raise ValueError(f"credit_diagnostics.log_interval must be >= 1, got {self.log_interval}")
        if int(self.min_reference_trajectories) < 1:
            raise ValueError(
                f"credit_diagnostics.min_reference_trajectories must be >= 1, got {self.min_reference_trajectories}"
            )
        for gamma in self.extra_gammas:
            _check_gamma(gamma)

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any] | None) -> "CreditDiagConfig":
        """Build from an OmegaConf/dict block; a missing block means disabled."""
        if cfg is None:
            return cls()
        extra = cfg.get("extra_gammas", None)
        return cls(
            enable=bool(cfg.get("enable", False)),
            log_interval=int(cfg.get("log_interval", 5)),
            min_reference_trajectories=int(cfg.get("min_reference_trajectories", 2)),
            extra_gammas=tuple(float(g) for g in extra) if extra is not None else (),
            fold_seed=int(cfg.get("fold_seed", 0)),
        )

    def gammas(self, training_gamma: float) -> list[float]:
        """The training gamma first, then the extra ones, without duplicates."""
        _check_gamma(training_gamma)
        out: list[float] = []
        seen: set[float] = set()
        for gamma in (float(training_gamma), *self.extra_gammas):
            key = round(float(gamma), 12)
            if key not in seen:
                seen.add(key)
                out.append(float(gamma))
        return out


def _check_gamma(gamma: float) -> None:
    if not (np.isfinite(gamma) and 0.0 < float(gamma) <= 1.0):
        raise ValueError(f"discount factors must be in (0, 1], got {gamma}")


def is_due(cfg: CreditDiagConfig | None, global_step: int | None) -> bool:
    """True on training updates that are multiples of `log_interval` (so a resumed run keeps the same cadence)."""
    return bool(cfg is not None and cfg.enable and global_step is not None and int(global_step) % cfg.log_interval == 0)


# --------------------------------------------------------------------------- #
# Folds and action keys
# --------------------------------------------------------------------------- #
def assign_folds(index: Sequence, traj_index: Sequence, seed: int = 0) -> np.ndarray:
    """
    Row-level fold (0/1) from the trajectory's fold. Within each task, trajectories are ordered by a hash of
    (seed, task id, trajectory id) and alternate between the folds, so folds are balanced to within one trajectory.
    The rule uses ids only: it is independent of rewards, outcomes and weights, and of the row order.
    """
    index = list(index)
    traj_index = list(traj_index)
    trajectories: dict[Any, set] = defaultdict(set)
    for task, traj in zip(index, traj_index):
        trajectories[task].add(traj)

    fold_of: dict[tuple, int] = {}
    for task, members in trajectories.items():
        ordered = sorted(members, key=lambda traj: _digest(seed, task, traj))
        for rank, traj in enumerate(ordered):
            fold_of[(task, traj)] = rank % NUM_FOLDS
    return np.asarray([fold_of[(task, traj)] for task, traj in zip(index, traj_index)], dtype=np.int64)


def _digest(seed: int, task: Any, traj: Any) -> bytes:
    return hashlib.blake2b(f"{seed}|{task}|{traj}".encode("utf-8"), digest_size=8).digest()


def action_key_ids(responses: torch.Tensor | np.ndarray, response_mask: torch.Tensor | np.ndarray) -> np.ndarray:
    """
    Integer id per row for the exact executed token sequence (`response_mask` selects the tokens, so padding is
    excluded). Equal ids <=> identical token sequences; keys are compared as raw bytes, never hashed lossily.
    """
    tokens = _to_numpy(responses)
    mask = _to_numpy(response_mask) > 0
    table: dict[bytes, int] = {}
    ids = np.empty(tokens.shape[0], dtype=np.int64)
    for row in range(tokens.shape[0]):
        key = np.ascontiguousarray(tokens[row][mask[row]], dtype=np.int64).tobytes()
        ids[row] = table.setdefault(key, len(table))
    return ids


def _to_numpy(values) -> np.ndarray:
    """A private copy: later code can never alias (or write into) a training tensor."""
    if isinstance(values, torch.Tensor):
        return values.detach().cpu().numpy().copy()
    return np.array(values)


def _as_float64(values) -> np.ndarray:
    return _to_numpy(values).astype(np.float64).reshape(-1)


def _factorize(values) -> np.ndarray:
    table: dict = {}
    values = list(values)
    return np.fromiter((table.setdefault(v, len(table)) for v in values), dtype=np.int64, count=len(values))


# --------------------------------------------------------------------------- #
# Gamma-independent structure: references and diagnostic groups
# --------------------------------------------------------------------------- #
@dataclass
class CreditStructure:
    """Who is a reference for whom, and which diagnostic groups exist. Independent of returns and of rho."""

    num_rows: int
    valid: np.ndarray  # (n,) bool: row has a non-empty response
    folds: np.ndarray  # (n,) int64
    n_ref: np.ndarray  # (n,) distinct other-fold trajectories with the same (task, anchor group, action key)
    ref_cell: np.ndarray  # (n,) index into `ref_cells`, -1 when the row has no valid reference
    ref_cells: list  # per reference cell: [(trajectory id, row indices), ...]
    elig_rows: np.ndarray  # rows inside eligible diagnostic groups
    elig_gid: np.ndarray  # compact group id of each row in `elig_rows`
    group_fully_retained: np.ndarray  # (K,) bool: no evaluation row of the group was dropped
    counts: dict

    @property
    def num_groups(self) -> int:
        return int(len(self.group_fully_retained))


def build_structure(
    task: Sequence,
    traj: Sequence,
    group: Sequence,
    key: Sequence,
    folds: Sequence,
    valid: Sequence | None = None,
    min_reference_trajectories: int = 2,
) -> CreditStructure:
    """
    Cross-fitted references and diagnostic groups from row metadata (any hashable ids).

    A row in fold f is referenced only by rows of fold 1-f sharing its task, anchor group and exact action key, so
    no trajectory is ever both reference and evaluation within a fold.
    """
    task = _factorize(task)
    traj = _factorize(traj)
    group = _factorize(group)
    key = _factorize(key)
    folds = np.asarray(folds, dtype=np.int64).reshape(-1)
    n = len(folds)
    valid = np.ones(n, dtype=bool) if valid is None else np.asarray(valid, dtype=bool).reshape(-1)
    if not (len(task) == len(traj) == len(group) == len(key) == len(valid) == n):
        raise ValueError("credit diagnostics need one task, trajectory, group, action key, fold and flag per row")
    if np.any((folds < 0) | (folds >= NUM_FOLDS)):
        raise ValueError("folds must be 0 or 1")
    if min_reference_trajectories < 1:
        raise ValueError("min_reference_trajectories must be >= 1")

    t_l, j_l, g_l, k_l, f_l = (a.tolist() for a in (task, traj, group, key, folds))
    valid_rows = np.flatnonzero(valid).tolist()

    occurrences: dict[tuple, dict[int, list[int]]] = defaultdict(lambda: defaultdict(list))
    for i in valid_rows:
        occurrences[(t_l[i], g_l[i], k_l[i], f_l[i])][j_l[i]].append(i)

    n_ref = np.zeros(n, dtype=np.int64)
    ref_cell = np.full(n, -1, dtype=np.int64)
    cell_index: dict[tuple, int] = {}
    ref_cells: list = []
    for i in valid_rows:
        cell_key = (t_l[i], g_l[i], k_l[i], 1 - f_l[i])  # the OTHER fold only
        other = occurrences.get(cell_key)
        if not other:
            continue
        n_ref[i] = len(other)
        if len(other) < min_reference_trajectories:
            continue
        cid = cell_index.get(cell_key)
        if cid is None:
            cid = cell_index[cell_key] = len(ref_cells)
            ref_cells.append([(tid, np.asarray(rows, dtype=np.int64)) for tid, rows in other.items()])
        ref_cell[i] = cid

    has_q = (ref_cell >= 0).tolist()
    eval_cells: dict[tuple, list[int]] = defaultdict(list)
    for i in valid_rows:
        eval_cells[(t_l[i], g_l[i], f_l[i])].append(i)

    elig_rows: list[int] = []
    elig_gid: list[int] = []
    retained: list[bool] = []
    singleton = insufficient = 0
    for rows in eval_cells.values():
        if len({j_l[i] for i in rows}) < 2:
            singleton += 1  # fewer than two evaluation trajectories even before reference filtering
            continue
        supported = [i for i in rows if has_q[i]]
        if len({j_l[i] for i in supported}) < 2:
            insufficient += 1  # enough evaluation trajectories, but references are missing
            continue
        gid = len(retained)
        elig_rows.extend(supported)
        elig_gid.extend([gid] * len(supported))
        retained.append(len(supported) == len(rows))

    counts = {
        "total_valid_steps": len(valid_rows),
        "steps_with_valid_q_reference": int(sum(has_q)),
        "steps_with_insufficient_reference": int(np.sum((n_ref > 0) & (ref_cell < 0))),
        "eligible_steps": len(elig_rows),
        "eligible_groups": len(retained),
        "evaluated_group_folds": len(eval_cells),
        "original_anchor_groups": len({(t_l[i], g_l[i]) for i in valid_rows}),
        "singleton_groups": singleton,
        "insufficient_reference_groups": insufficient,
    }
    return CreditStructure(
        num_rows=n,
        valid=valid,
        folds=folds,
        n_ref=n_ref,
        ref_cell=ref_cell,
        ref_cells=ref_cells,
        elig_rows=np.asarray(elig_rows, dtype=np.int64),
        elig_gid=np.asarray(elig_gid, dtype=np.int64),
        group_fully_retained=np.asarray(retained, dtype=bool),
        counts=counts,
    )


# --------------------------------------------------------------------------- #
# Per-gamma statistics
# --------------------------------------------------------------------------- #
@dataclass
class GroupStats:
    """Per-diagnostic-group moments (denominator n_g, float64)."""

    n: np.ndarray
    mse_gigpo: np.ndarray
    mse_relace: np.ndarray
    gap: np.ndarray
    var_d: np.ndarray
    cov_nd: np.ndarray


def cross_fitted_q(structure: CreditStructure, returns: np.ndarray) -> np.ndarray:
    """
    Q_hat per row (NaN where there is no valid reference). Trajectory-balanced: mean within each reference
    trajectory, then mean across reference trajectories. Only other-fold rows can enter.
    """
    returns = np.asarray(returns, dtype=np.float64)
    cell_q = np.empty(len(structure.ref_cells), dtype=np.float64)
    for cid, members in enumerate(structure.ref_cells):
        cell_q[cid] = np.mean([returns[rows].mean() for _, rows in members])
    q = np.full(structure.num_rows, np.nan, dtype=np.float64)
    ok = structure.ref_cell >= 0
    q[ok] = cell_q[structure.ref_cell[ok]]
    return q


def group_stats(structure: CreditStructure, rho: np.ndarray, returns: np.ndarray) -> GroupStats | None:
    """
    Moments of every eligible diagnostic group, or None when there is no eligible group (never zeros).

    A_G = G - mean(G), A_H = rho*G - mean(rho*G), A_ref = Q - mean(Q); all means over the same eligible rows.
    """
    if structure.num_groups == 0:
        return None
    rho = np.asarray(rho, dtype=np.float64)
    returns = np.asarray(returns, dtype=np.float64)
    rows, gid, k = structure.elig_rows, structure.elig_gid, structure.num_groups

    g = returns[rows]
    r = rho[rows]
    q = cross_fitted_q(structure, returns)[rows]
    n = np.bincount(gid, minlength=k).astype(np.float64)

    def gmean(x: np.ndarray) -> np.ndarray:
        return np.bincount(gid, weights=x, minlength=k) / n

    def centred(x: np.ndarray) -> np.ndarray:
        return x - gmean(x)[gid]

    rg = r * g
    a_g, a_h, a_ref = centred(g), centred(rg), centred(q)
    mse_gigpo = gmean((a_g - a_ref) ** 2)
    mse_relace = gmean((a_h - a_ref) ** 2)

    d = (r - 1.0) * g
    noise = g - q
    d_c, n_c = centred(d), centred(noise)
    return GroupStats(
        n=n,
        mse_gigpo=mse_gigpo,
        mse_relace=mse_relace,
        gap=mse_relace - mse_gigpo,
        var_d=gmean(d_c**2),
        cov_nd=gmean(n_c * d_c),
    )


def aggregate_stats(stats: GroupStats, relative_floor: float = RELATIVE_DENOMINATOR_FLOOR) -> dict[str, float]:
    """Occurrence-weighted aggregates sum_g(n_g * m_g) / sum_g(n_g) plus unweighted group summaries."""
    total = float(stats.n.sum())

    def agg(x: np.ndarray) -> float:
        return float(np.sum(stats.n * x) / total)

    mse_gigpo, mse_relace = agg(stats.mse_gigpo), agg(stats.mse_relace)
    out = {
        "proxy_mse_gigpo": mse_gigpo,
        "proxy_mse_relace": mse_relace,
        "proxy_mse_gap": agg(stats.gap),
        "correction_variance": agg(stats.var_d),
        "noise_correction_covariance": agg(stats.cov_nd),
        "proxy_gap_from_covariance": agg(stats.var_d + 2.0 * stats.cov_nd),
        "group_mean_proxy_gap": float(np.mean(stats.gap)),
        "fraction_groups_negative_gap": float(np.mean(stats.gap < 0.0)),
        "identity_max_abs_error": float(np.max(np.abs(stats.gap - (stats.var_d + 2.0 * stats.cov_nd)))),
    }
    if mse_gigpo >= relative_floor:  # from the aggregate MSEs, not an average of group percentages
        out["relative_mse_reduction_pct"] = 100.0 * (mse_gigpo - mse_relace) / mse_gigpo
    return out


def coverage_metrics(structure: CreditStructure) -> dict[str, float]:
    """Gamma-independent coverage. Always available, even when no error metric can be computed."""
    c = structure.counts
    has_q = structure.ref_cell >= 0
    out = {
        "total_valid_steps": c["total_valid_steps"],
        "steps_with_valid_q_reference": c["steps_with_valid_q_reference"],
        "steps_with_insufficient_reference": c["steps_with_insufficient_reference"],
        "eligible_steps": c["eligible_steps"],
        "eligible_step_fraction": c["eligible_steps"] / c["total_valid_steps"] if c["total_valid_steps"] else 0.0,
        "eligible_groups": c["eligible_groups"],
        "evaluated_group_folds": c["evaluated_group_folds"],
        "original_anchor_groups": c["original_anchor_groups"],
        "singleton_groups": c["singleton_groups"],
        "insufficient_reference_groups": c["insufficient_reference_groups"],
    }
    if has_q.any():
        counts = structure.n_ref[has_q]
        out["ref_trajectories_mean"] = float(counts.mean())
        out["ref_trajectories_min"] = int(counts.min())
        out["ref_trajectories_max"] = int(counts.max())
    if structure.num_groups:
        out["fraction_groups_fully_retained"] = float(structure.group_fully_retained.mean())
    return {f"{METRIC_PREFIX}/coverage/{name}": float(value) for name, value in out.items()}


def gamma_label(gamma: float) -> str:
    return f"{float(gamma):g}"


# --------------------------------------------------------------------------- #
# Entry points
# --------------------------------------------------------------------------- #
def compute_credit_diagnostics(
    *,
    components: Mapping[str, Any],
    responses: torch.Tensor,
    response_mask: torch.Tensor,
    gamma: float,
    cfg: CreditDiagConfig,
    folds: np.ndarray | None = None,
) -> dict[str, float]:
    """
    Diagnostics for one RELACE batch from the dict returned by `core_relace.compute_relace_components`
    (anchor groups `step_group_uids`, weights `rho`, `terminal_returns`, `timesteps`, `episode_lengths`, row ids)
    plus the executed tokens. Reads everything, writes nothing.

    `folds` overrides the hashed fold rule (tests only). Returns a flat dict of floats; the error metrics of a
    gamma are absent when no eligible group exists.
    """
    start = time.perf_counter()
    index = components["index"]
    traj_index = components["traj_index"]
    if folds is None:
        folds = assign_folds(index, traj_index, seed=cfg.fold_seed)

    valid = _to_numpy(response_mask).sum(axis=-1) > 0
    structure = build_structure(
        task=index,
        traj=traj_index,
        group=components["step_group_uids"],
        key=action_key_ids(responses, response_mask),
        folds=folds,
        valid=valid,
        min_reference_trajectories=cfg.min_reference_trajectories,
    )
    metrics = coverage_metrics(structure)

    rho = _as_float64(components["rho"])
    terminal = _as_float64(components["terminal_returns"])
    exponent = np.asarray(components["episode_lengths"], dtype=np.float64) - np.asarray(
        components["timesteps"], dtype=np.float64
    )
    for g in cfg.gammas(gamma):
        stats = group_stats(structure, rho, np.power(g, exponent) * terminal)
        if stats is None:
            continue
        for name, value in aggregate_stats(stats).items():
            metrics[f"{METRIC_PREFIX}/gamma_{gamma_label(g)}/{name}"] = value

    metrics[f"{METRIC_PREFIX}/compute_seconds"] = time.perf_counter() - start
    return metrics


def compute_credit_diagnostics_safe(**kwargs) -> dict[str, float]:
    """
    `compute_credit_diagnostics` that can never take down training: an error is printed and reported as
    `credit_diag/failed = 1` instead of propagating.
    """
    try:
        return compute_credit_diagnostics(**kwargs)
    except Exception:  # noqa: BLE001 - a diagnostic must not affect the training run
        print("[credit_diagnostics] failed; training is unaffected, metrics skipped for this update:")
        traceback.print_exc()
        return {f"{METRIC_PREFIX}/failed": 1.0}
