"""RELACE trajectory-analysis diagnostic.

Given one rollout group (rows in the GiGPO/RELACE row protocol), this module

* decides whether the group is informative: at least one successful and one failed
  trajectory, plus a matched state visited by several trajectories;
* runs `core_relace.compute_relace_components` twice on the identical inputs and state
  grouping, once as in training and once with rho fixed to 1;
* writes the whole group as JSON plus readable per-trajectory tables and prompts.

The GPU-side driver is `RayPPOTrainer.analyze_trajectories`; everything here runs on CPU.
A saved group can be re-analysed offline:

    python -m relace.trajectory_analysis replay <run_dir>/relace_inputs.pt --out <dir>
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
from collections import OrderedDict
from typing import Any

import numpy as np
import torch

from relace.core_relace import compute_relace_components

# Row-level quantities copied from compute_relace_components into the report.
_FULL_FIELDS = OrderedDict([
    ("p", "p"),
    ("p_hindsight", "p_hindsight"),
    ("raw_ratio", "raw_ratio"),
    ("raw_ratio_traj_mean", "raw_ratio_traj_mean"),
    ("rho", "rho"),
    ("discount", "discounts"),
    ("discounted_return", "discounted_returns"),
    ("weighted_return", "weighted_returns"),
    ("smoothed_return", "smoothed_returns"),
    ("macro_advantage", "macro_rows"),
    ("local_advantage_unmasked", "local_rows_unmasked"),
    ("local_advantage_masked", "local_rows"),
    ("final_advantage", "final_rows"),
])
# Quantities that change when rho is fixed to 1 (everything else is shared).
_UNIT_FIELDS = OrderedDict([
    ("rho", "rho"),
    ("weighted_return", "weighted_returns"),
    ("smoothed_return", "smoothed_returns"),
    ("local_advantage_unmasked", "local_rows_unmasked"),
    ("local_advantage_masked", "local_rows"),
    ("final_advantage", "final_rows"),
])


# ---------------------------------------------------------------------------
# Config handling (called from main_ppo before environments are built)
# ---------------------------------------------------------------------------

def prepare_config(config) -> bool:
    """Apply analysis-mode overrides in place. Returns True when analysis mode is enabled."""
    from omegaconf import open_dict

    cfg = config.get("trajectory_analysis", None)
    if cfg is None or not cfg.get("enable", False):
        return False
    if str(config.algorithm.adv_estimator) != "relace":
        raise ValueError("trajectory_analysis requires algorithm.adv_estimator=relace")
    if int(cfg.get("max_attempts", 1)) < 1:
        raise ValueError("trajectory_analysis.max_attempts must be >= 1")
    if int(cfg.get("min_shared_trajectories", 2)) < 2:
        raise ValueError("trajectory_analysis.min_shared_trajectories must be >= 2")
    with open_dict(config):
        if cfg.get("group_size", None) is not None:
            config.env.rollout.n = int(cfg.group_size)
        # One task per attempt; keep the (unused) validation environments minimal.
        config.data.train_batch_size = 1
        config.data.val_batch_size = 1
    if int(config.env.rollout.n) < 2:
        raise ValueError("trajectory_analysis needs a group size of at least 2 (trajectory_analysis.group_size or env.rollout.n)")
    return True


# ---------------------------------------------------------------------------
# Group selection
# ---------------------------------------------------------------------------

def trajectory_success(non_tensor_batch: dict) -> np.ndarray:
    """Per-row success flag: `episode_success` when the environment reports it, else reward > 0."""
    if "episode_success" in non_tensor_batch:
        return np.asarray(non_tensor_batch["episode_success"], dtype=np.float64) > 0
    return np.asarray(non_tensor_batch["episode_rewards"], dtype=np.float64) > 0


def shared_state_groups(step_group_uids, traj_index, min_trajectories: int = 2) -> dict[str, list[str]]:
    """Matched-state groups visited by at least `min_trajectories` distinct trajectories."""
    members: dict[str, list[str]] = OrderedDict()
    for gid, traj in zip(step_group_uids, traj_index):
        trajs = members.setdefault(str(gid), [])
        if str(traj) not in trajs:
            trajs.append(str(traj))
    return OrderedDict((gid, trajs) for gid, trajs in members.items() if len(trajs) >= min_trajectories)


def check_group(success, traj_index, step_group_uids, min_shared_trajectories: int = 2) -> dict[str, Any]:
    traj_index = np.asarray(traj_index, dtype=object)
    success = np.asarray(success, dtype=bool)
    traj_success = OrderedDict()
    for traj, ok in zip(traj_index, success):
        traj_success.setdefault(str(traj), bool(ok))
    n_success = sum(traj_success.values())
    n_failure = len(traj_success) - n_success
    shared = shared_state_groups(step_group_uids, traj_index, min_shared_trajectories)
    missing = []
    if n_success == 0:
        missing.append("no successful trajectory")
    if n_failure == 0:
        missing.append("no failed trajectory")
    if not shared:
        missing.append(f"no state shared by >= {min_shared_trajectories} trajectories")
    return {
        "qualifies": not missing,
        "n_trajectories": len(traj_success),
        "n_success": int(n_success),
        "n_failure": int(n_failure),
        "n_shared_state_groups": len(shared),
        "missing": missing,
    }


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def _row_values(tensor_or_array) -> list:
    if isinstance(tensor_or_array, torch.Tensor):
        tensor_or_array = tensor_or_array.detach().cpu().to(torch.float64).numpy()
    return [_clean_float(v) for v in np.asarray(tensor_or_array).reshape(-1)]


def _clean_float(value):
    value = float(value)
    return value if math.isfinite(value) else None


def _short_action(text: str, limit: int = 60) -> str:
    matches = re.findall(r"<action>(.*?)</action>", text, flags=re.S)
    if matches:
        text = matches[-1]
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def build_report(
    relace_inputs: dict,
    step_group_uids,
    original_prompts=None,
    hindsight_prompts=None,
    actions=None,
    final_obs=None,
    training_advantages: torch.Tensor | None = None,
    meta: dict | None = None,
) -> dict[str, Any]:
    """Run RELACE (as trained, and with rho=1) on one fixed batch and assemble a report."""
    step_group_uids = np.asarray(step_group_uids, dtype=object).reshape(-1)
    full = compute_relace_components(**relace_inputs, step_group_uids=step_group_uids)
    unit = compute_relace_components(**relace_inputs, step_group_uids=step_group_uids, unit_rho=True)

    bsz = len(step_group_uids)
    response_mask = relace_inputs["response_mask"]
    traj_index = full["traj_index"]
    timesteps = full["timesteps"]
    anchor_obs = np.asarray(relace_inputs["anchor_obs"], dtype=object).reshape(-1)
    returns = _row_values(full["terminal_returns"])
    success_src = relace_inputs.get("success_mask")
    success = (np.asarray(_row_values(success_src)) > 0) if success_src is not None else np.asarray(returns) > 0
    empty = [""] * bsz
    actions = [str(a) for a in (actions if actions is not None else empty)]
    original_prompts = [str(p) for p in (original_prompts if original_prompts is not None else empty)]
    hindsight_prompts = [str(p) for p in (hindsight_prompts if hindsight_prompts is not None else empty)]
    final_obs = [str(o) for o in (final_obs if final_obs is not None else empty)]
    n_tokens = response_mask.sum(dim=-1).to(torch.int64).tolist()

    # Readable labels: trajectories T0.. by first row, states S1.. by first visit in that order.
    traj_labels = OrderedDict()
    for traj in traj_index:
        traj_labels.setdefault(str(traj), f"T{len(traj_labels)}")
    order = sorted(range(bsz), key=lambda i: (int(traj_labels[str(traj_index[i])][1:]), int(timesteps[i]), i))
    state_labels = OrderedDict()
    for i in order:
        state_labels.setdefault(str(step_group_uids[i]), f"S{len(state_labels) + 1}")
    state_members: dict[str, list[str]] = OrderedDict()
    state_rows: dict[str, list[int]] = OrderedDict()
    for i in order:
        gid = str(step_group_uids[i])
        t_label = traj_labels[str(traj_index[i])]
        state_rows.setdefault(gid, []).append(i)
        if t_label not in state_members.setdefault(gid, []):
            state_members[gid].append(t_label)

    full_vals = {name: _row_values(full[key]) for name, key in _FULL_FIELDS.items()}
    unit_vals = {name: _row_values(unit[key]) for name, key in _UNIT_FIELDS.items()}
    unmasked = full_vals["local_advantage_unmasked"]
    masked = full_vals["local_advantage_masked"]

    rows = []
    for i in order:
        gid = str(step_group_uids[i])
        row = OrderedDict(
            row=i,
            trajectory=traj_labels[str(traj_index[i])],
            traj_uid=str(traj_index[i]),
            task_uid=str(full["index"][i]),
            t=int(timesteps[i]),
            episode_length=int(full["episode_lengths"][i]),
            observation=str(anchor_obs[i]),
            action=actions[i],
            action_short=_short_action(actions[i]),
            action_tokens=int(n_tokens[i]),
            state_group=state_labels[gid],
            state_group_uid=gid,
            state_group_size=len(state_rows[gid]),
            state_group_trajectories=state_members[gid],
            shared_state=len(state_members[gid]) >= 2,
            terminal_reward=returns[i],
            success=bool(success[i]),
        )
        for name in _FULL_FIELDS:
            row[name] = full_vals[name][i]
        row["local_suppressed_by_mask"] = bool(
            unmasked[i] is not None and masked[i] is not None and unmasked[i] != masked[i]
        )
        row["rho_1"] = OrderedDict((name, unit_vals[name][i]) for name in _UNIT_FIELDS)
        if row["final_advantage"] is not None and row["rho_1"]["final_advantage"] is not None:
            row["final_advantage_delta"] = row["final_advantage"] - row["rho_1"]["final_advantage"]
        else:
            row["final_advantage_delta"] = None
        row["original_prompt"] = original_prompts[i]
        row["hindsight_prompt"] = hindsight_prompts[i]
        row["final_obs"] = final_obs[i]
        rows.append(row)

    trajectories = []
    for traj_uid, label in traj_labels.items():
        t_rows = [r for r in rows if r["traj_uid"] == traj_uid]
        trajectories.append(OrderedDict(
            trajectory=label,
            traj_uid=traj_uid,
            length=len(t_rows),
            terminal_reward=t_rows[0]["terminal_reward"],
            success=t_rows[0]["success"],
            final_obs=t_rows[0]["final_obs"],
        ))

    state_groups = [
        OrderedDict(
            state_group=state_labels[gid],
            state_group_uid=gid,
            size=len(state_rows[gid]),
            trajectories=state_members[gid],
            rows=[(next(r["trajectory"] for r in rows if r["row"] == i), int(timesteps[i])) for i in state_rows[gid]],
            observation=str(anchor_obs[state_rows[gid][0]]),
        )
        for gid in state_labels
    ]

    crosscheck = None
    if training_advantages is not None:
        diff = (full["advantages"].detach().cpu().to(torch.float64) - training_advantages.detach().cpu().to(torch.float64)).abs()
        crosscheck = {
            "description": "max |token advantage from this analysis - advantage written by compute_advantage()|",
            "max_abs_diff": float(diff.max().item()) if diff.numel() else 0.0,
        }
    tokens_match = torch.allclose(
        full["advantages"],
        (full["macro_rows"].unsqueeze(-1) + full["omega"] * full["local_rows"].unsqueeze(-1)) * response_mask,
        atol=1e-6,
    )

    settings = {
        key: (value if isinstance(value, (bool, int, float, str)) or value is None else str(value))
        for key, value in relace_inputs.items()
        if key in ("epsilon", "temp", "clip_min", "clip_max", "omega", "gamma", "smooth", "alpha", "mode",
                   "enable_similarity", "similarity_thresh", "no_harm")
    }
    return OrderedDict(
        meta=meta or {},
        relace_settings=settings,
        definitions={
            "p": "exp(mean_token log pi(a_t | original prompt) / temp)",
            "p_hindsight": "exp(mean_token log pi(a_t | hindsight prompt) / temp)",
            "raw_ratio": "u = p_hindsight / (p + epsilon)",
            "rho": "clip(u / (mean of u over the trajectory + epsilon), clip_min, clip_max)",
            "discounted_return": "gamma^(T - t) * terminal_reward",
            "weighted_return": "rho * discounted_return",
            "smoothed_return": "alpha * weighted_t + (1 - alpha) * weighted_{t+1} if smooth, else weighted_return",
            "macro_advantage": "episode-level GRPO advantage of terminal_reward within the task group",
            "local_advantage_unmasked": "smoothed_return normalized within its matched-state group",
            "local_advantage_masked": "local advantage after zeroing negative values on successful trajectories (no_harm)",
            "final_advantage": "macro_advantage + omega * local_advantage_masked (applied to every action token)",
            "rho_1": "same quantities recomputed on the identical batch and state groups with rho = 1",
        },
        trajectories=trajectories,
        state_groups=state_groups,
        rows=rows,
        metrics={"rho": full["metrics"], "rho_1": unit["metrics"]},
        consistency={"training_crosscheck": crosscheck, "token_advantages_equal_row_formula": bool(tokens_match)},
    )


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def _fmt(value) -> str:
    if value is None:
        return "–"
    if isinstance(value, bool):
        return "✓" if value else "✗"
    if isinstance(value, float):
        return f"{value:+.4f}" if value != 0 else "0"
    return str(value)


def _cell(text, limit: int | None = 80) -> str:
    text = " ".join(str(text).split())
    if limit is not None and len(text) > limit:
        text = text[: limit - 1] + "…"
    return text.replace("|", "\\|")


def _md_table(headers: list[str], body: list[list]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in body]
    return "\n".join(lines)


def side_by_side_table(rows: list[dict]) -> str:
    headers = ["traj", "t", "state", "shared", "rho", "Q~ (rho)", "Q~ (rho=1)",
               "local post (rho)", "local post (rho=1)", "final (rho)", "final (rho=1)", "delta"]
    body = [[r["trajectory"], r["t"], r["state_group"], _fmt(r["shared_state"]), _fmt(r["rho"]),
             _fmt(r["smoothed_return"]), _fmt(r["rho_1"]["smoothed_return"]),
             _fmt(r["local_advantage_masked"]), _fmt(r["rho_1"]["local_advantage_masked"]),
             _fmt(r["final_advantage"]), _fmt(r["rho_1"]["final_advantage"]), _fmt(r["final_advantage_delta"])]
            for r in rows]
    return _md_table(headers, body)


def _trajectory_markdown(traj: dict, rows: list[dict], report: dict) -> str:
    status = "success" if traj["success"] else "failure"
    out = [
        f"# {traj['trajectory']} — {status}, terminal reward {traj['terminal_reward']}, {traj['length']} actions",
        "",
        f"- traj_uid: `{traj['traj_uid']}`",
        f"- final observation: {_cell(traj['final_obs'], 400)}",
        f"- prompts: [prompts/{traj['trajectory']}.md](../prompts/{traj['trajectory']}.md)",
        "",
        "## Per-action RELACE quantities",
        "",
        "Q~ = smoothed weighted return. Local (pre/post) = local advantage before/after the success mask. "
        "State groups marked * are shared with another trajectory.",
        "",
    ]
    headers = ["t", "observation", "action", "state", "R", "succ", "p", "pH", "u", "rho",
               "G", "rho·G", "Q~", "macro", "local pre", "local post", "final"]
    body = []
    for r in rows:
        state = r["state_group"] + ("*" if r["shared_state"] else "")
        body.append([r["t"], _cell(r["observation"], 60), _cell(r["action_short"], 50), state,
                     _fmt(r["terminal_reward"]), _fmt(r["success"]), _fmt(r["p"]), _fmt(r["p_hindsight"]),
                     _fmt(r["raw_ratio"]), _fmt(r["rho"]), _fmt(r["discounted_return"]), _fmt(r["weighted_return"]),
                     _fmt(r["smoothed_return"]), _fmt(r["macro_advantage"]), _fmt(r["local_advantage_unmasked"]),
                     _fmt(r["local_advantage_masked"]), _fmt(r["final_advantage"])])
    out += [_md_table(headers, body), "", "## RELACE vs rho = 1 (same batch, same state groups)", "",
            side_by_side_table(rows), ""]
    return "\n".join(out)


def _prompts_markdown(traj: dict, rows: list[dict]) -> str:
    out = [f"# {traj['trajectory']} prompts", "",
           "Original prompt X_t and the outcome-augmented (hindsight) prompt used to score each executed action.", ""]
    for r in rows:
        out += [f"## t = {r['t']} (state {r['state_group']})", "", "### Action", "", "~~~~text", r["action"], "~~~~", "",
                "### Original prompt", "", "~~~~text", r["original_prompt"], "~~~~", "",
                "### Hindsight prompt", "", "~~~~text", r["hindsight_prompt"], "~~~~", ""]
    return "\n".join(out)


def _csv_rows(rows: list[dict]) -> tuple[list[str], list[list]]:
    skip = {"original_prompt", "hindsight_prompt", "rho_1", "state_group_trajectories"}
    base = [k for k in rows[0].keys() if k not in skip]
    headers = base + ["state_group_trajectories"] + [f"rho_1.{k}" for k in _UNIT_FIELDS]
    body = [[r[k] for k in base] + [" ".join(r["state_group_trajectories"])] + [r["rho_1"][k] for k in _UNIT_FIELDS]
            for r in rows]
    return headers, body


def _write_csv(path: str, rows: list[dict]) -> None:
    headers, body = _csv_rows(rows)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(body)


def summary_markdown(report: dict) -> str:
    meta = report["meta"]
    out = ["# RELACE trajectory analysis", ""]
    for key, value in meta.items():
        out.append(f"- {key}: `{value}`")
    out += ["", "## Trajectories", ""]
    out.append(_md_table(["traj", "outcome", "R", "actions", "final observation"], [
        [t["trajectory"], "success" if t["success"] else "failure", _fmt(t["terminal_reward"]), t["length"],
         _cell(t["final_obs"], 100)] for t in report["trajectories"]]))
    shared = [g for g in report["state_groups"] if len(g["trajectories"]) >= 2]
    out += ["", f"## Shared matched states ({len(shared)})", ""]
    out.append(_md_table(["state", "visits (traj, t)", "observation"], [
        [g["state_group"], ", ".join(f"{tr}@{t}" for tr, t in g["rows"]), _cell(g["observation"], 100)] for g in shared]))
    out += ["", "## RELACE settings", "", "```json", json.dumps(report["relace_settings"], indent=2), "```", "",
            "## Metrics: RELACE vs rho = 1", ""]
    m_full, m_unit = report["metrics"]["rho"], report["metrics"]["rho_1"]
    out.append(_md_table(["metric", "RELACE", "rho = 1"], [[k, _fmt(m_full[k]), _fmt(m_unit.get(k))] for k in m_full]))
    out += ["", "## Consistency checks", "", "```json", json.dumps(report["consistency"], indent=2), "```", "",
            "## All actions: RELACE vs rho = 1", "", side_by_side_table(report["rows"]), "",
            "Per-trajectory tables: `trajectories/`. Prompts: `prompts/`. Everything: `group.json`.", ""]
    return "\n".join(out)


def write_report(report: dict, out_dir: str) -> str:
    os.makedirs(os.path.join(out_dir, "trajectories"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "prompts"), exist_ok=True)
    with open(os.path.join(out_dir, "group.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    with open(os.path.join(out_dir, "summary.md"), "w", encoding="utf-8") as f:
        f.write(summary_markdown(report))
    _write_csv(os.path.join(out_dir, "all_actions.csv"), report["rows"])
    for traj in report["trajectories"]:
        t_rows = [r for r in report["rows"] if r["traj_uid"] == traj["traj_uid"]]
        label = traj["trajectory"]
        with open(os.path.join(out_dir, "trajectories", f"{label}.md"), "w", encoding="utf-8") as f:
            f.write(_trajectory_markdown(traj, t_rows, report))
        _write_csv(os.path.join(out_dir, "trajectories", f"{label}.csv"), t_rows)
        with open(os.path.join(out_dir, "prompts", f"{label}.md"), "w", encoding="utf-8") as f:
            f.write(_prompts_markdown(traj, t_rows))
    return out_dir


def print_summary(report: dict, out_dir: str) -> None:
    print("=" * 100)
    print("RELACE trajectory analysis")
    for t in report["trajectories"]:
        print(f"  {t['trajectory']}: {'success' if t['success'] else 'failure'}  R={t['terminal_reward']}  actions={t['length']}")
    print()
    print(side_by_side_table(report["rows"]))
    print()
    print(f"consistency: {json.dumps(report['consistency'])}")
    print(f"written to: {out_dir}")
    print("=" * 100)


# ---------------------------------------------------------------------------
# Offline replay
# ---------------------------------------------------------------------------

def save_inputs(path: str, relace_inputs: dict, step_group_uids, extras: dict) -> None:
    payload = {"relace_inputs": relace_inputs, "step_group_uids": np.asarray(step_group_uids, dtype=object), "extras": extras}
    torch.save(payload, path)


def load_inputs(path: str) -> dict:
    # The file holds numpy object arrays, so it is unpickled: only load files you produced.
    return torch.load(path, map_location="cpu", weights_only=False)


def replay(path: str, out_dir: str) -> str:
    payload = load_inputs(path)
    extras = dict(payload.get("extras", {}))
    meta = dict(extras.pop("meta", {}))
    meta["replayed_from"] = os.path.abspath(path)
    report = build_report(payload["relace_inputs"], payload["step_group_uids"], meta=meta, **extras)
    write_report(report, out_dir)
    print_summary(report, out_dir)
    return out_dir


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p_replay = sub.add_parser("replay", help="re-run the analysis on a saved relace_inputs.pt (CPU only)")
    p_replay.add_argument("inputs")
    p_replay.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    if args.command == "replay":
        replay(args.inputs, args.out)


if __name__ == "__main__":
    main()
