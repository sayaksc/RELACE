import json
import os
from pathlib import Path
import shlex
import subprocess
import tempfile

import numpy as np
import pytest
import torch

from gigpo.core_gigpo import build_step_group
from relace import trajectory_analysis as ta
from relace.core_relace import compute_relace_advantage, compute_relace_components

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def group():
    # One task, three trajectories: A (success, 3 steps), B (failure, 2 steps), C (success, 2 steps).
    # States 'start' and 'kitchen' are shared across trajectories.
    trajs = [('A', 1.0, ['start', 'kitchen', 'fridge']), ('B', 0.0, ['start', 'garden']), ('C', 1.0, ['start', 'kitchen'])]
    rows = [(name, t + 1, len(obs), reward, o) for name, reward, obs in trajs for t, o in enumerate(obs)]
    bs, length = len(rows), 4
    rng = np.random.default_rng(0)
    mask = torch.zeros(bs, length)
    for i in range(bs):
        mask[i, :1 + i % length] = 1
    rewards = torch.zeros(bs, length)
    for i, row in enumerate(rows):
        rewards[i, int(mask[i].sum()) - 1] = row[3]
    inputs = dict(
        token_level_rewards=rewards, response_mask=mask,
        hindsight_log_probs=torch.tensor(rng.normal(-1, .6, (bs, length)), dtype=torch.float32),
        action_log_probs=torch.tensor(rng.normal(-1, .6, (bs, length)), dtype=torch.float32),
        anchor_obs=np.array([r[4] for r in rows], dtype=object), index=np.array(['task'] * bs, dtype=object),
        traj_index=np.array([r[0] for r in rows], dtype=object), timestep=np.array([r[1] for r in rows]),
        episode_lengths=np.array([r[2] for r in rows], dtype=np.float32),
        episode_returns=np.array([r[3] for r in rows], dtype=np.float32),
        success_mask=np.array([r[3] for r in rows], dtype=np.float32),
        clip_min=.8, clip_max=1.2, gamma=.6, smooth=True, alpha=.5, omega=1.0, no_harm=True,
        enable_similarity=False, similarity_thresh=.95, mode='mean_std_norm', temp=1.0, epsilon=1e-6,
    )
    return inputs


def test_components_match_training_advantage(group):
    expected, _, metrics = compute_relace_advantage(**group)
    components = compute_relace_components(**group)
    torch.testing.assert_close(components['advantages'], expected, rtol=0, atol=0)
    assert components['metrics'] == metrics
    # A fixed grouping with the same partition gives the same result.
    uids = build_step_group(group['anchor_obs'], group['index'], False, .95)
    fixed = compute_relace_components(**group, step_group_uids=uids)
    torch.testing.assert_close(fixed['advantages'], expected, rtol=0, atol=0)


def test_component_formulas(group):
    c = compute_relace_components(**group)
    eps = group['epsilon']
    torch.testing.assert_close(c['raw_ratio'], c['p_hindsight'] / (c['p'] + eps))
    torch.testing.assert_close(c['rho'], torch.clamp(c['raw_ratio'] / (c['raw_ratio_traj_mean'] + eps), .8, 1.2))
    expected_g = torch.tensor([.6 ** (T - t) * r for _, t, T, r in zip(group['traj_index'], group['timestep'],
                              group['episode_lengths'], group['episode_returns'])], dtype=torch.float32)
    torch.testing.assert_close(c['discounted_returns'], expected_g)
    torch.testing.assert_close(c['weighted_returns'], c['rho'] * c['discounted_returns'])
    success = torch.as_tensor(group['success_mask']) > 0
    assert torch.all(c['local_rows'][success] >= 0)
    torch.testing.assert_close(c['local_rows'][~success], c['local_rows_unmasked'][~success])
    torch.testing.assert_close(c['advantages'], (c['macro_rows'] + c['local_rows']).unsqueeze(-1) * group['response_mask'])


def test_unit_rho_only_removes_retrospective_weight(group):
    uids = build_step_group(group['anchor_obs'], group['index'], False, .95)
    full = compute_relace_components(**group, step_group_uids=uids)
    unit = compute_relace_components(**group, step_group_uids=uids, unit_rho=True)
    assert torch.all(unit['rho'] == 1)
    for key in ['p', 'p_hindsight', 'raw_ratio', 'discounted_returns', 'macro_rows']:
        torch.testing.assert_close(unit[key], full[key])
    torch.testing.assert_close(unit['weighted_returns'], unit['discounted_returns'])
    # Equivalent to clipping rho into [1, 1] in the training entry point.
    pinned, _, _ = compute_relace_advantage(**{**group, 'clip_min': 1.0, 'clip_max': 1.0})
    torch.testing.assert_close(unit['advantages'], pinned)


def test_group_selection(group):
    uids = build_step_group(group['anchor_obs'], group['index'], False, .95)
    success = group['success_mask'] > 0
    check = ta.check_group(success, group['traj_index'], uids)
    assert check['qualifies'] and check['n_success'] == 2 and check['n_failure'] == 1
    assert check['n_shared_state_groups'] == 2  # 'start' (A, B, C) and 'kitchen' (A, C)
    assert not ta.check_group(np.ones_like(success), group['traj_index'], uids)['qualifies']
    unique = np.array([f'u{i}' for i in range(len(uids))], dtype=object)
    assert ta.check_group(success, group['traj_index'], unique)['missing'] == ['no state shared by >= 2 trajectories']
    assert ta.check_group(success, group['traj_index'], uids, min_shared_trajectories=3)['n_shared_state_groups'] == 1


def test_report_and_replay(group):
    uids = build_step_group(group['anchor_obs'], group['index'], False, .95)
    bs = len(uids)
    extras = dict(
        original_prompts=[f'Your task is to: open fridge\nstep {i}' for i in range(bs)],
        hindsight_prompts=[f'Your task is to: reach | state\nstep {i}' for i in range(bs)],
        actions=[f'<think>...</think><action>go to {o}</action>' for o in group['anchor_obs']],
        final_obs=['done'] * bs,
        training_advantages=compute_relace_advantage(**group)[0],
        meta={'checkpoint': 'test'},
    )
    report = ta.build_report(group, uids, **extras)
    assert report['consistency']['training_crosscheck']['max_abs_diff'] == 0
    assert report['consistency']['token_advantages_equal_row_formula']
    assert [r['trajectory'] for r in report['rows']] == ['T0'] * 3 + ['T1'] * 2 + ['T2'] * 2
    first = report['rows'][0]
    assert first['action_short'] == 'go to start' and first['state_group'] == 'S1' and first['shared_state']
    assert first['state_group_trajectories'] == ['T0', 'T1', 'T2']
    for r in report['rows']:
        assert r['rho_1']['rho'] == 1
        assert r['final_advantage'] == pytest.approx(r['macro_advantage'] + r['local_advantage_masked'])

    with tempfile.TemporaryDirectory(dir=ROOT) as directory:
        out = os.path.join(directory, 'run')
        ta.write_report(report, out)
        saved = json.loads(Path(out, 'group.json').read_text())
        assert len(saved['rows']) == bs and saved['rows'][0]['hindsight_prompt'].startswith('Your task is to: reach')
        for name in ['summary.md', 'all_actions.csv', 'trajectories/T0.md', 'trajectories/T2.csv', 'prompts/T1.md']:
            assert Path(out, name).is_file(), name
        table = Path(out, 'trajectories/T0.md').read_text()
        assert 'final (rho=1)' in table and 'local pre' in table
        assert 'reach \\| state' not in table  # prompts live in prompts/, not in tables
        assert 'Hindsight prompt' in Path(out, 'prompts/T1.md').read_text()

        ta.save_inputs(os.path.join(out, 'relace_inputs.pt'), group, uids, extras)
        replayed = os.path.join(directory, 'replay')
        ta.main(['replay', os.path.join(out, 'relace_inputs.pt'), '--out', replayed])
        again = json.loads(Path(replayed, 'group.json').read_text())
        assert [r['final_advantage'] for r in again['rows']] == [r['final_advantage'] for r in saved['rows']]
        assert [r['rho_1'] for r in again['rows']] == [r['rho_1'] for r in saved['rows']]


def test_analysis_launcher():
    env = dict(os.environ, DRY_RUN='1', GROUP_SIZE='6', MAX_ATTEMPTS='5')
    for name in ['PROJECT_DIR', 'CHECKPOINT', 'OUTPUT_DIR', 'EXPERIMENT_NAME']:
        env.pop(name, None)
    command = subprocess.check_output(
        ['bash', str(ROOT / 'examples/relace_trainer/analyze_trajectories.sh'), 'alfworld', 'env.seed=3'], env=env, text=True)
    args = shlex.split(command)
    for expected in ['algorithm.adv_estimator=relace', 'trajectory_analysis.enable=True', 'trajectory_analysis.checkpoint=null',
                     'trajectory_analysis.group_size=6', 'trajectory_analysis.max_attempts=5', 'env.rollout.n=6',
                     'data.train_batch_size=1', 'algorithm.relace.hindsight_template=alfworld']:
        assert expected in args, expected
    assert args[-1] == 'env.seed=3'
