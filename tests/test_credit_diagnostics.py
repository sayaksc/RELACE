from collections import defaultdict

import numpy as np
import pytest
import torch

from relace import credit_diagnostics as cd
from relace.core_relace import compute_relace_advantage, compute_relace_components

GAMMA = 0.6
GAMMA_KEY = f'{cd.METRIC_PREFIX}/gamma_0.6/'
COVERAGE = f'{cd.METRIC_PREFIX}/coverage/'


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
def relace_rollout(seed=0, n_tasks=3, n_traj=8, steps=3, vocab=2, n_obs=2, max_len=2, unique_actions=False):
    """A small but realistic RELACE batch: inputs for `compute_relace_components` plus the executed tokens."""
    rng = np.random.default_rng(seed)
    rows = []  # (task, traj, t, T, obs, win)
    for task in range(n_tasks):
        for k in range(n_traj):
            length = int(rng.integers(2, steps + 1))
            win = float(rng.random() < 0.5)
            for t in range(1, length + 1):
                rows.append((f'task{task}', f'task{task}_traj{k}', t, length, f'obs{rng.integers(n_obs)}', win))
    bs = len(rows)
    responses = torch.zeros(bs, max_len, dtype=torch.long)
    mask = torch.zeros(bs, max_len)
    for i in range(bs):
        size = int(rng.integers(1, max_len + 1))
        tokens = np.arange(i * max_len, i * max_len + size) + 1 if unique_actions else rng.integers(1, 1 + vocab, size)
        responses[i, :size] = torch.as_tensor(tokens)
        mask[i, :size] = 1
    returns = np.array([10.0 * r[5] for r in rows], dtype=np.float32)
    token_rewards = torch.zeros(bs, max_len)
    token_rewards[:, 0] = torch.as_tensor(returns)
    generator = torch.Generator().manual_seed(seed)
    inputs = dict(
        token_level_rewards=token_rewards,
        response_mask=mask,
        hindsight_log_probs=-torch.rand(bs, max_len, generator=generator) * 2,
        action_log_probs=-torch.rand(bs, max_len, generator=generator) * 2,
        anchor_obs=np.array([r[4] for r in rows], dtype=object),
        index=np.array([r[0] for r in rows], dtype=object),
        traj_index=np.array([r[1] for r in rows], dtype=object),
        timestep=np.array([r[2] for r in rows]),
        episode_lengths=np.array([r[3] for r in rows]),
        episode_returns=returns,
        success_mask=(returns > 0).astype(np.float32),
        gamma=GAMMA,
        smooth=True,
        alpha=0.5,
        no_harm=True,
        enable_similarity=False,
    )
    return inputs, responses


def run_diagnostics(inputs, responses, **cfg_kwargs):
    components = compute_relace_components(**inputs)
    cfg = cd.CreditDiagConfig(enable=True, **cfg_kwargs)
    metrics = cd.compute_credit_diagnostics(
        components=components, responses=responses, response_mask=inputs['response_mask'], gamma=GAMMA, cfg=cfg
    )
    return components, metrics


def random_rows(seed, n_tasks=3, n_traj=8, steps=3, n_groups=2, n_keys=2):
    """Id arrays (task, traj, group, key) plus rho / G for the array-level API."""
    rng = np.random.default_rng(seed)
    task, traj, group, key = [], [], [], []
    for a in range(n_tasks):
        for k in range(n_traj):
            for _ in range(steps):
                task.append(f'task{a}')
                traj.append(f'task{a}_traj{k}')
                group.append(f'task{a}_group{rng.integers(n_groups)}')
                key.append(int(rng.integers(n_keys)))
    n = len(task)
    rho = rng.uniform(0.8, 1.2, n)
    returns = rng.choice([0.0, 0.6, 1.0, 6.0, 10.0], n)
    return np.array(task), np.array(traj), np.array(group), np.array(key), rho, returns


def oracle(task, traj, group, key, folds, rho, returns, min_ref=2):
    """Brute-force, loop-only restatement of the spec. Returns per-group dicts of the eligible groups."""
    n = len(task)
    q = [None] * n
    for i in range(n):
        refs = defaultdict(list)
        for j in range(n):
            if folds[j] != folds[i] and (task[j], group[j], key[j]) == (task[i], group[i], key[i]):
                refs[traj[j]].append(returns[j])
        if len(refs) >= min_ref:
            q[i] = float(np.mean([np.mean(v) for v in refs.values()]))
    cells = defaultdict(list)
    for i in range(n):
        cells[(task[i], group[i], folds[i])].append(i)
    out = []
    for rows in cells.values():
        rows = [i for i in rows if q[i] is not None]
        if len({traj[i] for i in rows}) < 2:
            continue
        g = np.array([returns[i] for i in rows])
        h = np.array([rho[i] * returns[i] for i in rows])
        qq = np.array([q[i] for i in rows])
        a_g, a_h, a_ref = g - g.mean(), h - h.mean(), qq - qq.mean()
        d = np.array([(rho[i] - 1) * returns[i] for i in rows])
        noise = g - qq
        mse_gigpo = np.mean((a_g - a_ref) ** 2)
        mse_relace = np.mean((a_h - a_ref) ** 2)
        out.append(dict(
            n=len(rows), mse_gigpo=mse_gigpo, mse_relace=mse_relace, gap=mse_relace - mse_gigpo,
            var_d=np.mean((d - d.mean()) ** 2), cov_nd=np.mean((noise - noise.mean()) * (d - d.mean())),
        ))
    return out


def structure_and_stats(task, traj, group, key, rho, returns, min_ref=2, seed=0):
    folds = cd.assign_folds(task, traj, seed=seed)
    structure = cd.build_structure(task, traj, group, key, folds, min_reference_trajectories=min_ref)
    return folds, structure, cd.group_stats(structure, rho, returns)


# --------------------------------------------------------------------------- #
# Numerics: oracle, identity, rho = 1, hand-computed values
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize('seed', range(5))
@pytest.mark.parametrize('min_ref', [1, 2])
def test_matches_brute_force_oracle(seed, min_ref):
    task, traj, group, key, rho, returns = random_rows(seed)
    folds, structure, stats = structure_and_stats(task, traj, group, key, rho, returns, min_ref=min_ref, seed=seed)
    expected = oracle(task, traj, group, key, folds, rho, returns, min_ref=min_ref)
    assert stats is not None and structure.num_groups == len(expected) > 0
    fields = ['n', 'mse_gigpo', 'mse_relace', 'gap', 'var_d', 'cov_nd']
    key_of = lambda row: tuple(round(float(row[f]), 9) for f in fields)  # noqa: E731
    actual_rows = [{f: getattr(stats, f)[g] for f in fields} for g in range(structure.num_groups)]
    assert sorted(map(key_of, actual_rows)) == sorted(map(key_of, expected))


@pytest.mark.parametrize('seed', range(5))
def test_mse_difference_equals_variance_plus_twice_covariance(seed):
    task, traj, group, key, rho, returns = random_rows(seed)
    _, structure, stats = structure_and_stats(task, traj, group, key, rho, returns, seed=seed)
    assert structure.num_groups > 0
    np.testing.assert_allclose(stats.gap, stats.var_d + 2 * stats.cov_nd, rtol=0, atol=1e-12)
    assert cd.aggregate_stats(stats)['identity_max_abs_error'] < 1e-12
    # the identity is an algebraic fact, so it also holds when rho is wild
    _, _, wild = structure_and_stats(task, traj, group, key, rho * 5 - 3, returns, seed=seed)
    np.testing.assert_allclose(wild.gap, wild.var_d + 2 * wild.cov_nd, rtol=0, atol=1e-10)


def test_unit_rho_gives_equal_mse_and_zero_gap():
    task, traj, group, key, _, returns = random_rows(1)
    _, structure, stats = structure_and_stats(task, traj, group, key, np.ones(len(task)), returns)
    assert structure.num_groups > 0
    np.testing.assert_array_equal(stats.mse_relace, stats.mse_gigpo)
    np.testing.assert_array_equal(stats.gap, 0.0)
    np.testing.assert_array_equal(stats.var_d, 0.0)
    np.testing.assert_array_equal(stats.cov_nd, 0.0)
    aggregate = cd.aggregate_stats(stats)
    assert aggregate['proxy_mse_gap'] == 0.0
    assert aggregate['fraction_groups_negative_gap'] == 0.0
    assert aggregate['relative_mse_reduction_pct'] == 0.0


def hand_fixture():
    """One task, one anchor group, one action. Fold 0 evaluates T0, T1; fold 1 evaluates T2 (twice) and T3."""
    traj = np.array(['T0', 'T1', 'T2', 'T2', 'T3'])
    folds = np.array([0, 0, 1, 1, 1])
    returns = np.array([1.0, 3.0, 2.0, 6.0, 2.0])
    rho = np.array([1.2, 0.8, 1.0, 1.0, 1.0])
    structure = cd.build_structure(['t'] * 5, traj, ['g'] * 5, [7] * 5, folds)
    return structure, rho, returns


def test_hand_computed_values_and_trajectory_balanced_reference():
    structure, rho, returns = hand_fixture()
    q = cd.cross_fitted_q(structure, returns)
    # fold 0 is referenced by fold 1: mean(mean(T2: 2, 6) = 4, T3: 2) = 3  (pooling occurrences would give 3.33)
    np.testing.assert_allclose(q[:2], 3.0)
    # fold 1 is referenced by fold 0: mean(T0: 1, T1: 3) = 2
    np.testing.assert_allclose(q[2:], 2.0)

    stats = cd.group_stats(structure, rho, returns)
    assert structure.num_groups == 2
    # group (fold 0): G=[1,3] rhoG=[1.2,2.4] Q=3 -> mse_gigpo=1, mse_relace=.36, var_D=.16, cov=-.4, gap=-.64
    # group (fold 1): G=[2,6,2], rho=1 -> mse_gigpo = mse_relace = 96/27, gap=0
    np.testing.assert_allclose(stats.n, [2, 3])
    np.testing.assert_allclose(stats.mse_gigpo, [1.0, 96 / 27])
    np.testing.assert_allclose(stats.mse_relace, [0.36, 96 / 27])
    np.testing.assert_allclose(stats.var_d, [0.16, 0.0], atol=1e-15)
    np.testing.assert_allclose(stats.cov_nd, [-0.4, 0.0], atol=1e-15)
    np.testing.assert_allclose(stats.gap, [-0.64, 0.0], atol=1e-15)


def test_aggregation_is_occurrence_weighted():
    structure, rho, returns = hand_fixture()
    aggregate = cd.aggregate_stats(cd.group_stats(structure, rho, returns))
    mse_gigpo = (2 * 1.0 + 3 * 96 / 27) / 5
    mse_relace = (2 * 0.36 + 3 * 96 / 27) / 5
    assert aggregate['proxy_mse_gigpo'] == pytest.approx(mse_gigpo)
    assert aggregate['proxy_mse_relace'] == pytest.approx(mse_relace)
    assert aggregate['proxy_mse_gap'] == pytest.approx(mse_relace - mse_gigpo)
    assert aggregate['proxy_mse_gap'] == pytest.approx((2 * -0.64 + 3 * 0.0) / 5)  # weighted, not -0.32
    assert aggregate['group_mean_proxy_gap'] == pytest.approx(-0.32)  # the unweighted mean
    assert aggregate['fraction_groups_negative_gap'] == pytest.approx(0.5)
    assert aggregate['correction_variance'] == pytest.approx((2 * 0.16) / 5)
    assert aggregate['noise_correction_covariance'] == pytest.approx((2 * -0.4) / 5)
    assert aggregate['proxy_gap_from_covariance'] == pytest.approx(aggregate['proxy_mse_gap'])
    # from the aggregate MSEs, not from averaging group percentages (which would give 32%)
    assert aggregate['relative_mse_reduction_pct'] == pytest.approx(100 * (mse_gigpo - mse_relace) / mse_gigpo)


# --------------------------------------------------------------------------- #
# Cross-fitting: leakage, action keys, tasks
# --------------------------------------------------------------------------- #
def test_assign_folds_is_balanced_deterministic_and_order_free():
    task, traj, _, _, _, _ = random_rows(0, n_tasks=4, n_traj=8)
    folds = cd.assign_folds(task, traj, seed=3)
    np.testing.assert_array_equal(folds, cd.assign_folds(task, traj, seed=3))
    for t in np.unique(task):
        per_traj = {tr: folds[(task == t) & (traj == tr)][0] for tr in np.unique(traj[task == t])}
        assert sorted(np.bincount(list(per_traj.values()), minlength=2)) == [4, 4]
        assert all(len(set(folds[traj == tr])) == 1 for tr in per_traj)  # whole trajectories, never split
    perm = np.random.default_rng(0).permutation(len(task))
    np.testing.assert_array_equal(cd.assign_folds(task[perm], traj[perm], seed=3), folds[perm])
    assert not np.array_equal(cd.assign_folds(task, traj, seed=4), folds)  # the seed salts the rule


@pytest.mark.parametrize('seed', range(3))
def test_no_trajectory_is_in_both_reference_and_evaluation(seed):
    task, traj, group, key, rho, returns = random_rows(seed)
    folds, structure, _ = structure_and_stats(task, traj, group, key, rho, returns, seed=seed)
    has_q = np.flatnonzero(structure.ref_cell >= 0)
    assert len(has_q) > 0
    for i in has_q:
        for _, rows in structure.ref_cells[structure.ref_cell[i]]:
            assert np.all(folds[rows] == 1 - folds[i])  # other fold only
            assert not np.isin(traj[rows], traj[i]).any()  # never its own trajectory
            assert np.all(task[rows] == task[i]) and np.all(group[rows] == group[i]) and np.all(key[rows] == key[i])

    # Perturbing any trajectory must leave Q_hat of every row in that trajectory's fold untouched ...
    q_before = cd.cross_fitted_q(structure, returns)
    moved_other_fold = False
    for victim in np.unique(traj):
        bumped = returns.copy()
        bumped[traj == victim] += 100.0
        q_after = cd.cross_fitted_q(structure, bumped)
        same_fold = folds == folds[traj == victim][0]
        np.testing.assert_array_equal(q_after[same_fold], q_before[same_fold])  # NaNs compare equal
        moved_other_fold |= bool(np.any(q_after[~same_fold] != q_before[~same_fold]))
    assert moved_other_fold  # ... while it does feed the other fold (the check is not vacuous)


def test_action_keys_use_the_full_token_sequence_without_padding():
    responses = torch.tensor([[1, 2, 3, 0], [1, 2, 3, 9], [1, 2, 4, 0], [1, 2, 0, 0], [1, 2, 3, 3]])
    mask = torch.tensor([[1, 1, 1, 0], [1, 1, 1, 0], [1, 1, 1, 0], [1, 1, 0, 0], [1, 1, 1, 1]])
    ids = cd.action_key_ids(responses, mask)
    assert ids[0] == ids[1]  # same executed tokens, different padding content
    assert len({ids[0], ids[2], ids[3], ids[4]}) == 4  # a changed token, a prefix, and an extension all differ


def test_different_reasoning_or_task_never_shares_a_reference():
    # rows: [task, traj, group, key, fold]. Fold-1 rows would be perfect references in every respect but one.
    rows = [
        ('t0', 'A', 'g', 1, 0),  # evaluated row
        ('t0', 'B', 'g', 1, 0),  # second evaluated trajectory
        ('t0', 'C', 'g', 2, 1), ('t0', 'D', 'g', 2, 1),  # same task+group, DIFFERENT action key (e.g. other <think>)
        ('t1', 'E', 'g', 1, 1), ('t1', 'F', 'g', 1, 1),  # same group id and key, DIFFERENT task
        ('t0', 'G', 'h', 1, 1), ('t0', 'H', 'h', 1, 1),  # same task and key, DIFFERENT anchor group
    ]
    task, traj, group, key, folds = map(np.array, zip(*rows))
    structure = cd.build_structure(task, traj, group, key, folds)
    assert (structure.ref_cell[:2] == -1).all()  # nothing leaks into the evaluated rows
    assert (structure.n_ref[:2] == 0).all()
    # and the same fixture with matching key, task and group does provide a reference
    key_ok = key.copy()
    key_ok[2:4] = 1
    matched = cd.build_structure(task, traj, group, key_ok, folds)
    assert (matched.n_ref[:2] == 2).all() and (matched.ref_cell[:2] >= 0).all()


# --------------------------------------------------------------------------- #
# Edge cases and coverage
# --------------------------------------------------------------------------- #
def test_coverage_counts_distinguish_singleton_and_insufficient_groups():
    #  (g, fold 0): T0, T1 but only ONE possible reference trajectory (T2) -> insufficient reference
    #  (g, fold 1): T2 alone -> singleton;  (h, *): one trajectory per fold -> singleton
    traj = ['T0', 'T1', 'T2', 'T0', 'T2']
    group = ['g', 'g', 'g', 'h', 'h']
    folds = [0, 0, 1, 0, 1]
    structure = cd.build_structure(['t'] * 5, traj, group, [1] * 5, folds)
    assert structure.num_groups == 0 and cd.group_stats(structure, np.ones(5), np.ones(5)) is None
    assert structure.counts['singleton_groups'] == 3
    assert structure.counts['insufficient_reference_groups'] == 1
    assert structure.counts['steps_with_valid_q_reference'] == 1  # T2's row, referenced by T0 and T1
    assert structure.counts['steps_with_insufficient_reference'] == 4
    assert structure.counts['original_anchor_groups'] == 2
    metrics = cd.coverage_metrics(structure)
    assert metrics[COVERAGE + 'eligible_groups'] == 0 and metrics[COVERAGE + 'eligible_step_fraction'] == 0.0
    assert COVERAGE + 'fraction_groups_fully_retained' not in metrics  # undefined without eligible groups


def test_missing_references_log_coverage_but_no_error_metrics():
    inputs, responses = relace_rollout(unique_actions=True)  # every executed sequence is unique: nothing repeats
    _, metrics = run_diagnostics(inputs, responses)
    assert metrics[COVERAGE + 'total_valid_steps'] == len(responses)
    assert metrics[COVERAGE + 'steps_with_valid_q_reference'] == 0
    assert metrics[COVERAGE + 'eligible_steps'] == 0 and metrics[COVERAGE + 'eligible_step_fraction'] == 0.0
    assert not [k for k in metrics if '/gamma_' in k]  # no MSE at all, in particular no zeros standing in for data
    assert COVERAGE + 'ref_trajectories_mean' not in metrics


def test_all_zero_returns_stay_eligible_but_have_no_relative_metric():
    task, traj, group, key, rho, _ = random_rows(2)
    _, structure, stats = structure_and_stats(task, traj, group, key, rho, np.zeros(len(task)))
    assert structure.num_groups > 0
    aggregate = cd.aggregate_stats(stats)
    assert aggregate['proxy_mse_gigpo'] == 0.0 and aggregate['proxy_mse_relace'] == 0.0  # a real zero, not missing
    assert aggregate['proxy_mse_gap'] == 0.0 and aggregate['identity_max_abs_error'] == 0.0
    assert 'relative_mse_reduction_pct' not in aggregate
    assert cd.coverage_metrics(structure)[COVERAGE + 'eligible_groups'] == structure.num_groups


def test_near_zero_denominator_omits_relative_reduction():
    task, traj, group, key, rho, returns = random_rows(3)
    _, _, stats = structure_and_stats(task, traj, group, key, rho, returns * 1e-6)  # MSE ~ 1e-12 < floor
    aggregate = cd.aggregate_stats(stats)
    assert 0.0 < aggregate['proxy_mse_gigpo'] < cd.RELATIVE_DENOMINATOR_FLOOR
    assert 'relative_mse_reduction_pct' not in aggregate and 'proxy_mse_gap' in aggregate
    assert 'relative_mse_reduction_pct' in cd.aggregate_stats(stats, relative_floor=0.0)


def test_fully_retained_fraction_counts_groups_that_lost_no_rows():
    # (g, fold 0): T0, T1 both supported by T2, T3 -> fully retained. (g, fold 1): T2, T3 are supported by T0/T1,
    # but T3's second row uses another action and has no reference -> the group keeps 3 of 4 rows.
    traj = ['T0', 'T1', 'T2', 'T3', 'T3']
    key = [1, 1, 1, 1, 2]
    structure = cd.build_structure(['t'] * 5, traj, ['g'] * 5, key, [0, 0, 1, 1, 1])
    assert structure.num_groups == 2
    np.testing.assert_array_equal(structure.group_fully_retained, [True, False])
    assert cd.coverage_metrics(structure)[COVERAGE + 'fraction_groups_fully_retained'] == 0.5


# --------------------------------------------------------------------------- #
# End to end on a RELACE batch
# --------------------------------------------------------------------------- #
def test_end_to_end_metrics_and_gamma_namespaces():
    inputs, responses = relace_rollout()
    components, metrics = run_diagnostics(inputs, responses, extra_gammas=(0.6, 0.95))
    # The training gamma is not duplicated by an equal extra gamma.
    namespaces = {k.split('/')[1] for k in metrics if k.split('/')[1].startswith('gamma_')}
    assert namespaces == {'gamma_0.6', 'gamma_0.95'}
    names = ['proxy_mse_gigpo', 'proxy_mse_relace', 'proxy_mse_gap', 'correction_variance',
             'noise_correction_covariance', 'proxy_gap_from_covariance', 'relative_mse_reduction_pct',
             'group_mean_proxy_gap', 'fraction_groups_negative_gap', 'identity_max_abs_error']
    for gamma in ('0.6', '0.95'):
        assert all(f'{cd.METRIC_PREFIX}/gamma_{gamma}/{name}' in metrics for name in names)
        assert metrics[f'{cd.METRIC_PREFIX}/gamma_{gamma}/identity_max_abs_error'] < 1e-10
    # Same trajectories and rho, a different discount: the numbers differ, the coverage is shared.
    assert metrics[GAMMA_KEY + 'proxy_mse_gigpo'] != metrics[f'{cd.METRIC_PREFIX}/gamma_0.95/proxy_mse_gigpo']
    for name in ['total_valid_steps', 'steps_with_valid_q_reference', 'steps_with_insufficient_reference',
                 'eligible_steps', 'eligible_step_fraction', 'eligible_groups', 'evaluated_group_folds',
                 'original_anchor_groups', 'singleton_groups', 'insufficient_reference_groups',
                 'ref_trajectories_mean', 'ref_trajectories_min', 'ref_trajectories_max',
                 'fraction_groups_fully_retained']:
        assert COVERAGE + name in metrics, name
    assert metrics[COVERAGE + 'total_valid_steps'] == len(responses)
    assert 0 < metrics[COVERAGE + 'eligible_step_fraction'] <= 1
    assert metrics[COVERAGE + 'ref_trajectories_min'] >= 2
    assert all(np.isfinite(v) for v in metrics.values())


def test_training_gamma_returns_match_the_components_convention():
    inputs, responses = relace_rollout()
    components = compute_relace_components(**inputs)
    exponent = components['episode_lengths'] - components['timesteps']
    g = np.power(GAMMA, exponent.astype(np.float64)) * components['terminal_returns'].double().numpy()
    np.testing.assert_allclose(g, components['discounted_returns'].double().numpy(), rtol=1e-6, atol=1e-7)


def test_end_to_end_agrees_with_array_level_pipeline():
    inputs, responses = relace_rollout(seed=2)
    components, metrics = run_diagnostics(inputs, responses, fold_seed=5)
    keys = cd.action_key_ids(responses, inputs['response_mask'])
    folds = cd.assign_folds(components['index'], components['traj_index'], seed=5)
    structure = cd.build_structure(components['index'], components['traj_index'], components['step_group_uids'],
                                   keys, folds, min_reference_trajectories=2)
    returns = components['discounted_returns'].double().numpy()
    stats = cd.group_stats(structure, components['rho'].double().numpy(), returns)
    expected = cd.aggregate_stats(stats)
    for name, value in expected.items():
        assert metrics[GAMMA_KEY + name] == pytest.approx(value, rel=1e-5, abs=1e-8), name


def test_diagnostics_do_not_mutate_training_tensors_or_advantages():
    inputs, responses = relace_rollout(seed=1)
    components = compute_relace_components(**inputs)
    advantages = components['advantages'].clone()
    snapshot = {k: v.clone() for k, v in inputs.items() if isinstance(v, torch.Tensor)}
    snapshot['responses'] = responses.clone()
    component_tensors = {k: v.clone() for k, v in components.items() if isinstance(v, torch.Tensor)}
    component_arrays = {k: v.copy() for k, v in components.items() if isinstance(v, np.ndarray)}

    cd.compute_credit_diagnostics(components=components, responses=responses, response_mask=inputs['response_mask'],
                                  gamma=GAMMA, cfg=cd.CreditDiagConfig(enable=True, extra_gammas=(0.95,)))

    torch.testing.assert_close(components['advantages'], advantages, rtol=0, atol=0)
    for name, before in snapshot.items():
        after = responses if name == 'responses' else inputs[name]
        assert torch.equal(after, before), name
    for name, before in component_tensors.items():
        assert torch.equal(components[name], before), name
    for name, before in component_arrays.items():
        assert np.array_equal(components[name], before), name


def test_training_advantages_do_not_depend_on_the_diagnostics_path():
    # The trainer takes advantages from `compute_relace_components` (so the diagnostics can reuse its anchor groups).
    # That must be exactly what `compute_relace_advantage` has always returned.
    inputs, _ = relace_rollout(seed=4)
    via_components = compute_relace_components(**inputs)
    advantages, returns, metrics = compute_relace_advantage(**inputs)
    torch.testing.assert_close(via_components['advantages'], advantages, rtol=0, atol=0)
    torch.testing.assert_close(returns, advantages, rtol=0, atol=0)
    assert via_components['metrics'] == metrics


# --------------------------------------------------------------------------- #
# Configuration, interval, logging
# --------------------------------------------------------------------------- #
def test_config_defaults_validation_and_gamma_deduplication():
    cfg = cd.CreditDiagConfig.from_config(None)
    assert (cfg.enable, cfg.log_interval, cfg.min_reference_trajectories, cfg.extra_gammas) == (False, 5, 2, ())
    parsed = cd.CreditDiagConfig.from_config(
        {'enable': True, 'log_interval': 3, 'min_reference_trajectories': 3, 'extra_gammas': [0.6, 0.95], 'fold_seed': 7}
    )
    assert (parsed.enable, parsed.log_interval, parsed.min_reference_trajectories, parsed.fold_seed) == (True, 3, 3, 7)
    assert parsed.gammas(0.6) == [0.6, 0.95]  # no duplicate of the training gamma
    assert parsed.gammas(0.9) == [0.9, 0.6, 0.95]
    assert cd.CreditDiagConfig.from_config({'enable': True, 'extra_gammas': None}).extra_gammas == ()
    for bad in ({'log_interval': 0}, {'min_reference_trajectories': 0}, {'extra_gammas': [1.5]}, {'extra_gammas': [0.0]}):
        with pytest.raises(ValueError):
            cd.CreditDiagConfig.from_config(bad)


def test_diagnostics_are_logged_only_at_the_configured_interval():
    inputs, responses = relace_rollout()
    components = compute_relace_components(**inputs)

    class FakeLogger:  # the trainer's `Tracking.log(data=..., step=...)`
        def __init__(self):
            self.calls = []

        def log(self, data, step):
            self.calls.append((step, dict(data)))

    def train(cfg, steps):
        logger = FakeLogger()
        for step in steps:
            metrics = {'critic/score/mean': 1.0}  # what the trainer logs every update anyway
            if cd.is_due(cfg, step):
                metrics.update(cd.compute_credit_diagnostics(
                    components=components, responses=responses, response_mask=inputs['response_mask'],
                    gamma=GAMMA, cfg=cfg))
            logger.log(data=metrics, step=step)
        return [step for step, data in logger.calls if any(k.startswith(cd.METRIC_PREFIX + '/') for k in data)]

    assert train(cd.CreditDiagConfig(enable=True), range(1, 13)) == [5, 10]  # default interval 5
    assert train(cd.CreditDiagConfig(enable=True, log_interval=3), range(1, 11)) == [3, 6, 9]
    assert train(cd.CreditDiagConfig(enable=True), range(33, 42)) == [35, 40]  # a resumed run keeps the cadence
    assert train(cd.CreditDiagConfig(enable=False), range(1, 13)) == []
    assert train(None, range(1, 13)) == []
    assert not cd.is_due(cd.CreditDiagConfig(enable=True), None)  # e.g. trajectory analysis passes no step


def test_failures_are_contained_and_reported(capsys):
    inputs, responses = relace_rollout()
    components = compute_relace_components(**inputs)
    components = {k: v for k, v in components.items() if k != 'rho'}  # sabotage
    metrics = cd.compute_credit_diagnostics_safe(
        components=components, responses=responses, response_mask=inputs['response_mask'], gamma=GAMMA,
        cfg=cd.CreditDiagConfig(enable=True))
    assert metrics == {f'{cd.METRIC_PREFIX}/failed': 1.0}
    assert 'training is unaffected' in capsys.readouterr().out
