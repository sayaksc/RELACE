import warnings

import numpy as np
import pytest
import torch

from gigpo.core_gigpo import (
    apply_no_harm_mask,
    build_step_group,
    compute_gigpo_outcome_advantage,
    episode_norm_reward,
    smooth_step_values,
    step_norm_reward,
)
from relace.core_relace import _apply_do_no_harm_mask, _smooth_q_values

GAMMA = 0.5

# Three trajectories of one task group:
#   A fails            : x -> b                (rows 0, 1)
#   B succeeds, 2 steps: a -> b                (rows 2, 3)
#   C succeeds, 3 steps: a -> c -> b           (rows 4, 5, 6)
# Discounted returns G_t = sum_k gamma^k r_{t+k} with gamma = 0.5 and terminal reward 1:
#   A: [0, 0]   B: [.5, 1]   C: [.25, .5, 1]
# Anchor-state groups: {x}: row 0, {a}: rows 2, 4, {b}: rows 1, 3, 6, {c}: row 5.
STEP_REWARDS = [0.0, 0.0, 0.5, 1.0, 0.25, 0.5, 1.0]
SUCCESS = [0, 0, 1, 1, 1, 1, 1]


@pytest.fixture
def batch():
    rewards = torch.zeros(7, 2)
    rewards[2:, 0] = 1.0  # per-row episode reward, as the episode reward manager writes it
    return dict(
        token_level_rewards=rewards,
        step_rewards=torch.tensor(STEP_REWARDS),
        response_mask=torch.tensor([[1., 1.], [1., 0.], [1., 1.], [1., 0.], [1., 1.], [1., 1.], [1., 0.]]),
        anchor_obs=np.array(['x', 'b', 'a', 'b', 'a', 'c', 'b'], dtype=object),
        index=np.array(['task'] * 7, dtype=object),
        traj_index=np.array(['A', 'A', 'B', 'B', 'C', 'C', 'C'], dtype=object),
        timestep=np.array([1, 2, 1, 2, 1, 2, 3]),
        mode='mean_norm',
        enable_similarity=False,
    )


def step_term(batch, **kwargs):
    """Step-level part of the advantage: full advantage minus the episode-only advantage."""
    full, _ = compute_gigpo_outcome_advantage(**batch, step_advantage_w=1.0, **kwargs)
    episode, _ = compute_gigpo_outcome_advantage(**batch, step_advantage_w=0.0, **kwargs)
    return full - episode


def expected(rows, batch):
    return torch.tensor(rows).unsqueeze(-1).tile([1, 2]) * batch['response_mask']


def test_defaults_reproduce_original_gigpo(batch):
    actual, returns = compute_gigpo_outcome_advantage(**batch)
    groups = build_step_group(batch['anchor_obs'], batch['index'], False)
    reference = episode_norm_reward(
        batch['token_level_rewards'], batch['response_mask'], batch['index'], batch['traj_index'], 1e-6, True
    ) + step_norm_reward(batch['step_rewards'], batch['response_mask'], groups, 1e-6, True)
    torch.testing.assert_close(actual, reference)
    torch.testing.assert_close(returns, reference)
    # Hand-computed step term: {a}: B +.125, C -.125; {b}: A -2/3, B +1/3, C +1/3; singletons 0.
    torch.testing.assert_close(
        step_term(batch), expected([0, -2 / 3, .125, 1 / 3, -.125, 0, 1 / 3], batch)
    )


def test_two_tuple_return_unless_metrics_requested(batch):
    assert len(compute_gigpo_outcome_advantage(**batch)) == 2
    assert len(compute_gigpo_outcome_advantage(**batch, return_metrics=True)) == 3


def test_smoothing_interpolates_with_next_step(batch):
    # q' = alpha * q_t + (1 - alpha) * q_{t+1}: A [0, 0], B [.75, 1], C [.375, .75, 1].
    rows = [0, -2 / 3, .1875, 1 / 3, -.1875, 0, 1 / 3]
    torch.testing.assert_close(step_term(batch, smooth=True, alpha=.5), expected(rows, batch))


def test_alpha_one_leaves_step_values_unchanged(batch):
    torch.testing.assert_close(step_term(batch, smooth=True, alpha=1.0), step_term(batch))


def test_smoothing_follows_timestep_not_row_order(batch):
    expected_adv, _ = compute_gigpo_outcome_advantage(**batch, smooth=True, alpha=.5, no_harm=True, success_mask=np.array(SUCCESS))
    perm = np.array([6, 3, 0, 5, 2, 4, 1])
    shuffled = dict(batch)
    for key in ('token_level_rewards', 'step_rewards', 'response_mask'):
        shuffled[key] = batch[key][torch.as_tensor(perm)]
    for key in ('anchor_obs', 'index', 'traj_index', 'timestep'):
        shuffled[key] = batch[key][perm]
    actual, _ = compute_gigpo_outcome_advantage(
        **shuffled, smooth=True, alpha=.5, no_harm=True, success_mask=np.array(SUCCESS)[perm]
    )
    torch.testing.assert_close(actual, expected_adv[torch.as_tensor(perm)])


def test_no_harm_zeroes_only_negative_step_advantages_of_successful_rows(batch):
    # Row 4 (successful, -.125) is masked; row 1 (failed, -2/3) is not; positive rows are kept.
    rows = [0, -2 / 3, .125, 1 / 3, 0, 0, 1 / 3]
    torch.testing.assert_close(
        step_term(batch, no_harm=True, success_mask=np.array(SUCCESS)), expected(rows, batch)
    )


def test_no_harm_never_pushes_successful_actions_below_the_episode_term(batch):
    episode, _ = compute_gigpo_outcome_advantage(**batch, step_advantage_w=0.0)
    protected, _ = compute_gigpo_outcome_advantage(**batch, no_harm=True, success_mask=np.array(SUCCESS))
    unprotected, _ = compute_gigpo_outcome_advantage(**batch)
    success = torch.tensor(SUCCESS, dtype=torch.bool)
    assert torch.all(protected[success] >= episode[success])
    assert (unprotected[success] < episode[success]).any()  # the mask is doing something here
    torch.testing.assert_close(protected[~success], unprotected[~success])


def test_no_harm_with_smoothing(batch):
    rows = [0, -2 / 3, .1875, 1 / 3, 0, 0, 1 / 3]
    torch.testing.assert_close(
        step_term(batch, smooth=True, alpha=.5, no_harm=True, success_mask=np.array(SUCCESS)),
        expected(rows, batch),
    )


def test_no_harm_metrics(batch):
    _, _, metrics = compute_gigpo_outcome_advantage(**batch, no_harm=True, success_mask=np.array(SUCCESS), return_metrics=True)
    assert metrics['gigpo/do_no_harm_success_frac'] == pytest.approx(5 / 7)
    assert metrics['gigpo/do_no_harm_suppressed_frac'] == pytest.approx(1 / 7)
    _, _, off = compute_gigpo_outcome_advantage(**batch, return_metrics=True)
    assert off == {'gigpo/do_no_harm_success_frac': 0.0, 'gigpo/do_no_harm_suppressed_frac': 0.0}


def test_success_mask_accepts_object_arrays(batch):
    # The rollout collator hands episode_success over as an object array of floats.
    actual, _ = compute_gigpo_outcome_advantage(**batch, no_harm=True, success_mask=np.array(SUCCESS, dtype=object).astype(float).astype(object))
    expected_adv, _ = compute_gigpo_outcome_advantage(**batch, no_harm=True, success_mask=np.array(SUCCESS))
    torch.testing.assert_close(actual, expected_adv)


def test_no_harm_without_success_mask_warns_and_is_a_no_op(batch):
    baseline, _ = compute_gigpo_outcome_advantage(**batch)
    with pytest.warns(UserWarning, match='no_harm'):
        actual, _ = compute_gigpo_outcome_advantage(**batch, no_harm=True)
    torch.testing.assert_close(actual, baseline)


def test_mask_and_smoothing_are_inert_when_disabled(batch):
    baseline, _ = compute_gigpo_outcome_advantage(**batch)
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        actual, _ = compute_gigpo_outcome_advantage(
            **batch, success_mask=np.array(SUCCESS), alpha=.1, smooth=False, no_harm=False
        )
    torch.testing.assert_close(actual, baseline)


@pytest.mark.parametrize('alpha', [-0.1, 1.1])
def test_invalid_alpha_is_rejected(batch, alpha):
    with pytest.raises(ValueError, match='alpha'):
        compute_gigpo_outcome_advantage(**batch, smooth=True, alpha=alpha)


def test_success_mask_length_is_validated():
    with pytest.raises(ValueError, match='success_mask'):
        apply_no_harm_mask(torch.zeros(3), np.ones(2))


def test_helpers_match_relace_implementations():
    # GiGPO's helpers are copies of RELACE's so the two estimators stay comparable.
    g = torch.Generator().manual_seed(0)
    values = torch.randn(12, generator=g)
    traj = np.array(list('aaabbbbccccc'), dtype=object)
    timesteps = np.array([1, 2, 3, 1, 2, 3, 4, 1, 2, 3, 4, 5])
    for alpha in (0.0, 0.3, 0.5, 1.0):
        torch.testing.assert_close(
            smooth_step_values(values, traj, timesteps, alpha), _smooth_q_values(values, traj, timesteps, alpha)
        )
    success = (torch.randn(12, generator=g) > 0).float()
    ours = apply_no_harm_mask(values, success)
    theirs = _apply_do_no_harm_mask(values, success)
    torch.testing.assert_close(ours[0], theirs[0])
    assert ours[1:] == theirs[1:]
