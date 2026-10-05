import numpy as np
import pytest
import torch

from relace.core_relace import compute_relace_advantage, grpo_macro_advantage


@pytest.fixture
def batch():
    return dict(
        token_level_rewards=torch.tensor([[1., 0.], [1., 0.], [3., 0.], [3., 0.]]),
        response_mask=torch.tensor([[1., 1.], [1., 0.], [1., 1.], [1., 0.]]),
        hindsight_log_probs=torch.tensor([[-.1, -.3], [-2., 0.], [-.4, -.5], [-1., 0.]]),
        action_log_probs=torch.full((4, 2), -.8),
        anchor_obs=np.array(['a', 'b', 'a', 'b']),
        index=np.array(['task'] * 4),
        traj_index=np.array(['a', 'a', 'b', 'b']),
        timestep=np.array([1, 2, 1, 2]),
        episode_lengths=np.array([2] * 4),
        enable_similarity=False,
    )


@pytest.mark.parametrize('mode', ['mean_norm', 'mean_std_norm'])
def test_zero_local_weight_matches_episode_advantage(batch, mode):
    actual, returns, _ = compute_relace_advantage(**batch, omega=0, mode=mode)
    expected, _ = grpo_macro_advantage(
        token_level_rewards=batch['token_level_rewards'], response_mask=batch['response_mask'],
        index=batch['index'], traj_index=batch['traj_index'],
        compute_mean_std_cross_steps=(mode == 'mean_std_norm'),
    )
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(returns, expected)


def test_padding_tokens_do_not_change_advantages(batch):
    expected, _, _ = compute_relace_advantage(**batch)
    mask = batch['response_mask'] == 0
    batch['hindsight_log_probs'][mask] = -1000
    batch['action_log_probs'][mask] = 1000
    actual, _, _ = compute_relace_advantage(**batch)
    torch.testing.assert_close(actual, expected)
    assert torch.count_nonzero(actual[mask]) == 0


def test_success_protection_preserves_macro_floor(batch):
    macro, _, _ = compute_relace_advantage(**batch, omega=0)
    protected, _, metrics = compute_relace_advantage(**batch, success_mask=np.ones(4), no_harm=True)
    assert torch.all(protected >= macro)
    assert metrics['relace/do_no_harm_success_frac'] == 1


@pytest.mark.parametrize('smooth', [False, True])
def test_retrospective_weights_are_clipped_and_finite(batch, smooth):
    advantages, _, metrics = compute_relace_advantage(**batch, clip_min=.8, clip_max=1.2, smooth=smooth, alpha=.5)
    assert torch.isfinite(advantages).all()
    assert metrics['relace/rho_min'] >= .8 - 1e-6
    assert metrics['relace/rho_max'] <= 1.2 + 1e-6
    assert all(key.startswith('relace/') for key in metrics)


def test_invalid_clip_interval_is_rejected(batch):
    with pytest.raises(ValueError, match='clip_min'):
        compute_relace_advantage(**batch, clip_min=2, clip_max=1)
