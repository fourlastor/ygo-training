from ygo_training.rewards import OutcomeRewardModel, terminal_outcome_reward


def test_terminal_rewards_depend_only_on_outcome():
    assert terminal_outcome_reward(None, 0) == -0.5
    assert terminal_outcome_reward(1, 0) == -1.0
    assert terminal_outcome_reward(0, 0) == 1.0
    assert OutcomeRewardModel().terminal(0, 0) == 1.0
    assert OutcomeRewardModel().cutoff() == -1.0
