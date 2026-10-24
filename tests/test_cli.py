import json
from argparse import Namespace
from typing import ClassVar

import pytest
import torch

from ygo_training.__main__ import (
    DEFAULT_LEARNER,
    _latest_checkpoint,
    _resume_path,
    _suite_path,
    _training_suite,
    measured_as_ever,
    run_defaults,
    train,
    trainer_config,
)
from ygo_training.checkpoints import save, write
from ygo_training.cli_parser import build_parser
from ygo_training.config import Calls, TrainerConfig
from ygo_training.model import PolicyValueNet
from ygo_training.run_repository import RunRepository
from ygo_training.schedules import decayed_teacher_imitation_weight


def parser():
    handlers = {
        name: lambda args: None
        for name in (
            "train",
            "status",
            "evaluate-suite",
            "evaluate-search",
            "edopro",
            "edopro-review",
            "shortlist",
        )
    }
    return build_parser(handlers=handlers, default_learner=DEFAULT_LEARNER)


def test_a_checkpoint_is_seated_in_a_room():
    args = parser().parse_args(["edopro", "model-00959.pt"])

    assert (args.learner, args.device) == ("blackwing", "cpu")
    assert (args.host, args.port, args.duels, args.wait) == ("127.0.0.1", 7911, 1, 60)
    assert (args.go_second, args.no_log, args.verbose, args.replay) == (False, False, False, None)
    assert str(args.log_dir) == "edopro-logs"
    # The other player's deck is no part of the model's input since format 15:
    # it is not said.
    with pytest.raises(SystemExit):
        parser().parse_args(["edopro", "model-00959.pt", "--opponent", "monarch"])


def test_a_kept_duel_is_reviewed_with_or_without_a_checkpoint():
    args = parser().parse_args(["edopro-review", "replay/_LastReplay.yrp"])

    assert str(args.replay) == "replay/_LastReplay.yrp"
    assert (args.checkpoint, args.player, args.decks, args.trace) == (None, None, None, None)
    assert (args.learner, args.device, str(args.output)) == ("blackwing", "cpu", "analysis/edopro")
    args = parser().parse_args(
        ["edopro-review", "saved.yrpX", "--checkpoint", "model-00959.pt", "--player", "ali"]
    )
    assert (str(args.checkpoint), args.player) == ("model-00959.pt", "ali")
    traced = parser().parse_args(["edopro-review", "saved.yrpX", "--trace", "logs/duel.trace"])
    assert str(traced.trace) == "logs/duel.trace"


def test_continue_selects_highest_numbered_checkpoint(tmp_path):
    checkpoints = tmp_path / "checkpoints"
    checkpoints.mkdir()
    (checkpoints / "model-00002.pt").touch()
    (checkpoints / "model-00011.pt").touch()
    (checkpoints / "not-a-model.pt").touch()

    args = Namespace(continue_run=True, resume=None)
    assert _resume_path(args, checkpoints) == checkpoints / "model-00011.pt"


def test_continue_requires_a_checkpoint(tmp_path):
    with pytest.raises(FileNotFoundError, match="No model-\\*\\.pt checkpoint"):
        _latest_checkpoint(tmp_path)


class _Trainer:
    """A stand-in for the trainer in `train`: an update leaves the model as
    it is and writes the checkpoint it is asked for, as the trainer does."""

    decks = ("blackwing", "monarch")
    recorded = None
    teacher_description = None
    held = None
    held_description = None
    critic_description = None
    card_head_description = None
    initial_model_fingerprint = "as built"
    device = "cpu"

    def __init__(self, root, config):
        self.opponents = ["monarch"]
        self.calls = config.calls(self.device)
        self.model = PolicyValueNet(
            embedding_dim=8, hidden_dim=16, board_representation=config.board_representation
        )
        self.optimizer = torch.optim.Adam(self.model.parameters())

    def begin_critic(self, beside=None):
        pass

    def begin_card_head(self, beside=None):
        pass

    def set_rollout_index(self, index):
        pass

    def set_training_schedule_index(self, index):
        pass

    # The duels the engine stops in the update of a step, as the pool reports them.
    stopped: ClassVar[dict[int, list[dict]]] = {}

    def train_update(self, checkpoint=None, step=1, *, finish=True, ahead=False):
        metrics = {"samples": 1.0}
        (save if finish else write)(checkpoint, self.model, self.optimizer, step, metrics)
        self._failed = self.stopped.get(step, [])
        return metrics

    def failed_duels(self):
        failed, self._failed = getattr(self, "_failed", []), []
        return failed

    def close(self, force=False):
        pass


def test_a_checkpoint_takes_its_name_once_its_update_is_evaluated(tmp_path, monkeypatch):
    suite = tmp_path / "suite.json"
    case = {"name": "mirror", "learner": "blackwing", "opponent": "monarch", "games": 2}
    suite.write_text(json.dumps({"cases": [case]}))
    run = RunRepository(tmp_path / "run")
    # The checkpoints there each time an update is evaluated; the run is
    # stopped in its second evaluation.
    there = []

    def evaluation(trainer, suite, *, seed):
        there.append(sorted(path.name for path in run.checkpoint_dir.glob("model-*.pt")))
        if len(there) == 2:
            raise KeyboardInterrupt
        return {"mirror": {"games": 2, "wins": 1}}

    monkeypatch.setattr("ygo_training.__main__.PpoTrainer", _Trainer)
    monkeypatch.setattr("ygo_training.__main__._run_evaluation_suite", evaluation)

    def trained(*arguments):
        """The run after the command: its files and the rows of its metrics."""
        train(
            parser().parse_args(
                ["train", "--root", str(tmp_path), "--run", str(run.run_dir)]
                + ["--eval-every", "1", "--eval-suite", str(suite), *arguments]
            )
        )
        rows = run.metrics_path.read_text().splitlines()
        return sorted(path.name for path in run.checkpoint_dir.iterdir()), [
            (row["version"], "evaluation" in row) for row in map(json.loads, rows)
        ]

    # Stopped while its second update is evaluated, the run holds what the
    # first left: the second's checkpoint had no name yet, and has none.
    with pytest.raises(KeyboardInterrupt):
        trained("--updates", "2")
    assert there == [[], ["model-00000.pt"]]
    first = ["model-00000.pt", "model-00000.pt.json"]
    assert sorted(path.name for path in run.checkpoint_dir.iterdir()) == first
    # The run says how its model is asked in the games, and goes on asked
    # another way: it then says that one.
    asked = {"pool_groups": 1, "pool_groups_over": 0, "inference_graphs": 0}
    assert json.loads(run.config_path.read_text())["model_calls"] == asked
    # Going on, it makes that update again and evaluates it: every checkpoint
    # has its row, and every row its evaluation.
    assert trained("--continue", "--updates", "2", "--inference-graphs", "8") == (
        [*first, "model-00001.pt", "model-00001.pt.json", "model-00002.pt", "model-00002.pt.json"],
        [(0, True), (1, True), (2, True)],
    )
    assert json.loads(run.config_path.read_text())["model_calls"] == {**asked, "inference_graphs": 8}
    assert there[2:] == [["model-00000.pt"], ["model-00000.pt", "model-00001.pt"]]


def test_teacher_imitation_decay_is_relative_to_the_branch_schedule():
    # ``--resume`` leaves the new trainer's schedule at zero; ``--continue``
    # explicitly restores the old schedule index. The helper's values make
    # that intended distinction concrete without spawning OCGCore workers.
    assert decayed_teacher_imitation_weight(0.02, 25, 0) == pytest.approx(0.02)
    assert decayed_teacher_imitation_weight(0.02, 25, 25) == 0.0


def test_cli_parser_keeps_public_training_defaults():
    args = parser().parse_args(["train"])

    assert args.workers == 16
    assert args.duels == 512
    assert (args.pool_threads, args.pool_slots) == (4, 16)
    assert args.ppo_epochs == 1
    # Left to the run (`run_defaults`).
    assert args.minibatch is None
    assert args.duels_in_flight is None
    assert args.learning_rate == pytest.approx(3e-4)
    assert args.learner == "blackwing"
    assert args.opponents is None
    assert args.eval_every == 5
    assert args.eval_suite is None
    assert args.duel_gae_lambda == pytest.approx(0.95)
    assert args.imitation_only_updates == 0
    assert args.teacher_imitation_weight == 0.0
    assert args.board_representation == "mean-pool"
    assert args.continue_run is False
    assert args.entropy_coefficient == pytest.approx(0.01)
    config = trainer_config(args)
    assert (config.duels_in_flight, config.minibatch, config.pool_slots) == (0, 256, 16)


def test_a_run_on_a_card_keeps_the_duels_in_flight_of_fast():
    # The run `fast` of 2026-10-23: twice an update's duels in flight, in
    # minibatches of 512, its calls answered as they are in two groups, the
    # pool's slots raised to hold them.
    line = ["train", "--device", "cuda", "--duels", "520", "--eval-every", "0"]
    args = parser().parse_args([*line, "--pool-threads", "20"])
    run_defaults(args)
    assert (args.duels_in_flight, args.minibatch) == (1040, 512)
    assert (args.pool_slots, args.inference_graphs, args.pool_groups) == (52, 0, 2)
    # Slots that hold them already are kept, an odd share rounded up for two groups.
    args = parser().parse_args([*line, "--pool-threads", "20", "--pool-slots", "64"])
    run_defaults(args)
    assert args.pool_slots == 64
    args = parser().parse_args([*line, "--pool-threads", "24"])
    run_defaults(args)
    assert args.pool_slots == 44
    # What the command line names stands.
    args = parser().parse_args([*line, "--duels-in-flight", "0", "--minibatch", "128"])
    run_defaults(args)
    assert (args.duels_in_flight, args.minibatch, args.pool_slots) == (0, 128, 16)
    assert (args.inference_graphs, args.pool_groups) == (None, None)


@pytest.mark.parametrize(
    "option",
    [
        ["--eval-every", "5"],
        ["--critic"],
        ["--card-head"],
        ["--opponent-checkpoint-duels", "4"],
        ["--teacher-imitation-weight", "0.1"],
        ["--rollout-estimates", "recomputed"],
        ["--imitation-only-updates", "240"],
    ],
)
def test_a_run_whose_updates_cannot_keep_duels_in_flight_plays_them_to_their_end(option):
    line = ["train", "--device", "cuda", "--duels", "520", "--eval-every", "0", *option]
    args = parser().parse_args(line)
    run_defaults(args)
    assert (args.duels_in_flight, args.minibatch, args.pool_slots) == (0, 256, 16)


def test_a_run_on_a_processor_keeps_no_duels_in_flight():
    args = parser().parse_args(["train", "--duels", "520", "--eval-every", "0"])
    run_defaults(args)
    assert (args.duels_in_flight, args.minibatch) == (0, 256)


def test_a_run_that_goes_on_keeps_what_it_began_with():
    line = ["train", "--device", "cuda", "--duels", "520", "--eval-every", "0", "--continue"]
    # all71, begun before the default: no duels in flight, minibatches of 256.
    args = parser().parse_args(line)
    run_defaults(args, {"minibatch": 256, "duels": 520})
    assert (args.duels_in_flight, args.minibatch) == (0, 256)
    args = parser().parse_args([*line, "--pool-threads", "20", "--pool-slots", "52"])
    run_defaults(args, {"minibatch": 512, "duels_in_flight": 1040, "duels": 520})
    assert (args.duels_in_flight, args.minibatch, args.pool_slots) == (1040, 512, 52)
    assert (args.inference_graphs, args.pool_groups) == (0, 2)


def test_cli_exposes_both_board_representations():
    for representation in ("mean-pool", "individual-tokens"):
        args = parser().parse_args(["train", "--board-representation", representation])
        assert args.board_representation == representation
    with pytest.raises(SystemExit):
        parser().parse_args(["train", "--board-representation", "legacy-mean-pool"])


@pytest.mark.parametrize(
    "option",
    [
        ["--gae-unit", "causal-transaction"],
        ["--puzzle-training", "0"],
        ["--opponent-policy", "ygo-policies"],
        ["--deck-pool", "ygo-policies"],
        ["--attack-target-mode", "joint-staged"],
        ["--mlflow"],
    ],
)
def test_retired_options_are_refused(option):
    with pytest.raises(SystemExit):
        parser().parse_args(["train", *option])


def test_cli_takes_the_value_heads_options_and_the_checkpoints_kept():
    args = parser().parse_args(["train"])
    assert (args.imitation_value_positions, args.imitation_value_detach) == (0, False)
    assert (args.ppo_value_positions, args.ppo_value_target) == (0, "return")
    assert args.keep_checkpoints == 1
    args = parser().parse_args(
        [
            "train",
            "--imitation-value-positions",
            "4",
            "--imitation-value-detach",
            "--ppo-value-positions",
            "3",
            "--ppo-value-target",
            "result",
            "--keep-checkpoints",
            "40",
        ]
    )
    assert (args.imitation_value_positions, args.imitation_value_detach) == (4, True)
    assert (args.ppo_value_positions, args.ppo_value_target) == (3, "result")
    with pytest.raises(SystemExit):
        parser().parse_args(["train", "--ppo-value-target", "anything"])
    assert args.keep_checkpoints == 40


def test_cli_resume_modes_remain_mutually_exclusive():
    with pytest.raises(SystemExit):
        parser().parse_args(["train", "--resume", "model.pt", "--continue"])


def test_status_views():
    assert parser().parse_args(["status", "metrics.jsonl", "--evaluation-suite"]).evaluation_suite
    assert parser().parse_args(["status", "metrics.jsonl", "--ppo"]).optimization
    with pytest.raises(SystemExit):
        parser().parse_args(["status", "metrics.jsonl", "--ppo", "--evaluation-suite"])


def test_evaluating_checkpoints_needs_a_suite(tmp_path):
    with pytest.raises(SystemExit):
        parser().parse_args(["evaluate-suite", "model.pt"])
    args = parser().parse_args(["evaluate-suite", "model.pt", "--suite", "suite.json"])
    assert args.suite.name == "suite.json"


def test_training_evaluates_on_the_learners_screen_unless_told_otherwise(tmp_path):
    assert _suite_path(tmp_path, None, "monarch") == (
        tmp_path / "eval_suites" / "ygo-policies-monarch-screen.json"
    )
    assert _suite_path(tmp_path, tmp_path / "mine.json", "monarch") == tmp_path / "mine.json"


def test_a_learner_of_several_decks_is_named_on_the_command_line():
    args = parser().parse_args(
        [
            "train",
            "--learner",
            "blackwing, monarch",
            "--opponents",
            "burn,heroes",
            "--pool-threads",
            "2",
        ]
    )
    config = trainer_config(args)
    assert (config.learner_deck, config.learner_decks) == ("blackwing", ["blackwing", "monarch"])
    assert config.learners == ["blackwing", "monarch"]
    one = trainer_config(parser().parse_args(["train", "--learner", "monarch"]))
    assert (one.learner_deck, one.learner_decks, one.learners) == ("monarch", None, ["monarch"])
    with pytest.raises(ValueError, match="require opponent_decks"):
        trainer_config(parser().parse_args(["train", "--learner", "blackwing,monarch"]))


def test_a_learner_of_several_decks_is_evaluated_on_the_screen_of_each(tmp_path):
    """Each deck's games are the ones its own screen deals: a network of
    several decks is set beside a network of one on the same deals."""
    folder = tmp_path / "eval_suites"
    folder.mkdir()
    for deck in ("blackwing", "monarch"):
        screen = {
            "name": f"{deck}-screen",
            "primary": f"{deck}-vs-burn",
            "cases": [
                {"name": f"{deck}-vs-{other}", "learner": deck, "opponent": other, "games": 4}
                for other in ("burn", "heroes")
            ],
        }
        (folder / f"ygo-policies-{deck}-screen.json").write_text(json.dumps(screen))

    alone, path = _training_suite(tmp_path, None, ["monarch"])
    assert path == str(folder / "ygo-policies-monarch-screen.json")
    assert all("seed_index" not in case for case in alone["cases"])
    both, paths = _training_suite(tmp_path, None, ["blackwing", "monarch"])
    assert paths == ", ".join(
        str(folder / f"ygo-policies-{deck}-screen.json") for deck in ("blackwing", "monarch")
    )
    assert (both["name"], both["primary"]) == (
        "blackwing-screen+monarch-screen",
        "blackwing-vs-burn",
    )
    assert [(case["name"], case["seed_index"]) for case in both["cases"]] == [
        ("blackwing-vs-burn", 0),
        ("blackwing-vs-heroes", 1),
        ("monarch-vs-burn", 0),
        ("monarch-vs-heroes", 1),
    ]
    # A suite that is asked for is the one played, whatever the decks.
    asked, where = _training_suite(
        tmp_path, folder / "ygo-policies-monarch-screen.json", ["blackwing", "monarch"]
    )
    assert asked == alone and where == path


def test_train_entropy_coefficient_defaults_to_the_trusted_value():
    base = ["train", "--run", "runs/x", "--learner", "blackwing"]
    assert parser().parse_args(base).entropy_coefficient == 0.01
    assert parser().parse_args([*base, "--entropy-coefficient", "0.001"]).entropy_coefficient == 0.001


def test_training_seed_defaults_to_the_evaluation_seed():
    assert parser().parse_args(["train"]).training_seed is None
    args = parser().parse_args(["train", "--seed", "2012", "--training-seed", "2013"])
    assert (args.seed, args.training_seed) == (2012, 2013)


def test_trainer_config_follows_the_train_command_line(tmp_path):
    args = parser().parse_args(
        [
            "train",
            "--seed",
            "2012",
            "--training-seed",
            "2013",
            "--learner",
            "monarch",
            "--opponents",
            "burn, crystal,",
            "--learning-rate",
            "3e-4",
            "--resume",
            str(tmp_path / "model.pt"),
            "--resume-learning-rate",
            "1e-4",
        ]
    )
    config = trainer_config(args)
    assert config.seed == 2013
    assert config.learner_deck == "monarch"
    assert config.opponent_decks == ["burn", "crystal"]
    assert config.learning_rate == 1e-4
    assert config.policy_loss_weight == 1.0
    assert trainer_config(parser().parse_args(["train"])).opponent_decks is None
    # How the model is asked in the games is left to the device unless named.
    pooled = ["train", "--pool-threads", "20"]
    assert trainer_config(parser().parse_args(pooled)).calls("cuda") == Calls(2, 64, 160)
    assert trainer_config(parser().parse_args(pooled)).calls("cpu") == Calls()
    named = parser().parse_args([*pooled, "--inference-graphs", "0"])
    assert trainer_config(named).calls("cuda") == Calls()
    named = parser().parse_args([*pooled, "--pool-groups", "1", "--inference-graphs", "96"])
    assert trainer_config(named).calls("cuda") == Calls(graphs=96)
    # A measuring command asks as ever on a card too.
    for command in (
        ["evaluate-suite", "a.pt", "--suite", "suite.json"],
        ["evaluate-search", "a.pt", "--records", "rows.jsonl"],
    ):
        asked = measured_as_ever(parser().parse_args([*command, "--pool-threads", "20"]))
        assert TrainerConfig(pool_threads=20, **asked).calls("cuda") == Calls()
    dagger = parser().parse_args(
        ["train", "--teacher-imitation-weight", "1", "--policy-loss-weight", "0"]
    )
    assert trainer_config(dagger).policy_loss_weight == 0.0
    assert trainer_config(dagger).teacher_label_fraction == 1.0
    thinned = parser().parse_args(
        ["train", "--teacher-imitation-weight", "1", "--teacher-label-fraction", "0.1"]
    )
    assert trainer_config(thinned).teacher_label_fraction == 0.1
    imitation = parser().parse_args(
        ["train", "--imitation-only-updates", "10", "--imitation-value-weight", "0.5"]
    )
    assert trainer_config(imitation).imitation_only_updates == 10
    assert trainer_config(imitation).imitation_value_weight == 0.5


def test_recorded_games_reach_the_trainer_config():
    args = parser().parse_args(
        [
            "train",
            "--imitation-only-updates",
            "4",
            "--imitation-records",
            "rows.jsonl",
            "--imitation-game",
            "baseline",
            "--imitation-games",
            "1000",
        ]
    )
    config = trainer_config(args)
    assert (config.imitation_records, config.imitation_game, config.imitation_games) == (
        "rows.jsonl",
        "baseline",
        1000,
    )
    plain = trainer_config(parser().parse_args(["train"]))
    assert (plain.imitation_records, plain.imitation_game) == (None, "search")
    with pytest.raises(SystemExit):
        parser().parse_args(["train", "--imitation-game", "pilot"])


def test_a_checkpoint_to_imitate_reaches_the_trainer_config():
    args = parser().parse_args(
        [
            "train",
            "--imitation-only-updates",
            "4",
            "--pool-threads",
            "2",
            "--imitation-teacher",
            "runs/specialist/model.pt",
            "--imitation-teacher-targets",
            "distribution",
        ]
    )
    config = trainer_config(args)
    assert (config.imitation_teacher, config.imitation_teacher_targets) == (
        "runs/specialist/model.pt",
        "distribution",
    )
    plain = trainer_config(parser().parse_args(["train"]))
    assert (plain.imitation_teacher, plain.imitation_teacher_targets) == (None, "choice")
    with pytest.raises(SystemExit):
        parser().parse_args(["train", "--imitation-teacher-targets", "logits"])


def test_a_checkpoint_for_each_deck_to_imitate_reaches_the_trainer_config():
    imitating = ["train", "--imitation-only-updates", "4", "--pool-threads", "2"]
    several = [*imitating, "--learner", "blackwing,cyber", "--opponents", "burn"]
    named = ["--imitation-teacher", "blackwing=runs/a/model.pt"]
    named += ["--imitation-teacher", "cyber=runs/b/model.pt"]
    config = trainer_config(parser().parse_args([*several, *named]))
    assert (config.imitation_teacher, config.imitation_teachers) == (
        None,
        {"blackwing": "runs/a/model.pt", "cyber": "runs/b/model.pt"},
    )
    for wrong in (
        ["--imitation-teacher", "a.pt", "--imitation-teacher", "b.pt"],
        ["--imitation-teacher", "blackwing=a.pt", "--imitation-teacher", "b.pt"],
        ["--imitation-teacher", "blackwing=a.pt", "--imitation-teacher", "blackwing=b.pt"],
    ):
        with pytest.raises(SystemExit):
            trainer_config(parser().parse_args([*several, *wrong]))


def test_measuring_on_a_search_run_needs_its_rows_and_a_checkpoint():
    with pytest.raises(SystemExit):
        parser().parse_args(["evaluate-search", "model.pt"])
    with pytest.raises(SystemExit):
        parser().parse_args(["evaluate-search", "--records", "rows.jsonl"])
    args = parser().parse_args(["evaluate-search", "a.pt", "b.pt", "--records", "rows.jsonl"])
    assert [path.name for path in args.checkpoints] == ["a.pt", "b.pt"]
    assert args.records.name == "rows.jsonl" and not args.allow_seen_deals


def test_a_network_in_the_other_seat_reaches_the_trainer_config():
    args = parser().parse_args(
        [
            "train",
            "--duels",
            "520",
            "--pool-threads",
            "2",
            "--self-play-duels",
            "260",
            "--opponent-checkpoint",
            "runs/long/model.pt",
            "--opponent-checkpoint-duels",
            "52",
        ]
    )
    config = trainer_config(args)
    assert (
        config.self_play_duels,
        config.opponent_checkpoint,
        config.opponent_checkpoint_duels,
    ) == (260, "runs/long/model.pt", 52)
    assert config.self_play_pairs == "all"
    mirrored = parser().parse_args(
        ["train", "--duels", "520", "--pool-threads", "2", "--self-play-duels", "260"]
        + ["--self-play-pairs", "mirrors"]
    )
    assert trainer_config(mirrored).self_play_pairs == "mirrors"
    plain = trainer_config(parser().parse_args(["train"]))
    assert (plain.self_play_duels, plain.opponent_checkpoint, plain.opponent_checkpoint_duels) == (
        0,
        None,
        0,
    )
    # A suite is measured against a held checkpoint too.
    measured = parser().parse_args(
        ["evaluate-suite", "a.pt", "--suite", "suite.json", "--opponent-checkpoint", "b.pt"]
    )
    assert str(measured.opponent_checkpoint) == "b.pt"
    assert (
        parser().parse_args(["evaluate-suite", "a.pt", "--suite", "s.json"]).opponent_checkpoint
        is None
    )


def _train(tmp_path, run, *arguments):
    """`train` with the stand-in trainer: the rows of the run's metrics."""
    train(
        parser().parse_args(
            ["train", "--root", str(tmp_path), "--run", str(run.run_dir), "--eval-every", "0"]
            + list(arguments)
        )
    )
    return [json.loads(row) for row in run.metrics_path.read_text().splitlines()]


def test_a_duel_the_engine_stops_is_kept_with_the_run_and_so_is_its_checkpoint(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.setattr("ygo_training.__main__.PpoTrainer", _Trainer)
    report = {"duel": 495, "seed": 47410671, "rollout_seed": 47410176, "failure": "a script failed"}
    # The second update's games, played with the first's checkpoint, have it.
    monkeypatch.setattr(_Trainer, "stopped", {2: [report]})
    run = RunRepository(tmp_path / "run")
    rows = _train(tmp_path, run, "--updates", "3", "--keep-checkpoints", "10")
    assert [row["failed_duels"] for row in rows] == [0, 1, 0]
    kept = [
        json.loads(line) for line in (run.run_dir / "failed-duels.jsonl").read_text().splitlines()
    ]
    assert kept == [{"version": 1, "played_with": str(run.checkpoint(0)), **report}]
    assert "update 1: the engine stopped duel 495 (a script failed)" in capsys.readouterr().err
    # Thinned, the run keeps one checkpoint in ten and its latest, and the
    # one the failed duel was played with.
    assert sorted(path.name for path in run.checkpoint_dir.glob("*.pt")) == [
        "model-00000.pt",
        "model-00002.pt",
    ]


def test_a_run_goes_on_across_a_move_of_the_pins_only_when_told(tmp_path, monkeypatch):
    monkeypatch.setattr("ygo_training.__main__.PpoTrainer", _Trainer)
    pinned = {"commit": "82a7c61", "vendor/ocgcore": "0e7286d"}
    monkeypatch.setattr("ygo_training.__main__.duel_revisions", lambda root: pinned)
    run = RunRepository(tmp_path / "run")
    _train(tmp_path, run, "--updates", "1")
    moved = {"commit": "a1b2c3d", "vendor/ocgcore": "8d14f6a"}
    monkeypatch.setattr("ygo_training.__main__.duel_revisions", lambda root: moved)
    with pytest.raises(ValueError, match="duel_revisions"):
        _train(tmp_path, run, "--continue", "--updates", "1")
    with pytest.raises(ValueError, match="--new-pins goes with --continue"):
        _train(tmp_path, RunRepository(tmp_path / "other"), "--new-pins", "--updates", "1")
    assert [
        row["version"]
        for row in _train(tmp_path, run, "--continue", "--new-pins", "--updates", "1")
    ] == [0, 1]
    described = json.loads(run.config_path.read_text())
    assert described["duel_revisions"] == moved
    assert described["pin_moves"] == [{"version": 1, "from": pinned, "to": moved}]
    # Told nothing more, it goes on at the pins it moved to, the move kept.
    _train(tmp_path, run, "--continue", "--updates", "1")
    assert json.loads(run.config_path.read_text())["pin_moves"] == described["pin_moves"]
