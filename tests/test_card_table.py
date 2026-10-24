"""What a model is told of a card beside its input, from checkpoint format 16
(experiments/2026-10-17-card-descriptions.md): the repository's table of
the cards' descriptions, a model that reads them or hides identities, the
checkpoints it leaves, and a run that hides a share of the cards in each
update and keeps which it trained on."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from tests.views import ACTIVATE, MONSTER, OTHER_MONSTER, PASS, batched, card, choice, named, view
from ygo_training.card_table import DESCRIPTIONS, CardTable
from ygo_training.checkpoints import (
    CARDS_FORMAT,
    FORMAT_VERSION,
    card_options,
    load,
    load_inference_model,
    save,
    saved_card_options,
)
from ygo_training.config import TrainerConfig
from ygo_training.environment import LIBRARY
from ygo_training.model import PolicyValueNet

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import card_embedding_check as texts
import export_checkpoint

TABLE = CardTable.read(DESCRIPTIONS)
# A Normal Monster, whose text is a story; and a card whose name is only
# treated as the first's, with an effect of its own.
HARPIE_LADY, HARPIE_LADY_1 = 76812113, 91932350
# No card has this code.
NO_CARD = 7
TOLD = [
    {"descriptions": True},
    {"hide_identities": 0.25},
    {"descriptions": True, "hide_identities": 0.25},
]
built = pytest.mark.skipif(
    not (ROOT / LIBRARY).exists(), reason="the environment is not built: cargo build --release"
)


def place(code: int) -> int:
    return int(torch.searchsorted(TABLE.codes, torch.tensor(code)))


def row(code: int) -> int:
    return int(TABLE.rows[place(code)])


def shown() -> dict:
    """A decision with two cards of the seat on the field, each named by an answer."""
    return view(
        cards=[card(MONSTER), card(OTHER_MONSTER, attacking=True)],
        choices=[
            choice(ACTIVATE, named(0)),
            choice(ACTIVATE, named(1, OTHER_MONSTER)),
            choice(PASS),
        ],
    )


def test_the_table_holds_every_code_of_the_database_in_order():
    assert TABLE.codes.tolist() == sorted(texts.database_cards())
    assert NO_CARD not in TABLE.codes.tolist()
    assert TABLE.width == 256
    assert not TABLE.descriptions[0].any()


def test_every_card_of_the_pool_with_an_effect_reads_its_own_text():
    cards = texts.database_cards()
    pool = texts.pool_codes()
    by_text: dict[str, int] = {}
    for code in pool:
        said = texts.said_of(cards[code])
        assert (row(code) > 0) == (said is not None), cards[code][3]
        if said is not None:
            # Cards whose texts read the same share their row.
            assert by_text.setdefault(said, row(code)) == row(code)
    # And a row is of one text.
    assert len(set(by_text.values())) == len(by_text) == len(TABLE.descriptions) - 1
    assert row(HARPIE_LADY) == 0
    assert row(HARPIE_LADY_1) > 0
    # A card outside the pool has a row only as an alternate artwork of one.
    rows = dict(zip(TABLE.codes.tolist(), TABLE.rows.tolist(), strict=True))
    for code in set(cards) - set(pool):
        _, alias, _, name, _ = cards[code]
        alternate = alias in rows and cards[alias][3] == name
        assert rows[code] == (rows[alias] if alternate else 0), name


@built
def test_every_card_of_every_deck_with_an_effect_has_a_description():
    from ygo_training.environment import Environment

    cards = texts.database_cards()
    with Environment(ROOT) as environment:
        decks = environment.decks
    for name, deck in decks.items():
        for code in {*deck["main"], *deck["extra"]}:
            assert (row(code) > 0) == (texts.said_of(cards[code]) is not None), (name, code)


@pytest.mark.parametrize("told", TOLD)
def test_a_model_told_more_of_a_card_begins_as_one_that_is_not(told):
    torch.manual_seed(3)
    plain = PolicyValueNet().state_dict()
    torch.manual_seed(3)
    more = PolicyValueNet(cards=TABLE, **told).state_dict()
    for name, value in plain.items():
        assert torch.equal(more[name], value), name


def test_a_card_s_description_is_added_to_its_identity():
    torch.manual_seed(3)
    model = PolicyValueNet(cards=TABLE, descriptions=True).eval()
    codes = torch.tensor([[HARPIE_LADY_1, HARPIE_LADY, 0, NO_CARD]])
    with torch.no_grad():
        read = model._identity(codes)
        identity = model.card_identity(codes)
        added = model.description(TABLE.descriptions[row(HARPIE_LADY_1)])
    assert torch.allclose(read[0, 0], identity[0, 0] + added, atol=1e-6)
    # Nothing is added for a Normal Monster, a card not seen, a code not in the table.
    assert torch.equal(read[0, 1:], identity[0, 1:])


def test_hidden_identities_are_read_as_one_and_at_play_those_never_trained_on():
    torch.manual_seed(3)
    model = PolicyValueNet(cards=TABLE, hide_identities=0.25).eval()
    with torch.no_grad():
        model.hidden_identity.copy_(torch.arange(96.0))
    blank = model.hidden_identity.detach()
    codes = torch.tensor([MONSTER, OTHER_MONSTER, 0, NO_CARD])
    identity = model.card_identity(codes).detach()
    hidden = torch.zeros(len(TABLE.codes), dtype=torch.bool)
    hidden[place(MONSTER)] = True
    model.hide(hidden)
    with torch.no_grad():
        read = model._identity(codes)
    assert torch.equal(read[0], blank)
    # In training, what is not drawn is shown, a code not in the table too.
    assert torch.equal(read[1:], identity[1:])
    trained = torch.zeros(len(TABLE.codes), dtype=torch.bool)
    trained[place(OTHER_MONSTER)] = True
    model.trained_on(trained)
    model.play()
    with torch.no_grad():
        read = model._identity(codes)
    assert torch.equal(read[0], blank)
    assert torch.equal(read[1], identity[1])
    # A card the seat does not see is as ever; a code not in the table was
    # never trained on.
    assert torch.equal(read[2], identity[2])
    assert torch.equal(read[3], blank)


def test_the_answers_change_where_a_card_named_is_hidden():
    torch.manual_seed(3)
    model = PolicyValueNet(cards=TABLE, hide_identities=0.25).eval()
    with torch.no_grad():
        model.hidden_identity.copy_(torch.ones(96))
        before, _ = model(batched(shown()))
        hidden = torch.zeros(len(TABLE.codes), dtype=torch.bool)
        hidden[place(OTHER_MONSTER)] = True
        model.hide(hidden)
        after, _ = model(batched(shown()))
    assert not torch.equal(before, after)


def test_a_model_told_more_of_a_card_is_answered_at_fixed_sizes_as_itself():
    # A call filled out to the sizes of the graphs holds more cards, of the
    # code 0: neither described nor hidden.
    from ygo_training.graphs import WIDTHS, Graphs

    torch.manual_seed(3)
    model = PolicyValueNet(cards=TABLE, descriptions=True, hide_identities=0.25).eval()
    hidden = torch.zeros(len(TABLE.codes), dtype=torch.bool)
    hidden[place(OTHER_MONSTER)] = True
    model.hide(hidden)
    batch = {name: value.numpy() for name, value in batched(shown(), view()).items()}
    graphs = Graphs(model, "cpu", largest=4)
    graphs.begin_update()
    made = graphs.answer(batch)
    with torch.no_grad():
        logits, values = model({name: torch.from_numpy(array) for name, array in batch.items()})
    assert made is not None
    answers = logits.shape[1]
    there = torch.isfinite(logits)
    assert made[0].shape == (2, WIDTHS["action"])
    assert torch.allclose(made[0][:, :answers][there], logits[there], atol=1e-4, rtol=0)
    assert torch.allclose(made[1], values, atol=1e-4, rtol=0)


@pytest.mark.parametrize("told", TOLD)
def test_a_checkpoint_of_a_model_told_more_loads_to_play_as_it_was_trained(tmp_path, told):
    torch.manual_seed(3)
    model = PolicyValueNet(cards=TABLE, **told)
    if model.hides_identities:
        trained = torch.zeros(len(TABLE.codes), dtype=torch.bool)
        trained[place(MONSTER)] = True
        model.trained_on(trained)
    path = tmp_path / "model-00001.pt"
    save(path, model, torch.optim.Adam(model.parameters()), 1, {})
    payload = torch.load(path, weights_only=False)
    assert payload["format_version"] == CARDS_FORMAT == 16
    assert saved_card_options(payload["metadata"]) == card_options(model)
    assert json.loads(path.with_suffix(".pt.json").read_text())["format_version"] == 16
    loaded, _ = load_inference_model(path)
    model.play()
    model.eval()
    with torch.no_grad():
        for ours, theirs in zip(model(batched(shown())), loaded(batched(shown())), strict=True):
            assert torch.equal(ours, theirs)
    if model.hides_identities:
        assert torch.equal(loaded.card_hidden, ~loaded.card_trained)
        assert bool(loaded.card_trained[place(MONSTER)])
    assert torch.equal(CardTable.read(path).codes, TABLE.codes)


def test_a_model_is_built_as_its_checkpoint_says(tmp_path):
    plain = PolicyValueNet()
    path = tmp_path / "model-00001.pt"
    save(path, plain, torch.optim.Adam(plain.parameters()), 1, {})
    # Told nothing more of a card, a model's checkpoints are of format 15.
    assert torch.load(path, weights_only=False)["format_version"] == FORMAT_VERSION == 15
    loaded, _ = load_inference_model(path)
    assert not loaded.reads_descriptions and not loaded.hides_identities
    with pytest.raises(ValueError, match="never reads the cards' descriptions, this one does"):
        load(path, PolicyValueNet(cards=TABLE, descriptions=True))
    told = PolicyValueNet(cards=TABLE, hide_identities=0.25)
    other = tmp_path / "model-00002.pt"
    save(other, told, torch.optim.Adam(told.parameters()), 2, {})
    with pytest.raises(ValueError, match="hides identities, this one does not"):
        load(other, PolicyValueNet())


def test_a_model_told_more_of_a_card_is_not_exported(tmp_path):
    model = PolicyValueNet(cards=TABLE, descriptions=True)
    path = tmp_path / "model-00001.pt"
    save(path, model, torch.optim.Adam(model.parameters()), 1, {})
    with pytest.raises(ValueError, match="it is not exported"):
        export_checkpoint.export(path, tmp_path / "exported", tmp_path / "decisions.json")


def test_what_a_model_is_told_of_a_card_reads_a_table():
    with pytest.raises(ValueError, match="read a card_table"):
        TrainerConfig(card_descriptions=True)
    with pytest.raises(ValueError, match="read a card_table"):
        TrainerConfig(hide_identities=0.25)
    with pytest.raises(ValueError, match="for card_descriptions or hide_identities"):
        TrainerConfig(card_table=str(DESCRIPTIONS))
    with pytest.raises(ValueError, match="a share of the cards"):
        TrainerConfig(card_table=str(DESCRIPTIONS), hide_identities=1.5)
    with pytest.raises(ValueError, match="is given a card table"):
        PolicyValueNet(hide_identities=0.25)


@built
def test_an_update_hides_its_own_cards_and_keeps_those_shown(tmp_path):
    from ygo_training.ppo_trainer import PpoTrainer

    config = TrainerConfig(
        workers=2,
        duels=2,
        pool_threads=2,
        pool_slots=2,
        seed=45,
        device="cpu",
        opponent_decks=["monarch"],
        card_table=str(DESCRIPTIONS),
        hide_identities=0.25,
    )
    with PpoTrainer(ROOT, config) as trainer:
        told = trainer._hide(5)
        first = trainer.model.card_hidden.clone()
        assert told == {"cards_hidden": float(round(0.25 * len(TABLE.codes)))}
        assert int(first.sum()) == told["cards_hidden"]
        trainer._hide(6)
        assert not torch.equal(trainer.model.card_hidden, first)
        trainer._hide(5)
        assert torch.equal(trainer.model.card_hidden, first)
        # One card of the decision hidden, the other shown: the one shown
        # is kept among those trained on.
        hidden = torch.zeros(len(TABLE.codes), dtype=torch.bool)
        hidden[place(MONSTER)] = True
        trainer._hidden = hidden.numpy()
        trainer._shown({name: value.numpy() for name, value in batched(shown()).items()})
        assert trainer._trained is not None
        assert np.flatnonzero(trainer._trained).tolist() == [place(OTHER_MONSTER)]
        trainer.model.trained_on(torch.from_numpy(trainer._trained))
        # Played outside the update, then hiding the update's own again.
        with trainer._playing():
            assert torch.equal(trainer.model.card_hidden, ~trainer.model.card_trained)
        assert torch.equal(trainer.model.card_hidden, first)


@pytest.mark.integration
@built
def test_a_run_told_more_of_a_card_is_checkpointed_and_measured(tmp_path):
    run = tmp_path / "run"
    common = ["--device", "cpu", "--workers", "2", "--pool-threads", "2", "--pool-slots", "2"]
    subprocess.run(
        [
            sys.executable,
            "-m",
            "ygo_training",
            "train",
            *common,
            "--duels",
            "2",
            "--updates",
            "2",
            "--imitation-only-updates",
            "1",
            "--ppo-epochs",
            "1",
            "--minibatch",
            "64",
            "--seed",
            "45",
            "--opponents",
            "monarch",
            "--eval-every",
            "0",
            "--card-table",
            str(DESCRIPTIONS),
            "--card-descriptions",
            "--hide-identities",
            "0.25",
            "--run",
            str(run),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=300,
        check=True,
    )
    first, second = (
        json.loads((run / "checkpoints" / f"model-{version:05d}.pt.json").read_text())
        for version in (0, 1)
    )
    for described in (first, second):
        assert described["format_version"] == 16
        assert described["card_descriptions"] is True
        assert described["hide_identities"] == 0.25
        assert described["cards_hidden"] == round(0.25 * len(TABLE.codes))
    assert 0 < first["cards_trained"] <= second["cards_trained"] < 300
    suite = tmp_path / "suite.json"
    suite.write_text(
        json.dumps(
            {
                "name": "tiny",
                "primary": "blackwing-vs-monarch",
                "cases": [
                    {
                        "name": "blackwing-vs-monarch",
                        "learner": "blackwing",
                        "opponent": "monarch",
                        "games": 2,
                    }
                ],
            }
        )
    )
    measured = tmp_path / "measured.json"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "ygo_training",
            "evaluate-suite",
            *common,
            "--seed",
            "45",
            "--suite",
            str(suite),
            "--output",
            str(measured),
            str(run / "checkpoints" / "model-00001.pt"),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=300,
        check=True,
    )
    record = json.loads(measured.read_text())["records"][0]
    assert record["checkpoint_step"] == 2
    assert record["evaluation"]["games"] == 2
