"""What `tools/call_timing.py` tells of the model's calls, with made-up
figures: the classes a call's size falls in, what is kept of the games'
batches, and the report."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("call_timing", ROOT / "tools/call_timing.py")
assert SPEC is not None and SPEC.loader is not None
tool = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(tool)


def test_a_calls_size_falls_in_a_class():
    assert [tool.of_class(size) for size in (1, 2, 8, 9, 64, 65, 320, 5000)] == [
        1, 8, 8, 32, 64, 128, 512, 100_000,
    ]  # fmt: skip
    assert [tool.named(largest) for largest in tool.CLASSES] == [
        "1", "2 to 8", "9 to 32", "33 to 64", "65 to 128", "129 to 256", "257 to 512",
        "513 to 1024", "1025 to 2048", "2049 or more",
    ]  # fmt: skip


def test_the_first_batch_of_each_class_is_kept_as_the_pool_gave_it():
    kept = tool.Kept()
    for size, answers in ((3, 4), (5, 9), (40, 2), (3, 6)):
        kept(
            None,
            (
                {
                    "history_lengths": torch.zeros(size, dtype=torch.long),
                    "scalars": torch.ones(size, 2),
                    "action_member_card": torch.zeros(size, answers, size % 4, dtype=torch.long),
                },
            ),
        )
    assert kept.sizes == [3, 5, 40, 3]
    assert sorted(kept.batches) == [8, 64]
    assert kept.batches[8]["scalars"].shape == (3, 2) and kept.batches[64]["scalars"].dtype.name == "float32"
    # The widest every array was in any call, kept or not, its decisions aside.
    assert kept.widest == {"history_lengths": (), "scalars": (2,), "action_member_card": (9, 3)}


def test_a_part_is_timed_and_the_report_weighs_a_call_by_its_class():
    assert tool.timed("cpu", 3, lambda: sum(range(1000))) > 0
    args = argparse.Namespace(
        checkpoint=Path("a/checkpoint.pt"), device="cpu", itself=260, duels=520, pool_threads=20,
        pool_slots=16, repeats=200,
    )  # fmt: skip
    parts = {"seeds": 0.01, "card": 0.5, "answer": 1.5, "draw": 0.2, "probability": 0.1, "back": 0.05}
    found = {
        "calls": {
            8: {"size": 4, "whole": 2.0, **parts},
            512: {"size": 320, "whole": 8.0, **parts},
        },
        "apart": {
            4: {
                "encoders": [("_history", 1, 0.3), ("_card_representation", 6, 0.9)],
                "launched": {"called": 321.0, "operations": 2096.0, "on_card": 410.0, "most": [("aten::embedding", 62.0)]},
            },
            320: {"encoders": [("_history", 1, 0.7)], "launched": "RuntimeError: no profiler"},
        },
    }
    played = {"sizes": [4, 2, 320, 300, 50], "games_seconds": 9.0, "model_seconds": 5.0}
    told = tool.report(args, played, found)
    assert "260 of them against itself" in told[1] and "676 decisions in 5 calls" in told[2]
    lines = {line.split("  ")[0] or line.strip().split("  ")[0]: line for line in told}
    small = next(line for line in told if line.strip().startswith("2 to 8"))
    assert small.split() == ["2", "to", "8", "2", "6", "4", "2.00", "0.01", "0.50", "1.50", "0.20", "0.10", "0.05"]
    large = next(line for line in told if line.strip().startswith("257 to 512"))
    assert large.split()[:7] == ["257", "to", "512", "2", "620", "320", "8.00"]
    # Two calls of 2 ms and two of 8 ms; the call of 50 has no class that was timed.
    assert any("the calls of an update take 0.02 s" in line for line in told)
    apart = told.index("The answer taken apart, a call of 4 decisions (how often, ms in all):")
    assert told[apart + 1].split() == ["_card_representation", "6", "0.90"]
    assert "calls 321 operations of PyTorch, which make 2,096" in told[apart + 3]
    assert "launches 410 kernels and copies on the card" in told[apart + 3]
    assert told[apart + 4].strip() == "the most called: embedding 62"
    assert any("was not counted: RuntimeError: no profiler" in line for line in told)
    assert told[-1] == "On a processor the times say nothing of a card, and no graph is recorded."
    assert lines
    # Nothing of graphs where none was tried.
    assert not any("CUDA graph" in line for line in told)


def test_two_answers_are_the_same_numbers_or_so_far_apart():
    none = float("-inf")
    one = (torch.tensor([[0.5, none], [1.0, 2.0]]), torch.tensor([0.25, -0.5]))
    assert tool.same_numbers(one, tuple(made.clone() for made in one)) == (True, 0.0)
    other = (torch.tensor([[0.5, none], [1.0, 2.5]]), torch.tensor([0.25, -0.5]))
    assert tool.same_numbers(one, other) == (False, 0.5)
    # An answer that is none on one side only is another answer, with nothing to measure.
    assert tool.same_numbers((torch.tensor([none]),), (torch.tensor([1.0]),)) == (False, 0.0)


def test_a_batch_goes_in_one_copy_as_bytes_its_arrays_are_views_of():
    arrays = {
        "history_lengths": np.array([3, 1], dtype=np.int64),
        "action_mask": np.array([[True, False, True], [False, True, True]]),
        "scalars": np.arange(10, dtype=np.float32).reshape(2, 5) / 7,
        "state_card": np.array([[70_000, 3], [5, 2**31 + 9]], dtype=np.int64),
        "action_member_card": np.zeros((2, 3, 0), dtype=np.int64),
    }
    call = argparse.Namespace(arrays=arrays, device="cpu")
    told = tool.in_one_copy(call, 3)
    # Each array starts at a multiple of eight: 16 bytes, 6 and 2 of filling, 40, 32, none.
    assert told["same"] and told["arrays"] == 5 and told["bytes"] == 16 + 8 + 40 + 32
    assert told["took"] > 0


def test_a_try_that_fails_is_told_and_the_report_stands_without_it():
    def failing(*_given):
        raise RuntimeError("the stream   is\ncapturing")

    assert tool.tried(failing, 1, 2) == "RuntimeError: the stream is capturing"
    assert tool.tried(lambda value: {"took": value}, 3) == {"took": 3}
    parts = {"seeds": 0.01, "card": 0.5, "answer": 1.5, "draw": 0.2, "probability": 0.1, "back": 0.05}
    found = {
        "calls": {
            8: {"size": 4, "whole": 2.0, **parts},
            32: {"size": 20, "whole": 2.5, **parts},
            512: {"size": 320, "whole": 8.0, **parts},
        },
        "graphs": {
            8: {"replay": 0.4, "place": 0.45, "whole": 1.0, "same": True, "apart": 0.0, "turned": True,
                "recording": 12.0, "keeps": 3 * 2**20},
            32: "RuntimeError: operation not permitted when stream is capturing",
            512: {"replay": 7.0, "place": 1.1, "whole": 9.0, "same": False, "apart": 2.5e-7,
                      "turned": False, "recording": 30.0, "keeps": 40 * 2**20},
        },
        "copies": {
            8: {"took": 0.06, "same": True, "bytes": 4096, "arrays": 50},
            32: {"took": 0.07, "same": False, "bytes": 8192, "arrays": 50},
            512: "ValueError: no view",
        },
    }  # fmt: skip
    played = {"sizes": [4] * 500 + [20, 320, 300], "games_seconds": 9.0, "model_seconds": 5.0}
    told = tool.graphs_report(played, found)
    small = next(line for line in told if line.strip().startswith("2 to 8"))
    assert small.split() == [
        "2", "to", "8", "4", "1.50", "0.40", "0.50", "0.45", "0.06", "2.00", "1.00",
        "yes,", "and", "for", "its", "rows", "turned", "round",
    ]  # fmt: skip
    middle = next(line for line in told if line.strip().startswith("9 to 32"))
    assert middle.split() == ["9", "to", "32", "20", "1.50", "-", "0.50", "-", "other", "2.50", "-", "not", "recorded"]
    large = next(line for line in told if line.strip().startswith("257 to 512"))
    assert large.split()[3:11] == ["320", "1.50", "7.00", "0.50", "1.10", "-", "8.00", "9.00"]
    assert large.endswith("NO, 2.5e-07 apart at most, NOT for its rows turned round")
    assert "Recording a graph took 12 to 30 ms, and one keeps 3.0 to 40.0 MiB of the card's memory." in told
    # 500 calls at 1 ms for 2 ms by a graph; the others the plain way, which is the shorter there.
    line = next(line for line in told if line.startswith("By those"))
    assert "take 0.52 s with a graph wherever it is the shorter way, where they take 1.02 s" in line
    assert told[-3:] == [
        "Not found (2):",
        "  9 to 32, the graph: RuntimeError: operation not permitted when stream is capturing",
        "  257 to 512, in one copy: ValueError: no view",
    ]
    # A processor's report has the copies and no graph.
    alone = tool.graphs_report(played, {"calls": found["calls"], "graphs": {}, "copies": {8: found["copies"][8]}})
    assert any(line.split()[-2:] == ["not", "recorded"] for line in alone)
    assert not any(line.startswith(("By those", "Recording")) for line in alone)


def test_a_replay_is_set_beside_the_models_own_answer(monkeypatch):
    """The steps around a graph, with a stand-in that records and replays
    nothing: on a card the recording leaves the answer to its replays."""

    class Graph:
        replays = 0

        def replay(self) -> None:
            Graph.replays += 1

    class Recording:
        def __init__(self, graph: Graph) -> None:
            assert isinstance(graph, Graph)

        def __enter__(self) -> None:
            return None

        def __exit__(self, *_ended: object) -> None:
            return None

    monkeypatch.setattr(tool.torch.cuda, "CUDAGraph", Graph)
    monkeypatch.setattr(tool.torch.cuda, "graph", Recording)
    monkeypatch.setattr(tool.torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(tool.torch.cuda, "memory_allocated", lambda: 0)
    asked: list[int] = []

    def model(given: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        asked.append(id(given["scalars"]))
        logits = given["scalars"].cumsum(-1).masked_fill(~given["action_mask"], float("-inf"))
        return logits, given["scalars"].sum(-1)

    arrays = {
        "history_lengths": np.array([2, 1, 3], dtype=np.int64),
        "scalars": np.array([[0.5, 1.0, 0.25], [2.0, 0.5, 1.5], [1.0, 1.0, 3.0]], dtype=np.float32),
        "action_mask": np.array([[True, True, False], [True, False, True], [True, True, True]]),
    }
    call = tool.Call(model, "cpu", arrays)
    told = tool.replayed(call, 3)
    # What the stand-in's recording answered is the model's answer to the batch,
    assert told["same"] is True and told["apart"] == 0.0
    # and a replay that does nothing leaves it for the rows turned round too.
    assert told["turned"] is False
    assert told["keeps"] == 0 and told["recording"] >= 0
    assert told["replay"] >= 0 and told["place"] > 0 and told["whole"] > told["place"]
    # One replay after the recording, one for the rows turned, and six for
    # each of the two times that hold a replay.
    assert Graph.replays == 2 + 6 + 6
    # The model answered the tensors that stay, not the call's own, while it was recorded.
    assert len(set(asked)) >= 2
    # A single decision has no rows to turn.
    alone = tool.Call(model, "cpu", {name: array[:1] for name, array in arrays.items()})
    assert tool.replayed(alone, 3)["turned"] is None


def test_a_batch_is_filled_out_to_the_widest_and_keeps_its_answers():
    small = {
        "history_lengths": np.array([1], dtype=np.int64),
        "scalars": np.array([[0.5, 1.0]], dtype=np.float32),
        "action_mask": np.array([[True, True]]),
        "action_member_card": np.array([[[7], [9]]], dtype=np.int64),
    }
    large = {
        "history_lengths": np.array([2, 1, 3], dtype=np.int64),
        "scalars": np.ones((3, 2), dtype=np.float32),
        "action_mask": np.ones((3, 3), dtype=bool),
        "action_member_card": np.ones((3, 3, 2), dtype=np.int64),
    }
    shapes = tool.widest({1: small, 8: large})
    assert shapes == {"history_lengths": (), "scalars": (2,), "action_mask": (3,), "action_member_card": (3, 2)}
    wide = tool.filled(small, shapes)
    # Zeros to the right of what was there, an answer that is none among them.
    assert wide["action_mask"].tolist() == [[True, True, False]]
    assert wide["action_member_card"].tolist() == [[[7, 0], [9, 0], [0, 0]]]
    assert wide["scalars"] is not small["scalars"] and wide["scalars"].tolist() == [[0.5, 1.0]]
    assert tool.filled(large, shapes)["action_member_card"].shape == (3, 3, 2)

    def model(given: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        scores = given["action_member_card"].sum(-1).to(torch.float32) * given["scalars"].sum(-1, keepdim=True)
        return scores.masked_fill(~given["action_mask"], float("-inf")), given["scalars"].sum(-1)

    own, wider = tool.Call(model, "cpu", small), tool.Call(model, "cpu", wide)
    told = tool.filled_out(own, wider, 3)
    assert told["same"] is True and told["apart"] == 0.0 and told["answer"] > 0

    def careless(given: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        # Scores an answer that is none, and reads the width of its batch.
        scores = given["action_member_card"].sum(-1).to(torch.float32) + 0.25 * given["action_mask"].shape[1]
        return scores, given["scalars"].sum(-1)

    told = tool.filled_out(tool.Call(careless, "cpu", small), tool.Call(careless, "cpu", wide), 3)
    assert told["same"] is False and told["apart"] == 0.25


def test_the_batches_filled_out_have_a_table_of_their_own():
    parts = {"seeds": 0.01, "card": 0.5, "answer": 1.5, "draw": 0.2, "probability": 0.1, "back": 0.05}
    graph = {"replay": 0.4, "place": 0.45, "whole": 1.0, "same": True, "apart": 0.0, "turned": True,
             "recording": 12.0, "keeps": 2**20}  # fmt: skip
    found = {
        "calls": {
            1: {"size": 1, "whole": 2.0, **parts},
            8: {"size": 4, "whole": 2.0, **parts},
            32: {"size": 20, "whole": 2.5, **parts},
            512: {"size": 320, "whole": 8.0, **parts},
        },
        "widest": {"state_card": (37,), "self_deck": (48,), "opponent_deck": (55,), "history_card": (32,),
                   "action_kind": (14,), "action_member_card": (14, 3)},
        "filled": {
            1: {"answer": 1.6, "same": False, "apart": 1.9e-6},
            8: {"answer": 1.7, "same": True, "apart": 0.0},
            32: "RuntimeError: out of memory",
            512: {"answer": 1.5, "same": True, "apart": 0.0},
        },
        "wide_graphs": {
            1: graph,
            8: {**graph, "same": False, "apart": 3.0e-7, "turned": False},
            512: "RuntimeError: capturing",
        },
        "graphs": {}, "copies": {},
    }  # fmt: skip
    played = {"sizes": [1] * 400 + [4] * 100 + [20, 320], "games_seconds": 9.0, "model_seconds": 5.0}
    failed: list[str] = []
    told = tool.wide_report(played, found, failed)
    assert told[1].startswith("The same with each batch filled out to the widest any call of the update is: 37 cards")
    assert "37 cards of the state, 32 earlier answers, 14 answers of 3 members (ms)." in told[1]
    one = next(line for line in told if line.split()[:2] == ["1", "1"])
    assert one.split()[:7] == ["1", "1", "1.50", "1.60", "0.40", "2.00", "1.00"]
    assert one.endswith("OTHERS, 1.9e-06 apart at most")
    few = next(line for line in told if line.strip().startswith("2 to 8"))
    assert few.endswith(
        "the same; the replay NOT the plain answer, 3.0e-07 apart at most; the replay NOT for its rows turned round"
    )
    middle = next(line for line in told if line.strip().startswith("9 to 32"))
    assert middle.split()[3:] == ["20", "1.50", "-", "-", "2.50", "-", "not", "answered"]
    large = next(line for line in told if line.strip().startswith("257 to 512"))
    assert large.split()[3:] == ["320", "1.50", "1.50", "-", "8.00", "-", "the", "same"]
    # 500 calls at 1 ms for 2 ms, the two others the plain way.
    assert told[-1] == (
        "By those, the calls of an update take 0.51 s with a graph of fixed sizes wherever it is the "
        "shorter way, where they take 1.01 s."
    )
    assert failed == [
        "  9 to 32, filled out: RuntimeError: out of memory",
        "  257 to 512, the graph of the batch filled out: RuntimeError: capturing",
    ]
    # They are told with the rest, what was not found last.
    whole = tool.graphs_report(played, found)
    assert told[1] in whole and whole[-3] == "Not found (2):" and whole[-2:] == failed
    # Nothing of it where no batch was filled out.
    assert tool.wide_report(played, {"calls": found["calls"]}, []) == []


class Tiny(torch.nn.Module):
    """A model of one weight a column, which scores the answers there are."""

    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([1.0, 0.5, 2.0]))

    def forward(self, given: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        scores = given["scalars"] * self.weight
        return scores.masked_fill(~given["action_mask"], float("-inf")), scores.sum(-1)


TINY = {
    "history_lengths": np.array([2, 1], dtype=np.int64),
    "scalars": np.array([[0.5, 1.0, 0.25], [2.0, 0.5, 1.5]], dtype=np.float32),
    "action_mask": np.array([[True, True, False], [True, False, True]]),
}


def test_a_step_of_a_pass_is_its_answer_and_its_gradient():
    model = Tiny()
    call = tool.Call(model, "cpu", TINY)
    told = tool.stepped(call, model, 4)
    assert told["answer"] > 0 and told["gradient"] > 0 and told["first"] > 0
    # No gradient is left on the model: the next step starts from none.
    assert model.weight.grad is None
    # Another way of answering is stepped with the same model's weights.
    asked = []

    def other(given: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        asked.append(len(given))
        return model(given)

    tool.stepped(call, other, 2)
    assert len(asked) == 4


def test_a_batch_is_answered_by_another_way_of_the_model_and_set_beside_it():
    model = Tiny()
    plain = tool.Call(model, "cpu", TINY)
    told = tool.compiled_call(plain, model, 3)
    assert told["same"] is True and told["apart"] == 0.0 and told["graph"] is None
    assert told["answer"] > 0 and told["whole"] > told["answer"] and told["compiling"] >= 0

    def off(given: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        logits, values = model(given)
        return logits + 0.125, values

    told = tool.compiled_call(plain, off, 3)
    assert told["same"] is False and told["apart"] == 0.125
    assert tool.headers() in (True, False)


def test_the_compiled_model_has_tables_of_its_own():
    parts = {"seeds": 0.01, "card": 0.5, "answer": 1.5, "draw": 0.2, "probability": 0.1, "back": 0.05}
    graph = {"replay": 0.2, "place": 0.45, "whole": 0.6, "same": True, "apart": 0.0, "turned": True,
             "recording": 12.0, "keeps": 2**20}  # fmt: skip
    found = {
        "calls": {8: {"size": 4, "whole": 2.0, **parts}, 512: {"size": 320, "whole": 8.0, **parts}},
        "filled": {8: {"answer": 1.7, "same": True, "apart": 0.0}, 512: "RuntimeError: too wide"},
        "compiled": {
            "the LSTM with the rest": {
                "calls": {
                    8: {"compiling": 11.5, "same": False, "apart": 1.2e-3, "answer": 0.9, "whole": 1.4, "graph": graph},
                    512: {"compiling": 14.0, "same": True, "apart": 0.0, "answer": 3.0, "whole": 9.5,
                              "graph": "RuntimeError: capturing"},
                },
            },
            "the LSTM left to cuDNN": "BackendCompilerFailed: no compiler",
        },
        "steps": {
            "size": 320,
            "ways": {
                "as a pass has it, the batch at its own sizes": {"answer": 4.4, "gradient": 9.0, "first": 0.02},
                "the batch filled out, compiled, the LSTM with the rest": {"answer": 2.5, "gradient": 70.0, "first": 41.5},
                "the batch filled out": "RuntimeError: out of memory",
            },
        },
    }  # fmt: skip
    played = {"sizes": [4] * 1000 + [320], "games_seconds": 9.0, "model_seconds": 5.0}
    failed: list[str] = []
    told = tool.compiled_report(played, found, failed)
    assert told[1] == (
        "The batches filled out, answered by the model compiled for their sizes, the LSTM with the rest (ms)."
    )
    small = next(line for line in told if line.strip().startswith("2 to 8"))
    assert small.split()[3:12] == ["4", "1.70", "0.90", "0.20", "2.00", "1.40", "0.60", "11.5", "1.2e-03"]
    assert small.endswith("1.2e-03 apart at most")
    large = next(line for line in told if line.strip().startswith("257 to 512"))
    assert large.split()[3:] == ["320", "-", "3.00", "-", "8.00", "9.50", "-", "14.0", "the", "same"]
    # A thousand calls at 0.6 ms by the compiled model's graph for 2 ms, and one the plain way.
    assert any(
        "take 0.61 s with the compiled model wherever it is the shorter way, where they take 2.01 s" in line
        for line in told
    )
    step = told.index(
        "A step of a pass at the batch of 320 decisions: the model's answer with its gradient kept, "
        "and the gradient (ms)."
    )
    assert told[step + 2].split()[-2:] == ["4.40", "9.00"]
    assert told[step + 3].endswith("2.50    70.00   the first took 41.5 s")
    assert told[step + 4].split()[-2:] == ["not", "made"]
    assert failed == [
        "  257 to 512, the graph of the model compiled, the LSTM with the rest: RuntimeError: capturing",
        "  compiled, the LSTM left to cuDNN: BackendCompilerFailed: no compiler",
        "  a step, the batch filled out: RuntimeError: out of memory",
    ]
    # Why nothing was compiled is told in a line; nothing of it where it was not asked.
    assert tool.compiled_report(played, {**found, "compiled": "no headers"}, []) == ["", "Not compiled: no headers"]
    assert tool.compiled_report(played, {"calls": found["calls"]}, []) == []
    assert "Not compiled: no headers" in tool.graphs_report(played, {"calls": found["calls"], "compiled": "no headers"})


class Tables(torch.nn.Module):
    """A model with a table, which scores the answers there are."""

    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(5)
        self.table = torch.nn.Embedding(7, 3)
        self.score = torch.nn.Linear(3, 3)

    def forward(self, given: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        seen = self.table(given["state_card"]).sum(1) * given["scalars"]
        scores = self.score(seen)
        return scores.masked_fill(~given["action_mask"], float("-inf")), seen.sum(-1)


WITH_TABLES = {
    **TINY,
    "history_lengths": np.array([2, 1, 3, 1], dtype=np.int64),
    "scalars": np.array([[0.5, 1.0, 0.25], [2.0, 0.5, 1.5], [1.0, 0.5, 0.5], [0.25, 2.0, 1.0]], dtype=np.float32),
    "action_mask": np.array([[True, True, False], [True, False, True], [True, True, True], [True, False, False]]),
    "state_card": np.array([[1, 2, 2], [3, 0, 6], [5, 5, 5], [4, 1, 0]], dtype=np.int64),
}


def test_a_table_looked_up_by_index_select_is_the_table():
    model = Tables()
    other = tool.with_lookups(Tables())
    assert isinstance(other.table, tool.Lookup) and isinstance(model.table, torch.nn.Embedding)
    given = {name: torch.from_numpy(array) for name, array in WITH_TABLES.items()}
    ours, theirs = model(given), other(given)
    assert all(torch.equal(one, two) for one, two in zip(ours, theirs, strict=True))
    assert tool.weights_beside(model, other) == (True, 0.0)
    for made, net in ((ours, model), (theirs, other)):
        made[1].sum().backward()
        assert net.table.weight.grad is not None
    assert torch.allclose(model.table.weight.grad, other.table.weight.grad, atol=1e-6)
    with torch.no_grad():
        other.score.bias += 0.5
    assert tool.weights_beside(model, other) == (False, 0.5)


def test_a_step_is_recorded_only_where_it_waits_for_nothing(monkeypatch):
    """The steps around a graph of a whole step, with stand-ins for the
    card's part: on a processor nothing is recorded, and a stand-in that
    replays nothing is told apart from the steps made one by one."""
    model = Tables()
    call = tool.Call(model, "cpu", WITH_TABLES)
    before = [parameter.detach().clone() for parameter in model.parameters()]
    told = tool.recorded_step(call, 3)
    assert told["waits"] is None and told["replay"] is None and "recording" not in told
    assert told["plain"] > 0 and told["able"] > 0 and told["able_same"] is True
    # The call's own model took no step: each way has a model of its own.
    assert all(torch.equal(was, now) for was, now in zip(before, model.parameters(), strict=True))
    assert tool.recorded_step(call, 3, lookups=True)["able_same"] is True

    class Graph:
        replays = 0

        def replay(self) -> None:
            Graph.replays += 1

    class Recording:
        def __init__(self, graph: Graph) -> None:
            assert isinstance(graph, Graph)

        def __enter__(self) -> None:
            return None

        def __exit__(self, *_ended: object) -> None:
            return None

    modes: list[str] = []
    monkeypatch.setattr(tool.torch.cuda, "CUDAGraph", Graph)
    monkeypatch.setattr(tool.torch.cuda, "graph", Recording)
    monkeypatch.setattr(tool.torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(tool.torch.cuda, "memory_allocated", lambda: 0)
    monkeypatch.setattr(tool.torch.cuda, "set_sync_debug_mode", modes.append)
    told = tool.recorded_step(call, 3, on_card=True)
    # Every wait was an error while a step was made, and is none again after.
    assert modes == ["error", "default"] and told["waits"] is None
    # Two replays on the batch, two on another, and six that are timed.
    assert Graph.replays == 2 + 2 + 6 and told["replay"] >= 0 and told["keeps"] == 0
    # A replay that takes no step does not leave the weights of the steps.
    assert told["same"] is False and told["another"] is False and told["another_apart"] > 0

    # A part that waits for the card is named, and nothing is recorded.
    clipping = tool.torch.nn.utils.clip_grad_norm_

    def waits(parameters, most):
        if modes[-1] == "error":
            raise RuntimeError("called a synchronizing CUDA operation")
        return clipping(parameters, most)

    monkeypatch.setattr(tool.torch.nn.utils, "clip_grad_norm_", waits)
    Graph.replays = 0
    told = tool.recorded_step(call, 3, on_card=True)
    assert told["waits"] == "the clipping: called a synchronizing CUDA operation"
    assert told["replay"] is None and Graph.replays == 0 and "recording" not in told
    assert told["plain"] > 0 and modes[-1] == "default"


def test_what_is_tried_last_is_told_after_the_report():
    parts = {"seeds": 0.01, "card": 0.5, "answer": 1.5, "draw": 0.2, "probability": 0.1, "back": 0.05}
    calls = {8: {"size": 4, "whole": 2.0, **parts}, 512: {"size": 320, "whole": 8.0, **parts}}
    step = {"waits": None, "replay": 4.5, "plain": 13.25, "able": 13.5, "able_same": True, "able_apart": 0.0,
            "recording": 40.0, "keeps": 64 * 2**20, "same": True, "same_apart": 0.0, "another": False,
            "another_apart": 2.5e-4}  # fmt: skip
    found = {
        "calls": calls,
        "trainer": {
            "widths": {"state": 96, "self_deck": 48, "opponent_deck": 56, "history": 32, "action": 32, "members": 2},
            "calls": {8: 1.25, 512: "wider than the sizes fixed"},
            "told": {"answered": 203, "recorded": 1, "checked": 1},
            "failed": None,
        },
        "step_graphs": {
            "size": 256,
            "ways": {
                "The tables looked up as the model looks them up": {
                    **step, "waits": "the gradient: called a synchronizing CUDA operation", "replay": None,
                    "able_same": False, "able_apart": 3.0e-9,
                },
                "The tables looked up by index_select": step,
                "Another way": "RuntimeError: out of memory",
            },
        },
    }  # fmt: skip
    told = tool.last_report(found)
    assert told[1].startswith("The trainer's own way (--inference-graphs), a call whole (ms): filled out to 96 cards")
    assert told[3].split() == ["2", "to", "8", "4", "2.00", "1.25"]
    assert told[4].split()[:6] == ["257", "to", "512", "320", "8.00", "-"] and told[4].endswith("wider than the sizes fixed")
    assert "It answered 203 calls from 1 graphs, each held against the model's own answer" in told[6]
    first = told.index("  The tables looked up as the model looks them up")
    assert told[first + 1].split()[-1] == "13.25"
    assert told[first + 2].endswith("weights 3.0e-09 apart at most after three steps")
    assert told[first + 3] == (
        "    not recorded, a step waits for the card: the gradient: called a synchronizing CUDA operation"
    )
    second = told.index("  The tables looked up by index_select")
    assert told[second + 2].endswith("the same weights after three steps")
    assert "4.50" in told[second + 3] and "the weights of the steps one by one on its batch" in told[second + 3]
    assert told[second + 3].endswith("NOT those weights on another of its sizes, 2.5e-04 apart at most")
    assert told[second + 4] == "    recording took 40 ms, and the graph keeps 64 MiB of the card's memory"
    assert told[-1] == "    not made: RuntimeError: out of memory"
    # A way that failed says so; a checkout without it says nothing.
    failed = tool.last_report({"calls": calls, "trainer": {**found["trainer"], "failed": "RuntimeError: capturing"}})
    assert failed[-1] == "IT FAILED, and answered the calls as they are from there: RuntimeError: capturing"
    assert tool.last_report({"calls": calls, "trainer": None}) == []
    assert tool.last_report({"calls": calls, "trainer": "ImportError: no"}) == [
        "", "The trainer's own way at fixed sizes was not run: ImportError: no",
    ]  # fmt: skip
