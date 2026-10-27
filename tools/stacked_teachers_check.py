"""A copy's teachers answered in one call for all of them, beside each
answering on its own: whether their first choices are the same.

    OMP_NUM_THREADS=8 uv run python tools/stacked_teachers_check.py --device cpu \\
        --teacher blackwing=release/champions/blackwing.pt --teacher cyber=... --updates 1

The games of an update of a copy, as `PARTS=copy` of
`experiments/commands/fourteen-decks.sh` plays them: each deck's champion in
the learner's seat, ITSELF of the DUELS each champion against itself in its
own mirror, the others against the thirteen decks' pilots. Every round of
the teachers' decisions is answered three ways: each teacher on its own
decisions, as a copy answers without graphs; each from its graphs of fixed
sizes (`graphs.Graphs`, replayed from CUDA graphs on a card), as the copy of
fourteen decks answered; and all of them in one call, their weights stacked
(`stacked.Stacked`). The games go on by the first. Told for each of the
other two: the first choices that are another than the first's, the rounds
answered to the last digit alike, the largest difference of a logit, the
widest a flip is (how far apart the first's own scores put the two first
choices that differ), and the seconds each way took.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ygo_training.checkpoints import load_inference_model
from ygo_training.graphs import Graphs
from ygo_training.pool import DuelPool, taught_games
from ygo_training.schedules import allocate_matchup_seats, rollout_games
from ygo_training.stacked import Stacked

THIRTEEN = (
    "blackwing,lightsworn,gishki,burn,crystal,dragunity,gladiator,heroes,infernity,monarch,"
    "morphtronic,rock-block,spellcaster"
)
WAYS = ("own", "graphs", "stacked")


class Checked(Stacked):
    """The stacked teachers answering beside each on its own and from its
    graphs, the games going on by each on its own."""

    def __init__(self, teachers: dict[str, torch.nn.Module], device: str, largest: int) -> None:
        super().__init__(teachers, device)
        self.teachers = teachers
        self.graphs = {
            deck: Graphs(model, device, largest=largest) for deck, model in teachers.items()
        }
        self.seconds = dict.fromkeys(WAYS, 0.0)
        self.told: dict[str, Any] = {"rounds": 0, "decisions": 0}
        for way in WAYS[1:]:
            self.told[way] = {
                "other first choices": 0,
                "rounds alike": 0,
                "largest": 0.0,
                "widest flip": 0.0,
            }

    def begin_update(self) -> None:
        for graphs in self.graphs.values():
            graphs.begin_update()

    def _timed(self, way: str, work: Any) -> torch.Tensor:
        if self.device.startswith("cuda"):
            torch.cuda.synchronize()
        started = time.perf_counter()
        made = work()
        if self.device.startswith("cuda"):
            torch.cuda.synchronize()
        self.seconds[way] += time.perf_counter() - started
        return made

    def _each(self, arrays: dict[str, np.ndarray], decks: np.ndarray, graphs: bool) -> torch.Tensor:
        """Each teacher's logits for its deck's decisions, on its own or from
        its graphs, in the decisions' order."""
        answered: torch.Tensor | None = None
        for deck in sorted(set(decks.tolist())):
            rows = np.flatnonzero(decks == deck)
            given = {name: array[rows] for name, array in arrays.items()}
            made = self.graphs[deck].answer(given) if graphs else None
            if made is None:
                tensor = {
                    name: torch.from_numpy(array).to(self.device) for name, array in given.items()
                }
                with torch.no_grad():
                    made = self.teachers[deck](tensor)
            # A graph's answers are filled out to its widths: the round's are
            # the first of them.
            logits = made[0][:, : arrays["action_mask"].shape[-1]].clone()
            if answered is None:
                answered = torch.empty(
                    (len(decks), logits.shape[-1]), dtype=logits.dtype, device=logits.device
                )
            answered[torch.from_numpy(rows).to(logits.device)] = logits
        assert answered is not None
        return answered

    def answer(self, arrays: dict[str, np.ndarray], decks: np.ndarray) -> torch.Tensor:
        own = self._timed("own", lambda: self._each(arrays, decks, graphs=False))
        made = {
            "graphs": self._timed("graphs", lambda: self._each(arrays, decks, graphs=True)),
            "stacked": self._timed("stacked", lambda: super(Checked, self).answer(arrays, decks)),
        }
        self.told["rounds"] += 1
        self.told["decisions"] += len(decks)
        first = own.argmax(-1)
        for way, logits in made.items():
            told = self.told[way]
            other = logits.argmax(-1)
            flipped = other != first
            told["other first choices"] += int(flipped.sum())
            if flipped.any():
                # How far apart the teacher's own scores put the two answers.
                rows = flipped.nonzero().flatten()
                gaps = own[rows, first[rows]] - own[rows, other[rows]]
                told["widest flip"] = max(told["widest flip"], float(gaps.max()))
            told["rounds alike"] += int(torch.equal(logits, own))
            finite = torch.isfinite(own)
            if not torch.equal(finite, torch.isfinite(logits)):
                raise RuntimeError(f"the {way} answers mask other answers")
            if finite.any():
                difference = float((logits[finite] - own[finite]).abs().max())
                told["largest"] = max(told["largest"], difference)
        return own


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--teacher", action="append", required=True, help="DECK=CHECKPOINT")
    parser.add_argument("--opponents", default=THIRTEEN)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--updates", type=int, default=1)
    parser.add_argument("--duels", type=int, default=520)
    parser.add_argument("--itself", type=int, default=260)
    parser.add_argument("--seed", type=int, default=2012)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--pool-threads", type=int, default=12)
    parser.add_argument("--pool-slots", type=int, default=26)
    parser.add_argument("--pool-groups", type=int, default=2)
    parser.add_argument("--graphs", type=int, default=130, help="the largest call of a graph")
    parser.add_argument("--limit", type=int, default=4096)
    args = parser.parse_args()
    teachers = {}
    for given in args.teacher:
        deck, _, path = given.partition("=")
        model, _ = load_inference_model(Path(path), device=args.device)
        model.requires_grad_(False)
        teachers[deck] = model
    decks = sorted(teachers)
    opponents = [deck for deck in args.opponents.split(",") if deck]
    checked = Checked(teachers, args.device, args.graphs)
    pool = DuelPool(ROOT, threads=args.pool_threads, slots=args.pool_slots, groups=args.pool_groups)
    try:
        for update in range(args.updates):
            rollout_seed = args.seed * 1_000 + update
            learners, others, seats = allocate_matchup_seats(
                decks,
                opponents,
                args.duels,
                itself=args.itself,
                seed=rollout_seed ^ 0x4F50504F,
                pairs="mirrors",
            )
            games = rollout_games(rollout_seed, args.workers, others, learners, seats)
            pool.forget()
            checked.begin_update()
            taught_games(
                pool,
                teachers,
                args.device,
                learner=decks[0],
                games=games,
                limit=args.limit,
                grouped=args.pool_groups > 1,
                stacked=checked,
            )
            told = checked.told
            print(
                f"update {update + 1}: {told['rounds']} rounds, {told['decisions']} decisions so far; "
                + "; ".join(
                    f"{way}: {told[way]['other first choices']} other first choices, "
                    f"{told[way]['rounds alike']} rounds alike, largest difference "
                    f"{told[way]['largest']:.3g}, widest flip {told[way]['widest flip']:.3g}"
                    for way in WAYS[1:]
                ),
                flush=True,
            )
    finally:
        pool.close()
    seconds = ", ".join(f"{way} {checked.seconds[way]:.1f} s" for way in WAYS)
    print(f"Seconds answering: {seconds}")


if __name__ == "__main__":
    main()
