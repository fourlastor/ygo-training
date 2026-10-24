"""A head that learns how cards work: its probe, its weight and its first
reading (experiments/2026-09-08-engine-transition-prediction.md).

    .venv/bin/python tools/card_head.py probe CHECKPOINT --run-json RUN.json --output DIR
    .venv/bin/python tools/card_head.py weight CHECKPOINT --run-json RUN.json --head DIR/card-head.pt
    .venv/bin/python tools/card_head.py read CHECKPOINT --run-json RUN.json --head HEAD --output FILE
    .venv/bin/python tools/card_head.py report FILE

`probe` builds the trainer as the run's `run.json` says (as
`tools/update_signal.py` does), the network as the checkpoint has it and
never changed, and plays the games of the run's next UPDATES updates with
the cards of each transaction followed to its horizon (`DuelPool.labels`).
After each update's games the head (`ygo_training/card_head.py`), begun
anew, is taught POSITIONS of each seat's labelled decisions a duel, drawn at
random, in one pass of minibatches at 3e-4: it reads the network's own parts
of those decisions, and nothing of its gradient reaches the network.  On
each update's games, before it is taught them, its log-loss of the cards'
places is kept beside that of the frequencies counted on the games before
(the frequencies of each place by the card's location, its side and the kind
of the answer chosen).  Then it reads every labelled decision of HELD
updates far from those (`--away`), which it never trained on, against the
frequencies counted on all the games it was taught.  The head is saved in
DIR (`card-head.pt`): the run begins its own as the probe left it.  The
labels are audited on the way: of the cards the seat knew both at the
decision and at the horizon, how many were found where they were followed
to with their identity, and how many cards were lost.

`weight` takes the games of the checkpoint's next update, and of them the
size of PPO's gradient on the parameters it shares with the head and of
the head's: their ratio is the weight at which the head's is as large as
PPO's, as the design fixes it.

`read` takes the first reading of a run's checkpoint and the head kept
beside it: the frequencies counted on the games of COUNTED updates of that
checkpoint, then the held games read as the probe reads them.

`report` gives what `probe` or `read` found: the share of the frequencies'
log-loss the head takes away, with its error (the duels drawn again), and
the first reading as it was fixed on 2026-10-22: it predicts when it takes
away at least half, less two standard errors.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from update_signal import flat, trainer_for, update_games

# The first reading's mark.
TAKES_AWAY = 0.5
ERRORS = 2.0
# The vocabularies the frequencies are counted by (`schema.py`).
LOCATIONS, KINDS = 16, 32


class Frequencies:
    """How often each place follows from a card's location and side and the
    kind of the answer chosen, counted one each to begin with."""

    def __init__(self) -> None:
        from ygo_training.card_head import PLACES

        self.counts = np.ones((LOCATIONS, 2, KINDS, PLACES))

    def add(self, keys: tuple[np.ndarray, ...], places: np.ndarray) -> None:
        np.add.at(self.counts, (*keys, places), 1)

    def surprise(self, keys: tuple[np.ndarray, ...], places: np.ndarray) -> np.ndarray:
        """The log-loss of each place given its keys."""
        cell = self.counts[keys]
        return -np.log(cell[np.arange(len(places)), places] / cell.sum(-1))


def decided(trajectories: list[dict[str, Any]]) -> list[tuple[int, int, int, int]]:
    """Every decision of the network in an update's games: its row, the
    answer chosen, its duel and its seat."""
    return [
        (int(step["row"]), int(step["chosen"]), duel, int(step["player"]))
        for duel, trajectory in enumerate(trajectories)
        for step in trajectory["steps"]
    ]


def labelled(trainer: Any, found: list[tuple[int, int, int, int]]) -> tuple[list, dict]:
    """Of those decisions, the ones with labels, and their labels."""
    labels = trainer._pool().labels([row for row, _, _, _ in found])
    keep = np.flatnonzero(labels["labelled"])
    return [found[index] for index in keep], {name: value[keep] for name, value in labels.items()}


def read(
    trainer: Any, head: Any, found: list, labels: dict, chunk: int
) -> tuple[Any, list[np.ndarray]]:
    """The network's parts of decisions ``found`` and what the head makes of
    them, without a gradient: each card's place logits, a chunk at a time,
    with its keys (location, side, kind of answer) and its label."""
    import torch

    from ygo_training.card_head import chosen_answers

    out = []
    with torch.no_grad():
        for start in range(0, len(found), chunk):
            part = found[start : start + chunk]
            rows = [row for row, _, _, _ in part]
            chosen = torch.tensor([answer for _, answer, _, _ in part], device=trainer.device)
            batch = trainer._kept(rows)
            _, _, state, tokens, action = trainer.model.parts(batch)
            place = labels["card_place"][start : start + chunk]
            width = min(tokens.shape[1], place.shape[1])
            logits = head(state, tokens[:, :width], chosen_answers(action, chosen))[0]
            kinds = batch["action_kind"][torch.arange(len(part)), chosen].cpu().numpy()
            out.append(
                (
                    torch.log_softmax(logits, -1).cpu().numpy(),
                    batch["state_location"][:, :width].cpu().numpy(),
                    batch["state_controller"][:, :width].cpu().numpy(),
                    kinds,
                    place[:, :width],
                    np.array([duel for _, _, duel, _ in part]),
                )
            )
    return out


def cards_of(parts: list) -> tuple[np.ndarray, ...]:
    """Every labelled card of those parts: the head's log-loss, its keys,
    its place and its duel."""
    losses, keys, places, duels = [], ([], [], []), [], []
    for log_probs, location, side, kind, place, duel in parts:
        known = place >= 0
        rows, cards = np.nonzero(known)
        losses.append(-log_probs[rows, cards, place[rows, cards]])
        keys[0].append(location[rows, cards])
        keys[1].append(side[rows, cards])
        keys[2].append(kind[rows])
        places.append(place[rows, cards])
        duels.append(duel[rows])
    return (
        np.concatenate(losses),
        tuple(np.concatenate(key) for key in keys),
        np.concatenate(places),
        np.concatenate(duels),
    )


def taken_away(head_loss: np.ndarray, frequency_loss: np.ndarray, duels: np.ndarray,
               draws: int = 1000, seed: int = 2012) -> tuple[float, float]:  # fmt: skip
    """The share of the frequencies' log-loss the head takes away, over every
    card, and its error with the duels drawn again."""
    names, places = np.unique(duels, return_inverse=True)
    mine = np.bincount(places, head_loss, len(names))
    theirs = np.bincount(places, frequency_loss, len(names))
    share = 1.0 - mine.sum() / theirs.sum()
    generator = np.random.default_rng(seed)
    drawn = []
    for _ in range(draws):
        pick = generator.integers(0, len(names), len(names))
        drawn.append(1.0 - mine[pick].sum() / theirs[pick].sum())
    return float(share), float(np.std(drawn, ddof=1))


def audit(labels: dict, counts: dict[str, int]) -> None:
    """What the labels of one update say of themselves."""
    counts["checked"] += int(labels["label_checked"].sum())
    counts["agreed"] += int(labels["label_agreed"].sum())
    place = labels["card_place"]
    counts["cards"] += int((place >= 0).sum())


def held(trainer: Any, head: Any, frequencies: Frequencies, args: argparse.Namespace) -> dict:
    """The head and the frequencies on the games of the held updates."""
    head.eval()
    losses, references, duels = [], [], []
    for place in range(args.held):
        seed, games = update_games(trainer, args.away + place)
        trajectories, _ = trainer._learner_games(games, seed, labels=False, follow=True)
        found, labels = labelled(trainer, decided(trajectories))
        loss, keys, places, duel = cards_of(read(trainer, head, found, labels, args.chunk))
        losses.append(loss)
        references.append(frequencies.surprise(keys, places))
        duels.append(duel + place * 100_000)
        print(f"held update {place + 1} of {args.held}: {len(found)} decisions", flush=True)
    head_loss, frequency_loss = np.concatenate(losses), np.concatenate(references)
    share, error = taken_away(head_loss, frequency_loss, np.concatenate(duels))
    return {
        "cards": len(head_loss),
        "head_log_loss": float(head_loss.mean()),
        "frequencies_log_loss": float(frequency_loss.mean()),
        "share": share,
        "error": error,
    }


def probe(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    from ygo_training.card_head import CardHead, card_loss, chosen_answers

    torch.manual_seed(args.seed)
    torch.set_num_threads(args.torch_threads)
    described = json.loads(args.run_json.read_text())
    trainer = trainer_for(args.checkpoint, described, args)
    trainer.model.eval()
    head = CardHead(trainer.model).to(trainer.device)
    optimizer = torch.optim.Adam(head.parameters(), lr=args.learning_rate)
    frequencies = Frequencies()
    generator = random.Random(args.seed)
    counts = {"checked": 0, "agreed": 0, "cards": 0}
    curve = []
    started = time.perf_counter()
    for later in range(args.updates):
        seed, games = update_games(trainer, later)
        trajectories, _ = trainer._learner_games(games, seed, labels=False, follow=True)
        found, labels = labelled(trainer, decided(trajectories))
        audit(labels, counts)
        # What it makes of these games before it is taught them.
        head.eval()
        loss, keys, places, _ = cards_of(read(trainer, head, found, labels, args.chunk))
        before = {
            "head": float(loss.mean()),
            "frequencies": float(frequencies.surprise(keys, places).mean()),
        }
        frequencies.add(keys, places)
        # Taught so many of each seat's labelled decisions a duel.
        by_seat: dict[tuple[int, int], list[int]] = {}
        for index, (_, _, duel_of, seat) in enumerate(found):
            by_seat.setdefault((duel_of, seat), []).append(index)
        taught = [
            index
            for indices in by_seat.values()
            for index in (
                indices
                if len(indices) <= args.positions
                else generator.sample(indices, args.positions)
            )
        ]
        generator.shuffle(taught)
        head.train()
        losses = []
        for start in range(0, len(taught), args.minibatch):
            part = [found[index] for index in taught[start : start + args.minibatch]]
            picked = taught[start : start + args.minibatch]
            rows = [row for row, _, _, _ in part]
            chosen = torch.tensor([answer for _, answer, _, _ in part], device=trainer.device)
            batch = trainer._kept(rows)
            with torch.no_grad():
                _, _, state, tokens, action = trainer.model.parts(batch)
                answers = chosen_answers(action, chosen)
            told = {
                name: torch.from_numpy(labels[name][picked]).to(trainer.device)
                for name in ("labelled", "card_place", "card_change", "life_change", "count_change")
            }
            result = card_loss(head, state, tokens, answers, told)
            if result is None:
                continue
            optimizer.zero_grad()
            result[0].backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            optimizer.step()
            losses.append(float(result[0].detach()))
        curve.append(
            {"update": later + 1, "decisions": len(found), **before, "loss": float(np.mean(losses))}
        )
        print(
            f"update {later + 1} of {args.updates}: {len(found)} decisions, before it is taught "
            f"them the head's log-loss {before['head']:.4f}, the frequencies' "
            f"{before['frequencies']:.4f}; {time.perf_counter() - started:.0f} s",
            flush=True,
        )
    args.output.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "card-head": head.state_dict(),
            "checkpoint": str(args.checkpoint),
            "updates": args.updates,
        },
        args.output / "card-head.pt",
    )
    found_held = held(trainer, head, frequencies, args)
    trainer.close()
    return {
        "kind": "probe",
        "checkpoint": str(args.checkpoint),
        "updates": args.updates,
        "positions": args.positions,
        "learning_rate": args.learning_rate,
        "held_updates": args.held,
        "seconds": time.perf_counter() - started,
        "audit": counts,
        "curve": curve,
        "held": found_held,
    }


def weight(args: argparse.Namespace) -> dict[str, Any]:
    """PPO's gradient on the shared parameters and the head's, over the
    games of the checkpoint's next update."""
    import torch

    from ygo_training.card_head import CardHead, card_loss, chosen_answers
    from ygo_training.optimization import HEAD_PARAMETERS

    torch.set_num_threads(args.torch_threads)
    described = json.loads(args.run_json.read_text())
    trainer = trainer_for(args.checkpoint, described, args)
    head = CardHead(trainer.model).to(trainer.device)
    head.load_state_dict(
        torch.load(args.head, map_location=trainer.device, weights_only=False)["card-head"]
    )
    shared = [
        parameter
        for name, parameter in trainer.model.named_parameters()
        if parameter.requires_grad and not name.startswith(HEAD_PARAMETERS)
    ]
    seed, games = update_games(trainer, 0)
    trajectories, _ = trainer._learner_games(games, seed, labels=False, follow=True)
    optimizer = trainer.policy_optimizer
    optimizer.begin_update(trainer.measured_from)
    samples, advantages, _ = optimizer.prepare_samples(trajectories)
    trainer.model.train()
    indices = list(range(len(samples)))
    keys = {samples[index].step["_transaction_key"] for index in indices}
    ppo = np.zeros(sum(parameter.numel() for parameter in shared))
    for part in optimizer._chunks(samples, indices, args.chunk):
        share = len({samples[index].step["_transaction_key"] for index in part}) / len(keys)
        total = optimizer.objective(samples, advantages, part).components["total"]
        ppo += flat(torch.autograd.grad(total * share, shared, allow_unused=True), shared)
    found, labels = labelled(trainer, decided(trajectories))
    taught = np.zeros_like(ppo)
    for start in range(0, len(found), args.chunk):
        part = found[start : start + args.chunk]
        rows = [row for row, _, _, _ in part]
        chosen = torch.tensor([answer for _, answer, _, _ in part], device=trainer.device)
        _, _, state, tokens, action = trainer.model.parts(trainer._kept(rows))
        told = {
            name: torch.from_numpy(labels[name][start : start + args.chunk]).to(trainer.device)
            for name in ("labelled", "card_place", "card_change", "life_change", "count_change")
        }
        result = card_loss(head, state, tokens, chosen_answers(action, chosen), told)
        if result is not None:
            share = len(part) / len(found)
            taught += flat(
                torch.autograd.grad(result[0] * share, shared, allow_unused=True), shared
            )
    trainer.close()
    sizes = {"ppo": float(np.linalg.norm(ppo)), "head": float(np.linalg.norm(taught))}
    return {
        "checkpoint": str(args.checkpoint),
        "head": str(args.head),
        "samples": len(samples),
        "labelled": len(found),
        "gradient": sizes,
        "cosine": float(ppo @ taught / max(sizes["ppo"] * sizes["head"], 1e-30)),
        "weight": sizes["ppo"] / sizes["head"],
    }


def reading(args: argparse.Namespace) -> dict[str, Any]:
    """The first reading of a checkpoint and the head kept beside it."""
    import torch

    from ygo_training.card_head import CardHead

    torch.set_num_threads(args.torch_threads)
    described = json.loads(args.run_json.read_text())
    trainer = trainer_for(args.checkpoint, described, args)
    trainer.model.eval()
    head = CardHead(trainer.model).to(trainer.device)
    head.load_state_dict(
        torch.load(args.head, map_location=trainer.device, weights_only=False)["card-head"]
    )
    head.eval()
    frequencies = Frequencies()
    counts = {"checked": 0, "agreed": 0, "cards": 0}
    started = time.perf_counter()
    for later in range(args.counted):
        seed, games = update_games(trainer, later)
        trajectories, _ = trainer._learner_games(games, seed, labels=False, follow=True)
        found, labels = labelled(trainer, decided(trajectories))
        audit(labels, counts)
        _, keys, places, _ = cards_of(read(trainer, head, found, labels, args.chunk))
        frequencies.add(keys, places)
        print(f"counted update {later + 1} of {args.counted}: {len(found)} decisions", flush=True)
    found_held = held(trainer, head, frequencies, args)
    trainer.close()
    return {
        "kind": "reading",
        "checkpoint": str(args.checkpoint),
        "head": str(args.head),
        "counted_updates": args.counted,
        "held_updates": args.held,
        "seconds": time.perf_counter() - started,
        "audit": counts,
        "held": found_held,
    }


def verdict(share: float, error: float) -> str:
    """The first reading: it predicts at half the frequencies' log-loss taken
    away, less two standard errors."""
    return "it predicts" if share - ERRORS * error >= TAKES_AWAY else "not shown to predict"


def report(found: dict[str, Any]) -> str:
    told = found["held"]
    checked = found["audit"]
    lines = [f"{found['kind']} of {found['checkpoint']}:"]
    for point in found.get("curve", [])[:: max(1, len(found.get("curve", [])) // 10)]:
        lines.append(
            f"  update {point['update']:3d}: the head's log-loss {point['head']:.4f}, the "
            f"frequencies' {point['frequencies']:.4f} (taught to {point['loss']:.4f})"
        )
    lines += [
        (
            f"The labels: {checked['agreed']:,} of {checked['checked']:,} cards known at both "
            f"ends found where they were followed to, with their identity."
        ),
        (
            f"On the held games ({told['cards']:,} cards): the head's log-loss "
            f"{told['head_log_loss']:.4f}, the frequencies' {told['frequencies_log_loss']:.4f}; "
            f"it takes away {told['share']:.3f} ± {told['error']:.3f} of theirs"
        ),
        f"The reading: {verdict(told['share'], told['error'])}",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument("checkpoint", type=Path)
    shared.add_argument("--run-json", type=Path, required=True, help="the run's run.json")
    shared.add_argument("--chunk", type=int, default=1024, help="decisions read at a time")
    shared.add_argument("--learners", default=None, help="comma-separated; the run's by default")
    shared.add_argument("--opponents", default=None, help="comma-separated; the run's by default")
    shared.add_argument("--threads", type=int, default=20, help="the pool's threads")
    shared.add_argument("--slots", type=int, default=16)
    shared.add_argument("--torch-threads", type=int, default=8)
    shared.add_argument("--device", default="cpu", help="where the networks run: cpu or cuda")
    shared.add_argument(
        "--scratch-run",
        type=Path,
        default=ROOT / "runs" / "card-head",
        help="the run the trainer is told of; nothing is written there",
    )
    shared.add_argument("--held", type=int, default=4, help="updates' games read, never trained on")
    shared.add_argument("--away", type=int, default=10_000, help="updates away for the held games")
    probed = commands.add_parser("probe", parents=[shared])
    probed.add_argument("--output", type=Path, required=True)
    probed.add_argument("--updates", type=int, default=40)
    probed.add_argument("--positions", type=int, default=8, help="of each seat's decisions a duel")
    probed.add_argument("--learning-rate", type=float, default=3e-4)
    probed.add_argument("--minibatch", type=int, default=256)
    probed.add_argument("--seed", type=int, default=2012)
    weighed = commands.add_parser("weight", parents=[shared])
    weighed.add_argument("--head", type=Path, required=True, help="the probe's card-head.pt")
    weighed.add_argument("--output", type=Path, required=True)
    read_one = commands.add_parser("read", parents=[shared])
    read_one.add_argument("--head", type=Path, required=True, help="the head kept beside it")
    read_one.add_argument("--counted", type=int, default=8, help="updates the frequencies count")
    read_one.add_argument("--output", type=Path, required=True)
    reported = commands.add_parser("report")
    reported.add_argument("found", type=Path)
    args = parser.parse_args()
    if args.command == "report":
        print(report(json.loads(args.found.read_text())))
        return
    if args.command == "weight":
        found = weight(args)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(found, indent=2))
        print(json.dumps(found, indent=2))
        return
    found = probe(args) if args.command == "probe" else reading(args)
    path = args.output / "probe.json" if args.command == "probe" else args.output
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(found))
    print(report(found))


if __name__ == "__main__":
    main()
