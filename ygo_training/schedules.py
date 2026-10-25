"""Pure training schedules with no engine or PyTorch dependencies."""

from __future__ import annotations

import random

MAX_SEEDED_WORKERS = 64
# Every rollout owns a run of this many duel seeds.
MAX_EPISODES_PER_ROLLOUT = 1_024
ROLLOUT_SEED_STRIDE = MAX_EPISODES_PER_ROLLOUT
ROLLOUT_SEED_SLOTS = (0x7FFF_FFFF + 1) // ROLLOUT_SEED_STRIDE
# Low 20 bits of Knuth's multiplicative hash; odd, so it permutes all slots.
RUN_SEED_MULTIPLIER = 489_905
WORKER_RNG_DOMAIN = 1 << 31
POLICY_SEQUENCE_LIMIT = 1_000_000
POLICY_WORKER_STRIDE = POLICY_SEQUENCE_LIMIT
# Decisions a network takes in one duel, at most, for the seeds of its draws.
DUEL_DECISION_BITS = 20


def allocate_balanced_opponents(
    opponents: list[str], episodes: int, *, seed: int
) -> list[str]:
    """Allocate an exact, shuffled, worker-independent opponent population."""
    if episodes < 0:
        raise ValueError("episodes must be non-negative")
    if not opponents:
        if episodes:
            raise ValueError("cannot allocate episodes from an empty opponent pool")
        return []
    base, remainder = divmod(episodes, len(opponents))
    extra_order = list(opponents)
    random.Random(seed ^ 0x52454D41).shuffle(extra_order)
    extras = set(extra_order[:remainder])
    assignments = [
        opponent
        for opponent in opponents
        for _ in range(base + int(opponent in extras))
    ]
    random.Random(seed).shuffle(assignments)
    return assignments


def allocate_balanced_pairs(
    pairs: list[tuple[str, str]], episodes: int, *, seed: int
) -> list[tuple[str, str]]:
    """Allocate an exact, shuffled population of ``pairs``: every pair as
    often as any other, to within a game."""
    drawn = allocate_balanced_opponents(
        [str(index) for index in range(len(pairs))], episodes, seed=seed
    )
    return [pairs[int(index)] for index in drawn]


def allocate_weighted_pairs(
    weights: dict[tuple[str, str], float], episodes: int, *, seed: int
) -> list[tuple[str, str]]:
    """Allocate an exact, shuffled population of the pairs of ``weights``:
    each pair as often as its share of the weights, the duels a share leaves
    over going to the largest remainders, ties broken by the seed."""
    if episodes < 0:
        raise ValueError("episodes must be non-negative")
    total = sum(weights.values())
    if not weights or total <= 0 or min(weights.values()) < 0:
        raise ValueError("pairs are weighted by numbers not below zero, not all zero")
    pairs = sorted(weights)
    order = list(range(len(pairs)))
    random.Random(seed ^ 0x52454D41).shuffle(order)
    exact = [episodes * weights[pair] / total for pair in pairs]
    counts = [int(share) for share in exact]
    left = episodes - sum(counts)
    # The largest remainders first, and among equal ones the seed's order.
    ranked = sorted(order, key=lambda place: counts[place] - exact[place])
    for place in ranked[:left]:
        counts[place] += 1
    drawn = [pair for pair, count in zip(pairs, counts, strict=True) for _ in range(count)]
    random.Random(seed).shuffle(drawn)
    return drawn


def allocate_balanced_matchups(
    learners: list[str], opponents: list[str], episodes: int, *, seed: int
) -> list[tuple[str, str]]:
    """Allocate an exact, shuffled population of the learner's decks against
    the opponents: every pairing as often as any other, to within a game."""
    pairs = [(learner, opponent) for learner in learners for opponent in opponents]
    return allocate_balanced_pairs(pairs, episodes, seed=seed)


# Who sits in the seat the learner does not: the pilot of the deck there,
# the learner's own network, or a checkpoint held as it is.
OTHER_SEATS = ("pilot", "self", "checkpoint")
# Which of a learner's decks meet in its duels against itself: any two of
# them, one in each seat; each deck itself alone; or two different decks
# alone (`allocate_matchup_seats`).
SELF_PLAY_PAIRS = ("all", "mirrors", "cross")


def allocate_other_seats(
    pilots: list[str], learner: str, *, itself: int, checkpoint: int, seed: int
) -> tuple[list[str], list[str]]:
    """The duels of an update with who sits in the other seat of each.

    ``pilots`` are the decks met as they were allocated, each played by its
    pilot.  ``itself`` duels more are the learner against its own network and
    ``checkpoint`` against a checkpoint held as it is, each with the learner's
    deck on both sides.  The three are shuffled among each other: a duel
    with a network in both seats asks twice as much of it as one against a
    pilot, and the duels are played in the order of this list.  The deck of
    the other seat in each duel, and who plays it (``OTHER_SEATS``).
    """
    if itself < 0 or checkpoint < 0:
        raise ValueError("the duels against a network must not be negative")
    duels = (
        [(deck, "pilot") for deck in pilots]
        + [(learner, "self")] * itself
        + [(learner, "checkpoint")] * checkpoint
    )
    random.Random(seed ^ 0x53454154).shuffle(duels)
    return [deck for deck, _ in duels], [seat for _, seat in duels]


def allocate_matchup_seats(
    learners: list[str],
    opponents: list[str],
    episodes: int,
    *,
    itself: int,
    seed: int,
    pairs: str = "all",
    weights: dict[tuple[str, str], float] | None = None,
) -> tuple[list[str], list[str], list[str]]:
    """The duels of an update of a learner of several decks, some of them
    against itself.

    ``itself`` of the ``episodes`` duels have the learner's own network in
    both seats, each with one of the learner's decks: every pair of them,
    which deck is in the learner's seat and which in the other, as often as
    any other to within a game, a deck against itself among them.  So a seat
    has each deck as often as any other, and two different decks meet twice
    as often as a deck meets itself.  With ``pairs`` `mirrors` each deck
    meets only itself, every deck as often as any other; with `cross` only
    the other decks, every ordered pair of two decks as often as any other.
    With ``weights``, each ordered pair of the learner's decks by name, the
    duels against itself are shared out by them instead
    (``allocate_weighted_pairs``).  The other duels are the learner's decks
    against the opponents' pilots, every pairing as often as any other.  The
    two kinds are shuffled among each other, as ``allocate_other_seats``
    shuffles them for a learner of one deck.

    For each duel: the deck in the learner's seat, the deck in the other
    seat, and who plays that one (``OTHER_SEATS``).
    """
    if not 0 <= itself <= episodes:
        raise ValueError("the duels against itself are of the update's duels")
    met = {
        "all": [(own, other) for own in learners for other in learners],
        "mirrors": [(own, own) for own in learners],
        "cross": [(own, other) for own in learners for other in learners if other != own],
    }
    if pairs not in met:
        raise ValueError(f"pairs must be one of {SELF_PLAY_PAIRS}")
    if itself and not met[pairs]:
        raise ValueError("two different decks meet where the learner has two")
    duels = [
        (own, other, "pilot")
        for own, other in allocate_balanced_matchups(
            learners, opponents, episodes - itself, seed=seed
        )
    ] + [
        (own, other, "self")
        for own, other in (
            allocate_balanced_pairs(met[pairs], itself, seed=seed ^ 0x53454C46)
            if weights is None
            else allocate_weighted_pairs(weights, itself, seed=seed ^ 0x53454C46)
        )
    ]
    random.Random(seed ^ 0x53454154).shuffle(duels)
    return (
        [own for own, _, _ in duels],
        [other for _, other, _ in duels],
        [seat for _, _, seat in duels],
    )


def allocate_episodes(workers: int, episodes: int) -> list[int]:
    """Distribute a global episode count with at most one episode of imbalance."""

    base, remainder = divmod(episodes, workers)
    return [base + (worker < remainder) for worker in range(workers)]


def episode_offsets(counts: list[int]) -> list[int]:
    """Return each worker's first episode index within one rollout."""

    offsets = []
    offset = 0
    for count in counts:
        offsets.append(offset)
        offset += count
    if offset > MAX_EPISODES_PER_ROLLOUT:
        raise ValueError(
            f"seed schedule supports at most {MAX_EPISODES_PER_ROLLOUT} episodes per rollout"
        )
    return offsets


def episode_seed(rollout_seed: int, episode: int) -> int:
    """The seed of one duel of a rollout: its deal and both pilots' ties."""
    if not 0 <= episode < MAX_EPISODES_PER_ROLLOUT:
        raise ValueError(f"episode must be in [0, {MAX_EPISODES_PER_ROLLOUT})")
    result = int(rollout_seed) + episode
    if not 0 <= result <= 0x7FFF_FFFF:
        raise ValueError("episode seed exceeds the signed 32-bit range")
    return result


def worker_rng_seed(rollout_seed: int, worker_id: int) -> int:
    """Pack rollout and worker into a separate signed-64-bit generator seed."""
    if not 0 <= worker_id < MAX_SEEDED_WORKERS:
        raise ValueError(f"worker_id must be in [0, {MAX_SEEDED_WORKERS})")
    result = (int(rollout_seed) << 32) | WORKER_RNG_DOMAIN | worker_id
    if result > 0x7FFF_FFFF_FFFF_FFFF:
        raise ValueError("worker RNG seed exceeds Python's signed 64-bit range")
    return result


def rollout_games(
    rollout_seed: int,
    workers: int,
    opponents: list[str],
    learners: list[str] | None = None,
    others: list[str] | None = None,
) -> list[dict[str, int | str]]:
    """The duels of a rollout against the listed opponents, one each, in the
    order of their episodes: as ``workers`` worker processes come to them.

    A worker plays a run of the episodes and draws the learner's seat in each
    from a generator of its own, so the seats follow from the number of
    workers.  Played elsewhere, the duels are the same ones.  ``learners``
    names the deck the learner plays in each, where it plays several, and
    ``others`` who sits in the other seat (``OTHER_SEATS``), where it is not
    the pilot in every duel.
    """
    counts = allocate_episodes(workers, len(opponents))
    games: list[dict[str, int | str]] = []
    for worker, (offset, count) in enumerate(zip(episode_offsets(counts), counts, strict=True)):
        drawn = random.Random(worker_rng_seed(rollout_seed, worker))
        for episode in range(offset, offset + count):
            game: dict[str, int | str] = {
                "episode_index": episode,
                "worker": worker,
                "seed": episode_seed(rollout_seed, episode),
                "opponent": opponents[episode],
                "learner_player": drawn.randrange(2),
            }
            if learners is not None:
                game["learner"] = learners[episode]
            if others is not None:
                game["other"] = others[episode]
            games.append(game)
    return games


def keeps_label(duel_seed: int, decision: int, fraction: float) -> bool:
    """Whether a learner decision keeps its teacher label when only a fraction do.

    The choice depends on the duel and the decision alone and draws from no
    random stream, so thinning the labels leaves the games unchanged.
    """
    if fraction >= 1.0:
        return True
    mask = 0xFFFF_FFFF_FFFF_FFFF
    mixed = (duel_seed * 0x9E3779B97F4A7C15 + (decision + 1) * 0xD1B54A32D192ED03) & mask
    mixed ^= mixed >> 31
    mixed = (mixed * 0xBF58476D1CE4E5B9) & mask
    mixed ^= mixed >> 29
    return mixed / 2.0**64 < fraction


def policy_sample_seed(rollout_seed: int, worker_id: int, sequence: int) -> int:
    """Pack rollout/worker/prompt into a separate signed-64-bit Torch seed."""
    if not 0 <= worker_id < MAX_SEEDED_WORKERS:
        raise ValueError(f"worker_id must be in [0, {MAX_SEEDED_WORKERS})")
    if not 0 <= sequence < POLICY_SEQUENCE_LIMIT:
        raise ValueError(f"policy sequence must be in [0, {POLICY_SEQUENCE_LIMIT})")
    local = worker_id * POLICY_WORKER_STRIDE + sequence
    result = (int(rollout_seed) << 32) | local
    if result > 0x7FFF_FFFF_FFFF_FFFF:
        raise ValueError("policy sample seed exceeds Torch's signed 64-bit range")
    return result


def duel_sample_seed(rollout_seed: int, episode: int, decision: int) -> int:
    """The seed of the draw at one decision of one duel of a rollout.

    ``decision`` is the decision's place among those the network takes in the
    duel, so the draw follows from the duel alone, whatever is played beside
    it and in whatever order: what duels taken forward together need.  The
    workers' ``policy_sample_seed`` counts a worker's requests instead.
    """
    if not 0 <= episode < MAX_EPISODES_PER_ROLLOUT:
        raise ValueError(f"episode must be in [0, {MAX_EPISODES_PER_ROLLOUT})")
    if not 0 <= decision < 1 << DUEL_DECISION_BITS:
        raise ValueError(f"decision must be in [0, {1 << DUEL_DECISION_BITS})")
    result = (int(rollout_seed) << 32) | (episode << DUEL_DECISION_BITS) | decision
    if not 0 <= result <= 0x7FFF_FFFF_FFFF_FFFF:
        raise ValueError("duel sample seed exceeds the signed 64-bit range")
    return result


def decayed_teacher_imitation_weight(
    initial_weight: float, decay_updates: int, update_index: int
) -> float:
    if initial_weight <= 0.0:
        return 0.0
    if decay_updates <= 0:
        return initial_weight
    return initial_weight * max(0.0, 1.0 - update_index / decay_updates)


class TrainingSchedule:
    """Own rollout seed and teacher-schedule counters across updates."""

    def __init__(self, *, seed: int) -> None:
        if seed < 0:
            raise ValueError("seed must be non-negative")
        self.seed = seed
        self.rollout_index = 0
        self.update_index = 0

    def set_rollout_index(self, index: int) -> None:
        if index < 0:
            raise ValueError("rollout index must be non-negative")
        self.rollout_index = index

    def set_update_index(self, index: int) -> None:
        if index < 0:
            raise ValueError("training schedule index must be non-negative")
        self.update_index = index

    def rollout_seed(self, index: int) -> int:
        """The seed of the rollout of ``index``: its duels own the run of
        ``ROLLOUT_SEED_STRIDE`` seeds that begins there."""
        rollout_slot = (self.seed * RUN_SEED_MULTIPLIER + index) % ROLLOUT_SEED_SLOTS
        return rollout_slot * ROLLOUT_SEED_STRIDE

    def next(
        self, *, teacher_weight: float, teacher_decay_updates: int, warmup_updates: int
    ) -> tuple[int, int, float, bool]:
        rollout_index = self.rollout_index
        update_index = self.update_index
        rollout_seed = self.rollout_seed(rollout_index)
        imitation_weight = decayed_teacher_imitation_weight(
            teacher_weight, teacher_decay_updates, update_index
        )
        warmup = update_index < warmup_updates
        self.rollout_index += 1
        self.update_index += 1
        return rollout_index, rollout_seed, imitation_weight, warmup
