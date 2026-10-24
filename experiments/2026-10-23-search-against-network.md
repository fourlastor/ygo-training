# The staged search against the network

**Status:** ACTIVE (made beside [Duels in flight](2026-10-23-duels-in-flight.md),
as it trains nothing and touches nothing of that one; beside
[Overlap](2026-10-23-overlap.md) since that one was closed)

**Proposed:** 2026-10-23

**Activated:** 2026-10-23

**Last updated:** 2026-10-23 (design and reading fixed before anything was
built)

Proposed on 2026-10-23, while the run of duels in flight went and
[the credit offline](2026-10-23-credit-offline.md) was being measured: to
try the policies' new search algorithm against the best network there is,
and see whether it is stronger, weaker or not told apart. The search is the
policies' staged search (`crates/ygo-policies-search`, seated as one more
seat of a duel since the policies' `a36ce95`, in the pin): at a decision its
pilot's answer and up to seven others are played to the end of the duel in
worlds dealt again from what the seat cannot see, 8, then 24, then 64 of
them, and an alternative takes the pilot's place only when it wins clearly
more often. Upstream measured it, with the generic pilot, at 62.7% against
each deck's specialist in its own mirror, where the generic pilot alone wins
36.9%, at some 61 s of a processor a game against 70 ms.

Read beside it on 2026-10-23: Dustin Rubin's posts on search for a card game
(The Fox in the Forest, `dustinrubin.github.io/ai.cardgames`). A
determinized search on top of his strongest network first won only 35%
against it; one taught from such a search beat it 67% once and then stopped
improving; and the pilot a search plays its worlds with is what upstream's
own figures find decides it (on the 19 decks where the generic pilot is
weakest the search wins 47.2% with it and 69.7% with each deck's
specialist). So two arms. The generic pilot is the setting for play without
knowing the opponent's deck; in training the decks are known, and the better
pilot can be used.

## Design (fixed on 2026-10-23)

- `all71`'s last checkpoint (`model-27599.pt`, after 10,560 updates of
  PPO), giving its first choice as a measured suite does, against the
  staged search at the settings upstream measured (`SearchOptions::MEASURED`:
  stages of 8, 24 and 64 worlds, the best 2 kept after the first, a paired
  z of 1.645, inside open chains, the other player's Set monsters dealt
  again, 8 candidates), head to head.
- Two arms:
  - **generic**: the search played with the generic pilot, the other player
    taken to be the generic pilot, as upstream measured it;
  - **specialist**: the same search played with the deck's own specialist,
    the other player taken to be the other deck's specialist.
  In both the search deals its worlds from the other deck's list, which the
  network is not given.
- 1,040 deals of `all71`'s run's own, as it drew its duels against itself
  (any two of the 71 decks), 30,000 updates on; each played twice, the
  network and the search swapping decks: the same seed, the same decks in
  the same seats, the other one playing each. 2,080 games an arm. A duel the
  limit of 4,096 answers stops is a draw.

## Reading (fixed on 2026-10-23, before anything was built)

For each arm: the network is **stronger** when it wins at least 52% of the
games, by two standard errors (the deals drawn again, each deal's two games
together); **weaker** when it wins 48% or less, by two standard errors;
otherwise **not told apart**.

Told beside it, without a mark: the draws; the mirrors and the other
pairings; how often the search changed its pilot's answer and what a
searched decision took; what a game took.

## The build (2026-10-23)

- **The policies** (`21d49d3`, local): `SearchSeat::with_pilots` seats the
  staged search with a pilot named in the generic one's place and the other
  player's model named too, as the bench's `--pilot specialist` and
  `--opponent-model specialist` have them; `SearchSeat::new` is the generic
  seat as it was (the search's own tests unchanged). A pilot or a model
  other than the generic policy is told nothing of what the cards do, which
  a specialist would refuse.
- **The environment** (`84eec25c`): a search's seat (`control: search`,
  its pilot `policy`, its model `model`, its settings `search`) is played
  by that search: fed every message of the duel, it answers its seat's
  questions with its worlds played on the duel itself, which the episode
  now shares with it, and its answers are given as a recorded seat's are.
  The end of a duel says what each search's seat did. Held to the duel the
  search plays fed as the bench feeds it, digest for digest, with the
  generic pilot and with a deck's specialist (`crates/ygo-env/tests/duels.rs`,
  a few worlds and candidates to play it in seconds); the crate's other
  tests and the pool's Python ones unchanged.
- **`tools/search_against_network.py`** (`play`, `report`; tests in
  `tests/test_search_against_network.py`). Tried small, three deals of each
  arm both ways: every game ran to its end, the search searching about a
  third of its seat's questions (75 of 238 with the generic pilot, 93 of 234
  with the specialist), about 0.5 to 0.6 s a searched decision, some 47 s a
  game.

## Results

### The generic arm (2026-10-23)

Played on the development machine from 13:22 to 17:06, at `84eec25c` with the
policies at `21d49d3` (`analysis/search-against-network/generic.json`, 22
threads).

**Reading: the network is stronger.** `all71` wins 54.8% ± 0.8% of the 2,080
games, the deals' two games counted together; with two standard errors taken
away, 53.2%, at or over 52%. It won 1,140, lost 938, and 2 were drawn.

Beside it, without a mark:

- **By pairing:** in the 26 games of a deck against itself 50.0% ± 12.7%; in
  the other pairings 54.9% ± 0.8% (2,054 games).
- **The search, a game:** asked 223 times, it searched 65 of them and changed
  its pilot's answer 2.4 times, taking 0.69 s a searched decision and some 45
  s a game.
- **What a game took:** 3.73 h on 22 threads, 142 s of a thread a game: the
  search's seat holds its thread while the pool's rounds go in step.

### The specialist arm (2026-10-23)

Played on the development machine from 14:33 to 17:46 beside the generic
one, at the same revisions, 48 slots on 22 threads
(`analysis/search-against-network/specialist-run.json`).

**Reading: not told apart.** `all71` wins 48.7% ± 0.8% of the 2,080 games;
with two standard errors taken away 47.1%, under 52%, and with them added
50.3%, over 48%. It won 1,012 and lost 1,068, none drawn.

Beside it, without a mark:

- **By pairing:** in the 26 games of a deck against itself 42.3% ± 9.6%; in
  the other pairings 48.7% ± 0.8% (2,054 games).
- **The search, a game:** asked 220 times, it searched 68 of them and changed
  its pilot's answer 2.3 times, taking 0.68 s a searched decision and some 46
  s a game.
- **What a game took:** 3.23 h on 22 threads, 123 s of a thread a game.

## What it says

The staged search at upstream's settings plays weaker than `all71` with the
generic pilot (the network 54.8% ± 0.8%), and as well as it with each deck's
own specialist (48.7% ± 0.8%, not told apart, the network a point under
even). The pilot a search plays its worlds with is worth some 6 points of
the network's share here, as upstream found it decides the search. Neither
arm plays clearly better than the network, at some 46 s of a processor a
game where the network takes milliseconds: as a teacher, the specialist's
search would teach about `all71`'s strength, not more. Stronger pilots would
raise it, as the pilots improved on the policies' branch
([The improved pilots](2026-10-23-improved-pilots.md)) may.
