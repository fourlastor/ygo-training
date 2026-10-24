# The improved pilots on all71's yardstick

**Status:** ACTIVE (made beside [Overlap](2026-10-23-overlap.md), as it
trains nothing and touches nothing of that one)

**Proposed:** 2026-10-23

**Activated:** 2026-10-23

**Last updated:** 2026-10-24 (Malefic measured: the gap closed 56% ± 16%)

Work in the policies brings their pilots up towards `all71`: a census of
where a deck's pilot and the network part, and rules measured there, pushed
to the policies' branch `pilots-against-the-network` a deck at a time. This
measures each pushed deck here, on the games
`all71` was measured on: the share of its gap to `all71` the improved pilot
closes, on the yardstick the gap was found with. Cyber's pilot was 38.7
points under `all71` there (18.8% against 57.5%); the branch's own measure
had it 26 points up, one combination of three rules worth 16.7 ± 2.3 alone.

## Design (decided on 2026-10-23)

- **The yardstick.** The suites of seventy-one decks: the forty decks' and
  the eighteen decks', each deck 16 games against each of the thirteen
  decks' pilots, seed 2012, 24 workers, as `all71` and the pilots played them
  there (`analysis/seventy-one-decks/new` and `old`). `all71`'s games are its
  R3 measurement's, by its three checkpoints.
- **Two plays of the suites, at the same trainer code.** The pilots as the
  trainer pins them, once a pin; the branch's, built in a worktree of its
  own with the policies at the branch, nothing of this checkout touched
  (`experiments/commands/improved-pilots.sh`).
- **Only the pilots the branch changed are measured.** A deck the branch left
  as it was must win exactly the games it won; one that does not is told,
  and its pilot is measured too. A change in the policies' shared code
  could change the thirteen opponents, whose games would then not be those
  `all71` played.
- **Told, without a mark:** each changed deck's pilot as pinned and as the
  branch has it, the gain with its standard error (the same games, paired),
  `all71`'s share of them, and the share of the gap the branch's pilot closes
  (`tools/improved_pilots.py`). A deck has 208 games on it, some ±3.4 points.

## The pilots as pinned (2026-10-23)

Played at `9a9ce101`, the policies at `21d49d3`
(`analysis/improved-pilots/pinned-21d49d3`, 75 s): all 58 decks' pilots win
exactly the games they won for seventy-one decks, Cyber's 18.8% among them.
The yardstick stands as it was found.

## Cyber (2026-10-23)

The branch at `d696a4b`, five rule commits on Cyber and the branch's
write-up (`benchmarks/pilots-against-the-network.md` there): Chimeratech
fusions summoned in Attack; Chimeratech Fortress Dragon made only of the
other player's Machines; Zwei or Proto summoned before fusing; every
Machine it may banished to Overload Fusion; Cyber Twin Dragon brought with
Future Fusion. They change `decks/cyber.rs`, and add focused tests to
`lib.rs`. Played at `38728d3e` (`analysis/improved-pilots/branch-d696a4b`,
2 min with the build).

- **The other 57 decks win exactly the games they won**: nothing of the
  branch reaches another pilot, the thirteen opponents among them.
- **Cyber: 40.4% against 18.8% as pinned, +21.6 ± 3.2 points over its 208
  games.** `all71` wins 57.5% of the same games, 38.8 ± 3.2 over the pinned
  pilot: **the improved pilot closes 56% ± 8% of its gap.** The branch's own
  development pairs had it +21.2 (20.9% to 42.1%), about what the yardstick
  finds.

## The search with Cyber's improved pilot (designed 2026-10-23)

Once Cyber was measured, the search test was to be run again with Cyber's
improved pilot, and not with the other decks', which did not change; the
reading was fixed that evening.

- **What is played.** The specialist arm of
  [the staged search against the network](2026-10-23-search-against-network.md),
  on 520 of `all71`'s own deals that Cyber is in, in either seat, 30,000
  updates on, each played twice with the decks swapped: once with the
  policies as pinned and once with the branch's at `d696a4b` (the worktree's
  library), the same deals. The improved pilot is the search's own pilot
  where it plays Cyber, and its model of the network where the network does.
- **Reading.** The improved pilot **makes the search stronger** when the
  network wins at least 2 points less against the search with it than
  against the search with the pilot as pinned, by two standard errors, each
  deal's two games together and the deals paired; otherwise **not shown**
  (`tools/search_against_network.py compare`). Beside it, without a mark:
  each search's share against the network, and what the search did a game.
- **Where.** The development machine's processor, some 3 hours: the search is
  played by its processor, and the network's calls are a small part of a
  game.

## The search with Cyber's improved pilot: stronger (2026-10-23)

Played on the development machine's processor, on 22 threads: 0.86 hours
with the pilot as pinned and 1.13 with the branch's, not the 3 hours
reckoned (`analysis/search-against-network/cyber-pinned.json`,
`cyber-improved.json` and `cyber-compare.txt`).

- **The improved pilot makes the search stronger.** Against the search with
  it the network wins 5.7 ± 1.6 points less than against the search with
  the pilot as pinned, the 520 deals paired: 48.2% ± 1.3% of the 1,040
  games, against 53.8% ± 1.3%. By that experiment's own reading neither
  search is told apart from the network.
- **Beside it, without a mark**, by the seat Cyber is in, the same games
  paired. Where the search plays Cyber, the network wins 55.6% against
  63.6%, 8.0 ± 2.2 points less. Where the network plays Cyber, and the
  improved pilot is only the search's model of it, the network wins 41.4%
  against 45.3%, 3.9 ± 2.2 points less. The 18 games of the mirror (33.3%
  against 22.2%) are too few to say anything.
- **What the search did a game** changed little: asked 208 times against
  204, searched 63 against 60, its pilot's answer changed 2.5 times either
  way, 0.46 s a searched decision against 0.42. A game took 86 s of a
  thread against 66.

What it says: the better pilot carries into the search, the most of it
where the search plays Cyber itself.

## Zombie, Chain Burn, Herald and Iron Chain (2026-10-24)

The branch at `76090d2`: nine rule commits on four decks, and the
session's write-up of each.
- **Zombie:** Il Blud tributed for from the hand, waiting recruiters
  flipped to attack, Card Trooper summoned against small monsters, Ryko
  Set first, a weaker attacker to Defense after battle.
- **Chain Burn:** a chain opened only with three links ready, Poison of the
  Old Man Set and chained from the field only, Swift Scarecrow and Waboku
  kept for attackers worth half its Life Points, no draw or burn cards in
  the Draw and Standby Phases, any chain joined only when Fortune can come
  fourth.
- **Herald:** face-down monsters attacked with its Fairies.
- **Iron Chain:** Ryko Set first, Card Trooper against small monsters,
  Debris Dragon summoned for a small monster, Dandylion Set early.

They change those four decks' files and add tests to `lib.rs`. Played at
`a3395eec` (`analysis/improved-pilots/branch-76090d2`, 2 minutes with the
build).

- **The other 53 decks win exactly the games they won**, Cyber's the same
  games as at `d696a4b`.
- **Each deck on its 208 games**, the same games paired:

| Deck | As pinned | Improved | Gain | `all71` | Gap closed |
|---|---|---|---|---|---|
| Chain Burn | 25.0% | 40.4% | +15.4 ± 3.5 | 56.9% | 48% ± 9% |
| Iron Chain | 35.1% | 46.6% | +11.5 ± 3.4 | 59.8% | 47% ± 12% |
| Herald | 30.3% | 38.5% | +8.2 ± 2.2 | 54.5% | 34% ± 9% |
| Zombie | 48.1% | 55.8% | +7.7 ± 3.1 | 80.9% | 23% ± 8% |

The branch's own census, on new deals against the thirteen decks' pilots,
had the gaps closed 55%, 39%, 35% and 24%: about what the yardstick finds.

## Amazoness (2026-10-24)

The branch at `1b68829`, the first of the median decks: Amazoness's
monsters Normal Summoned face-up, never Set, and Special Summoned in
Attack; the branch's write-up. Played at `e72757e9`
(`analysis/improved-pilots/branch-1b68829`).

- **Only Amazoness plays other games than at `76090d2`**; the other 52
  decks win exactly the games they won, the five improved before as they
  were.
- **Amazoness: 54.8% against 49.0% as pinned, +5.8 ± 2.9 points** over its
  208 games; `all71` wins 62.8% of them, 13.8 ± 3.3 over the pilot as
  pinned: **the gap closed 42% ± 17%**, wide as the gap is small. The
  session's own census had it 30% ± 6%.

## Malefic (2026-10-24)

The branch at `0a529a1`: Terraforming for Malefic World whenever it is
neither in hand nor on the field, Malefic Parallel Gear Set, its own Field
Spell played whatever the opponent's; the branch's write-up. Played at
`c5fdd81c` (`analysis/improved-pilots/branch-0a529a1`).

- **Only Malefic plays other games than at `1b68829`.**
- **Malefic: 64.9% against 57.2% as pinned, +7.7 ± 2.4 points** over its
  208 games; `all71` wins 71.0%, 13.8 ± 3.3 over the pilot as pinned: **the
  gap closed 56% ± 16%**. The branch's own census had it 49% ± 9%.
