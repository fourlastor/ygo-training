# The improved pilots on all71's yardstick

**Status:** ACTIVE (made beside [Overlap](2026-10-23-overlap.md), as it
trains nothing and touches nothing of that one)

**Proposed:** 2026-10-23

**Activated:** 2026-10-23

**Last updated:** 2026-10-25 (Fish, Dragunity, Gladiator and Lightsworn
measured, and the operation stopped at `cbc71f0`)

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

## Fish, Dragunity, Gladiator and Lightsworn (2026-10-25)

The branch at `cbc71f0`, its last push, and the branch's write-up of each:

- **Fish:** Coelacanth recruits Royal Swamp Eel, then Unshaven Angler,
  before Oyster Meister; Stardust Dragon first among its Synchros; Foolish
  Burial sends Coelacanth, for a revival.
- **Dragunity:** Stardust Dragon before Scrap Dragon and Trident Dragion.
- **Gladiator:** Gladiator Beast Gyzarus made only against a card of the
  opponent's.
- **Lightsworn:** Judgment Dragon's wipe before the Normal Summon when the
  opponent has more cards than its other monsters, kept though the branch's
  held-out reading had it far under that pass's mark (+0.78 ± 0.45 points).
- In Tele-DAD, Dark World and Heroes no rule paid: their pilots are as they
  were.

Dragunity, Gladiator and Lightsworn are three of the thirteen decks whose
pilots are every deck's opponents, so the measurement changed in two ways
(`experiments/commands/improved-pilots.sh`, `tools/improved_pilots.py`):

- **The thirteen decks' own screens are played too**, 128 games against
  each of the thirteen pilots, as `all71` played them for seventy-one decks
  (`analysis/seventy-one-decks/screens`): a deck of the thirteen has no
  games on the two suites.
- **Games against a changed opponent are told apart.** `all71` played the
  pilots as pinned, so a deck's games against the branch's Dragunity,
  Gladiator or Lightsworn are not those it was measured on. Each deck is
  measured on its games against the other ten: 160 games for a deck of the
  suites, 1,280 for one of the thirteen on its screen. The games against
  each changed opponent are told by themselves, those of the decks whose
  own pilots are as they were.

The trainer pins the policies at `eebace8` since its publication on
2026-10-24: `21d49d3`, which the branch was cut from, with the policies'
license, and CardScripts at its own mend of Batteryman Fuel Cell in place
of the policies' mended copy. The pilots as pinned at `eebace8` win exactly
the games they won at `21d49d3`, all 520 and 234 cases
(`analysis/improved-pilots/pinned-eebace8`). Played at `0b5f89ed`
(`analysis/improved-pilots/branch-cbc71f0`).

- **The 60 decks whose pilots the branch left as they were win exactly the
  games they won** against the ten; the eleven it changed play other games.
- **The four, each against the ten**, the same games paired:

| Deck | Games | As pinned | Improved | Gain | `all71` | Gap closed |
|---|---|---|---|---|---|---|
| Fish | 160 | 54.4% | 61.9% | +7.5 ± 2.4 | 67.7% | 56% ± 18% |
| Dragunity | 1,280 | 62.9% | 66.6% | +3.8 ± 0.9 | 75.3% | 30% ± 7% |
| Gladiator | 1,280 | 60.9% | 63.9% | +3.0 ± 0.6 | 72.4% | 26% ± 5% |
| Lightsworn | 1,280 | 65.2% | 65.6% | +0.5 ± 0.3 | 76.4% | 4% ± 2% |

  The branch's own census, on new deals against the thirteen, had Fish's
  gap closed 33% ± 7%, Dragunity's 12% ± 10% and Gladiator's 26% ± 6%, and
  Lightsworn's improved pilot 0.6 points under its pilot before.

- **The seven decks improved before, against the ten**, read about as on
  their 208 games above, within their errors:

| Deck | As pinned | Improved | Gain | `all71` | Gap closed |
|---|---|---|---|---|---|
| Cyber | 21.2% | 44.4% | +23.1 ± 3.9 | 57.9% | 63% ± 10% |
| Chain Burn | 26.2% | 43.1% | +16.9 ± 4.1 | 57.3% | 54% ± 11% |
| Iron Chain | 39.4% | 49.4% | +10.0 ± 3.8 | 63.5% | 41% ± 14% |
| Malefic | 58.1% | 66.9% | +8.8 ± 2.9 | 68.8% | 82% ± 26% |
| Herald | 33.8% | 40.6% | +6.9 ± 2.4 | 57.1% | 29% ± 9% |
| Zombie | 48.1% | 54.4% | +6.2 ± 3.5 | 80.8% | 19% ± 10% |
| Amazoness | 53.8% | 59.4% | +5.6 ± 3.1 | 65.8% | 47% ± 22% |

- **Against the changed opponents**, the 2,080 games of the 60 decks whose
  own pilots are as they were: Dragunity's improved pilot is a harder
  opponent, the others winning 36.1% against 38.9%, 2.8 ± 0.7 points less;
  Gladiator's harder still, 36.6% against 40.7%, 4.1 ± 0.5 points less;
  Lightsworn's about the same, 31.9% against 32.2%, 0.2 ± 0.2 points less.
  Counted with the 512 games of the eleven improved decks, Lightsworn's
  had read 1.0 ± 0.3 points easier: those decks' own gain (+6.1 ± 1.4 on
  those games), not its pilot's.

## The thirteen, and the stop (2026-10-25)

Of the thirteen decks' pilots, three are improved (Dragunity, Gladiator,
Lightsworn), and in two no rule paid (Heroes, Infernity). The branch's
census of the other eight, `all71` against each deck's own pilot on the
same 1,040 deals, and beside it any kind of decision whose answers by the
network alone carry enough of the gap to be worth a rule:

| Deck | Pilot | `all71` | Gap | Worth a rule |
|---|---|---|---|---|
| Blackwing | 59.3% | 70.1% | +10.8 ± 1.6 | none |
| Gishki | 42.3% | 53.0% | +10.7 ± 1.7 | none |
| Rock-Block | 58.8% | 66.0% | +7.1 ± 1.6 | chain windows, +5.2 ± 1.3 |
| Crystal Beasts | 35.6% | 41.9% | +6.3 ± 1.5 | main-phase commands, +4.7 ± 1.3 |
| Spellcaster | 22.7% | 28.4% | +5.7 ± 1.4 | none |
| Monarch | 66.7% | 72.3% | +5.6 ± 1.7 | none |
| Morphtronic | 37.6% | 40.7% | +3.1 ± 1.3 | none |
| Burn | 23.0% | 22.1% | −0.9 ± 1.2 | none |

Blackwing's and Gishki's gaps are spread as Tele-DAD's, Dark World's,
Infernity's and Heroes' were, where no rule paid; Rock-Block's and
Crystal's are small, so a good rule adds a few points.

**The operation stopped there on 2026-10-25**, the returns diminishing:
eleven pilots improved on the branch at `cbc71f0`, the gap to `all71`
closed from 4% (Lightsworn) to over half (Cyber, Chain Burn, Fish,
Malefic). The trainer's pin is still `eebace8`. A move to the branch
changes every run's opponents and the pilots copied, and every reading
against the pilots: against Dragunity's and Gladiator's improved pilots
the other decks win 2.8 and 4.1 points less.
