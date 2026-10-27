# A network of fourteen decks: the champions copied, then PPO against itself alone

**Status:** ACTIVE

**Proposed:** 2026-10-26

**Activated:** 2026-10-26, when [A shorter copy](2026-10-26-shorter-copy.md)
was read and closed; design and readings fixed that evening, before
anything was built.

**Last updated:** 2026-10-27 (the copy and the arm measured: the mirrors
lost in PPO, `all71` not decided; the recipe arm held; two arms that give
the mirrors a share of their duels, design and readings fixed and queued)

The network of four decks, its champions copied and then PPO on every
pairing of its decks, played each mirror about as well as its champion and
beat `all71` on all ten pairings. A network of all seventy-one decks has
5,041 ordered pairings: at the games of each pairing that gave the four
decks most of their gain, its PPO with half of each update's duels against
the pilots would take some 400 hours. This takes the recipe to fourteen
decks, the thirteen of training and Cyber, with a cheaper PPO: every duel
against itself and none against a pilot, half the updates for as many
duels against itself. At fourteen decks the pilots play no deck the network
does not play itself, which at four they did; so dropping their games was
not tested there. The cheaper arm is read against yardsticks fixed
beforehand, and the four decks' recipe is made as its control only where it
falls short.

## Design (fixed on 2026-10-26)

- **The decks**: Blackwing, Cyber, Lightsworn, Infernity, Gishki, Burn,
  Crystal Beasts, Dragunity, Gladiator Beasts, HEROes, Monarchs,
  Morphtronics, Rock-Block and Spellcasters, each taught by its champion of
  [the champions' release](../models/champions.md) (56.5% to 76.4% in its
  mirror against `all71`).
- **The copy (`taught`)**: one network of the fourteen decks from scratch,
  seed 2012, each deck's games played by its champion by its first choice
  and imitated. 2,240 updates of 520 duels, 160 a deck, the length the
  shorter copy found as good as 240; at 3e-4, in minibatches of 256 with no
  duels in flight, one pass, the value taught at four positions a game. Of
  an update's duels, 260 are each champion against itself in its own
  mirror, both seats imitated, and 260 against the thirteen decks' pilots.
  The teachers' games in two groups and from CUDA graphs, as for four decks.
  A checkpoint every 80 updates.
- **The arm (`selfplay`)**: PPO from the copy's last checkpoint, 4,320
  updates of 520 duels at 1e-4 with `fast`'s settings, seed 2012; every duel
  against itself, on the 196 ordered pairings of its decks each as often as
  any other, the mirrors among them; none against a pilot. Some 11,500
  games of each ordered pairing, as many as the four decks' uniform arm had
  after about 700 updates, by when it had most of its gain. A checkpoint
  every 240 updates of the trainer's count, which goes on from the copy's
  2,240: after 160, 400 and so on to 4,240 updates, and the last after
  4,320.
- **The control (`recipe`), made only where reading 3 asks for it**: the
  four decks' recipe from the same copy, 8,640 updates, of each update's
  520 duels 260 against itself on the same pairings and 260 against the
  thirteen decks' pilots: as many duels against itself as the arm.
- **Measured** (`experiments/commands/fourteen-decks.sh`, `PARTS` copy,
  selfplay and measure; `tools/fourteen_decks_report.py`): each teacher,
  and the copy after 560, 1,120, 1,680 and 2,240 updates (40, 80, 120 and
  160 a deck), against `all71` in each mirror, 2,048 deals from both seats;
  the arm's last three checkpoints, after 4,000, 4,240 and 4,320 updates, on
  every ordered pairing against `all71`, 1,024 games each
  (`eval_suites/fourteen-decks-pairings-checkpoint.json`); the copy's last
  checkpoint and every one the arm keeps on 64 games of each
  (`eval_suites/fourteen-decks-pairings-quick-checkpoint.json`); the arm's
  last checkpoint head to head against each champion in its mirror, 2,048
  deals from both seats; it and `all71` against the thirteen decks' pilots,
  each deck against each, 128 games
  (`eval_suites/fourteen-decks-against-pilots.json`); it head to head
  against the network of four decks on that network's 16 ordered pairings,
  both ways.
- On the CUDA machine's queue at high priority, by reckoning: about 11
  hours for the copy, 10 for the arm and 5 to measure.

Not measured here: transfer between pairings, the share of a trained
pairing's gain that a pairing never played gets. Holding pairings out of the
arm's games would change two things at once against the four decks' recipe,
the pilots' games and the pairings played, and leave the two not told
apart. It waits for an experiment of its own.

## Readings (fixed on 2026-10-26, before anything was built)

1. *The mirrors are kept*: the arm's last checkpoint head to head against
   each champion in its mirror, 2,048 deals from both seats: **kept** when
   the mean over the fourteen mirrors is at least 48% by two standard
   errors and no mirror is at most 45% by two standard errors; **lost**
   when the mean is at most 48% by two standard errors or a mirror at most
   45%; otherwise **not decided**. A mark for the mean and a floor for each
   mirror: over fourteen mirrors a mark of 48% for each would fail by
   chance alone, and the network of four decks was itself 48.1% against
   Infernity's champion. By this rule that network reads kept, and its
   weighted arm (a mean of 48.3%, Cyber 45.3%) not decided.
2. *It beats `all71` on every pairing*: each of the 105 pairings against
   `all71` by the last three checkpoints, a deal's games at the three
   together (a mirror from both seats, two decks the mean of their two
   seatings): it **beats `all71`** when every pairing is over 50% by two
   standard errors, **falls short** when one is at most 50% by two standard
   errors, otherwise **not decided**.
3. *Only if reading 1 is not kept or reading 2 does not beat `all71`*: the
   recipe arm is made and read by readings 1 and 2 too, and the arm is
   compared with it by the mean over the 105 pairings against `all71` by
   their last three checkpoints, the deals paired: **as good** when the two
   are within 1 point of each other by two standard errors, **worse** when
   the arm is at least 1 point under by two standard errors, otherwise
   **not decided**.

Told beside them, without a mark: each pairing against `all71` and the
weakest; each arm's curve along its updates; the copy in each mirror along
its updates, less its teacher; against the thirteen decks' pilots beside
`all71`; head to head against the network of four decks; the hours.

## Results (read on 2026-10-27)

The copy took 13.4 hours and the arm 9.3 (`analysis/fourteen-decks/report.txt`
on the CUDA machine); the measure 2 hours 15 minutes, with up to six
evaluations at once (below).

1. **The mirrors: lost.** The arm's last checkpoint head to head against each
   champion in its mirror, 2,048 deals from both seats: a mean of 44.9% ±
   0.2%, Cyber the lowest at 40.3% ± 0.7%. Blackwing 45.9%, Lightsworn
   47.8%, Infernity 46.9%, Gishki 49.0%, Burn 42.9%, Crystal Beasts 41.3%,
   Dragunity 44.8%, Gladiator Beasts 44.8%, HEROes 42.7%, Monarchs 45.2%,
   Morphtronics 48.5%, Rock-Block 43.0%, Spellcasters 45.9%.
2. **Against `all71`: not decided.** The mean over the 105 pairings by the
   last three checkpoints 56.9%; the weakest, Lightsworn against
   Spellcasters, 50.1% ± 0.4%. The mirrors against `all71` from 56.7%
   (Rock-Block) to 70.8% (Cyber).
3. Reading 1 not kept, **the recipe arm is to be made**, and read by
   readings 1 and 2 and against the arm.

Beside them:

- The arm along its updates, the mean over the pairings against `all71` on
  64 games each: 53.0% from the copy, 55.2% after 1,120 updates, 56.5% after
  2,080, at its highest 57.3% after 3,760 and 56.7% after 4,320. Most of the
  gain came in its first half.
- Against the thirteen decks' pilots 64.1%, `all71` 58.3% on the same
  games: +5.8 ± 0.3 points. Burn (29.0%) and Spellcasters (31.0%) the
  weakest there, as for `all71` (26.0%, 29.7%).
- Head to head against the network of four decks on that network's
  pairings: 45.2% ± 0.2%, Cyber's mirror the lowest at 35.5%.
- The copy in each mirror against `all71` after 2,240 updates, less its
  teacher: from −0.2 points (Crystal Beasts) to −2.9 (Cyber, Rock-Block),
  most of the way after 1,120 updates (80 a deck).
- Several evaluations at once through MPS (`analysis/fourteen-decks/at-once.txt`),
  against one alone at 68.3 games a second: three at once 2.07 times as
  many games a second, five 2.74 times, six 3.31 times. One evaluation of
  each level was played again alone and gave the same 2,048 games.
- The mirrors along the arm, the mean of the fourteen against `all71` on 64
  games each (about ±1.6 points): 65.7% from the copy, then between 63.4%
  and 65.8% at every checkpoint kept, 64.8% after 4,320 updates; the other
  pairings from 51.0% to 55.5%. PPO took its gain to the other pairings and
  left the mirrors where the copy had them.

The recipe arm was held on 2026-10-27, before it was queued: its duels
against itself have the arm's share of mirrors, and it would not tell
whether the mirrors were lost in the copy or in PPO.

- The copy's last checkpoint head to head against each champion in its
  mirror, on the deals the arm played them (`PARTS=copy-heads`, measured
  on 2026-10-27, seven evaluations at once, five minutes): a mean of 48.4%
  ± 0.1%, the lowest Cyber and Rock-Block at 47.1%; by reading 1's rule
  the copy would read kept. The arm less the copy, deal by deal: −3.5 ±
  0.2 points over the mirrors; Crystal Beasts −7.2, Cyber −6.8, Burn −6.2,
  HEROes −5.5, Rock-Block −4.1, Dragunity −4.0, Monarchs −3.6, Gladiator
  Beasts −3.2, Blackwing −2.3, Spellcasters −2.1, Infernity −1.8,
  Lightsworn −0.6, Gishki −0.6, Morphtronics −0.5. The copy held the
  mirrors and PPO lost them.

## The mirrors' arms (design and readings fixed on 2026-10-27, before anything was built)

The copy held the mirrors head to head and PPO lost them, its mirrors 14 of
the 196 ordered pairings and so 7% of its duels. Two arms give them a share
of their own.

- **Design**: from the copy's last checkpoint, each as the selfplay arm
  (4,320 updates of 520 duels at 1e-4 with `fast`'s settings, seed 2012,
  every duel against itself and none against a pilot, its checkpoints
  where the arm's are), but for the duels against itself shared out by
  weights of the ordered pairings (`--self-play-weights`): **`quarter`**, a
  quarter of them in the mirrors (each mirror 13, each other pairing 3:
  some 9.3 duels of each mirror an update and 2.1 of each other pairing),
  and **`half`**, half of them (13 and 1: some 18.6 and 1.4). The two
  trained at once through MPS (`PARTS=mirrors`, `AT_ONCE=2`), then
  measured as the selfplay arm, seven evaluations at once
  (`PARTS=mirrors-measure`, `AT_ONCE=7`). About 13 hours for the two and
  3 to measure, by reckoning.
- **Readings**, for each of the two: readings 1 and 2 as they are; and
  4. *Against the selfplay arm*: the mean over the 105 pairings against
  `all71` by their last three checkpoints, the deals paired: **as good**
  when the two are within 1 point of each other by two standard errors,
  **worse** when the arm is at least 1 point under the selfplay arm by two
  standard errors, otherwise **not decided**.
- Told beside them, without a mark: each arm less the copy head to head in
  the mirrors, deal by deal; each arm's curve along its updates.

## Beside it: luna's queue

The champions of more decks queued at low priority beside the shorter copy
go on behind this experiment's runs. Claudi-oh's Countdown was made on
2026-10-26, a champion at 54.1% ± 0.4% in its mirror against `all71`; then
Claudi-oh's Verdict, Astra's Exodia, Machina Gadgets, Between Light and
Dark, D.D. Border Patrol, Legendary Six Samurai, Gravekeeper's Tomb, Scrap
Renaissance, Heaven's Dispatch and Quickdraw Plants. Verdict was made that
evening, a champion at 61.1% ± 0.5%. On 2026-10-27 the other 46 decks of the
pool were queued behind them, each with a mirror suite of its own
(`a9c72e05`), so that every deck of the 71 has a champion to teach a network
of all of them.

Countdown's pilot was thought to play its deck near perfectly, and the
champion to stand well under it. Measured on the development machine on
2026-10-26, into `analysis/champions/countdown/pilot`, the champion's last
checkpoint plays the mirror as well as the pilot, and the other decks a
little under it and far over `all71`:

- In the mirror, head to head, 2,048 deals from both seats
  (`eval_suites/countdown-against-pilot.json` for the champion in the
  learner's seat, `evaluate-suite --pilot` against it on
  `eval_suites/countdown-against-checkpoint.json` for the other way): the
  champion 50.7% ± 0.2%, `all71` on the same deals 46.6% ± 0.4%, the
  champion 4.2 ± 0.4 points over it. Player 0's seat wins some 64% of the
  games whoever holds it, which keeps every share near half. The champion
  plays the pilot's answer at 59% of its decisions, `all71` at 85%.
- Against the thirteen decks' pilots, 128 games of each pairing, the same
  1,664 deals for all three (`eval_suites/countdown-against-pilots.json`):
  the pilot 89.7% ± 0.7%, the champion 84.4% ± 0.9%, 5.2 ± 0.9 points
  under the pilot and 16.5 ± 1.2 over `all71`, and `all71` 68.0% ± 1.1%,
  21.7 ± 1.2 under the pilot, as seventy-one decks found it on 208 of the
  deals (64.4% against 84.6%, the one deck its PPO took down from its
  copy's 78.4%). The champion's 86.1% ± 2.4% on those 208 deals was within
  their noise of either. `all71` is 21 to 33 points under the pilot on ten
  of the thirteen pairings and within 6 of it on Lightsworn, HEROes and
  Dragunity. The champion is within 9 points of the pilot on all but Burn
  (29.7% against 46.9%), and over `all71` by 14 to 30 points on the ten
  `all71` lost, most on Morphtronics, Spellcasters and Gladiator Beasts.
  `all71`'s games last 37 turns to the pilot's 26, and it loses 32 of them
  by deck-out to the pilot's one.
