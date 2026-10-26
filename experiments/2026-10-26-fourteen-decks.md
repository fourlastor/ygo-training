# A network of fourteen decks: the champions copied, then PPO against itself alone

**Status:** ACTIVE

**Proposed:** 2026-10-26

**Activated:** 2026-10-26, when [A shorter copy](2026-10-26-shorter-copy.md)
was read and closed; design and readings fixed that evening, before
anything was built.

**Last updated:** 2026-10-26 (design and readings fixed; the runs queued)

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

## Beside it: luna's queue

The champions of more decks queued at low priority beside the shorter copy
go on behind this experiment's runs. Claudi-oh's Countdown was made on
2026-10-26, a champion at 54.1% ± 0.4% in its mirror against `all71`; then
Claudi-oh's Verdict, Astra's Exodia, Machina Gadgets, Between Light and
Dark, D.D. Border Patrol, Legendary Six Samurai, Gravekeeper's Tomb, Scrap
Renaissance, Heaven's Dispatch and Quickdraw Plants.

Countdown's pilot was thought to play its deck near perfectly, and the
champion to stand well under it. Measured on the development machine on
2026-10-26, into `analysis/champions/countdown/pilot`, the champion's last
checkpoint plays the deck as well as the pilot, in the mirror and against
the other decks:

- In the mirror, head to head, 2,048 deals from both seats
  (`eval_suites/countdown-against-pilot.json` for the champion in the
  learner's seat, `evaluate-suite --pilot` against it on
  `eval_suites/countdown-against-checkpoint.json` for the other way): the
  champion 50.7% ± 0.2%, `all71` on the same deals 46.6% ± 0.4%, the
  champion 4.2 ± 0.4 points over it. Player 0's seat wins some 64% of the
  games whoever holds it, which keeps every share near half. The champion
  plays the pilot's answer at 59% of its decisions, `all71` at 85%.
- Against the thirteen decks' pilots, on the 208 deals where seventy-one
  decks found `all71` 20.2 points under the pilot (64.4% against 84.6%, the
  one deck its PPO took down from its copy's 78.4%): the champion 86.1% ±
  2.4%, 1.4 ± 2.8 points over the pilot.
