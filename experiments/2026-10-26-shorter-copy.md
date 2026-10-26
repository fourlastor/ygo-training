# A shorter copy: four decks on from the copy's 640th update

**Status:** COMPLETED

**Proposed:** 2026-10-26

**Activated:** 2026-10-26, when [A network of four decks](2026-10-24-four-decks.md)
was read and closed; design and reading fixed that morning, before anything
was built.

**Last updated:** 2026-10-26 (read: as good; closed)

The network of four decks imitated its champions for 960 updates before its
PPO, 240 a deck, as specialist distillation's copy had. Along that copy its
mirrors came within about a point of their teachers by 640 updates, Cyber's
within 3.1. A copy of 160 updates a deck would save a third of its time:
1.4 hours at four decks, some 5 at fourteen. This measures what the shorter
copy costs at the end of PPO.

## Design (fixed on 2026-10-26)

- **One arm (`short`)**: PPO from the four-deck copy's checkpoint after 640
  updates (`runs/four-decks/taught`), as the uniform arm was made from the
  copy's last: 2,880 updates at 1e-4 with `fast`'s settings, seed 2012, a
  checkpoint every 240; of an update's 520 duels, 260 against itself on the
  16 ordered pairings of its four decks as often as any other, and 260
  against the thirteen decks' pilots, 5 in each pairing.
- **Measured as the uniform arm was**: its last three checkpoints on every
  ordered pairing against `all71`, 1,024 games each
  (`eval_suites/four-decks-pairings-checkpoint.json`); every checkpoint on
  128 games of each; its last checkpoint head to head against each champion
  in its mirror (`experiments/commands/four-decks.sh`, `PARTS` short and
  short-measure; `tools/shorter_copy_report.py`). The uniform arm's
  measurements of 2026-10-26 stand as they are.
- On the CUDA machine's queue at high priority: about 5.3 hours, and 1 to
  measure.

## Reading (fixed on 2026-10-26, before anything was built)

1. *The shorter copy costs nothing*: the short arm against the uniform arm,
   by their checkpoints after 2,400, 2,640 and 2,880 updates of PPO, by the
   mean over the ten pairings of the learner's share against `all71` (a
   mirror from both seats, two decks the mean of their two seatings), the
   deals paired: **as good** when the two are within 1 point of each other
   by two standard errors, **worse** when the short arm is at least 1 point
   under by two standard errors, otherwise **not decided**.

The short arm's checkpoints after 2,400 and 2,640 updates were never kept.
The trainer numbers a run's updates on from the checkpoint it goes on from,
here the copy's 640th, and keeps a checkpoint where that number closes 240,
and the last: the short arm's after 80, 320 and so on to 2,720 updates, and
after 2,880. Decided on 2026-10-26, after the arm was made and before its
pairings were played: it is read by its last three checkpoints, after 2,480,
2,720 and 2,880 updates, beside the uniform arm's after 2,400, 2,640 and
2,880, the mark unchanged. Its three are 53 updates later on average; along
its last 1,700 updates the uniform arm's curve stays within its noise (61.4%
to 64.2% on 128 games of each pairing), which makes the shift worth well
under a tenth of a point. Its curve begins at the copy's checkpoint after
640, on the same games.

Told beside it, without a mark: each pairing in both arms; each arm's curve
along its updates; each arm's last checkpoint head to head against each
champion in its mirror; the hours.

## Results (2026-10-26)

The arm made on the CUDA machine's queue from 09:27 to 14:46 on 2026-10-26
at `0f789d1a`, into `runs/four-decks/short`; measured at `9b507e8c` by
17:27, into `analysis/four-decks/network` (`short-report.txt`). The first
measurement, at `0f789d1a`, stopped where it asked for the checkpoint after
2,400 updates; the games it had played against the champions are kept. No
duel failed.

1. **The shorter copy costs nothing: as good.** By the mean over the ten
   pairings against `all71`, the short arm 62.8% and the uniform arm 63.0%,
   −0.2 ± 0.3 points, the deals paired (−0.8 to +0.3 by two standard
   errors).

Beside it, without a mark:

- **Every pairing against `all71`**, by each arm's last three checkpoints:

  | Pairing | From 640 | From 960 (uniform) |
  |---|---|---|
  | Blackwing mirror | 66.7% | 65.4% |
  | Blackwing and Cyber | 61.7% | 63.2% |
  | Blackwing and Lightsworn | 57.1% | 58.1% |
  | Blackwing and Infernity | 60.6% | 61.6% |
  | Cyber mirror | 81.1% | 79.7% |
  | Cyber and Lightsworn | 57.8% | 57.2% |
  | Cyber and Infernity | 62.9% | 62.8% |
  | Lightsworn mirror | 55.6% | 56.4% |
  | Lightsworn and Infernity | 60.8% | 60.7% |
  | Infernity mirror | 64.2% | 65.5% |

- **Head to head against each champion in its mirror**, by each arm's last
  checkpoint, the deals paired: Blackwing 50.3% against the uniform arm's
  50.1% (+0.2 ± 0.9), Cyber 50.3% against 52.7% (−2.4 ± 1.0), Lightsworn
  49.8% against 49.5% (+0.3 ± 0.7), Infernity 49.1% against 48.1% (+1.0 ±
  0.8). Against `all71` the Cyber mirror goes the other way (81.1% against
  79.7%).
- **Along the updates**, the mean over the pairings against `all71` on 128
  games of each: the short arm from its copy's 56.6% (the uniform arm's
  copy 58.1%) to 60.7% after 320 updates, then between 58.9% and 62.9%,
  62.9% after 2,880; the uniform arm between 61.4% and 64.2% from 720
  updates, 64.2% after 2,880.
- **The training games against the pilots**, sampled and so only a
  diagnostic: the short arm 0.5 to 1.1 points under the uniform arm in each
  stretch of 480 updates (some 125,000 games each), 73.6% against 74.5% in
  the last.
- **Hours:** the copy to 640 updates 2.85 of its 4.27, 1.42 saved; the
  short arm's PPO 5.32, the uniform arm's 5.26.

What it says: a copy of 160 updates a deck rather than 240 costs the network
of four decks nothing measurable against `all71` after the same PPO, and
saves a third of the copy. The short arm begins 1.5 points under and is
level by the end. Head to head against Cyber's champion and in the sampled
games against the pilots it is behind by small amounts.

## Beside it: luna's queue

Behind the arm, at low priority, champions of ten more decks by the recipe,
the strongest of the policies' tier list outside the thirteen: Claudi-oh's
Countdown, Claudi-oh's Verdict, Astra's Exodia, Machina Gadgets, Between
Light and Dark, D.D. Border Patrol, Legendary Six Samurai, Gravekeeper's
Tomb, Scrap Renaissance and Heaven's Dispatch (`runs/champions`,
`analysis/champions`, their mirror suites `eval_suites/DECK-against-checkpoint.json`).
Each is read by the rule of a champion of the four decks' record, without a
mark: they are the teachers of networks of more decks. Countdown's duels run
18 to 28 turns, about twice Lightsworn's, and its run may take nearer 3 hours.

## Closed (2026-10-26)

Closed that evening, once read: a copy of 160 updates a deck is the copy of
the networks of more decks. The next experiment,
[A network of fourteen decks](2026-10-26-fourteen-decks.md), takes it.
