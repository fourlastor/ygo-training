# A shorter copy: four decks on from the copy's 640th update

**Status:** ACTIVE

**Proposed:** 2026-10-26

**Activated:** 2026-10-26, when [A network of four decks](2026-10-24-four-decks.md)
was read and closed; design and reading fixed that morning, before anything
was built.

**Last updated:** 2026-10-26 (the arm made; read by the checkpoints it keeps)

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
