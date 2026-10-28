# A network of the seventy-one decks: their champions copied, then PPO

**Status:** FUTURE

**Proposed:** 2026-10-28

**Last updated:** 2026-10-28 (the copy's design and readings fixed that
day, while the fourteen decks' mirror arms were measured, before anything
was built, and the copy to be made before this record is active; PPO's
fixed that afternoon, once those arms were read, and built)

The goal since 2026-10-24 is one network of all seventy-one decks that plays
each deck's mirror about as well as its champion and beats `all71` on every
pairing. In [a network of fourteen decks](2026-10-26-fourteen-decks.md) the
champions' copy kept the mirrors (48.4% ± 0.1% head to head against each
champion, the lowest 47.1%), and PPO with every duel against itself lost
them (44.9%) while it gained on the other pairings; of two arms that gave
the mirrors a share of PPO's duels, read on 2026-10-28, a quarter kept
them at no cost to the other pairings. A run of the
champions' recipe is being made for every deck of the seventy-one, the last
expected on 2026-10-29, and each run's copy games are recorded once
(`tools/record_champion.py`: in its mirror and against each of the thirteen
decks' pilots, kept as the answers it gave), so that a copy plays them again
with no teacher asked (`train --copy-records`).

The copy does not depend on what PPO is found to need. It is made as soon
as the last recording is, while the fourteen decks' record is still active
if it is: agreed on 2026-10-28, so that no time is lost waiting for it.

## Design (the copy's fixed on 2026-10-28, before anything was built)

- **The decks and their teachers**: the seventy-one decks of the pool, each
  taught by its run's last checkpoint: the fourteen of
  [the champions' release](../models/champions.md), Countdown and Verdict
  (made on 2026-10-26), and the fifty-five being made since 2026-10-27 on
  the CUDA machine, a lent machine and a rented one (`runs/champions/DECK`
  on the CUDA machine). A run is a champion when it is over 52% by two
  standard errors against `all71` in its mirror. One
  that is not is copied all the same where it is not under `all71` by two
  standard errors (Tele-DAD: 51.0% ± 0.5%, not told apart); one under it
  would be taught by `all71` instead, its games recorded the same way.
- **The copy (`taught`)**: one network of the seventy-one decks from
  scratch, seed 2012, each deck's games its teacher's recorded games played
  again with its answers, and imitated. 11,360 updates of 520 duels, 160 a
  deck, the length the shorter copy found as good as 240. That is longer
  than the 4,800 updates a copy was held to on 2026-10-20: played again
  from its recordings, an update costs a fraction of a copy that asks its
  teachers, and the full length was agreed on 2026-10-28. At the fourteen
  decks' copy's settings: 3e-4, in minibatches of 256 with no duels in
  flight, one pass, the value taught at four positions a game; of an
  update's duels 260 in the mirrors, both seats imitated, and 260 against
  the thirteen decks' pilots. The copy takes 41,600 of each deck's 44,000
  recorded mirror games and 3,200 of the 4,000 against each pilot. A
  checkpoint every 160 updates. At the commit whose engine keeps each
  thread's last duel arena for its next duel: the same games as the
  engine the recordings were made with, played again and held to their
  digests on 2026-10-28.
- **Measured** (`experiments/commands/seventy-one-champions.sh`, `PARTS`
  copy and copy-measure, the teachers held to the rule above by
  `tools/seventy_one_teachers.py` before the copy;
  `tools/seventy_one_champions_report.py`), on the CUDA machine, several at
  once through MPS: the copy's last checkpoint head to head against each
  teacher in its mirror, 2,048
  deals from both seats; it and its checkpoints after 40, 80 and 120 updates
  a deck against `all71` in each mirror, 2,048 deals from both seats, beside
  each teacher's own figure from its making; it against the thirteen decks'
  pilots, each deck against each, 128 games
  (`eval_suites/seventy-one-decks-against-pilots.json`), beside `all71` on
  the same games.
- By reckoning: some 20 to 30 hours for the copy (the games played again at
  about 3.5 seconds an update on the development machine, the step 2 to 3
  on the CUDA machine), 2 to 3 to measure.

## Readings (the copy's fixed on 2026-10-28, before anything was built)

1. *The copy keeps the teachers*: the copy's last checkpoint head to head
   against each teacher in its mirror, 2,048 deals from both seats:
   **kept** when the mean over the seventy-one mirrors is at least 48% by
   two standard errors and no mirror is at most 45% by two standard errors;
   **lost** when the mean is at most 48% by two standard errors or a mirror
   at most 45%; otherwise **not decided**. The fourteen decks' rule for
   their arm, which their copy met.

Told beside it, without a mark: each mirror against `all71` less its
teacher's figure; the copy along its updates; against the pilots beside
`all71`.

## PPO (design and readings fixed on 2026-10-28, once the fourteen decks' mirror arms were read, before anything was built)

In the fourteen decks a quarter of PPO's duels in the mirrors kept them,
and the other pairings gained as much as when the duels were spread
evenly; half took the mirrors further for 1.3 points of the other
pairings. The seventy-one decks' PPO takes the quarter.

- **Design**: from the copy's last checkpoint, 21,900 updates of 520 duels
  at 1e-4 with the fourteen decks' arms' settings, seed 2012, every duel
  against itself and none against a pilot, a quarter of them in the
  mirrors (`--self-play-weights`, each mirror 70 and each other ordered
  pairing 3: some 1.8 duels of each mirror an update and 0.08 of each
  other pairing); a checkpoint every 240 updates. At a fixed share of
  mirrors each deck's mirror stands to its other pairings as in the
  fourteen decks' `quarter`, one duel to six, whatever the number of
  decks; at seventy-one decks each deck has some five times fewer duels an
  update than at fourteen, and 21,900 updates are the fourteen decks'
  4,320 by the deck. Some two days by reckoning, at the fourteen decks'
  pace. Neither of the fourteen decks' arms had a duel against the pilots,
  and both stayed 4 to 6 points over `all71` against them.
- **The checks**, so that a run that gets worse is not left to run for
  days: PPO is trained in stretches to 2,880, 5,760, 8,640, 11,520,
  14,400, 17,280 and 20,160 updates, some six hours each at the fourteen
  decks' pace, and after each its last checkpoint is measured, the
  training paused, on the same deals every time: head to head against
  each teacher in its mirror, 128 deals from both seats (18,176 games),
  and against `all71` on 71 of the ordered pairings, each deck against the
  deck after it in the list and the last against the first, 256 games each
  (18,176 games; `eval_suites/seventy-one-decks-check-pairings-checkpoint.json`).
  The copy's last checkpoint is measured once on the same deals before PPO
  begins. Head to head, because against `all71` the fourteen decks'
  selfplay arm held its mirrors where the copy had them while it lost them
  head to head against the champions. Every ordered pairing at 4 games
  each, as first agreed, was dropped before anything was built: the
  evaluator plays a suite's cases one after the other, each waiting for
  its slowest duel, some 5 seconds a case on the CUDA machine seven at
  once, and 5,041 cases would take about an hour a check.
- **The rule**: PPO stops when the mean over the mirrors head to head, or
  the mean over the pairings against `all71`, is under the copy's by at
  least 1 point by two standard errors, the deals paired, the same of the
  two at two checks in a row (`tools/seventy_one_checks.py`). Its
  checkpoints are kept, and a run so stopped can be taken on from its last.
  Where it stops for the mirrors, half of its duels in the mirrors (each
  mirror 70, each other pairing 1) is the next to try.
- **Measured** at its end (`PARTS=ppo-measure`, seven at once, some three
  hours by reckoning): its last checkpoint head to head against each
  teacher in its mirror, 2,048 deals from both seats; it and the copy
  against `all71` on every ordered pairing, 64 games each (322,624 games;
  `eval_suites/seventy-one-decks-pairings-quick-checkpoint-1.json` to
  `-7.json`, one suite's deals in seven parts by the learner's deck that go
  at once); it against the thirteen decks' pilots, 128 games each, beside
  the copy and `all71`.
- The copy is not checked so: it imitates its teachers' recorded answers.

### Readings (fixed on 2026-10-28, before anything of PPO was built)

By PPO's last checkpoint:

1. *The mirrors kept*: the copy's reading 1.
2. *Against `all71`, deck by deck*: each deck's share, played by the
   network against `all71` playing each of the seventy-one decks (4,544
   games a deck): **beats all71** when every deck is over 50% by two
   standard errors, **falls short** when one is at most 50% by two
   standard errors, otherwise **not decided**. Every pairing on its own, as
   the fourteen decks were read, would take some 15 million games.
3. *PPO against the copy*: the mean over the ordered pairings against
   `all71`, the deals paired: **pays** when PPO is at least 1 point over
   the copy by two standard errors, **does not pay** when it is at most 1
   point over it by two standard errors, otherwise **not decided**.

Told beside them, without a mark: each check, and its gain over the copy
against `all71`; the mirrors against `all71` beside the copy's; against
the pilots beside the copy and `all71`; the hours.
