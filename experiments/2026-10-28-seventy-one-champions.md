# A network of the seventy-one decks: their champions copied, then PPO

**Status:** ACTIVE

**Proposed:** 2026-10-28

**Activated:** 2026-10-28, when [A network of fourteen decks](2026-10-26-fourteen-decks.md)
was read and closed; the copy's design and readings fixed that morning and
PPO's that afternoon, before anything was built.

**Last updated:** 2026-10-30 (the copy read: reading 1 lost, 46.9% ± 0.1%
head to head; 60.2% against `all71` in the mirrors, 45.7% ± 0.4% over every
ordered pairing; PPO's reading 2 amended, its reading 4 added, PPO queued)

The goal since 2026-10-24 is one network of all seventy-one decks that plays
each deck's mirror about as well as its champion and beats `all71` on every
pairing. In [a network of fourteen decks](2026-10-26-fourteen-decks.md) the
champions' copy kept the mirrors (48.4% ± 0.1% head to head against each
champion, the lowest 47.1%), and PPO with every duel against itself lost
them (44.9%) while it gained on the other pairings; of two arms that gave
the mirrors a share of PPO's duels, read on 2026-10-28, a quarter kept
them at no cost to the other pairings. A run of the
champions' recipe was made for every deck of the seventy-one, the last on
2026-10-29, and each run's copy games were recorded once
(`tools/record_champion.py`: in its mirror and against each of the thirteen
decks' pilots, kept as the answers it gave), so that a copy plays them again
with no teacher asked (`train --copy-records`).

The copy does not depend on what PPO is found to need. It is made as soon
as the last recording is: agreed on 2026-10-28 that it would be even while
the fourteen decks' record was still active, so that no time was lost
waiting for it, and that record was closed the same afternoon.

## Design (the copy's fixed on 2026-10-28, before anything was built)

- **The decks and their teachers**: the seventy-one decks of the pool, each
  taught by its run's last checkpoint: the fourteen of
  [the champions' release](../models/champions.md), Countdown and Verdict
  (made on 2026-10-26), and the fifty-five made from 2026-10-27 to
  2026-10-29 on the CUDA machine, a lent machine and a rented one
  (`runs/champions/DECK` on the CUDA machine). A run is a champion when it
  is over 52% by two standard errors against `all71` in its mirror. One
  that is not is copied all the same where it is not under `all71` by two
  standard errors (Tele-DAD, 51.0% ± 0.5%, and Gem-Knight, 50.3% ± 0.5%,
  not told apart; the other sixty-five are champions); one under it
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
  checkpoint kept every 142 updates, two a deck, so that each point
  measured below is one: every 160 at first, refused at the copy's start
  on 2026-10-29 by the command's own check, 160 dividing none of 2,840,
  5,680 and 8,520. The trainer saves every update and keeps every 142nd
  and the latest, so the step changes no game and no update. At the
  commit whose engine keeps each thread's last duel arena for its next
  duel: the same games as the engine the recordings were made with,
  played again and held to their digests on 2026-10-28.
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
`all71`. Added on 2026-10-29, while the copy was made: the copy as PPO's
checks measure it (below), against `all71` on every ordered pairing and
head to head in the mirrors, before PPO is decided on. The copy has played
its mirrors and the thirteen decks' pilots alone, and is not expected to
hold the other pairings: a network of fourteen decks lost its whole lead
over `all71` against the decks it had never met (+5.0 ± 0.4 points
against the thirteen decks' pilots, −0.2 ± 0.3 against the other
fifty-seven's; `tools/attack_targets.py`, measured on 2026-10-29).

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
  and both stayed 4 to 6 points over `all71` against them. An update's
  duels are whole, and those a share leaves over go to the largest
  remainders: read from the allocation on 2026-10-30, before PPO began,
  each mirror has 2 duels an update and each other pairing 0.076, in a
  different draw each update, so that 27.3% of the duels are in the mirrors
  and not the quarter reckoned (some 43,800 duels of each mirror and 1,670
  of each other pairing, against 40,100 and 1,720). The weights stand.
- **The checks**, so that a run that gets worse is not left to run for
  days: PPO is trained in stretches to 2,880, 5,760, 8,640, 11,520,
  14,400, 17,280 and 20,160 updates, some six hours each at the fourteen
  decks' pace, and after each its last checkpoint is measured, the
  training paused, on the same deals every time: head to head against
  each teacher in its mirror, 128 deals from both seats (18,176 games),
  and against `all71` on every ordered pairing, 4 games each (20,164
  games), one case a deck against all seventy-one
  (`eval_suites/seventy-one-decks-check-pairings-checkpoint.json`). The
  copy's last checkpoint is measured once on the same deals before PPO
  begins: since 2026-10-29 a part of its own (`PARTS=copy-check`), queued
  behind the copy's measurements, which PPO's part finds made. Head to
  head, because against `all71` the fourteen decks'
  selfplay arm held its mirrors where the copy had them while it lost them
  head to head against the champions. Some ten minutes a check by
  reckoning, seven at once. The cases of a suite go one after the other,
  each waiting for its slowest duel, some 5 seconds a case on the CUDA
  machine seven at once, and 5,041 cases of one pairing each would take
  about an hour a check: since 2026-10-28 a case may name several
  opponents, its games all played at once, and on the processor such a
  case plays the same games as each opponent alone.
- **The rule**: PPO stops when the mean over the mirrors head to head, or
  the mean over the pairings against `all71`, is under the copy's by at
  least 1 point by two standard errors, the deals paired, the same of the
  two at two checks in a row (`tools/seventy_one_checks.py`). Its
  checkpoints are kept, and a run so stopped can be taken on from its last.
  Where it stops for the mirrors, half of its duels in the mirrors (each
  mirror 70, each other pairing 1) is the next to try.
- **Measured** at its end (`PARTS=ppo-measure`, seven at once, about an
  hour by reckoning): its last checkpoint head to head against each
  teacher in its mirror, 2,048 deals from both seats; it and the copy
  against `all71` on every ordered pairing, 64 games each (322,624 games;
  `eval_suites/seventy-one-decks-pairings-quick-checkpoint-1.json` to
  `-7.json`, one case a deck against all seventy-one, one suite's deals in
  seven parts that go at once); it against the thirteen decks' pilots, 128
  games each, beside the copy and `all71`.
- The copy is not checked so: it imitates its teachers' recorded answers.

### Readings (fixed on 2026-10-28, before anything of PPO was built)

By PPO's last checkpoint:

1. *The mirrors kept*: the copy's reading 1.
2. *Against `all71`, deck by deck*: each deck's share from both seats, the
   network playing it against `all71` playing each of the seventy-one decks
   and playing each of the others against `all71` playing it (9,024 games a
   deck): **beats all71** when every deck is over 50% by two standard
   errors, **falls short** when one is at most 50% by two standard errors,
   otherwise **not decided**. Every pairing on its own, as the fourteen
   decks were read, would take some 15 million games. Amended on
   2026-10-30, before PPO began: fixed as each deck in the learner's seat
   alone (4,544 games a deck), it told the deck more than the network (the
   copy's results, below). On the same games.
3. *PPO against the copy*: the mean over the ordered pairings against
   `all71`, the deals paired: **pays** when PPO is at least 1 point over
   the copy by two standard errors, **does not pay** when it is at most 1
   point over it by two standard errors, otherwise **not decided**.
4. *The mirrors against the copy*: the mean over the mirrors head to head
   against each teacher, the games of reading 1, the deals paired with the
   copy's: **raised** when PPO is at least 1 point over the copy by two
   standard errors, **not raised** when it is at most 1 point over it by two
   standard errors, otherwise **not decided**. Added on 2026-10-30, with PPO
   begun and before its first check, on games already played for reading 1.
   Reading 1 asks of PPO some 1.3 points over the copy where the fourteen
   decks' `quarter` gained 0.7 over theirs, so it can be lost with PPO
   closing the gap, and this tells the two apart. Raised, the network holds
   better play of the mirrors than the copy reached, so that its size was
   not what stopped the copy there, and more PPO or half of the duels in the
   mirrors is next to try. Not raised says nothing of its size, and a wider
   network is next to try.

Told beside them, without a mark: each check, and its gain over the copy
against `all71`; each mirror less the copy's; the mirrors against `all71`
beside the copy's; against the pilots beside the copy and `all71`; the
hours.

## The copy: results (read on 2026-10-30)

The copy took 35.3 hours, from 2026-10-29 05:03 to 2026-10-30 16:25, with no
update holding a failed duel (`analysis/seventy-one-champions/report.txt` and
`checks/report.txt` on the CUDA machine). Its measurements took two and a half
hours at seven at once. Five of their 712 files, four of Cloudian's points
against `all71` and one of Exodia's, ran out of the card's memory, and were
played again two at a time the same evening; no other file was touched, none
of the five head to head. `evaluate-suite` goes on past a case that fails, so
the command now plays such a file again and fails its run by name (commit
5fa832b6). Two evaluations of Cloudian's mirror alone hold 11.8 of the card's
12.3 GB, which seven at once may meet again in `ppo-measure`.

1. **The teachers: lost.** The copy's last checkpoint head to head against
   each teacher in its mirror, 2,048 deals from both seats: a mean of 46.9%
   ± 0.1%, under the mark of 48%, and five mirrors at most 45% by two
   standard errors: `destiny-hero` 42.3% ± 0.7%, `zombie` 42.5%, `deckout`
   43.3%, `pyramid` 43.6%, `arcana` 43.7%. The highest `countdown` 50.0%,
   `destiny-board` 49.7%, `exodia` 49.2%.

Beside it:

- The same fourteen decks and teachers, deal by deal: 46.4% where the
  fourteen decks' copy had 48.4%, −2.0 ± 0.2 points, from −0.2 (Lightsworn)
  to −3.1 (Burn), at the same 160 updates a deck.
- The teachers' answers in the games imitated, on those fourteen decks at
  the same updates a deck (each copy's metrics): 95.4% against the fourteen
  decks' copy's 96.1% at 20 to 40 updates a deck, 96.2% against 97.4% at 140
  to 160. The gap is there early and does not close: over the last 20 updates
  a deck this copy gained 0.04 points and that one 0.07. The same network
  fits each of seventy-one decks less well than each of fourteen, on the same
  games of each; what limits it, its size or a rate held at 3e-4 through five
  times the steps, was not measured.
- The stronger the teacher, the more of it lost: over the seventy-one decks,
  head to head against the teacher's own figure against `all71`, r = −0.53.
  The third of the teachers strongest against `all71` (68.5%) 45.8%, the
  weakest third (56.1%) 47.7%.
- Against `all71` in each mirror along the copy: 58.6% after 2,840 updates,
  59.5% after 5,680, 60.2% after 8,520 and after 11,360, so nothing gained in
  its last 40 updates a deck. After 11,360, 1.9 ± 0.1 points under the
  teachers' figures, from +0.9 (`gemini`) to −4.3 (`arcana`, `herald`,
  `pyramid`). Sixty-nine of the seventy-one over 50%; Gem-Knight 47.9%
  under it by two standard errors, Tele-DAD 48.9% not told apart.
- Against the thirteen decks' pilots 59.0%, `all71` 59.7% on the same games:
  −0.7 ± 0.1 points.
- The copy on PPO's checks' deals (`checks/copy`), what PPO will be read
  against: head to head 47.1%; against `all71` over all 5,041 ordered
  pairings, 4 games each, 45.7% ± 0.4%. By how many of a pairing's two decks
  are among the thirteen whose pilots the copy played, both seats of each, so
  that what a deck is worth cancels (`tools/seventy_one_checks.py pairings`):

  | Pairing | Against `all71` | Games |
  |---|---|---|
  | A mirror | 57.0% ± 2.9% | 284 |
  | Both decks met | 50.3% ± 2.0% | 624 |
  | One met | 47.8% ± 0.6% | 6,032 |
  | Neither met | 44.2% ± 0.4% | 13,224 |

  The fourteen decks' copy stood at 51.0% on its other pairings, every deck
  of which it had met.
- Each deck over its pairings from both seats, 564 games a deck: none over
  50% by two standard errors, thirty under, from 29.1% (`deckout`) to 52.1%
  (Crystal Beasts). The four lowest win by other means than battle: `deckout`
  29.1%, `chain-burn` 37.2%, `countdown` 37.8%, `exodia` 38.7%.

PPO's reading 2 as fixed on 2026-10-28 reads each deck from the learner's seat
alone, and the check shows that this reads the deck more than the network:
`arcana` 10.6%, with which `all71` itself wins 24.3% against the copy, and
twenty-four decks under 50% by two standard errors, twenty over. No network
near `all71` could meet it. The fourteen decks' reading 2 took the mean of a
pairing's two seatings, and that was lost when the reading was cut to deck by
deck. It was amended on 2026-10-30, before PPO began, to each deck from both
seats: level with `all71`, a deck and the deck it meets win between the two
seatings of their pairing as much as they lose, whatever each is worth. The
games played do not change.

**Current next step:** PPO, decided on 2026-10-30 once this reading was seen,
is queued on the CUDA machine that evening, its measurements behind it. A check
tells a gap of about a point from none, two standard errors of a gap being 0.8
points in the mirrors and 1.0 over the pairings, and the fourteen decks'
`quarter` gained 0.7 in the mirrors over its whole length, so that a check
without a gap does not say the mirrors stand still.
