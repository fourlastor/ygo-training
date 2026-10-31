# A network of the seventy-one decks: their champions copied, then PPO

**Status:** ACTIVE

**Proposed:** 2026-10-28

**Activated:** 2026-10-28, when [A network of fourteen decks](2026-10-26-fourteen-decks.md)
was read and closed; the copy's design and readings fixed that morning and
PPO's that afternoon, before anything was built.

**Last updated:** 2026-10-31 (PPO stopped by its checks after 5,760 updates:
the mirrors head to head 44.6%, 2.5 ± 0.4 points under the copy, every ordered
pairing against `all71` 50.6%, 5.0 ± 0.4 points over it; the mirrors along the
runs read the same day: `quarter` never fell by the rule, and PPO's fall is
`selfplay`'s, so that it is not taken on)

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

## PPO: stopped by its checks (2026-10-31)

PPO was decided on 2026-10-30 once the copy's reading was seen, and ran on the
CUDA machine from that evening at 22:48 to 2026-10-31 14:28, 9.5 seconds an
update and twelve minutes a check, with no update holding a failed duel
(`analysis/seventy-one-champions/checks/report.txt`). The rule was met after
5,760 updates, by the mirrors, at the second check. Its checkpoints are kept,
the last `model-17119.pt`. Its four readings, which ask for the last
checkpoint of the whole run, were not read, and its measurements were
cancelled.

| On the check's deals | The copy | After 2,880 updates | After 5,760 updates |
| --- | --- | --- | --- |
| The mirrors head to head against the teachers | 47.1% | 44.9% (−2.2 ± 0.4 points) | 44.6% (−2.5 ± 0.4 points) |
| Every ordered pairing against `all71` | 45.7% | 50.1% (+4.4 ± 0.3 points) | 50.6% (+5.0 ± 0.4 points) |

Beside the rule:

- Told by how many of a pairing's two decks the copy met, both seats of each,
  at the copy and the two checks: neither 44.2%, 49.5%, 50.5% (± 0.4); one
  47.8%, 50.9%, 50.6% (± 0.6); both 50.3%, 51.8%, 50.8% (± 2.0); the mirrors
  against `all71` 57.0%, 59.9%, 57.4% (± 2.9).
- Each deck from both seats on the check's 564 games, by reading 2's rule: the
  copy none over 50% and thirty under, the lowest `deckout` 29.1%; after 2,880
  updates two over and three under, the lowest `chain-burn` 42.2%; after 5,760
  one over and none under, the lowest `countdown` and `deckout` 47.0%.
- Against `all71` PPO took most of its gain in the first stretch, where
  the decks the copy never met gained most, and by the second every group of
  pairings stood about 50.5%. The second stretch moved neither the mirrors nor
  the pairings by more than a check can tell.
- On the fourteen decks of the earlier network the mirrors fell 4.3 points on
  128 deals, where over its 4,320 updates that network's `selfplay` fell 3.5 ±
  0.2 and `quarter` gained 0.7 ± 0.2. The mirrors' share of the duels was
  `quarter`'s, but not their duels an update: each mirror had 2 here, 2.65 in
  `selfplay`, 9.3 in `quarter` and 18.6 in `half`, and half of the duels here
  would give it 4 (drawn from the allocation). Whether that or the number of
  duels decides is open: each mirror had played 11,520 duels when PPO stopped,
  of the 43,800 reckoned, `selfplay` 11,400 by its end and `quarter` some
  40,100.

### The mirrors along the runs (fixed on 2026-10-31, before any of it was played)

The design's next to try is half of the duels in the mirrors (each mirror 70,
each other pairing 1). No head to head was measured along the fourteen decks'
arms, so that it is not known whether `quarter` fell and came back or never
fell. Decided first, on checkpoints already kept and with no training
(`PARTS=along`, seven at once, about an hour and a half by reckoning): each of
`quarter`, `selfplay` and `half` after 160, 640, 1,120 and 2,080 of its 4,320
updates, head to head against each of the fourteen teachers on the 2,048 deals
of its reading, less the fourteen decks' copy deal by deal; and PPO after 160,
640, 1,360 and 2,080 updates on the checks' deals, less the copy's.

- Of `quarter`: **it fell and came back** when at any of the four points it is
  under its copy by at least 1 point by two standard errors, the line of PPO's
  rule; **it never fell** when it is at none.
- Fell and came back: the stop says little of where PPO would have ended, and
  PPO is taken on from its last checkpoint. Never fell: the fall here is not
  one `quarter` had and got over, half of the duels (4 of each mirror an update)
  may not be enough, and a lower rate or a wider network is weighed against it.
- Told beside them, without a mark: how soon PPO and `selfplay` fell.

A lower rate comes in because an update of 520 duels in the seventy-one decks'
mix was about 2.5% signal ([The signal of an update](2026-10-22-update-signal.md):
a noise scale of 20,018 ± 6,347 duels on `all71`, 758 ± 96 on one matchup), and
a fall that comes early and then stands is what noise against a weak pull would
look like, which has not been shown. More duels an update would not change that
by themselves, an update being cut into some 160 minibatches of 512 decisions.

#### Read on 2026-10-31

Played on the CUDA machine from 14:41 to 15:55, 904 files with no failed case
(`analysis/seventy-one-champions/along/report.txt`). Head to head, in points
against each run's own copy on the same deals, each run with its duels of each
mirror an update:

| After updates | `half` (18.6) | `quarter` (9.3) | `selfplay` (2.65) | PPO, the fourteen decks (2) | PPO, the seventy-one (2) |
| --- | --- | --- | --- | --- | --- |
| 160 | −0.0 ± 0.2 | −0.9 ± 0.2 | −2.3 ± 0.2 | −0.5 ± 0.8 | −1.2 ± 0.3 |
| 640 | +0.3 ± 0.2 | −1.2 ± 0.2 | −2.9 ± 0.2 | −2.7 ± 0.8 | −1.9 ± 0.4 |
| 1,120 | +0.6 ± 0.2 | −1.0 ± 0.2 | −2.9 ± 0.2 | | |
| 1,360 | | | | −3.7 ± 0.9 | −1.6 ± 0.4 |
| 2,080 | +1.4 ± 0.2 | −0.7 ± 0.2 | −2.8 ± 0.2 | −3.2 ± 0.8 | −2.4 ± 0.4 |
| 2,880 | | | | −4.5 ± 0.9 | −2.2 ± 0.4 |
| 4,320 | +2.7 ± 0.2 | +0.7 ± 0.2 | −3.5 ± 0.2 | | |
| 5,760 | | | | −4.3 ± 0.9 | −2.5 ± 0.4 |

- **`quarter` never fell**, by the rule: at its lowest, after 640 updates, it is
  0.8 points under by two standard errors, where the line is 1. It did stand
  about a point under for its first 2,080 updates, and only its last 2,240 took
  it over the copy.
- That is not PPO's fall. On the same fourteen decks PPO stood 3.2 to 4.5 points
  under from 1,360 updates on, and its curve is `selfplay`'s, which lost 2.3 of
  its 3.5 points in its first 160 updates and did not come back in 4,320. PPO is
  not taken on from its last checkpoint.
- PPO fell soon, as `selfplay` did: over the seventy-one decks 1.2 points after
  160 updates and 1.9 after 640. From 2,080 updates on its gap moves by no more
  than a check can tell.
- The more duels of each mirror, the higher the arm at every point. After 640
  updates a straight line in the logarithm of the duels through `selfplay` and
  `quarter` puts 4 of each mirror, which half of the duels would give here, at
  about −2.4 on the fourteen decks, some 0.6 points over `selfplay`: half of the
  duels alone is not reckoned to hold them. That line is a reckoning, not a
  measurement.

A lower rate is the next to try, agreed on 2026-10-31 once this was read. It
tests whether the noise of an update is what costs the mirrors, and it is
expected to want more updates for the same gain. Its design and readings are
to be fixed before it is built.

### The mean of PPO's weights (fixed on 2026-10-31, before any of it was played)

Agreed once the mirrors along the runs were read, before a lower rate is
designed. At a fifth of the rate, if gain goes with rate and updates, a run
would take some 14,400 updates, 38 hours, to gain what PPO's first 2,880 did,
and part of what it asks can be asked of checkpoints already kept. PPO's mirrors
stand about where they stood from 2,080 updates on, so that what its later
checkpoints share is where the run stands, and what differs between them is in
good part the noise of their updates. The mean of their weights keeps the first
and little of the second, with no training.

The network is the mean, weight by weight, of the thirteen checkpoints PPO kept
every 240 updates from 2,800 to 5,680, `model-14159.pt` to `model-17039.pt`
(`tools/mean_checkpoint.py`, `PARTS=mean`, seven at once, some fifteen minutes by
reckoning). It is measured as a check is, on the checks' deals, less PPO's last
checkpoint's on the same deals.

- The mirrors head to head are **raised** when at least 1 point over PPO's last
  checkpoint by two standard errors and **not raised** when at most 1 point over
  it by two standard errors. Between the two they are **not told**, and the mean
  and PPO's last checkpoint are played again on 512 deals of each mirror, about
  an hour and a half.
- Where its pairings against `all71` are under PPO's last checkpoint's by at
  least 1 point by two standard errors, the mean has broken the network, the
  checkpoints being too far apart to be meaned. The test then says nothing and a
  lower rate goes ahead as agreed.
- Raised: the noise of the updates is costing the mirrors. A run at PPO's rate
  that keeps a running mean of its weights is designed beside the lower rate,
  since it would not want more updates.
- Not raised: the noise about where the run stands costs less than a point of
  the 2.5. A lower rate is then a long shot, not ruled out, since noise can steer
  where a run goes as well as shake it where it stands, and it is weighed
  against a wider network.

**Current next step:** the mean of PPO's weights, then the design of what it
calls for.
