# A network of the seventy-one decks: their champions copied, then PPO

**Status:** FUTURE

**Proposed:** 2026-10-28

**Last updated:** 2026-10-28 (the copy's design and readings fixed that
day, while the fourteen decks' mirror arms were measured, before anything
was built; PPO's to be drawn once those arms are read)

The goal since 2026-10-24 is one network of all seventy-one decks that plays
each deck's mirror about as well as its champion and beats `all71` on every
pairing. In [a network of fourteen decks](2026-10-26-fourteen-decks.md) the
champions' copy kept the mirrors (48.4% ± 0.1% head to head against each
champion, the lowest 47.1%), and PPO with every duel against itself lost
them (44.9%) while it gained on the other pairings; two arms that give the
mirrors a share of PPO's duels are measured on 2026-10-28. A run of the
champions' recipe is being made for every deck of the seventy-one, the last
expected on 2026-10-29, and each run's copy games are recorded once
(`tools/record_champion.py`: in its mirror and against each of the thirteen
decks' pilots, kept as the answers it gave), so that a copy plays them again
with no teacher asked (`train --copy-records`).

## Design (the copy's fixed on 2026-10-28, before anything was built)

- **The decks and their teachers**: the seventy-one decks of the pool, each
  taught by its run's last checkpoint: the fourteen of
  [the champions' release](../models/champions.md), Countdown and Verdict
  (made on 2026-10-26), and the fifty-five being made since 2026-10-27 on
  the CUDA machine, a lent machine and a rented one (`runs/champions/DECK`
  on the CUDA machine). A run is a champion when it
  is over 52% by two standard errors against `all71` in its mirror. One
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
- **Measured**, on the CUDA machine, several at once through MPS: the copy's
  last checkpoint head to head against each teacher in its mirror, 2,048
  deals from both seats; it and its checkpoints after 40, 80 and 120 updates
  a deck against `all71` in each mirror, 2,048 deals from both seats, beside
  each teacher's own figure from its making; it against the thirteen decks'
  pilots, each deck against each, 128 games, beside `all71` on the same
  games.
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

## PPO

Drawn once the fourteen decks' mirror arms are read, before anything of it
is built: the share of its duels in the mirrors, whether any are against
the pilots, its length over the 5,041 ordered pairings, and its readings
(the mirrors head to head against each teacher, and every pairing against
`all71`).
