# Training command reference

Run commands from the repository root. `python -m ygo_training <command> --help`
is the executable source of truth; this document explains what the options
mean for an experiment.

Decks are named by the id of their pilot in ygo-policies (`blackwing`,
`monarch`, ...). Every opponent is played by the pilot of its own deck.

## `train`

```sh
uv run python -m ygo_training train --device cuda \
  --run runs/example --learner blackwing \
  --opponents 'blackwing,burn,crystal,dragunity,gishki,gladiator,heroes,infernity,lightsworn,monarch,morphtronic,rock-block,spellcaster' \
  --workers 24 --duels 520 --updates 50
```

`--duels` is the total number of duels per update, the opponent decks in
equal numbers. Every game is played through the library's pool
(`--pool-threads`, see `evaluate-suite`). `--workers` names the shares an
update's duels are dealt in, each drawing the learner's seat in its duels
from a generator of its own, as the worker processes that played them until
2026-10-16 did: the duels follow from it, and every script names 24.
`--updates` is the number of updates of this invocation, also after
`--resume` or `--continue`. Without `--opponents` the learner meets every
other deck of the pool.

`--learner blackwing,monarch` names several decks for one learner: it plays
each against every opponent, in equal shares of an update's games (520
games, two decks and thirteen opponents are twenty games a pairing). The
network is told the deck it plays by its list, as it always is. In an
imitating update each game is played by the pilot of its own deck, and is
the game a learner of that deck alone has of the same deal; in the others
the model plays each deck itself. Such a run needs named `--opponents`,
and imitates neither recorded duels nor a checkpoint.
An update reports the games and the wins deck by deck
(`learner_deck_duels`, `learner_deck_win_rate`) and, when it imitates, how
often the first choice is the pilot's for each deck
(`imitation_model_teacher_agreement_by_deck`, over
`imitation_samples_by_deck` decisions).

| Setting | Default |
| --- | --- |
| learner | `blackwing` |
| workers / duels per update | 16 / 512 |
| PPO | 1 epoch (`--ppo-epochs`: one pass over an update's games), minibatch 256, learning rate 3e-4, entropy 0.01 |
| GAE lambda (`--duel-gae-lambda`) | 0.95 |
| rollout temperature | 1 |
| evaluation | the learner's screen (`eval_suites/ygo-policies-<learner>-screen.json`) every 5 updates |
| board representation | `mean-pool` |
| the model's calls in an update's games | on a card, through the pool: from CUDA graphs up to half the pool's duels, the duels in two groups (`--inference-graphs`, `--pool-groups`, `--pool-groups-over`; the section on calls from CUDA graphs and the duels in two groups); on a processor as they are |
| seed | 2011 |

### Who plays the learner's seat

- `--imitation-only-updates N`: for the run's first `N` updates the pilot of
  the learner's deck plays its seat and the learner is trained on the pilot's
  decisions. `--imitation-value-weight` also trains the value head on the
  duels' outcomes: a duel's outcome is the target at the first step of
  every transaction of the seat, or with `--imitation-value-positions K`
  at `K` of them drawn at random, and with `--imitation-value-detach` the
  head reads the shared state and sends no gradient into it. A duel has one
  outcome and some seventy transactions; shown at every one of them and
  several times over (`--ppo-epochs`), a head learns the duels by heart and
  says nothing of others. An update reports the head's explained variance
  and its correlation with the outcomes on that update's duels before it
  trains on them.
- `--imitation-records ROWS` makes those updates imitate recorded duels
  instead: see [Imitating recorded duels](#imitating-recorded-duels).
- `--imitation-teacher CHECKPOINT` makes them imitate a network instead:
  see [Imitating a checkpoint](#imitating-a-checkpoint).
- Otherwise the learner plays, with sampled answers, and is trained by PPO.
  `--teacher-imitation-weight W` adds the pilot's answers to the learner's own
  decisions as an auxiliary loss of weight `W` (an anchor);
  `--teacher-imitation-decay-updates` lets it fall linearly to zero.
  An update makes one pass over its games unless `--ppo-epochs` says more:
  from a copy of the pilot, PPO in four passes undoes the copy and in one
  it adds to it, with no anchor
  ([PPO with a head that predicts](../experiments/2026-10-09-ppo-with-a-head-that-predicts.md)).
  Four was the default until 2026-10-10; the scripts of earlier experiments
  that took it now name it.
  PPO from a copy runs at a learning rate of 1e-4 since 2026-10-11, named by
  its script (`--resume-learning-rate 1e-4`, since a copy's optimizer comes
  with the 3e-4 it imitated at): over a million games it ends 4.8 ± 0.7
  points of the held-out deals over the same run at 3e-4. The default stays
  3e-4, which the imitating runs are made with.
- `--ppo-value-positions K` makes the same choice for those updates as
  `--imitation-value-positions` for the imitating ones: the value head is
  taught at `K` transactions of each seat of a duel, drawn for the update,
  where it is otherwise taught at every one. Every transaction keeps its
  advantage, measured against the head, and its part in the policy's
  loss. What the head is taught is the return, which is the duel's outcome
  with `--duel-gae-lambda 1` and leans on the head's own estimates below
  that; `--ppo-value-target result` teaches it the duel's outcome whatever
  the advantages use, which is what the imitating updates teach it.
  An update reports `duel_outcome_explained_variance` and
  `duel_outcome_correlation`: the head against how its seat's duels ended,
  on that update's games before it trains on them, whatever the returns
  are made of.
- `--policy-loss-weight 0` with a teacher weight trains on those labels alone
  (DAgger); `--entropy-coefficient 0` is then usual.
- `--teacher-label-fraction F` keeps the label on a share `F` of the
  decisions. The choice depends on the duel and the decision alone, so the
  duels are the same whatever the share.

Imitation-only updates and a teacher weight cannot be combined in one
invocation.

A decision is something the network could decide: the zone a card goes to is
left to the pilot, and a decision with a single answer is not asked. The
imitation losses and PPO count a command together with the selections it
leads to (an attack and its target) as one action.

A command of the Main or the Battle Phase that the network's seat gave at
the same position earlier in the turn is not offered there again (since
2026-10-15; `Trail::offered` of `native/ygo-model-input/seat`, so here and
at a table alike), nor since 2026-10-16 is an activation at a chain window
(below). A position is what the seat is shown of the duel and the question,
not its own earlier answers nor what happened since it last answered: a
command that brought it back there changed nothing, and a network that
takes its first choice would give it again until the cap of decisions ended
the duel. The network of three decks of the experiment on more decks did so
in a sixth of its games as Lightsworn against Rock-Block, activating Lyla,
Lightsworn Sorceress again and again under Skill Drain, which negates her
effect and lets her activate it again. With the rule, of 60 such games
played by its first choice (`analysis/loop/`), none reaches the cap where
10 did, the ten end in 9 to 25 turns (six won), and of the other 50, 47 are
played as before and three part only where it gave such a command a second
time; Lightsworn wins 37 for 31. Where every command was given at a
position already, all stay.

Chain windows were left alone at first, since the seat is asked at windows
that look alike while the duel goes on around it. The first arms of the
experiment on thirteen decks looped at one all the same: activating Lyla at
a chain window of the seat's own Main Phase brought it back to the same
window, in about one duel an update of `half` from its seventieth (and a
third of one of `itself`), each run to the cap of 4,096 decisions, which
the duel then played alone while the others waited: an update of `half`
took 16.3 s where it had taken 13.4. So since 2026-10-16 an activation the
seat gave at a chain window is not offered again at the same position
either. Passing a chain window is always offered, a window with one answer
is not touched, and steps of a selection and pilots' seats never are. Three
updates of `half`'s games played again with the rule from its checkpoint
after 240 updates: none was cut off where two had been, and an update took
728 to 967 calls of the model where it had taken 1,622 to 3,274.

Since 2026-10-23, by a decision of the day before, a monster face-up
on the field is not offered the activation of its effect while its effects
are negated, where the seat may decline it (a window that must be answered
keeps it). The engine lets such an effect activate, and it does nothing as
it resolves. The field is asked for with each card's status
(`QUERY_STATUS` in the policies' `query::RECOMMENDED`, which EDOPro's server
sends too), and the projection keeps its `STATUS_DISABLED`
(`CardView::negated`). `all71` had learnt to loop so as Psychic:
Destructotron's effect under Fiendish Chain, its cost paid by Brain
Research Lab with a Psychic Counter each time, so that no activation came
back to the same position (the record of seventy-one decks, in its section
on the duels it cut). Of 600 duels of Psychic against its usual partners,
`all71`'s network after 10,320 updates in both seats drawing its answers
(`analysis/negated-rule/`), 5 ran to the cap before and none does now;
Destructotron's effect was taken 3,987 times in 217 of them, 790 times in
one, and is taken 276 times in 184, 7 at most in one. The pilots' games are
as they were (8,320 of the forty, each of them), and take 1% longer, the
status asked for with the field. The network's first choices give another
result in 21 and 4 of the 1,664 games of two screens: Rock-Block's, whose
deck plays Skill Drain, and Lightsworn's, whose Lyla met it (wins 1,129 to
1,124 and 1,259 to 1,259).

`--target-kl T` stops an update's remaining epochs once a minibatch's
approximate KL to the rollout policy exceeds `1.5 T`.

Two settings of ygo-agent's trainer (read in its source at `26293f8`),
each its default here off (2026-10-16): `--clip E` sets PPO's clip on the
ratio of an answer's chances to the rollout's (0.2 here, 0.25 theirs), and
`--upgo` adds UPGO to each transaction's advantage as theirs does: the
seat's return followed on through the transactions after it for as long as
each went at least as well as the head expected (its reward and the head's
estimate after it at least the head's estimate before), and where one went
worse, that estimate; less the head's estimate at the transaction. The
head's target stays what it was. A seat's transactions are all its own, so
nothing is turned round for the other player as theirs does;
`tests/test_upgo.py` holds the rest to a transcription of their loop, to
the bit. An update with UPGO reports what it added
(`upgo_advantage_mean`, `upgo_advantage_positive_rate`, before the
advantages are normalised, which theirs are not).

Two settings for speed (2026-10-18: built to be tried as an arm).
`--rollout-estimates kept` trains an update of PPO with the chances and the
value its games were answered with: **the default since 2026-10-20**, once
the two were measured the same to their last digits (in the list of what is
left to try for speed), which is all a game's calls, in batches of other
sizes, change in them. An update so depends in its last digits on how its
games' calls were batched (`--pool-groups`, the graphs, the threads and
slots), where with them worked out again it did not; its games did already.
`--rollout-estimates recomputed` works them out again over the update's
decisions at once, as the trainer did before (about 1 s of an update alone,
2 s of `half`'s three at once): with it, and
`--inference-graphs 0 --pool-groups 1`, the trainer is as it was. And
`--duels-in-flight N`, another training when on, a multiple of `--duels`
and twice them at least,
keeps N duels in flight from one update to the next (`pool.Stream`): the
games of the updates ahead are under way while an update's are, so that
the model's calls stay full where an update played to its end answers
fewer and fewer decisions a call as its duels end, and an update is made
of the next `--duels` duels to end, each trained on once, whole, its every
decision answered by the weights the model had when it asked. It needs
`--rollout-estimates kept` (a duel begun under older weights is trained
with what they gave), as many slots in the pool (`--pool-threads` times
`--pool-slots`), the pilots and the model itself in the other seat alone,
and no evaluation as the run goes. The duels end in the same order however
fast the threads are, and by the same weights they are the duels the same
games played to their end are (`tests/test_stream.py`); an update reports
how many were in flight at its end (`stream_in_flight`), how many had ended
and wait for the next (`stream_carried`), and how many updates ago its
oldest duel began (`stream_oldest`, on average `stream_age`). The library
lets go of the decisions of the duels an update trained on
(`Pool::forget_rows`) and keeps the others. Calls of hundreds of decisions
want graphs of their own sizes or none (the item on many duels in flight in
the list of what is left to try for speed): with many duels in flight
`--inference-graphs` is worth setting at the calls it should take.

**On a card they are the default since 2026-10-23** (decided after the run
`fast` of experiments/2026-10-23-duels-in-flight.md learned as `all71` did
and played 1.46 times the duels an hour). Where the command line does not
say, a run on a card whose updates can keep them (of PPO from its start,
`--eval-every 0`, its estimates kept, no critic, card head, held
checkpoint's duels or teacher; a copy keeps the earlier settings) keeps twice `--duels` in flight, in minibatches of 512, its model's
calls answered as they are (`--inference-graphs 0`) in two groups, the
pool's slots raised to hold them (`run_defaults` in
`ygo_training/__main__.py`). Every other run, and every run on a processor,
plays an update's games to their end in minibatches of 256, as before. A
run that goes on keeps the duels in flight and the minibatches it began
with. `--duels-in-flight 0 --minibatch 256` gives the earlier settings to a
command of an earlier experiment run at today's code.

`--overlap thread`, another training, plays the next update's games beside
an update's pass (experiments/2026-10-23-overlap.md). A second thread plays
them with a copy of the weights from before the pass, its calls on a CUDA
stream of their own on a card, and the pool's calls are made one at a time
(`pool.Locked`): a duel's answers are one update older than with duels in
flight alone. `--overlap in-turn` plays the same games on one thread before
the pass, which gives the same updates to the bit (`tests/test_overlap.py`).
It needs duels in flight, updates of PPO from the run's start, a model that
hides nothing and calls answered as they are. An update reports how long it
waited for its games (`overlap_wait_seconds`), how long they went beside the
pass before it (`overlap_beside_seconds`), how long each side waited for the
other's calls into the pool (`overlap_games_lock_seconds`,
`overlap_pass_lock_seconds`) and the weights' copy (`overlap_copy_seconds`).
A duel the engine stops in the games beside a pass is told with that pass's
update, and with the checkpoint it was played with. Measured on 2026-10-23
it learned as duels in flight alone did and was not faster by the
experiment's rule (1.17 times the duels an hour, the pass slowed by the
games sharing the card with it), so it is not a default.

### Who plays the other seat

The pilot of the deck there, in every duel, unless one of these says
otherwise for a share of the duels of the updates whose games the model
plays. Both go through the library's pool (`--pool-threads`). A checkpoint
held in the other seat is for a learner of one deck; against itself a
learner plays as many decks as it has.

```sh
uv run python -m ygo_training train --device cuda --run runs/itself \
  --learner blackwing --workers 24 --pool-threads 20 --duels 520 --updates 240 \
  --opponents 'blackwing,burn,crystal,dragunity,gishki,gladiator,heroes,infernity,lightsworn,monarch,morphtronic,rock-block,spellcaster' \
  --resume runs/long/checkpoints/model-02279.pt --resume-learning-rate 1e-4 \
  --self-play-duels 260
```

- `--self-play-duels N`: in `N` of an update's duels the learner plays
  against itself. Its own network answers both seats, drawing an answer at
  each decision, each seat has the learner's deck, and every decision of
  the duel is one it learns from: a duel has a result for each seat, and
  one wins what the other loses. The other `--duels` less `N` are against
  the pilots of `--opponents`, in equal numbers. A learner of several
  decks (`--learner blackwing,lightsworn`) has one of its decks in each
  seat of such a duel: every pair of them, which deck is in the learner's
  seat and which in the other, as often as any other, so that a seat has
  each deck as often and two different decks meet twice as often as a
  deck meets itself. Each seat is asked with the list of its own deck.
  The others are its decks against the pilots, every pairing as often.
- `--self-play-pairs mirrors`: such a learner plays its duels against
  itself with one deck in both seats, every deck as often as any other,
  so each deck has its mirror and never meets another of its decks
  (experiments/2026-10-24-two-decks.md). `cross`: two different decks
  alone, every ordered pair as often as any other, so no deck meets itself
  (experiments/2026-10-24-specialist-distillation.md). `all`, the default,
  is every pair as above. The duels against the pilots are the same either
  way.
- `--opponent-checkpoint CHECKPOINT --opponent-checkpoint-duels N`: in `N`
  of an update's duels the other seat is a checkpoint held as it is. It
  plays the learner's deck and takes its first choice at every decision, as
  it does when it is measured; the learner draws its answers, and only its
  own decisions are learned from. With every duel of the update against it
  (`N` the number of `--duels`) the learner is an exploiter of that
  checkpoint: what it wins after a fixed number of updates says how far a
  learner that looks for the checkpoint's habits gets against it. The
  checkpoint is one that played the learner's deck; it may read the board
  another way than the learner.
- `--opponent-checkpoint` with no duels of an update holds the checkpoint
  for the evaluation alone: the cases of the suite that are played against
  a checkpoint ([`evaluate-suite`](#evaluate-suite)).

The three kinds of duel are shuffled among each other. An imitating update
(`--imitation-only-updates`) plays the pilots throughout, as it does without
these options. Building the checkpoint draws nothing: the learner starts
from the network it starts from without it.

An update reports its duels and the wins of the learner's seat by who sat
in the other (`other_seat_duels`, `other_seat_win_rate`, under `pilot`,
`self` and `checkpoint`). `learner_win_rate` stays the rate over all of an
update's duels: against itself the learner's seat is one of its two and
wins about half, so with such duels it is `other_seat_win_rate.pilot` that
is set beside the `learner_win_rate` of a run against pilots alone. Of the
duels against itself an update tells how often the seat that opens wins
(`self_play_first_seat_win_rate`) and how long they are
(`self_play_mean_turns`); `held_decisions` is how many decisions the
checkpoint answered. `run.json` names the checkpoint by its path, its
updates and a fingerprint of its parameters. Where a learner of several
decks plays itself, `learner_deck_duels` and `learner_deck_win_rate` are
of its duels against the pilots alone, and `self_play_deck_seats` and
`self_play_deck_win_rate` tell the seats each deck took against itself
and how many of them won.

### A critic that sees the hidden cards

```sh
uv run python -m ygo_training train --device cuda --run runs/critic \
  --learner blackwing --workers 24 --pool-threads 20 --duels 520 --updates 240 \
  --resume runs/long/checkpoints/model-02279.pt --resume-learning-rate 1e-4 \
  --self-play-duels 260 --critic --critic-deck 5
```

`--critic` measures the advantages of an update of PPO against a network
of its own that sees the hidden cards, in training only
(`ygo_training/critic.py`, [the experiment](../experiments/2026-10-22-privileged-critic.md)):
it reads each decision as the policy has it and the same decision with
nothing hidden but the Decks' order, and with `--critic-deck N` the next
`N` cards of each Deck as well. It begins as the value head of the model
the run begins with. After each update's games, once it has given their
credit, it is taught the results of `--critic-positions` (8) of each
seat's transactions a duel, drawn at random, in one pass of minibatches at
`--critic-learning-rate` (3e-4), with an optimizer of its own. The value
head is taught as without it, its target its own return: only the credit
changes. The critic is kept beside each checkpoint (`critic-NNNNN.pt`),
never in one, and a run goes on from it (`--continue`, with the same
options); nothing that plays reads it. An update tells what the critic and
the value head explain of its games' results before the critic is taught
them (`critic_outcome_explained_variance`,
`critic_head_outcome_explained_variance`), the positions taught, the
critic's loss and the seconds it took. It is for updates of PPO played to
their end: not with imitating updates, duels in flight or a model that
hides identities.

### A head that learns how cards work

```sh
uv run python -m ygo_training train --device cuda --run runs/cards \
  --learner blackwing --workers 24 --pool-threads 20 --duels 520 --updates 240 \
  --resume runs/long/checkpoints/model-02279.pt --resume-learning-rate 1e-4 \
  --self-play-duels 260 --card-head --card-head-weight 0.2 \
  --card-head-begun analysis/card-head/run/probe/card-head.pt
```

`--card-head` teaches beside PPO a head of its own (`ygo_training/card_head.py`,
[the experiment](../experiments/2026-09-08-engine-transition-prediction.md)):
at the last decision of each of the seat's transactions, from the network's
own card tokens, state and chosen answer, it predicts what the seat will see
at the horizon (its next decision of another transaction with no chain
open, the end of the turn or of the duel): where each card of the
decision's state is, the change of the ATK and DEF of those on the field,
and of each seat's life points and cards in each location. The labels come
from the engine's messages: `crates/ygo-env/src/follow.rs` moves each card
of the state as the seat's projection moves it (`Setup::follow`,
`DuelPool.labels`), and nothing of them reaches the input. The head is
taught in PPO's own minibatches; `--card-head-weight` scales the gradient it
gives the network beside PPO's, while the head itself learns as much
whatever the weight (at 0 the network's update is the one without it, bit
for bit). It learns at 3e-4 with an optimizer of its own, begins anew or as
`--card-head-begun` names it (a probe's, below), and is kept beside each
checkpoint (`card-head-NNNNN.pt`), never in one. An update tells the head's
losses (`card_head_places_loss`, `_changes_loss`, `_life_loss`,
`_counts_loss`) and how many decisions taught it.

`tools/card_head.py` goes with it: `probe` teaches the head alone on a
checkpoint whose network it holds as it is, and reads it on games it never
trained on against the frequencies of each place by a card's location, its
side and the kind of answer (and audits the labels: every card the seat
knew at both ends must be found where it was followed to); `weight` sets
the size of PPO's gradient on the shared parameters beside the head's, over
an update's games; `read` takes the experiment's first reading of a run's
checkpoint and the head beside it.

### Imitating a checkpoint

```sh
uv run python -m ygo_training train --device cuda --run runs/copy \
  --learner blackwing --workers 24 --pool-threads 20 --duels 520 --updates 200 \
  --imitation-only-updates 1000 --imitation-value-weight 0.5 --imitation-value-positions 4 \
  --imitation-teacher runs/specialist/checkpoints/model-02279.pt
```

`--imitation-teacher CHECKPOINT` puts a network where the pilot is in the
imitation-only updates: it plays the learner's seat, taking its first choice
at every decision as it does when it is measured, and the learner is trained
on its answers. The checkpoint is one that played the learner's deck; it
may read the board another way than the learner
(`--board-representation`). Its games go through the library's pool, so
`--pool-threads` is needed.

- The deals are the ones a run that imitates the pilot plays with the same
  seed, workers, duels and opponents, and the learner starts from the same
  network: the two runs differ in who is imitated. The games are the
  teacher's whatever the learner has become, so two runs with one teacher
  train on the same games.
- `--imitation-teacher-targets choice` (the default) teaches the teacher's
  first choice, as a pilot's answer is taught. `distribution` teaches what
  the teacher gives every answer: the learner's cross-entropy against the
  teacher's probabilities, which also tells it what the teacher nearly
  chose. A learner taught the choice alone grows surer than its teacher.
- The value head is taught the results of the teacher's games, as it is
  taught those of the pilot's (`--imitation-value-weight` and the options
  beside it).
- The deck's pilot is still asked alongside. An update reports, measured
  before it on games new to the learner: how often its first choice is the
  teacher's (`imitation_model_teacher_agreement`); at the decisions where
  the teacher leaves its pilot (`imitation_departures`), how often it is
  the teacher's answer (`imitation_model_departure_agreement`) and how
  often still the pilot's (`imitation_model_departure_pilot_agreement`);
  the divergence of the teacher's answers from the learner's, in nats a
  decision (`imitation_teacher_divergence`); and the entropy of the
  teacher's own (`imitation_teacher_entropy`). `learner_win_rate` is the
  teacher's, in the games it is imitated in.
- `run.json` names the teacher by its path, its updates and a fingerprint
  of its parameters, and a run is not continued with another.
- A learner of several decks has a teacher for each of them,
  `--imitation-teacher DECK=CHECKPOINT` once for each deck: each deck's
  seats are played by its own teacher, and each is asked again at the
  decisions of its deck in the update. `run.json` names them by deck.
- With `--self-play-duels N`, `N` of an update's duels are a teacher against
  itself: it plays both seats, and both are imitated
  (`--self-play-pairs mirrors` for each deck's own teacher in its mirror).
- What the teachers give every answer is asked again in the training pass
  for `distribution` alone; for `choice` the answers are the ones the games
  played, and the teachers are asked once, before the update, for what it
  reports of them.
- A copy keeps its earlier calls: the teachers' duels go forward by the
  pool's groups only where the run names `--pool-groups`, and their calls
  are answered from CUDA graphs only where it names `--inference-graphs`.
  Either gives the teachers other batches, and a first choice between two
  answers that score the same to their last digits, two copies of a card,
  can then go the other way: the games are the teachers' still, and not
  those of a run that names neither.

### Imitating recorded duels

```sh
uv run python -m ygo_training train --device cuda \
  --run runs/search-arm --resume runs/seed/checkpoints/model-00029.pt \
  --learner blackwing --workers 24 --pool-threads 20 --duels 512 --updates 32 \
  --imitation-only-updates 32 --imitation-value-weight 0.5 \
  --imitation-records analysis/search/train.jsonl --imitation-games 4000
```

`--imitation-records` names the rows of a `policy-bench search --record true`
run. Every row is one deal played twice by the learner's deck: by its pilot
with the search, and by its pilot alone. `--imitation-game search` (the
default) imitates the first, `--imitation-game baseline` the second, on the
same deals in the same updates, so two runs that differ in that option alone
differ only in whose answers they learn.

The games are put in an order that does not depend on how the search run was
scheduled: each opponent's games by seed, the opponents taking turns.
`--imitation-games N` takes the first `N` of that order, an equal share of
every opponent and of both seats when `N` is a multiple of twice the number
of opponents, and a smaller `N` is part of a larger. An update imitates the
next `--duels` games of a shuffled pass over them; when a pass is through,
another begins in another order. `--updates` times `--duels`, divided by the
number of games, is the number of passes.

A record is refused unless it plays again with the digest it was recorded
with and leaves its pilot where its row says. Records therefore belong to
the revisions of the engine, the scripts, the card database and the pilots
they were made with.

Such updates report, beside the usual imitation metrics, `recorded_pass`,
`recorded_departures` (decisions at which the recorded duels left their
pilot) and, measured before the update, how often the model's first choice
at those decisions is the recorded answer
(`imitation_model_departure_agreement`) or still the pilot's
(`imitation_model_departure_pilot_agreement`). In the first pass the model
has not seen the games it is measured on.

`--imitation-label pilot` has the policy imitate the pilot's own answer at
every decision of a recorded game, also where a search played another; the
steps the pilot was never asked, after such an answer, are left out. The
default, `played`, imitates the answer the game was played with.

With rows of a search run that logged what it tried (`--log true`, or `full`
for every alternative: see [The search](#the-search)), each decision the
search examined carries the *lead* it measured for every answer it tried:
how much more often that answer won than the pilot's in the worlds of the
first stage, a draw counting as half a win, and 0 for the pilot's own.
`--imitation-lead-weight W` trains the lead head on them by squared error,
with weight `W` beside the imitation loss; with `--imitation-lead-detach`
the head reads the policy's view of an answer and sends no gradient of its
own into it. Such updates report `imitation_lead_loss`,
`imitation_lead_targets`, the head's explained variance before the update
(`imitation_lead_explained_variance`) and, from the update's first
minibatch, the size of the two losses' gradients on the parameters they
share (`imitation_shared_gradient_norm`,
`imitation_lead_shared_gradient_norm`). Without a weight the head is
measured and left as it is. A log that names other alternatives than the
decision offers is refused.

### Resume and run integrity

- `--continue` loads the latest numbered checkpoint below `--run` and goes on
  with the run's schedule (imitation-only updates, decay) where it stood.
- `--resume PATH` starts a branch in a new `--run`: its schedule begins at
  the resumed checkpoint.
- `--resume-learning-rate` replaces the learning rate saved with the
  optimizer.
- `--new-pins`, with `--continue`, goes on although the pins have moved
  since the run was last made: for a move whose duels are the old ones,
  which the section on moving the pin checks. `run.json` keeps each move
  in `pin_moves`: the version it went on from, and the revisions before
  and after.
- `--allow-overwrite` is required to replace existing versions.

A checkpoint is `model-NNNNN.pt-tmp` until it is written whole and its
update has been evaluated, where an evaluation is due; its row in
`metrics.jsonl` follows at once. A run stopped before that, killed or
interrupted, goes on from the checkpoint before and makes the update again:
no checkpoint is there without its row and its evaluation.

`run.json` records the run's protocol, and a continuation that changes it is
refused. The protocol includes `duel_revisions`: the commits of the pilots,
the engine, the card scripts and the card database the duels run on, since
at other revisions a seed can be another duel. Two changes are taken: a
run that has made only imitation-only updates may be continued with a
larger `--imitation-only-updates`, and goes on imitating; and one told
`--new-pins` goes on at the pins it is made at. Only checkpoints of the
current format load; see [architecture.md](architecture.md).

### Duels the engine stops

A duel in which a card's script fails, or whose arena fills, cannot go on.
The engine logs why, and the duel ends as one the limit stops: no winner,
and in a suite a game that is not terminal, scored as a draw. Everything
else goes on (it stopped the run until 2026-10-21, when Batteryman Fuel
Cell's script stopped `all71` of seventy-one decks after 957,000 duels).
An error that is not the engine's, in the messages read, still stops it.

- **A run keeps each in `failed-duels.jsonl`** beside `metrics.jsonl`, one
  line a duel: the update (`version`) and the checkpoint its games were
  played with (`played_with`, which thinning then keeps); the game as the
  update dealt it (`episode_index`, `seed` its deal, `learner`, `opponent`,
  `learner_player`, `other`) and the update's `rollout_seed`, which draw the
  network's answers again; the `setup` it was submitted as; what the engine
  said (`failure`), its `decisions` and `turns`; and its `record` up to the
  failure. A metrics row counts its update's (`failed_duels`), and each is
  said on standard error as it is kept. `evaluate-suite` adds them to a
  checkpoint's record (`failed_duels`) where there are any, and says them
  too.
- **To see one**, play its record again with the policies' bench, which
  tells the duel with nothing hidden as far as it goes and ends with why it
  went no further:

  ```sh
  sed -n 1p runs/NAME/failed-duels.jsonl | python3 -c 'import json, sys; json.dump(json.load(sys.stdin)["record"], sys.stdout)' > analysis/failed-duel.json
  cd native/ygo-policies && cargo build --release -p ygo-policies-bench && target/release/policy-bench replay --input ../../analysis/failed-duel.json
  ```

- **A card's script that fails is mended in the policies** where the fault
  is the script's: see the section on moving the pin.
- **To look for them**, `tools/fuzz_duels.py` plays every pair of the
  pool's decks with seats that answer at random among the legal answers,
  against the deck's pilot or both seats so, and keeps each duel the engine
  stops with its record (`--per-duel` draws each answer from its duel
  alone, for two builds' games side by side). On 2026-10-21, 1,008,200
  duels in 68 minutes here (four processes of five threads): no card's
  script failed. The engine stopped two. One was a loop of the rules, which
  the engine guards against (`recursive event trigger detected.`): a
  Wave-Motion Cannon's damage turned into a gain of Life Points, and gains
  setting off gains (Fire Princess), 500 at a time; EDOPro stops it the
  same way. The other was a loop of the engine: Malefic Cyber End Dragon,
  which destroys itself with no face-up Field Spell, with Machina Gearframe
  equipped, destroyed in its place. The destruction of Gearframe asked the
  Dragon to destroy itself again before Gearframe had gone, and Gearframe
  was to be destroyed instead again, without end; the arena filled in two
  minutes, and with no arena the duel never comes back. Upstream's engine is
  the same there. The core now has the substitute go first and the monster
  destroy itself after (fourlastor/ygopro-core `904b336`, the policies'
  `ce26706`, a test of the duel beside it); 80,656 duels of random answers,
  the check suite's 3,692 pilot games and the bench's games for moving the
  pin are the same with and without it. Read by hand besides: of the 23
  cards of the pool whose effect pays a cost that moves cards, then targets
  and reads its first target unchecked, only Batteryman Fuel Cell can have
  its cost take away its only target.

`--seed` sets the evaluation games and, unless `--training-seed` says
otherwise, the training randomness (rollout duels, sampling, minibatches).

### Periodic evaluation

`--eval-every N` plays a held-out suite with the model's first choice every
`N` updates; `--eval-suite PATH` names it and `--eval-every 0` disables it.
A learner of several decks is evaluated on the screens of them all, when no
suite is named: each deck's games are the ones its own screen deals, so its
rows are set beside those of a learner of that deck alone on the same deals.
The games are the same for every checkpoint of a run. The suite is the
evidence a model is selected on; the sampled training win rate is a
diagnostic. Every checkpoint is written, and none is marked best.
`--keep-checkpoints N` keeps the checkpoint of every `N`th update and the
latest, and removes the others of its own invocation as it goes; give it
the `N` of `--eval-every` and the checkpoints kept are those evaluated.

## `evaluate-suite`

```sh
uv run python -m ygo_training evaluate-suite \
  runs/example/checkpoints/model-00029.pt runs/example/checkpoints/model-00049.pt \
  --device cuda --workers 24 \
  --suite eval_suites/ygo-policies-blackwing-screen.json \
  --output analysis/example-evaluation.json
```

Plays every case of a suite with each checkpoint. A case that fails is
recorded with its error and the others go on. Each case reports wins, losses
and draws overall, by opponent and by seat, the learner's answers by kind,
and `pilot_agreement`: how often the model's first choice was the answer of
its deck's pilot.

```sh
uv run python -m ygo_training evaluate-suite --pilot --workers 24 \
  --suite eval_suites/ygo-policies-blackwing-screen.json
```

With `--pilot` and no checkpoint, the pilot of each learner deck plays its
seat on the same deals. This is the reference a model is compared with.
A case of such a measurement also tells how each of its games ended
(`outcomes`, in the order of the games): two players of one suite with one
seed are then set beside each other game by game, which tells a smaller
difference than their two rates do. The rows of a training run hold the
counts alone.
A suite's deals follow from `--seed`: a training run plays its suite with
its training seed, and its reference is the pilot's with the same one.
A case is dealt from the run of seeds of its place among the suite's cases,
or of the place its `seed_index` names: what keeps a case on its deals when
suites are joined.

`eval_suites/ygo-policies-blackwing-unseen.json` is 128 games against the
pilot of each of the eighteen decks the Blackwing learner does not train
against: what it wins against decks, and mostly cards, it was never shown.

```sh
uv run python -m ygo_training evaluate-suite --pilot --workers 24 --seed 2012 \
  --suite eval_suites/ygo-policies-blackwing-wide.json \
  --learner-list data/decks/darker-than-black.ydk
```

With `--learner-list` the learner's seat plays the deck list of that `.ydk`
file in place of its deck's own: a list no pilot was written for. The pilot
of the learner's deck plays it with `--pilot`, or is asked alongside a
checkpoint, and the other seat is as it was. The result names the list (its
path and SHA-256). `data/decks/darker-than-black.ydk` is a variant of the
Blackwing list, and `eval_suites/ygo-policies-blackwing-wide.json` the
Blackwing screen with 512 games a pairing, its first 128 the screen's own
([Darker Than Black](../experiments/2026-10-19-darker-than-black.md)).

```sh
uv run python -m ygo_training evaluate-suite \
  runs/a/checkpoints/model-02279.pt runs/b/checkpoints/model-02279.pt \
  --device cuda --workers 24 --pool-threads 20 --seed 2012 \
  --suite eval_suites/blackwing-against-checkpoint.json \
  --opponent-checkpoint runs/held/checkpoints/model-02279.pt \
  --output analysis/example-against-checkpoint.json
```

A case whose `opponent_policy` is `checkpoint` is played against a network
in the other seat, where a case is otherwise played against the pilot of
the opponent's deck. The suite does not say which: `--opponent-checkpoint`
names it, for `evaluate-suite` and for `train`, and such a case is refused
without it. Both networks take their first choice, each at the decisions of
its own seat, and what a case reports of answers and of agreement with the
pilot is of the measured seat alone. With `--pilot` the deck's pilot is
measured against the same checkpoint on the same deals. The games go
through the pool (`--pool-threads`). The result names the checkpoint
(`opponent_checkpoint`: its path, its updates and a fingerprint).
`eval_suites/blackwing-against-checkpoint.json` is 2,048 games of the
Blackwing mirror against the checkpoint held; its first 128 are the deals
of the screen's mirror against the pilot. Two checkpoints measured against
one held checkpoint with one seed are set beside each other game by game
(`tools/suite_beside.py`). Two checkpoints measured against each other,
each in turn the one held, play every deal twice: the measured seat is the
same in both, so each network has the deal once from either seat.

On a processor (`--device cpu`) give PyTorch fewer threads than it takes:
on the development machine, whose 24 cores are of two kinds, 128 games
through the pool take 33 s with its 24 threads and 5.5 s with
`OMP_NUM_THREADS=8`, whatever `--pool-threads` is. A suite played there
has the games of the same suite played on the CUDA machine's card.

Every game is played through the library's pool: `--pool-threads` threads
(4 unless named; every script names 20 on the CUDA machine), each with up
to `--pool-slots` duels in flight (16), the model answering a round's
decisions together. In a batch the last digit of what a network answers
changes with what shares it, and a game can go another way where two
answers are rated alike. A pool's rounds do not depend on the threads'
pace: with the same threads and slots it plays the same games every time.
An update's decisions stay in the library, and its minibatches are built
there. A recorded duel that is watched (`evaluate-search`) is asked in one
call of the model, all its decisions together; where a search's log or the
lead head is read, a duel shows the pool the views of its decisions, and
what is read of them is read there (`ygo_training/searched.py`).

Until 2026-10-16 the games went to worker processes, one duel each, unless
`--pool-threads` was named, and recorded duels and the lead head stayed
with them. They were dropped that day: every script had used
the pool since it came. What the two routes gave was held together until
then, and once more that day on the development machine before the
workers went:

- an update that imitates the pilot, or recorded duels with the lead head
  taught, is the workers' update to the last digit: the same games (the
  seats follow from `--workers`, as the workers drew them), the same
  minibatches, the same model after it;
- an update whose games the model plays has the workers' deals and other
  draws: a draw is seeded by its duel and its place in it, not by a
  worker's count of requests;
- `evaluate-search` plays the same deals, and by the lead head the same
  games, and reads the recorded duels the same way, but at a decision where
  the model rates two answers alike: the workers' batches followed the
  clock, and answers that score the same to the bit (copies of a card) part
  in the last digit in some of them. On 48 held-out deals one decision of
  3,403 was read otherwise, and in the lead head's games one answer was its
  pilot's copy of a card where it had been the other copy; the pool's are
  the same every time.

`experiments/commands/pool-check.sh`, which timed the two routes, runs at
the revision before (`60413ff8`). It took about ten minutes: one checkpoint
on the suite, updates that imitate, and updates whose games the model
plays, for several values of `--pool-slots`, and said whether the
imitating runs ended with the same model (`analysis/pool-check/report.txt`).

On the CUDA machine (an RTX 4070 SUPER and 20 processor threads) on
2026-10-09, with 520 games an update and `--pool-threads 20 --pool-slots 16`:

| | Workers | Pool |
|---|---|---|
| The suite, 1,664 games | 74 s | 31 s, the same 954 wins |
| An update that imitates: its games, the update, all of it | 3.4 s, 6.1 s, 9.9 s | 1.8 s, 5.1 s, 7.2 s |
| An update whose games the model plays, one pass: its games, the update, all of it | 20.9 s, 5.1 s, 28.8 s | 6.3 s, 4.7 s, 13.3 s |

- The two imitating runs ended with the same model after three updates.
- All of an update is more than its games and its update: before an
  update whose games the model plays, the games' estimates are worked out
  again in batches that do not depend on how the games were played, 2.1 s.
- Of the 6.3 s of pooled games the engines take 3.1 s, the model 2.3 s,
  and the batches with their copies 0.6 s. With 8 duels a thread the games
  take 7.2 s, with 32, 6.1 s.
- A run through the pool repeats itself on the card: a second check gave,
  for the imitating run and the three others through the pool, every
  figure and the model of the first.

**Three changes of 2026-10-13 leave an update as it is and shorten it.**
An independent review of where an update's time goes proposed them
(`docs/2026-10-13-training-performance-investigation.md`, not committed).
Each was measured alone on the same machine from the branch
`speed-trials`, where a change is kept beside the code it replaces behind
a switch: PPO against the pilots on from a trained checkpoint, six updates
made twice, each beside the same update as it was.

| | An update | Its games | The pass |
|---|---|---|---|
| As it was | 13.95 s | 6.38 s | 5.10 s |
| A refresh of the field decoded once for both seats | -0.69 ± 0.02 s | 5.69 s | |
| A transaction's members given their advantage in one read | -0.48 ± 0.03 s | | 4.63 s |
| The next minibatch built by another thread while the card works on this one | -0.42 ± 0.02 s | | 4.67 s |

- **The refresh** is most of what a seat is fed. A seat on its own redacts
  the message and decodes it, so two seats decoded it twice; it is decoded
  once and each seat shown its part, as the policies' own seats do
  (`Table::update` in `crates/ygo-env/src/seat.rs`).
- **The advantages** were assigned on the card once for every one of some
  40,000 transactions (`PpoOptimizer.prepare_samples`).
- **The next minibatch**: the pass's minibatches and their order are known
  before it starts, so one thread builds each from the decisions the pool
  keeps and sends it to the card a step ahead (`PpoOptimizer.update`). In
  a pass the pool is asked by that thread alone. An update's
  `pool_minibatch_seconds` is still the time the building took, and is no
  longer time the pass waits for. The card holds a megabyte or two more.
- **The same updates.** On the card the games, every figure, the model,
  the optimizer and the random state were those of the trainer as it was
  after each of the twelve updates of every change. On a processor two
  updates of `main` before and after the changes were set beside each
  other in seven ways (against the pilots, against itself, with an anchor,
  with the experiment's settings, in two passes, with a KL target that
  ends an update early, against a held checkpoint): the same again.
  `tests/test_pool.py` holds a pass built ahead to one whose minibatches
  are built when they are asked for.
- **Together** they were measured only with four other changes on:
  12.11 s an update (-1.85 ± 0.03 s, -13.2%). Those four gained nothing
  that can be told from the noise and are left out: the distribution over
  the answers not checked a second time (-0.01 ± 0.03 s), a minibatch's
  figures read from the card after the last step (-0.05 ± 0.02 s), the
  lead head not worked out where it is not read (+0.01 ± 0.03 s), and
  PyTorch's inference mode in the games (-0.06 ± 0.03 s).
- Not measured: the other ways of an update (against itself, a held
  checkpoint, imitation), and the three alone together.

**The LSTM is given a packed batch** (decided on 2026-10-13):
the same steps of the same rows, which a card's library then works out
its standard way (`PolicyValueNet._history`). On a card it is another
run, by the third decimal of what the LSTM gives; on a processor nothing
changes.

- **Why.** One update against the pilots traced under PyTorch's profiler
  on the CUDA machine: the card's kernels ran 6.2 s, and one of them,
  `RNN_blockPersist_fp_LSTM`, 3.0 s of it. It ran 691 times, once for
  every call of the model (335 rounds of the games, 178 minibatches of
  the estimates worked out again, 178 of the pass), at 4.3 ms a call. The
  gradient of the same LSTM took 1.1 ms a call. PyTorch chooses the way
  cuDNN works an LSTM out (`get_algo` in
  `aten/src/ATen/native/cudnn/RNN.cpp`, read on its main branch): for a
  padded batch and at most 192 hidden units, which is what the model's
  has, the forward is one kernel that keeps the weights at hand for all
  the steps (`CUDNN_RNN_ALGO_PERSIST_STATIC_SMALL_H`); the gradient, above
  128 units, and any batch given packed (`pack_padded_sequence`) are
  worked out the standard way. The checkpoint's own LSTM both ways
  (`tools/lstm_timing.py` of the branch `speed-trials`, 32 steps a
  sequence, the median of 200 runs):

| Sequences in a batch | 16 | 64 | 128 | 256 | 320 |
|---|---|---|---|---|---|
| A forward as PyTorch chooses it | 0.36 ms | 0.83 ms | 3.25 ms | 6.34 ms | 7.84 ms |
| A forward the standard way | 0.28 ms | 0.31 ms | 0.38 ms | 0.59 ms | 0.75 ms |
| With its gradient, as chosen | 0.68 ms | 1.25 ms | 3.95 ms | 7.42 ms | 9.07 ms |
| With its gradient, the standard way | 0.71 ms | 0.72 ms | 0.99 ms | 1.63 ms | 1.94 ms |

- **What it saves: 2.41 ± 0.14 s of an update's 13.94 s against the
  pilots** (-17.3%), measured as the variant `lstm` of the branch, six
  updates twice beside the same update as it was: the games 5.86 s for
  6.37 s, the estimates worked out again 1.18 s for 2.14 s, the pass
  4.16 s for 5.10 s. The card holds 373 MiB at most where it held 442.
  Not measured: beside the three changes above, and where the model is
  called more (against itself, a held checkpoint).
- **On a card it is another run, and not by a last digit.** The two ways'
  outputs are up to 6e-3 apart (an output lies between -1 and 1), and
  each is that far, 1e-3 to 2e-2, from the same LSTM worked out in double
  precision on the processor. cuDNN works the LSTM out at the reduced
  precision of tensor cores (TF32), which PyTorch lets it unless
  `torch.backends.cudnn.allow_tf32` is turned off: with it off the plain
  model's answers to a minibatch of real decisions moved by up to 1.3e-3
  in a score and 6.6e-5 in the value. The gradients of the weights are
  off the reference by about a thousandth of the largest of them, either
  way. A run taken on after the change is not the run it would have
  been, and a checkpoint measured on a card after it is not asked to the
  digit as before.
- **On a processor the two ways are one code**: what the model reads is
  what the LSTM gives the padded batch, to the last digit
  (`tests/test_ppo.py`), and two updates of `main` before and after were
  the same in the seven ways named above.
- **A measurement on a card is the same measurement.** The screen's
  1,664 games by a checkpoint's first choice, played both ways on the
  CUDA machine the same day (`PARTS=suite bash
  experiments/commands/speed-trials.sh` of the branch): 1,336 wins both
  ways, no game ended otherwise, and everything told of the games is the
  same, the answers by kind and the turns too.

**Calls from CUDA graphs, and the duels in two groups** (decided on
2026-10-13): on a card, a run whose games go through the pool asks
its model the way measured fastest there, unless it names another.

- **What a run on a card takes where it names nothing.** Calls of the
  model of up to half the pool's duels (160 of 320) are answered at sizes
  fixed in advance, from CUDA graphs (`--inference-graphs`). A thread's
  duels go in two groups, one taken forward while the model answers the
  other (`--pool-groups`), and the groups go on together once their
  rounds hold 64 decisions or fewer between them (`--pool-groups-over`).
  Half the pool's duels is the most a group of two asks at once, so that
  every call of an update's games is then answered from a graph. There
  are nine graphs for 160 decisions, and one keeps some 32 MiB of the
  card's memory.
- **Where nothing changes.** On a processor: one group, and every call as
  it is. With a thread's duels an odd
  number, or one queue for the threads (`pool_shared` of the trainer's
  configuration), the duels stay in one group and the graphs are used all
  the same.
- **Named, an option is what it says.** `--inference-graphs 0` is every
  call as it is, and then one group, since a round split in two is the
  slower way with calls as they are; `--pool-groups 1` is one group with
  the graphs; `--pool-groups-over 0` is by groups to an update's last
  duel. `--inference-graphs 0 --pool-groups 1` is the trainer as it was
  before that day.
- **What it saves**, half of the duels against itself, on from the arm's
  checkpoint: a run alone takes 15.5 s an update where it took 18.6 s,
  and three runs at once through MPS 23.0 s each for 26.0 s. The
  measurements are in the list of what is left to try for speed (the
  item on a call of the model costing the same whatever it holds).
- **A run is not to the digit the run it would have been.** Filled out to
  fixed sizes a decision's scores differ in their last digits from those
  of its own batch, by up to 1.3e-5 where it was measured, and a draw
  within that of its boundary goes the other way: now and then another
  game, of the same model. In the runs measured no draw did, in seven
  updates. An update of given games is the same update, and two groups
  are the games of one.
- **A measurement is the measurement it was.** A suite, the held-out
  games and a run's own evaluations are played with every call as it is
  and every duel together. `evaluate-suite` and `evaluate-search` take
  one group and no graph unless they name another way themselves; a
  run's own evaluations go through its pool of two groups, taken forward
  together, which plays a measurement's games as a pool of one does
  (`tests/test_pool.py`): a thread takes its duels forward in the order
  it started them, whatever their groups.
- **A run says how it was asked.** Its `run.json` has `model_calls`
  (`pool_groups`, `pool_groups_over`, `inference_graphs`), and each
  update's figures `pool_graph_batches`, the calls answered from graphs,
  `pool_grouped_batches`, those that answered a group, and
  `pool_graph_failed`. A graph whose answer is not the model's ends all
  of them: the calls are answered as they are from there, the run goes
  on, and says so once on its error stream and in every update's figures.
  A run goes on (`--continue`) asked another way than it began with, and
  its description then says the last way.
- **What measures a way names it**: `tools/side_by_side.py` gives every
  run all three options, and `tools/call_timing.py` and
  `tools/update_timing.py` play their games with every call as it is.
  `tools/benchmark.sh` takes the trainer as it is,
  so that on a card its figures for the modes whose games the model plays
  are with the graphs and the groups from that day, and not those of the
  report of 2026-10-12 below.
- **A deck list of more than 48 cards misses the graphs** (found
  2026-10-15 in the arms of `experiments/commands/more-decks.sh`). The
  width of a seat's own deck is 48, Blackwing's list, the longest in the
  games it was fixed from, and a call that holds one decision of a seat
  playing a longer list is answered as it is. Five of the thirteen decks of
  training are longer: Lightsworn (50 cards) and Dragunity, Gladiator,
  Heroes and Infernity (55 each). Three runs at once through MPS, over 100
  updates of each: the network of Blackwing and Lightsworn had 63 of an
  update's 2,023 calls answered from graphs, that of those two and Gishki
  59 of 1,379, that of Blackwing alone all but one call in the hundred
  updates; their games took 16.0
  and 11.4 s an update for 6.9, with fewer decisions (61,000 for 72,000),
  and the whole update 25.2 and 20.9 s for 17.0. The pass over the
  decisions is the same (6.5 to 7.1 s). The games are the model's own.
  A width of 56, the other seat's, holds all thirteen: not done, since a
  call filled out differs in its last digits (now and then another game),
  so it waits for a decision and a point between experiments.
  Since format 15 (2026-10-16) the input holds no deck list, and no deck
  keeps a call off the graphs.
- **An answer of more than two member cards misses them** (found
  2026-10-16, the first updates of the arms of
  `experiments/commands/thirteen-decks.sh`, 81% of their calls from
  graphs). The width of an answer's members is 2, fixed from Blackwing's
  games. Played on the processor through a pool of the CUDA machine's
  shape with the network of thirteen decks (the copy's last checkpoint),
  three updates of `half`'s games: 485 of 2,538 calls missed the graphs,
  the large ones (151 decisions at the median, where the calls answered
  from graphs hold 31), so 36% of the decisions. Each was kept off by one
  or two decisions with answers of 3 to 5 members: of 599 such decisions,
  450 were a selection of cards, 134 an order of cards to set (a Sort),
  13 counters to place; by the seat's deck Gishki 245, Morphtronic 179,
  Monarch 88, Crystal 63, Spellcaster 13, Dragunity 9. Nothing else did:
  no state over 96 cards, no call over 160 decisions, and answers over 32
  only twice (a Sort of Gishki with 120 answers, as many as five cards
  have orders). A members width of 3 would take in 94% of the calls, 4
  98%, and 8 with 48 answers all of them. Not done: a call filled out
  wider costs more arithmetic and memory, unmeasured, and differs in its
  last digits from what it is now (now and then another game), so it
  waits for a decision and a point between experiments.

`tools/update_timing.py CHECKPOINT --device cuda --pool-threads 20` times
the pass over an update's decisions by the size of its minibatches: one
update's games played with the checkpoint's model, what an update does
before its first pass, then one pass for each size from the checkpoint's
weights, with a step's seconds, the card's memory and where a step's time
goes. It measures speed only. On the same machine, 44,651 decisions:

| Decisions a step | 256 | 512 | 1,024 | 4,096 |
|---|---|---|---|---|
| A pass | 5.0 s | 4.2 s | 4.5 s | 4.6 s |
| A step | 28 ms | 47 ms | 103 ms | 417 ms |

- A step's cost grows with its size, so larger minibatches gain little:
  512 saves 0.8 s of a pass and more saves nothing.
- At 256, of a step's 28 ms the model's answer takes 10 ms, the gradient
  9 ms, the batch with its way to the card 5 ms and the optimizer 1 ms.
- The advantages take 0.65 s an update. They took 2 s more while they were
  worked out on the card one number at a time: a pass was 7.0 s then, at
  every size. Since 2026-10-13 a transaction's members are given theirs
  in one read, which took another 0.48 s off (above).

With `--compile` it measures all of it a second time, on the same games,
with the model compiled by `torch.compile`, which training does not use:
what compiling saves, what it costs when a process starts, and whether the
model keeps its numbers. The tool makes three settings the model needs:
every size of a batch left free, the LSTM compiled with the rest, and more
than the eight versions PyTorch allows before it goes back to the plain
model without a word. The compiler wants a C compiler and Python's own
headers, which a Python of uv's has; the head of the tool says how to
borrow them for another. On the same machine on 2026-10-09
(`analysis/update-timing/report-compiled.txt`):

| Part of an update of 520 games | Plain | Compiled |
|---|---|---|
| The model's turns in the games | 3.0 s | 1.4 s |
| The games' estimates worked out again | 2.06 s | 0.95 s |
| A pass, 256 decisions a step | 4.99 s | 17.33 s |
| Of a step: the model's answer | 10.2 ms | 2.5 ms |
| Of a step: the gradient | 9.4 ms | 85.8 ms |

- **Compiled, the model answers two to four times faster and its gradient
  takes nine times longer**, so an update compiled throughout is slower:
  about 24 s where it takes 14.6 s. Why the gradient is slow is not known.
  The games and the recomputation compiled with a plain pass would come to
  about 11.8 s.
- **Its answers are not the plain model's in the third decimal**: up to
  0.002 in an answer's score and 0.0008 in the value, where on a processor
  the two differ in the last digit. A compiled model gives the same
  answers twice. Why the two differ that much on the card is not known.
- **Compiling took 530 s for 39 versions**, at the start of the process.
- **It is not in use.** A fifth of an update is what there is to gain, and
  games played with one model and trained with another by 0.002 are not
  something to take on without knowing where the difference comes from.

Run again on 2026-10-13 with parts of the model left out of what is
compiled (`--leave` of the tool on the branch `speed-trials`), on from a
trained checkpoint, an update of 46,540 decisions
(`analysis/speed-trials/compile-*.txt`):

| | Plain | The LSTM left | The LSTM and the identity tables left | All compiled |
|---|---|---|---|---|
| The model's turns in the games | 2.6 s | 2.3 s | 2.7 s | 1.1 s |
| The games' estimates worked out again | 2.16 s | 2.37 s | 2.57 s | 1.01 s |
| A pass, 256 decisions a step | 5.1 s | 16.3 s | 13.6 s | 15.1 s |
| Of a step: the model's answer | 10.2 ms | 10.3 ms | 12.0 ms | 2.5 ms |
| Of a step: the gradient | 9.0 ms | 68.9 ms | 52.6 ms | 69.8 ms |
| Its answers off the plain model's, in a score | | 9.5e-6 | 9.5e-6 | 1.2e-3 |
| Versions compiled, and the time it took | | 20, 97 s | 112, 66 s | 38, 459 s |

- **All that compiling gained was the LSTM.** With the LSTM left to cuDNN
  the model answers no faster compiled than plain. Compiled, the LSTM is
  taken apart into plain steps, and that is faster than the kernel
  PyTorch has cuDNN make a padded batch's forward with, which was the
  slow thing: the trainer now gives the LSTM a packed batch, for the same
  gain without compiling (the paragraph above on the LSTM given a packed
  batch).
- **The gradient is slow whatever is left out**: seven times slower with
  the LSTM left, six with the identity tables left too. The LSTM is not
  the cause, and the tables are a part of it at most. Not known.
- **The third decimal was cuDNN's reduced precision.** With the LSTM left
  the compiled model's answers are the plain model's to the last digits.
  The plain model's own answers move by 1.3e-3 in a score and 6.6e-5 in
  the value when `torch.backends.cudnn.allow_tf32` is turned off, which
  is how far the model compiled whole is from it.
- **Compiling stays out of the trainer.**

`--matmul-precision high` runs the timing with
`torch.set_float32_matmul_precision('high')`, which the compiler advises
on a card with tensor cores, and says how far one minibatch's answers move
from training's precision. Not measured on the card yet.

**Left open, to look into before compiling is taken up again** (noted on
request, 2026-10-09). Nothing below is measured.

- **The LSTM is the first lead, for all three surprises.** (Followed on
  2026-10-13, above: it is why the versions are many and the answers
  differ, and it is not why the gradient is slow.) Compiled, an
  LSTM is not handed to cuDNN whole as the plain model's is: PyTorch takes
  it apart and walks the sequence one step at a time
  (`one_layer_lstm` in `torch/_decomp/decompositions.py`; a processor has a
  library call for it, a card has not). The tool asks for that with
  `torch._dynamo.config.allow_rnn = True`. What would follow from it:
  - *39 versions*: a version for each length of the seat's history, and an
    update's games open with every length from 1 to 32.
  - *the slow gradient*: backward through up to 32 separate steps where
    cuDNN has one call. It grows with the batch (86 ms at 256 decisions,
    177 ms at 512), so it is work and not overhead.
  - *the third decimal*: the plain LSTM is cuDNN's and the compiled one
    plain float32 steps. Whether cuDNN runs it at the reduced precision of
    tensor cores (`torch.backends.cudnn.allow_tf32` is on unless turned
    off) is the thing to check: the plain model's own answers with it
    off, against its answers with it on.

  The try that follows: leave the LSTM out of what is compiled (`allow_rnn`
  off, the limit of versions still raised), so that it stays cuDNN's and
  the rest is compiled around it, and run the tool again.
- **`torch.use_deterministic_algorithms(True)`**, which training sets, may
  send parts of a compiled gradient down slow paths. One run of the tool
  without it would tell.
- **The compiler's own warnings**, as leads:
  - `TensorFloat32 tensor cores for float32 matrix multiplication
    available but not enabled`: what `--matmul-precision high` times. It
    is told at every compiling on a card with tensor cores and measures
    nothing. What it could save is bounded by the traced update of
    2026-10-13: PyTorch's own matrix products (`addmm`, `mm`) were 0.48 s
    of the 6.2 s the card's kernels ran, in an update of 14 s. cuDNN's
    LSTM, half of the card's work, is at that precision already (the list
    of what is left to try for speed had it; the trainer now gives it a
    packed batch).
  - `Not enough SMs to use max_autotune_gemm mode`: the card is under the
    size at which the compiler writes its own matrix products, so
    `mode="max-autotune"` has less to offer on it.
  - `_POSIX_C_SOURCE redefined`, from the C compiler on two headers: no
    lead.
- **`TORCH_LOGS=recompiles`** on the card names what each version was
  compiled for.
- **The use the figures point to** is the compiled model where no gradient
  is needed, in the games and the recomputation, and the plain one in the
  pass. It needs the two to give the same numbers, or the difference
  between them accepted.
- **Compiled for sizes fixed in advance** (proposed on 2026-10-13): every
  figure above is of the model compiled with every size left free.
  `tools/call_timing.py CHECKPOINT --itself 260 --compile` measures the
  model compiled with none left free
  (`torch.compile(model, dynamic=False)`), on the batches of an update's
  calls filled out to the widest any call of the update is (the list of
  what is left to try for speed has the tool, the calls and what filling
  out does): an answer alone and replayed from a CUDA graph, how far its
  numbers are from the plain model's, what compiling a version took, and a
  step of a pass at a minibatch's sizes. On the CUDA machine the same day
  (`analysis/call-timing/half-itself-compiled.txt`; 89 cards of the
  state, decks of 48 and 55, 32 earlier answers, 27 answers of 2
  members), a call whole:

  | Decisions in a call | As it is | By a graph of the plain model | Compiled | By a graph of the compiled model |
  |---|---|---|---|---|
  | 1 | 3.13 ms | 1.10 ms | 1.67 ms | 0.70 ms |
  | 8 | 3.31 ms | 1.29 ms | 1.80 ms | 0.85 ms |
  | 32 | 3.39 ms | 1.56 ms | 1.89 ms | 1.06 ms |
  | 64 | 3.55 ms | 1.93 ms | 2.05 ms | 1.32 ms |
  | 128 | 3.77 ms | 2.68 ms | 2.48 ms | 1.88 ms |
  | 256 | 4.31 ms | 4.43 ms | 3.98 ms | 3.19 ms |
  | 320 | 4.25 ms | 5.49 ms | 4.92 ms | 4.05 ms |

  - **For the answers the sizes were what stood in the compiler's way.**
    A version for each of the seven batches compiled in 7 to 11 s, where
    it took 38 versions and 459 s with the sizes free. The answer alone
    takes 0.91 to 0.99 ms up to 64 decisions where the plain model takes
    2.4 ms, and 0.27 to 0.61 ms replayed from a graph, where the plain
    model's replay takes 0.60 to 1.14 ms.
  - **An update's calls: 2.20 s by graphs of the compiled model, 2.99 s
    by graphs of the plain one, 5.78 s as they are**, each way taken
    wherever it is the shorter.
  - **Its numbers are the plain model's but for last digits**: 1.9e-6 to
    2.3e-5 apart in a score on the same batch, where the model compiled
    whole was 1.2e-3 off. Compiled with the LSTM and with the LSTM left to
    cuDNN came out the same, in every time and every number, and the
    second took its versions from the compiler's cache in 1.1 to 1.6 s:
    given a packed batch, as the trainer now gives it, the LSTM stays
    cuDNN's either way, it seems. Not looked into.
  - **The gradient compiled takes 107 ms a step where it takes 8.6 ms**
    (10.1 ms with the batch filled out), at fixed sizes too: the free
    sizes were not why. The answer with its gradient kept takes 1.95 ms
    for 2.95 ms. On a processor the compiled gradient takes what the
    plain one does (68 ms for 73 ms at 255 decisions), so it is the
    card's. Compiled, the gradient of every lookup in a table is written
    into a table of zeros the general way, 62 of them a step, where the
    plain model's embedding has a kernel of its own; and that is the
    cost: with the tables looked up by `index_select`, which takes the
    general way too, a plain step takes 108.5 ms where it takes 15.4 ms
    (measured the same day; in the list of what is left to try for
    speed, a step of a pass as a graph).
  - **What follows.** Compiling stays out of the pass. For the games'
    calls it adds 0.8 s an update to what graphs of the plain model give,
    for a compiler at the trainer's start and numbers a further 2e-5
    from the plain model's; a compiled answer is recorded in a graph as
    the plain one is, so it can come after.

A suite is a JSON file:

```json
{"name": "tiny", "primary": "blackwing-vs-monarch", "cases": [
  {"name": "blackwing-vs-monarch", "learner": "blackwing", "opponent": "monarch", "games": 128}]}
```

`games` is even: the learner goes first in half of them.

**Left to try for speed** (kept on request, 2026-10-09). Nothing below
is measured unless it says so, and none of it is in the trainer.

- **Many duels in flight, an update every so many of their decisions**
  (2026-10-16, after another trainer's figures were given: 230
  million decisions an hour of PPO on an RTX 4070 and an i9, from 4,096
  duels played without a stop, each update made of the decisions since
  the one before, an unfinished duel's value standing in for its result,
  and the decisions' arrays kept on the card). Ours the same day, the
  three arms of thirteen decks at once through MPS on the RTX 4070 SUPER:
  44 million decisions an hour (401,000 duels), 14 to 15 million a run.
  An update of `half`, 13.0 s on average, is 5.6 s of games (the model's
  calls 2.8 s, 764 of 70 decisions on average and 160 at most; the
  engines 3.6 s beside them), 5.2 s of the pass (1.3 s of it gathering
  the minibatches from the library's keeping and taking them to the
  card) and 2.1 s more, most of it the games' estimates worked out again.
  Played again here, three of its updates had 12 to 47% of their calls
  answer fewer than ten decisions and 11 to 23% full; the last 40% of the
  calls held 2 to 12% of the decisions. Duels in flight would fill every
  call, about 2 s of an update by those costs; the pass and the second
  estimates are untouched, and are what arrays kept on the card and the
  games' own estimates would take away (the second changes the numbers,
  so a variant). The processor is not near its end: the copy's pilots
  played 520 duels in 1.2 s, about 1.5 million an hour. What is not known
  is the model's cost a decision in calls and minibatches of thousands on
  this card, which decides whether the whole could be several times
  faster; `experiments/commands/wide-calls.sh` measures it (four minutes,
  the card free). **Measured on 2026-10-17** (`analysis/wide-calls`, the
  thirteen-decks copy, an update of 1,020 duels in flight, the most an
  update's seeds allow): a call of 1,020 decisions takes 5.5 ms whole,
  where one of a decision takes 2.5 and one of 255 3.3, so 5 µs a decision
  against 19 at the 160 a call holds now; the model's answer in it is 1.8
  to 2.0 ms up to 256 decisions and 3.0 ms at 1,020. Filled out to the
  trainer's fixed sizes (96 cards of the state, 32 answers) a call of
  1,020 takes 16.0 ms: wide calls want graphs of their own sizes or none.
  Outside the model's turns those games took 5.5 s for 120,390 decisions,
  45 µs a decision with their tail, which wide calls leave the larger part.
  Its training is another training, an experiment: a duel would be stopped only between transactions and go
  on with the next update's weights, and a looping duel would no longer
  slow an update, only hold a place, so the decisions cut off would be
  the sign to watch. **Built on 2026-10-18 as options, not yet tried on
  the card** (`train --duels-in-flight N --rollout-estimates kept`, under
  `train` above): a duel is not stopped at all but trained on once, when it
  has ended, with what the weights that answered it gave, so that no
  transaction is cut and nothing is valued in a duel's stead. By the
  measurements here, with minibatches of 1,024 beside it, an update of
  `half` might go from 13 s to 7 or 8: about 1.7 times, the engines'
  45 µs a decision then the larger part. To be tried as an arm of the next
  experiment, once it is decided.
- **The games' own estimates are the recomputed ones to the last digits**
  (measured 2026-10-19 on the CUDA machine, nothing else running there:
  `half`'s settings from its checkpoint after 1,920 updates of PPO, the
  games of three updates, the weights held). Over some 57,000 decisions an
  update, the chance of an answer as its game gave it and as the update
  works it out again differ by 1e-7 on average in the logarithm, 1.7e-4 at
  most; the value by 5e-8, 1.1e-4 at most; the ratio of PPO is moved from
  1 by 1.7e-4 at most, against its clip of 0.2. Working them out again
  took 1.1 s of an update played alone, some a tenth of it. So
  `--rollout-estimates kept` without duels in flight changes an update by
  its last digits only, as the calls from graphs do. **Adopted as the
  default on 2026-10-20**, after seventy-one decks had started with it.
- **A card's description costs the pass two thirds more** (measured
  2026-10-17 on the CUDA machine, the three copies of
  [the experiment on card descriptions](../experiments/2026-10-17-card-descriptions.md)
  at once through MPS, over their updates 180 to 229): an update of a copy
  told the descriptions takes 9.8 s, its pass 6.9 to 7.1 s, where that of
  the copy that only hides identities takes 6.3 s, its pass 4.2 s; their
  games are within half a second of each other. A call works every
  description of the table out once (3,762 rows of 256 numbers to 96) and
  gathers each card's from them, and the gather's backward scatters every
  card of the minibatch into the table's rows, which under deterministic
  algorithms PyTorch sorts first. Where the time goes is not measured. To
  try on a free card: gathering each card's fixed row first and projecting
  it there, which leaves a matrix product in the backward. It sums a
  gradient in another order, so it is for a later run, once it is decided.
- **Larger minibatches pay now** (measured 2026-10-17,
  `analysis/wide-calls/pass*.txt`, the thirteen-decks copy, 45,566
  decisions, the card alone): a pass takes 2.63 s in minibatches of 256,
  1.83 s of 512, 1.68 s of 1,024 and 1.63 s of 4,096, where on 2026-10-09,
  with the deck lists in the input, none was shorter than 512's. A step of
  1,024 is 10.5 ms gathering the minibatch, 3.7 taking it to the card, 8.4
  the answer, 17.9 the gradient and 1.4 the optimizer: the gathering and
  the copy, what arrays kept on the card would take away, are 38% of it.
  Another training (fewer, larger steps); ygo-agent's minibatch of 512,
  with a clip of 0.25, was an arm of the training-differences experiment
  and not told apart from its control. Float32 matrix products in TF32 save
  nothing (2.62 s for 2.63) and move a score by up to 0.2; Adam fused saves
  0.4 ms a step, 0.03 s a pass.
- **A wider members width for the graphs** (agreed on 2026-10-16: to be
  measured once the experiment on thirteen decks is over). Answers of more
  than two member cards keep a fifth of the calls of the network of
  thirteen decks off the graphs (the section above on calls from CUDA
  graphs). **Measured on 2026-10-17 and not worth it**
  (`analysis/wide-calls/calls-members-N.txt`): a call from a graph of 2, 4
  and 8 members takes 1.02 ms at one decision whatever the width, but 2.44,
  2.81 and 3.58 ms at 128 and 4.05, 4.96 and 6.97 at 254. The calls a
  width of 4 would take in are the large ones (151 decisions at the
  median), where a graph of 4 members (about 3.2 ms) is no shorter than the
  call as it is (3.1 ms), and every other call is longer from it: a wash
  or a loss. By the rate at
  which the deck lists' calls came onto the graphs, the calls left off
  would save about 0.4 s of a 13.4 s update, less what the wider filling
  costs every call. A width that pays would be a variant for a run after
  it, once it is decided, since a call filled out wider differs in its
  last digits.
- **One queue for the pool's threads** (proposed; built and measured on
  2026-10-09, not in use). A duel goes to the thread of its number, dealt
  in advance, and a thread that has played its share cannot take
  another's. `DuelPool(..., shared=True)`, which a trainer opens with
  `TrainerConfig.pool_shared` (not on the command line), has every thread
  draw its next duel from one queue as a slot of its own comes free. A
  round holds the same duels every time either way, and games the pilot
  plays are the same games kept under the same rows: an imitating update
  is the same update, to the last digit (`tests/test_pool.py`). On the
  development machine, 520 games:

  | Threads | 8 | 12 | 20 | 24 |
  |---|---|---|---|---|
  | Games the pilot plays, duels dealt | 2.35 s | 1.85 s | 1.38 s | 1.37 s |
  | The same, one queue | 2.14 s | 1.69 s | 1.25 s | 1.22 s |

  - **Games the pilot plays are 9 to 11% shorter.** A thread plays its
    whole share in one go, and dealt duels wait for the thread whose share
    is longest. On the CUDA machine that would be about 0.15 s of an
    imitating update's 7.2 s; not measured there.
  - **Games the model plays do not change.** With a decision of the
    learner's seat answered every round (the pilot's answer given for the
    model's, 20 threads of 16 duels): 40,569 decisions in 341 rounds either
    way, the engines 2.4 to 2.6 s both ways. Every duel in flight is
    answered once a round, so what counts is how full the slots are, and
    they are full a little longer (77 rounds of 320 decisions against 67).
  - **What an update whose games the model plays loses is its last
    duels**, with either queue: 115 of the 341 rounds hold eight decisions
    or fewer, 29 of them one, about 1% of the decisions for a third of the
    calls to the model. At the 3.3 ms a call costs on the card whatever it
    holds, that is about 0.4 s of the model's 2.2 s.
- **In games the model plays, a round waits for its slowest thread.**
  Every duel stops at each decision of the model's seat, and the pool
  gathers one from every duel in flight for the model to answer together.
  The engines then stand still while the model answers (2.3 s an update)
  and the card while the engines play (3.1 s). Answering decisions as
  they come ready would end both waits. A run then repeats itself only if
  a decision's answer does not depend on what shares its batch, which
  batches of one fixed shape might give and today's do not.
- **The games' estimates are worked out again before an update** in
  batches that do not depend on how the games were played, 2.1 s. On the
  pool's route the games are played in the same batches every time, so the
  estimates of the games themselves could serve. It changes last digits.
  Measured on the CUDA machine on 2026-10-13 (the variant `estimates` of
  the branch `speed-trials`): an update is 2.04 ± 0.12 s shorter (-14.6%)
  and is another update that repeats itself. Working the estimates out
  again moved a probability's logarithm by 6.2e-5 at most (7.1e-8 on
  average) and a value by 7.0e-5, over 46,168 decisions. Not taken up:
  a decision.
- **Minibatches of 512 decisions** save 0.8 s of a pass (measured, above)
  and are another update.
- **Half precision** on the card: not tried.
- **Compiling the model**: the list above.
- **With a checkpoint held in the other seat the model's part of the games
  triples** (measured in the base of the experiment on self-play,
  2026-10-11): 7 to 8 s an update where it is 2.5 s with a pilot there, in
  twice as many batches of half the size (700 to 850 of 65 to 76
  decisions, for 400 of 115). The engines take about the same, 3.3 to
  3.5 s for 2.9 s. A round's decisions are split between the two networks
  and each answers its own in a call of its own; whether a round could be
  gathered so that each answers fuller batches is not looked into. All of
  such an update is 21 to 22 s for 14.0 s.
- **Lua is a quarter to a third of the engines' part, and the engines are
  a fifth of an update** (profiled on the development machine,
  2026-10-12). The duels run on the reference Lua 5.4.8 with its plain
  interpreter, built as C++ inside OCGCore; a card script is compiled once
  a process and every duel loads its bytecode (`script_bytecode_cache.h`
  of the core). The engines' part of an update is the CUDA machine's own
  figure: 2.9 s of 14.0 s against the pilots, 3.3 to 3.5 s of 21 to 22 s
  against a held checkpoint. Where the processor's time goes inside it, by
  the code it is spent in, with the handling of memory given to whoever
  asked for it:

  | | 768 games between pilots, one thread | 208 games the model plays, through the pool |
  |---|---|---|
  | The C++ core | 40% | 31% |
  | Lua | 33% | 25% |
  | The seats (Rust: what a seat is told and sees) | 20% | 32% |
  | Setting a duel up, the pilots' answers, the model's input | 7% | 12% |

  - **The interpreter's loop is a seventh of Lua's time.** Of all the
    time between pilots: tables 7%, garbage collection 5%, the C API 5%,
    the loop of the VM 4%, strings 3%, calls 2%, loading the kept
    bytecode 2%. A faster interpreter or a JIT works on the loop and the
    calls. LuaJIT is no replacement as it stands: it is Lua 5.1, the card
    scripts are written for 5.3 and later (bitwise operators in 9,561 of
    22,793 files, integer division) and so is the core's use of the C
    API.
  - **The largest single thing is the field, asked for and read after
    every step.** The duel asks the core for every location
    (`Duel::field`, `OCG_DuelQueryLocation`, `card::get_infos`) and the
    seats read the answer (`feed_update_pair`, `query::read_record`, in
    buffers grown by `realloc`): 36% of the time between pilots (19%
    asking, 17% reading) and 41% through the pool (11% and 30%, the
    reading under `ygo_env::seat::Table::feed`).
  - **The pilots' own answers are 1 to 2%**, and the model's input
    (`Environment::pack`) 2%.
  - **What it bounds.** By those shares all of Lua is about 5% of an
    update against the pilots (0.7 s of 14.0 s) and the field asked for
    and read about 8% (1.2 s). Between pilots one thread plays 62 games a
    second.
  - **How.** `gprofng collect app -p hi` (sampling by the clock: `perf` is
    not open to users on that machine), a function's own time sorted by
    the archive that defines it and the memory handling by its callers,
    which is approximate (a third of the stacks end at the VM's loop). One
    run each; the lists and the two scripts are in
    `analysis/engine-profile/` there.
  - **Profiled again on 2026-10-20** at the policies' `82a7c61` (the
    engine's arena reused, the seat of that pin), the same 768 games
    between pilots on one thread (`analysis/engine-profile/2026-10-20/`):
    the core 41.7%, Lua 33.9%, the seat layer 14.1% (it was 19.9%), the
    pilots 5.9%. Inside Lua's own 32%: tables 7.5%, garbage collection
    5.2%, the VM's loop 4.7%, the C API 4.7%, strings 3.4%, calls 2.4%,
    loading and compiling 2.1%. Lua is entered from the core's filters
    run card by card (half), effects' functions (a quarter) and loading a
    duel's scripts, 8.4% of the whole by itself. The field asked for is
    18% inclusive (`card::get_infos` 16%, half of that writing the values
    into its byte buffer, `insert_value`, and the other half working out
    a card's level, ATK and DEF through its effects), read by the seats
    9% (it was 17%).
  - **Compiling the card scripts** (asked on 2026-10-20). Into
    C on Lua's own runtime (as an ahead-of-time compiler for Lua 5.4
    does; LuaJIT cannot, being 5.1) takes away the VM's loop and a part of
    the calls and loading: some 7% of the engines at most, the tables,
    the collector, the strings and the API staying as they are. Into C++
    that calls the core itself, which no tool does, all of Lua's 32% at
    most, the engines 1.4 times as fast; a compiler of a dynamic language
    for the 22,793 scripts and the shared procedures, every card to play
    as before (the digests would tell where one does not). By the
    engines' 3.6 s of a 13 s update of `half` (above), at most 0.25 s and
    1.2 s an update; more of an update if duels in flight make the engines
    its larger part. Before either, cheaper and exact: the field's values
    written straight into a buffer of the right size (done on 2026-10-20,
    the item below on the field written straight into its buffer), the
    field asked for less often (not exact as it would be done: the same
    item), a duel's scripts loaded once for many, the collector's settings.
- **The engine built from a profile of its own games is a tenth faster,
  and plays the same games** (measured on the development machine on
  2026-10-20, by flags alone: no file changed). The core and Lua are
  built at `-O2` by `crates/ygo-policies-duel/build.rs` of the policies,
  the Rust with the default release profile. Five builds of
  `policy-bench` in a scratch target dir, the same 768 games
  between pilots on one thread (seed 730000) three times each,
  interleaved, the median:

  | Build | 768 games | Against now |
  |---|---|---|
  | As now | 11.30 s | |
  | Rust with fat LTO, one codegen unit | 11.34 s | none |
  | That, and the C and C++ at `-O3` | 10.95 s | −3.1% |
  | Fat LTO, and the C and C++ built from a profile (GCC's `-fprofile-generate` on games of another seed, then `-fprofile-use`) | 10.13 s | −10.4% |
  | That, and the Rust from a profile too (`-Cprofile-generate`, `llvm-profdata`) | 10.24 s | −9.4% |

  Every build played the 768 games game for game as the build as it is
  (`tools/compare_bench_rows.py`). The C and C++ are three quarters of
  the engines' time between pilots and a little over half through the
  pool, so the trainer's engines would gain some 6 to 10%: 0.2 to 0.35 s
  of `half`'s 3.6 s, more of an update where the engines are its larger
  part. GCC names a profile after the object it was made for, whose path
  holds a hash of the Rust flags: the build that uses it takes the
  instrumented build's flags (`-C link-arg=-lgcov` in both), or the
  profiles are moved to the new path.

  **A build made on each machine from its own games** (on request,
  2026-10-20): `bash tools/engine_pgo_build.sh` from the repository root,
  with nothing else using the build, and again after the pins move. It
  builds the trainer's library as it is and plays a check suite of pilot
  games with it (the 71 decks against the thirteen decks' pilots, 3,692
  games); builds it to count what it runs and plays two games of every
  such pairing with it (some 4 minutes, the counters shared by 20
  threads); writes `.cargo/config.toml` (machine-local, git-ignored),
  with which every `cargo build --release` in the tree builds the engine
  from that profile in `target/engine-pgo/data`; builds the library so;
  and plays the check suite again. Its games must be the same, or the
  config is moved aside and the library built as it was. The wide
  Blackwing screen is timed with both. On the development machine its
  6,656 pilot games through the pool, on 20 threads, took 7.40 s for 7.77 s
  and, the script run as written, 7.48 s for 7.74 s, less the 1.7 s a run
  starts in: 3 to 5% less, under half the tenth of `policy-bench` on one
  thread, the C and C++ being a smaller part of the pool's engines. In an
  update of `half`, some 0.1 to 0.2 s of the engines' 3.6 s. To build the engine as it is again: move the config aside and
  `cargo build --release`. A build of the policies' bench in the tree
  takes the config's flags too, whose profiles name other paths: it is
  built as it is, quietly.
- **The field written straight into its buffer, and decoded only when it
  changes: the pool's pilot games a ninth to a seventh quicker, the same
  games** (on request, 2026-10-20; measured on the development
  machine with plain builds, without its profile). Two changes:
  - **The core writes a card's record straight into room made for it**
    (`card::get_infos`, fourlastor/ygopro-core `8d14f6a`, pinned by the
    policies' `697fe49`). Each value went in with a `resize` of the query
    buffer, some 27 calls a card. Where the buffer already has room for the
    most a record can hold, the values now go straight into that room;
    written one at a time they would not have grown the buffer either, so
    the duel's arena, from which the buffer grows and which is never given
    back, is used as it was. Elsewhere, or past that room, they go one at a
    time as before. The field the policies ask for before a selection
    (`Duel::field`, twelve locations) took 5.7 µs and takes 3.9 µs, timed
    by asking for it again at every selection of 192 games between pilots.
    The sampling profile had put the buffer's growth at a tenth of the
    bench and half of the query; a second pass of writing, timed, costs
    4.2 µs of the 5.7. A first version that wrote a value a byte at a time
    gained nothing.
  - **The trainer decodes a location's refresh only when it differs from
    that location's last one** (`Refreshes` in
    `crates/ygo-env/src/episode.rs`). 84.2% of the refreshes before a
    selection are byte for byte those of the selection before (1.6 million
    selections of the 71 decks against the thirteen decks' pilots); a
    decoding depends on its bytes alone, so what was kept is what the
    decoding gives. Decoding the field took 2.2 µs a selection.
  - **Measured.** Through the pool, `evaluate-suite --pilot`: 384 games of
    Blackwing against the twelve others on one thread 6.84 s for 6.10 s
    (the core's change alone 6.38 s), the medians of three, interleaved;
    less the 1.7 s a run starts in, the games 5.1 s for 4.4 s (-14%). The
    wide Blackwing screen on 20 threads, 6,656 games, 9.47 s for 8.63 s:
    the games 7.8 s for 6.9 s (-11%). This machine's profile, made again
    after the change (`tools/engine_pgo_build.sh`, one run), takes it to
    7.88 s for the plain build's 8.44 s: with both, the games a fifth
    shorter than in the morning (6.2 s for 7.8 s). Not measured on the CUDA machine: in
    an update the engines are some 3.6 s of `half`'s 13 s, run partly
    while the model answers, and the field was a quarter of the pool's
    pilot games, so perhaps 0.3 to 0.5 s.
  - **The same games.** Every message of 3,640 games of the 71 decks
    against the thirteen pilots, the refreshes included, and every refresh
    as decoded, byte for byte (a scratch tool beside the bench, kept in
    `analysis/field-read/field_digest.rs`); through the pool the check
    suite of `tools/engine_pgo_build.sh` (3,692 games), the wide screen,
    96 games of a network of thirteen decks and two updates of PPO from
    it, a quarter of the duels against itself: the same records, model,
    optimizer, figures and fingerprints; the bench's games for moving the
    pin, plain and searched, identical at both pins; the tests of both
    workspaces and the trainer's.
  - **What is left of the field**: some 0.8 s of the 384 games on one
    thread (of 4.4 s). The query is 0.63 s of it, most of that its other
    part, working out a card's level, ATK and DEF through its effects,
    which is the rules and is left as it is; then copying the kept
    refreshes (0.07 s), showing them to the two seats (0.06 s) and the
    refreshes still decoded (0.04 s). The policies' own seats
    (`Seat::feed_update_pair`, which `policy-bench` uses) still decode
    every refresh: the same keeping there would speed the bench, not
    training.
  - **Asking for the field less often is not exact.** A selection that
    follows nothing but hints has the field of the one before in all but
    47 of 972,252 cases (the same 3,640 games): some value changes without
    a message. Asking only after other messages would show the seats a
    stale field those 47 times. Not taken up.
- **Several runs at once on one machine** (measured on a rented machine on
  2026-10-13, and not on the CUDA machine). A run leaves most of a machine
  idle: while an update's games were played 3.6 to 3.9 processor threads
  were busy on average and the card 10% to 16%, and during the pass 1.1 to
  1.6 threads and the card about 30%. Three runs of PPO against the pilots
  side by side, each with all of its pool threads, took 1.4 times as long
  an update each and did 2.1 times the work together; two took 1.2 times
  as long and did 1.7 times. A run whose pool threads are unchanged plays
  the games it plays alone. The figures are in the section on a rented
  machine. **On the CUDA machine it gained next to nothing as processes
  have the card** (2026-10-13, half of the duels against itself: 1.08
  times the work for two runs), **and through NVIDIA's MPS three runs do
  2.11 times the work of one** (the same day, after that day's changes).
  The figures are in the section on several runs at once.
- **The duels in two groups, one advanced while the model answers the
  other** (proposed on 2026-10-13; double-buffered sampling in
  Sample Factory). Built as `--pool-groups 2`. Measured on the CUDA
  machine the same day with every round split and calls as they are: the
  same games and the same model, and a longer update. With the calls
  answered from graphs it is the shorter way, and what a run on a card
  takes since that day (the section above on calls from CUDA graphs and
  the duels in two groups).
  - **What is built.** Every thread's duels are shared out evenly among
    the groups (`Pool::grouped` in `crates/ygo-env/src/pool.rs`), and where
    the model plays an update's games the groups are taken forward one at
    a time (`begin`, which does not wait, and `finish`, which does;
    `pool.sampled_games`). A duel stays in the group whose slot it took,
    and which duels are in a group follows from the order the groups are
    begun in, so a run repeats itself. Measurements and updates that
    imitate still take every duel forward together.
  - **Held to the same games**: the duels through groups are the duels
    played one at a time (`crates/ygo-env/tests/pool.rs`); on a processor
    two updates in the seven ways named above are the same games, figures,
    model and optimizer as with one group, with more and smaller calls of
    the model, and `tests/test_pool.py` holds an update to that; on the
    card the run alone played the same games and left the same model
    after each of seven updates.
  - **Measured** (`bash experiments/commands/two-groups.sh`: the runs of
    the section on several runs at once with one group and with two, half
    of the duels against itself, on from the arm's checkpoint, six timed
    updates a run):

    | Runs at once | One group, an update | Two groups, every round split |
    |---|---|---|
    | 1 | 18.6 s | 20.6 s |
    | 3, as processes have the card | 43.7 s each | 55.8 s each |
    | 3, through MPS | 26.0 s each | 27.9 s each |

    Of the run alone, an update: the model was called 2,318 times where
    it is 1,298, and its part of the games took 8.9 s where it is 5.2 s;
    what the games stood waiting for the engines fell from 3.0 s to
    0.55 s. The engines did run under the model's calls. The calls that
    were added cost more than that saved: 12.0 s of games for 10.0 s.
  - **Why: a call costs the same whatever it holds, and a second call a
    round takes back most of what the engines' time saves** (the next
    point has the costs with one group). One group takes a round of n
    decisions forward in 3.94 + 0.061 n ms. The run above, fitted the
    same way, none of its updates more than 0.25 s from it: a round split
    in two takes 7.46 + 0.041 n ms, and a round only one group has duels
    left for, 3.83 ms. A split round is 0.020 n - 3.5 ms shorter: 2.9 ms
    of 23.5 ms at 320 decisions, and longer under about 175. Of the
    engines' 0.034 ms a decision 0.020 ms is saved: the model's own part
    of a decision takes 0.021 ms where it takes 0.013 ms, which engines
    busy on every thread beside it would explain (not looked into), and
    the copies take more. Of the 1,683 rounds of the update taken apart
    below, 224 hold more than 128 decisions and 1,278 hold 32 or fewer.
    Sample Factory's case is the other one: environments that take long
    beside a call that costs little.
  - **The groups can go on together once their rounds are small**
    (`--pool-groups-over N`; measured on the card with the calls from
    graphs, in the next point).
    When the groups' last rounds hold N decisions or fewer between them,
    those on their way are waited for and every duel is taken forward
    together for the rest of the update's games (`Pool::step` takes over
    from groups that wait; `pool_grouped_batches` counts the calls before
    it). Held to the same games the same ways. One update's games at the
    experiment's size on the development machine (the second long run's
    last checkpoint, half against itself, N = 128): 914 calls where one
    group makes 683, 462 of them by groups, of 62 to 160 decisions, for
    the rounds one group answers in 231 calls. Reckoned by the costs
    measured above: about 0.45 s of an update's games less than one group
    with half against itself (the arm's checkpoint and that one alike)
    and 0.3 s against the pilots, where every round split is 2.0 s more;
    N between 128 and 256 changes that by 0.03 s.
    `OVER=176 HALVES=two bash experiments/commands/two-groups.sh` measures
    it beside the one group measured before, in about a quarter of an
    hour. What the engines take is 0.034 ms a decision, 2.7 s an update:
    no sharing out of the duels saves more, and a second call a round is
    3.5 ms.
  - **A cheaper call changes the reckoning** (the same day, reckoned from
    the costs above and from what a call takes from a CUDA graph, the
    last point below; not measured). The second call of a split round is
    what the groups lose by, and from a graph a call of 160 decisions
    takes 3.1 ms where it takes 3.9 ms, and one of a few decisions 1.1 ms
    for 3.1 ms. For the update whose calls are counted below, its rounds
    in all:

    | The calls answered | One group | Split where it is the shorter | From how many decisions |
    |---|---|---|---|
    | As they are | 11.5 s | 11.1 s | 171 |
    | From graphs of the plain model | 8.7 s | 7.7 s | 71 |
    | From graphs of the compiled model | 7.9 s | 6.7 s | 52 |

    Splitting every round stays the longer way, 9.5 s and 7.9 s: under
    some fifty decisions the engines take less than any call. The order
    that follows is the graphs first and the groups on top of them.
- **A call of the model costs the same whatever it holds, and an
  update's games take as many calls as its longest duel has decisions**,
  since a duel's next decision waits for the answer to the one before.
  Measured on the CUDA machine on 2026-10-13, half of the duels against
  itself, on from the arm's checkpoint.
  - **An update's games take 3.94 ms a call and 0.061 ms a decision**,
    and nothing else: fitted to seven updates of a run alone, none of
    which is more than 0.14 s from it. They had 795 to 1,748 calls for
    75,000 to 87,000 decisions, and their games took 7.8 s to 11.6 s. Of
    a call 3.2 ms are the model's turn, 0.3 ms the copy of the round's
    arrays, 0.2 ms the threads and 0.2 ms the rest; of a decision
    0.034 ms are the engines and 0.013 ms the model. The calls are half
    of an update's games, 4.8 s of 9.7 s on average, and a quarter of the
    update.
  - **Most calls hold next to nothing.** `tools/call_timing.py CHECKPOINT
    --itself 260` plays one update's games, keeps a batch of every size
    the model was given, and answers each again with the parts timed one
    by one (the seeds, the batch taken to the card, the answer and the
    encoders inside it, the draw, its probability, what is read back):

    | Decisions in a call | Calls | Their decisions | A call | The answer in it | The batch taken to the card |
    |---|---|---|---|---|---|
    | 1 | 518 | 518 | 3.06 ms | 2.25 ms | 0.43 ms |
    | 2 to 8 | 470 | 1,260 | 3.22 ms | 2.38 ms | 0.45 ms |
    | 9 to 32 | 290 | 5,643 | 3.31 ms | 2.36 ms | 0.53 ms |
    | 33 to 64 | 108 | 4,941 | 3.47 ms | 2.39 ms | 0.64 ms |
    | 65 to 128 | 73 | 6,673 | 3.67 ms | 2.39 ms | 0.78 ms |
    | 129 to 256 | 69 | 12,934 | 4.19 ms | 2.49 ms | 1.14 ms |
    | 257 or more | 155 | 48,425 | 4.13 ms | 2.34 ms | 1.14 ms |

    Of 1,683 calls 988 hold eight decisions or fewer, 2% of the
    decisions, for 3.1 s of the 5.6 s the calls take; 518 hold one, a
    duel played on alone after the others. A pool of 320 duels in flight
    answers 320 decisions in a call only while 320 are left to play.
  - **An answer is its operations.** It calls 321 operations of PyTorch
    whatever it holds, which launch 522 kernels and copies on the card
    for one decision and 455 for 320. A card's representation is made six
    times an answer (the state's cards, the context's, the two decks, the
    answers and their members): 168 of the 321 operations, 42 of the 62
    lookups in embedding tables. Timed one by one the six take 1.97 ms,
    where the state's encoder, which holds four of them, takes 2.66 ms: a
    part timed alone waits for the card after it, so the parts come to
    more than the 2.25 ms of an answer timed whole.
  - **What would shorten it; none of it is built.** One pass for all the
    cards of a batch where there are six: a third of an answer's
    operations less. It would move a gradient's last digits, a table's
    gradient being summed in another order. Fewer calls take another
    training. Duels that carry over the end of an update, the pool kept
    full and an update made of the duels that ended in it: the first
    decisions of a long duel would be an earlier model's, and by the
    costs above an update's games would take about 6 s where they take
    9.7 s. Or a lower limit on a duel's decisions (`--limit`, 4,096): a
    duel it stops is counted as lost.
  - **An answer replayed from a CUDA graph: a call of a few decisions
    takes a third of what it takes.** A graph is the card's work for one
    answer, recorded once and launched again in one call, where the model
    launches its 500 kernels one by one. It is recorded for one set of
    sizes, and a call's batch has seven (its decisions, and what the
    state's cards, the two decks, the earlier answers, the answers and
    their members are padded to): the small calls, which are most, would
    be filled out to sizes fixed in advance, a graph for each count of
    decisions, and the largest left as they are. It is a card's alone, and
    the development machine has no card this PyTorch can use. Measured on
    the CUDA machine on 2026-10-13 with the tool above, twice
    (`analysis/call-timing/half-itself-graphs.txt`, and
    `half-itself-compiled.txt`, whose figures these are): each kept
    batch's answer replayed at its own sizes, and filled out with zeros to
    the widest any call of the update is (89 cards of the state, decks of
    48 and 55, 32 earlier answers, 27 answers of 2 members).

    | Decisions in a call | The answer | Replayed | A call whole | By a graph | By a graph of fixed sizes |
    |---|---|---|---|---|---|
    | 1 | 2.33 ms | 0.57 ms | 3.13 ms | 1.07 ms | 1.10 ms |
    | 8 | 2.49 ms | 0.70 ms | 3.31 ms | 1.23 ms | 1.29 ms |
    | 32 | 2.45 ms | 0.85 ms | 3.39 ms | 1.48 ms | 1.56 ms |
    | 64 | 2.49 ms | 1.05 ms | 3.55 ms | 1.78 ms | 1.93 ms |
    | 128 | 2.49 ms | 1.32 ms | 3.77 ms | 2.20 ms | 2.68 ms |
    | 256 | 2.55 ms | 2.03 ms | 4.31 ms | 3.30 ms | 4.43 ms |
    | 320 | 2.43 ms | 1.55 ms | 4.25 ms | 2.84 ms | 5.49 ms |

    - **A replay gives the model's own numbers to the last digit** at the
      sizes it was recorded at, and for another batch of those sizes.
      Recording takes 4 to 15 ms, and a graph keeps 32 MiB of the card's
      memory.
    - **By those times an update's calls take 2.99 s where they take
      5.78 s**, with a graph of fixed sizes wherever it is the shorter
      way, which is up to 128 decisions.
    - **Filled out, a decision's scores are not the same numbers**: up to
      5.7e-6 apart in batches of up to 64 decisions and 1.3e-5 at 128 (on
      a processor up to 3.8e-6). A draw within that of its boundary goes
      the other way, so games answered through graphs of fixed sizes are
      now and then other games, of the same model; an update of given
      games stays the same update.
    - **The batch goes to the card in one copy in a tenth of the time**,
      laid out as one run of bytes its arrays are views of: 0.03 ms for a
      decision and 0.05 ms for 8, where array by array into tensors that
      stay takes 0.26 and 0.29 ms, and 0.43 ms as the trainer takes it.
      From 256 decisions on the two take the same.
    - **Compiled for those sizes the model's replay takes half as long
      again**: the compile's own section, above, on compiling for sizes
      fixed in advance.
    - **Built as `--inference-graphs N`** (`ygo_training/graphs.py`), and
      since 2026-10-13 what a run on a card takes where it names nothing
      (the section above on calls from CUDA graphs and the duels in two
      groups). A call of an update's
      games of up to N decisions is filled out to the next of a few
      counts of decisions (the powers of two and N) and to fixed widths
      (96 cards of the state, decks of 48 and 56, 32 earlier answers, 32
      answers of 2 members), taken to the card in one copy and answered
      from the graph of those sizes, recorded the first time they are
      asked for; a call of more decisions, or wider, is answered as it
      is. Once an update each graph's answer is held against the model's
      own to the same batch, and a graph that gives another ends all of
      them: the calls are answered as they are from there, the run goes
      on, and its figures say so (`pool_graph_failed`;
      `pool_graph_batches` counts the calls answered from graphs). Graphs
      are recorded again if the weights are no longer where they were. On
      a processor there is no graph and the calls are filled out and
      answered by the model, which is what the tests hold: a call filled
      out is the library's own batch of it in a wider one, array for
      array; and an update's games through it are the games, with the
      same figures, model and optimizer after them, in two groups too.
    - **Measured in runs** (`bash experiments/commands/call-graphs.sh`
      on the CUDA machine, 2026-10-13: the runs of the section on several
      runs at once, half of the duels against itself, on from the arm's
      checkpoint, six timed updates a run; calls of up to 160 decisions
      from graphs, and the two groups while their rounds hold more than 64
      decisions). An update in each run, and the duels an hour of all of
      them:

      | Runs at once | As it is | Calls from graphs | And the duels in two groups |
      |---|---|---|---|
      | 1 | 18.6 s, 100,800 | 16.3 s, 114,500 | 15.5 s, 120,700 |
      | 3, as processes have the card | 43.7 s, 128,600 | 32.9 s, 170,700 | 32.5 s, 173,000 |
      | 3, through MPS | 26.0 s, 215,700 | 23.6 s, 237,800 | 23.0 s, 244,000 |

      - **A run alone is an eighth shorter with the graphs and a sixth
        with the groups on top**: its games take 7.8 s and 6.9 s where
        they take 10.0 s, the model's part of them 3.0 s and 3.6 s for
        5.2 s. With every duel together 84% of the 1,298 calls of an
        update are answered from graphs; in two groups all 1,604 are.
      - **The same games every way**: the run alone played the games of
        the run as it is in each of its seven updates, with the graphs
        and with the groups, so that no draw fell the other way there.
      - **No graph failed**, alone, three at once or through MPS, and
        three runs through MPS with graphs and groups play 244,000 duels
        an hour: 2.4 times a run alone as it is, where they played 2.15
        times before. As processes have the card, three runs gain a third
        from the graphs: fewer of the card's jobs to take turns over.
      - **Through the trainer's own way a call takes** 1.26 ms for a
        decision, 1.31 ms for 8, 1.59 ms for 32, 2.00 ms for 64 and
        2.96 ms for 128 (the tool, the same day), where the call as it is
        takes 3.07 to 3.69 ms.
    - **A step of a pass as a graph: it can be recorded, and saves a
      fifth of a step** (the tool, the same day, at a batch of 256
      decisions filled out; the answer with its gradient kept, a loss,
      the gradient, its clipping and the optimizer's step). No part of a
      step waits for the card, against what was expected of the tables'
      gradient. Recorded whole, a step replays in 12.6 ms where it takes
      15.4 ms step by step, with the weights of the steps made one by
      one, on its batch and on another of its sizes; recording takes
      32 ms and the graph keeps 91 MiB. A step is the card's work then,
      not its launching: 0.85 s of an update's pass by that. It would
      take minibatches filled out to fixed sizes, a loss whose shapes do
      not follow the minibatch's transactions, and an optimizer that can
      be recorded (`capturable=True`), whose steps are not the same
      numbers as the trainer's: 3.8e-6 apart in a weight after three
      steps. Not built.
    - **Why the compiled gradient is slow is told by the same try**: with
      the tables looked up by `index_select`, whose gradient is the
      general kernel's and not the embedding's own, a step takes 108.5 ms
      where it takes 15.4 ms. That is the compiled gradient's 107 ms: a
      compiled model takes a table's gradient the general way.
- **A run needs few threads, and its duels in flight** (CUDA machine,
  2026-10-13, the pool with 320 duels in flight every time, against the
  pilots). The engines' part of an update's games is 2.87 s with 20
  threads of 16 duels, 3.64 s with 10 of 32, 3.95 s with 8 of 40, 4.73 s
  with 5 of 64 and 5.47 s with 4 of 80; the model's part (2.5 s in 394
  calls of 117 decisions) does not change. Half the threads cost 5% of an
  update and a fifth of them 18%. What the model's batches follow is the
  duels in flight: a machine with fewer threads keeps them by more duels
  a thread. A run alone keeps 2.2 of the machine's 20 threads busy.
- **Most of a batch is padding** (the same day, an update against the
  pilots). Of what a minibatch is padded to, the share that is there: the
  answers' members 1%, the context's cards 7%, the answers 24%, the
  state's cards 57%, its earlier answers 82%, the other deck 86%, its own
  deck all. Every row is worked out to its batch's longest. Not looked
  into: encoding only what is there.
- **The machine.** Per thread the Core Ultra 9 285K does the processor's
  parts 1.3 to 1.4 times as fast as the CUDA machine's processor (the
  games above, and minibatches built in 2.4 ms where it takes 3.4 ms). By
  that an update of 13.3 s would take about 10.8 s there; more threads
  than 20 add little, and a larger card takes at most the third of an
  update that is the card's arithmetic.

## `evaluate-search`

```sh
uv run python -m ygo_training evaluate-search \
  runs/search-arm/checkpoints/model-00031.pt runs/pilot-arm/checkpoints/model-00031.pt \
  --device cuda --workers 24 \
  --records analysis/search/heldout.jsonl \
  --output analysis/search/heldout-evaluation.json
```

Measures checkpoints on the rows of a search run they did not train on, in
two ways:

- **Its deals.** The model plays every deal of the run with its first
  choice. The rows hold what the pilot and the search made of the same
  deals, so the three are compared deal by deal (`deals.references`,
  `deals.paired`).
- **The answer key.** Every searched duel is played again as recorded, and
  at each decision of the learner's seat the model is asked what it would
  answer. Nothing it says is played. Where the search left its pilot's
  answer (a deviation) the model's first choice is the search's, still the
  pilot's, or neither; everywhere else the search kept the pilot's answer,
  and the model's is that one or another. Both are given overall, by kind of
  decision and by what the pilot answered and the search played instead
  (`answer_key.by_change`), and every deviation is listed with the
  probability the model gave either answer.

The answer key is not reproduced to the last answer. The decisions of the
recorded duels reach the model in batches made up as they arrive, the
probabilities of one checkpoint differ from one measurement to the next in
their ninth decimal, and a first choice between two answers that close
falls either way: a small, barely trained model measured three times had
another answer than the pilot's at 184, 184 and 185 of 867 decisions.

If the run was made with `--log true` the rows also say which decisions the
search examined, and the kept decisions are given for those apart. The games
are the same with and without the log.

With the log the model's answers are also read by what the search measured
(`answer_key.tried`). At every decision it examined, the search played the
pilot's answer and up to eleven alternatives out in the same worlds. Where
the model's first choice is another answer than the pilot's, it is worth the
lead the search measured for it in the worlds of its first stage: how much
more often it won there than the pilot's answer. The table gives, for the
decisions examined:

- `others`: the model's answers other than the pilot's, and how many a game;
- `outright`: those given where the pilot's answer won every world, without
  a lead. Nothing can be ahead there, and a search tries nothing else unless
  it ran with `--log full`; with the full log there are none, and a line
  below the table says how many of the answers tried were given there and
  what they add up to a game;
- `untried`: those the search did not try, past its twelfth alternative;
- `tried`: those it did try, with how many were `ahead` of the pilot's
  answer, `level` with it or `behind`, their mean `lead` in points of win
  rate, and the leads of a game added up (`sum a game`).

Below the first choice come the model's heaviest alternative, taken once the
model gives it half, a fifth, a tenth and a twentieth of what it gives the
pilot's answer; the search's own choices; and any other answer the search
tried. With the full log a model that has not seen the games is not
flattered: its sum estimates what its answers gain or lose against the
pilot's, the pilots playing on. Without it the leads are measured only where
the pilot's answer lost some world, which flatters every alternative alike:
read a model beside another model, and beside any other answer. The search's
own choices were taken on those very worlds and are flattered either way. A
like card from the same pile counts as the same answer, as the search tries
them. A log whose alternatives are not the ones the decision offers is
refused, as a record with another digest is.

The model's lead head is asked at the same decisions (`answer_key.tried.led`
and `.head`). A model that follows its head gives its policy's first choice
unless it predicts for another alternative a lead over that choice of at
least a margin, and then the alternative it predicts the largest lead for.
For margins of 5 to 30 points the second table gives the first choices so
`changed`, the answers other than the pilot's that result (`others`), what
those are worth (`sum a game`) and the `gain`: what the answers taken lead
the first choices they replace by, added up over a game. A third table has
the changes apart by where they start (`split`): from the pilot's answer to
another, from another back to the pilot's, and from one other answer to
another, each with how many and what they gain a game. A model whose first
choice is not its pilot's answer gains by going back to it, which is its
copy repaired and nothing the search found. The last line is
what the head knows: how its predictions correlate with the measured leads
over every alternative, and the mean measured lead of the alternatives it
ranks highest.

Then the value head against the worlds. At a decision the search examined,
the rate at which the pilot's answer wins the worlds of the first stage
estimates what the decision is worth to a seat that goes on as its pilot
does. Over 8 worlds a rate is not that worth: of the spread of the rates
over the decisions, what is real is what remains without the spread of a
mean of so few results. The model's value is read against that part, at
the examined decisions that open a transaction, which is where the head is
trained: its correlation with the rates, the share of what can be told
that a straight line through its values explains, the share its values
explain as they stand, and beside them the share of a single result that a
perfect predictor would explain there. The last line gives the first by
turn.
`tools/policy_search_predictability.py TRAIN MEASURE` gives the references
for it from two logged runs alone: what a perfect predictor explains at
the decisions of MEASURE, and what a small network tells that sees only
the log's `situation` (life points, counts of cards, turn and phase),
taught on TRAIN the games' results or the worlds' rates, without and with
the opponent's deck.

`tools/learner_report.py DIRECTORY` is the report of the experiment on the
learner itself, for the folder `experiments/commands/learner-value-board.sh`
leaves: every run's value head by what it explains of the results of games
new to it and against the worlds, the copies by their agreement, the suite
and the held-out deals, the lead heads, and the readings as that
experiment's record fixes them.

`tools/learner_follow_up_report.py DIRECTORY` is the same for its follow-up,
the folder `experiments/commands/learner-follow-up.sh` leaves: four runs of
one pass, the board by the positions a game's result is taught at, each
measured after the same updates. A copy is read deal by deal over all the
checkpoints measured.

`tools/ppo_head_report.py DIRECTORY` is the report of the experiment on PPO
with a head that predicts, for the folder `experiments/commands/ppo-head.sh`
leaves: five runs of PPO from one checkpoint by how the value head is
taught. It says whether the arms' first update had the same games; what
each head explains of the results of its own games, the entropy of the
sampled answers and the games won while sampling; the held-out deals each
arm wins after the same updates, read deal by deal against the checkpoints
of its start; where the first choice leaves the pilot's answer, by the kind
of the decision; and the readings as that experiment's record has them.

`tools/ppo_long_report.py DIRECTORY --long UPDATES` is the report of its
follow-up, for the folder `experiments/commands/ppo-long.sh` leaves: two
runs as long as `one-pass` and from the same checkpoint, one with another
seed for its games and one at another learning rate, read as the arms
were; and `one-pass` made long, a row for each of its checkpoints (the
held-out deals it wins and how many more than the pilot on the same deals,
the suite, where its first choice leaves the pilot's answer, and the games
it wins while sampling), read by its last three once it has had the
updates of `--long`. A measurement keeps each reference's result on every
deal (`deals.references.pilot.outcomes`), which that reading is made on.
`--first NAME` reads another run made long than `one-pass` (its
continuation is `NAME-long`), and `--beside DIRECTORY` sets it beside the
first long run of that folder: the deals each wins after the same updates,
this run over the other deal by deal, and over it at the end. The script
makes such a run with `NAME`, `FIRST`, `FIRST_OUT` and `BESIDE`, and one
short run alone with `SHORT`.

`tools/specialist_copy_report.py DIRECTORY --updates UPDATES` is the report
of the copies of a specialist, for the folder
`experiments/commands/specialist-copy.sh` leaves: new networks that imitate
a checkpoint as the pilot's copy imitated its pilot, one taught the
teacher's choice and one its distribution. It says whether the copies
started as the pilot's copy did and were taught on the same games; how
each learned (its agreement with the teacher, whose answer it gives where
the teacher leaves its pilot, how far its answers are from the teacher's);
the held-out deals each wins after the same updates, beside the pilot's
copy after as many, the pilot, the teacher and the runs the teacher came
from; and the readings, taken deal by deal over a copy's last two
checkpoints once it has had the updates of `--updates`.

`tools/suite_beside.py FILE... [--names NAME...]` sets measurements of one
suite beside each other: every record of the files `evaluate-suite --output`
left, with one seed, so the same deals. It prints each player's rate case by
case and over all, and each player over the first and over the one before
it, game by game, with the standard error of that difference. A file made
before a measurement kept each game's result is refused.

`tools/self_play_report.py DIRECTORY --arm UPDATES --step UPDATES` is the
report of the experiment on self-play
([Self-play, and the named plays](../experiments/2026-10-11-self-play.md)),
for the folder `experiments/commands/self-play.sh` leaves. Of the base,
which has no reading: what the deck's pilot, earlier checkpoints of the
start's run and the start itself win of the mirror against the start held
in the other seat; and each exploiter of the start, a learner that begins
as the start and plays every duel against it, at each of its evaluations:
what its first choices win of those deals against the start and of the
suite against the pilots, and what it won of the duels it trained on since
the evaluation before. Of the arms, `with-itself` and `pilots-alone`:
after every `--step` updates the held-out deals each wins, the suite
against the pilots and the mirror against the start; and the readings, by
an arm's last three checkpoints once it has had the updates of `--arm`.
With-itself has lost against the pilots when it wins at least 2 points and
two standard errors fewer of the held-out deals than the start, deal by
deal. It is the better player of the mirror when it wins at least 52% of
the games against pilots-alone and two standard errors over half, every
deal played from either seat (its last three checkpoints against the
other's last, each in turn the one held; a drawn game is half a win), and
the worse one at 48% and two under. The decks outside training are told
beside them, game by game. Arms taken on from a shorter length are read
the same way at the new one, in a folder of their own
(`ARM=1920 SINCE=960 OUT=analysis/self-play-further PARTS=arms` of the
script): with `--since UPDATES` the report also says whether with-itself
has gained on itself, its last three checkpoints with its own checkpoint
after so many updates, each in turn the one held, by the marks of the
mirror. Arms of another training seed are made and read the same way, in
folders of their own (`ARM_SEED=2013 RUNS=runs/self-play-2013
OUT=analysis/self-play-2013 PARTS=arms`): other deals and other draws from
the same start, measured on the same deals. The exploiters of the arms'
checkpoints after a longer arm are made with that `ARM` and
`PARTS=exploiters`, in run folders beside those of the first 960 updates
(`exploiter-ARM-SEED.LENGTH`), and read by the same marks in that length's
folder. `AT_ONCE` has the trainings of a part go at the same time (the
section on an experiment's trainings at once). Where both arms are
measured on the held-out deals the report also says whether with-itself is
under the control against the pilots: under when it wins at least 1 point
and two standard errors fewer of them than pilots-alone, deal by deal over
each arm's last three checkpoints, and over it the same the other way
(fixed on 2026-10-13 for
[a second pair of arms](../experiments/2026-10-13-self-play-second-seed.md)).
`experiments/commands/self-play-second-seed.sh` makes that pair and then
the exploiters of the first pair's checkpoints after 1,920 updates, one
command for both.
`eval_suites/ygo-policies-blackwing-screen-and-checkpoint.json` is the
screen with the mirror against a held checkpoint beside it, the same 2,048
deals as in `eval_suites/blackwing-against-checkpoint.json`.

`tools/several_decks_report.py DIRECTORY --decks DECK... --updates UPDATES`
is the report of one network of several decks beside a network of each, for
the folder `experiments/commands/several-decks.sh` leaves: every network
imitates pilots from the same first network, the one of all the decks for
as many updates as there are decks times those of a deck's own. For each
deck it sets the two beside each other after as many games of the deck:
how often the first choice is the pilot's on games new to the network, and
the games of the deck's screen each wins, beside the pilot's. A deck is
read game by game of its screen over the last two checkpoints: the network
of all is worse on it when it is 2 points and two standard errors under the
deck's own.

`tools/more_decks_report.py DIRECTORY` is the report of one network for
more decks with half of its duels against itself
([the experiment](../experiments/2026-10-14-more-decks.md)), for the folder
`PARTS=measure` of `experiments/commands/more-decks.sh` leaves: each
network's copy at its last checkpoint and its checkpoints after its last
three of 1,920 updates of PPO, on the held-out deals of the first deck
(Blackwing) and on the screen of each of its decks, and each deck's pilot
on its screen. A game is read by the share of a network's last three
checkpoints that win it. A network of more decks is worse on Blackwing
when it wins at least 2 points and two standard errors fewer of the
held-out deals than the network of Blackwing alone, deal by deal; the
network of all the decks has gained on its copy, and is over or under the
decks' own pilots, at 2 points and two standard errors over all its
screens together, game by game. Beside the readings it sets out how fast
each network learns: its other checkpoints of PPO on the held-out deals
(`CURVE=0` of the measuring part leaves them out, two hours less), by its
updates and by the seats of Blackwing it had played by then, of which a
network of N decks has an Nth. The measuring part takes a network only
once it is made to its end, and a finished measurement is not made again.

`tools/deck_lists_report.py DIRECTORY [--runs RUNS]` is the report of the
model without the deck lists
([the experiment](../experiments/2026-10-15-without-deck-lists.md)), for
the folder `PARTS=measure` of `experiments/commands/without-deck-lists.sh`
leaves. Copies whose model leaves parts of its input out are each set
beside the copy of the experiment on more decks of the same seed and
settings, game by game on the same deals: a part can go (equivalent) when
the gap less two standard errors is above −2 points, and stays (worse)
when the copy that reads less wins at least 2 points and two standard
errors fewer. The other player's list (`blackwing-own-list`) and the
seat's own earlier answers (`blackwing-no-history`) are read on the
held-out deals and on the eighteen decks outside training, both lists in a
network of three decks (`three-no-lists`) over its three screens. With
`--runs` (the runs' `run.json` and `metrics.jsonl`, which the measuring
part copies) it also says whether each copy began as the same network as
its control and played the same games in every update, which is what makes
the two differ only in what their models read.

`train --leave-out PARTS` names parts of its input the model does not
read, comma-separated: since format 15 only `history` (the seat's own
earlier answers), the two deck lists it was built for being out of the
input (`own-deck` and `opponent-deck` before, of which the copies of that
experiment were made); nothing where it is not given. The input holds every part
either way: a part left out is given nothing where it was, and the network
keeps every weight it has, so that models that leave out more begin as the
same network under the same seed, and an imitating update plays the same
games whatever its model reads (`tests/test_leave_out.py`). A checkpoint
keeps what its model left out (`leave_out` of its description, and of its
input schema where it is not nothing): loaded, a model reads as it was
taught, and a run that goes on is refused another. `tools/export_checkpoint.py`
refuses one that leaves anything out, which a forward pass written
elsewhere does not know.

`train --card-table FILE` with `--card-descriptions`, `--hide-identities
SHARE` or both tells the model more of a card than its input gives, by the
code the input carries: checkpoint format 16 (`ygo_training/card_table.py`;
[the experiment](../experiments/2026-10-17-card-descriptions.md)).
`data/card-descriptions.npz` is the repository's table: every code of the
card database and for each the row of its description, a fixed vector of
what its text says, which `tools/card_descriptions.py` made from the texts
and voyage-4-nano's vectors of `experiments/commands/card-embeddings.sh`
(an alternate artwork reads its original's row, a Normal Monster none).
With `--card-descriptions` a card's description goes through a layer of its
own and is added to its identity wherever the identity is read. With
`--hide-identities SHARE` that share of the table's cards, drawn again each
update from the run's seed and the update, is read with one learnt vector
for an identity in the update's games and in its training, and the model
keeps which cards it was trained on, shown with their identity; at play
(any checkpoint loaded, and a run's own evaluations) it hides the identity
of every card it was not trained on, and of every code not in the table but
0, which is a card the seat does not see. The parameters these bring are
the model's last: under one seed it begins as one told nothing more. A
checkpoint carries its table, and `evaluate-suite` and `evaluate-search`
build their model as the first checkpoint they are given says; a model
told nothing more is still written in format 15, which revisions before
format 16 load. `tools/export_checkpoint.py` refuses a checkpoint told
more, which a forward pass written elsewhere does not look up.

`--lead-margin M` also plays the run's deals by that rule (`led_deals`),
beside the games played by the policy alone. The rule is applied only at
decisions a strict one-step search examines, told from the decision as the
network sees it: a command, a chain window that may be passed, a yes/no or a
battle position, with more than one alternative, no chain under way and no
face-down monster on the other side. The head is taught nowhere else.
`evaluate-search` refuses a logged run in which the search examined other
decisions than these.

A checkpoint is refused if its own run imitated any deal of these rows
(`run.json` names the games it imitated): what a model does on games it was
trained on says nothing of others. `--allow-seen-deals` measures anyway, for
a look at what was memorised, and the result says how many deals were seen.

`tools/search_imitation_report.py DIRECTORY` puts such results in one table
when the directory holds `seed.json` for the model the runs start from and
`search-N.json` and `pilot-N.json` for two arms on `N` games: every run at
every measured checkpoint, and the first arm against the second deal by deal
and deviation by deviation, with standard errors. With `--records ROWS`, the
rows the measurements were made on, it also sets the deviations apart by the
z the search took them at, and gives every run's rate at the deviations
where the starting model gave the search's answer under 1% of its weight:
those a run has from the games it imitated. Where the rows were logged it
gives every run's other answers by the search's worlds after its last
measured checkpoint, and at every measured checkpoint the two arms'
added-up leads against each other game by game, beside how many answers a
game each gives where the pilot's wins outright, which have no lead. A
`seed-continued.json` in the directory, the starting model's own run
measured further on, is given beside `seed.json`.

`tools/search_leads_report.py` is the search-leads experiment's, for arms
whose lead head is trained:

- `seed PILOT.json MODEL.json --checkpoint PATH` holds the model the arms
  start from against its pilot, from the `evaluate-suite` results of the
  two on the same suite, and fails where the model wins more than 5 points
  less or the result is of another checkpoint.
- `weight METRICS.jsonl` sets the weight of the lead loss from a
  calibration run, which trained the head with the weight 1. Every update
  reports how hard the imitation loss and the lead loss pull on the
  parameters they share, before it moves anything
  (`imitation_shared_gradient_norm`, `imitation_lead_shared_gradient_norm`);
  the weight is the first over the second, over the run's updates but the
  first, where the head is still untrained.
- `read DIRECTORY` takes `DIRECTORY/validation/ARM.json`, the arms `pilot`,
  `search`, `leads` and `leads-read` at every measured update on a
  validation run with the full log, and `seed.json` for the model they
  start from. For `leads` and `leads-read` it gives what the rule gains at
  every update and margin and what the head knows, and chooses the update
  and the margin at which the rule gains most a game: of equal gains the
  earlier update, then the larger margin, and none where it gains nothing
  above zero. `--choice FILE` writes the choices. Where
  `DIRECTORY/heldout/` holds those arms at the choice in
  `DIRECTORY/choice.json`, measured with `--lead-margin`, and the others at
  the updates so chosen, it applies the experiment's readings: the mark is
  a quarter of what the search gains over its pilot on those deals, the
  rule's gain is held against it and three standard errors, and the deals
  won by the rule against the `pilot` arm's, deal by deal, against it and
  two. A result at another checkpoint or margin than the choice is refused.
- `more DIRECTORY` takes `DIRECTORY/more/ARM.json`, every arm at every
  measured update on the held-out run, measured after the readings
  (`experiments/commands/search-leads-more.sh`). It sets each arm against
  the `pilot` arm at the same update and over all of them, by the deals won
  deal by deal and by what its other answers are worth game by game, gives
  the rule's gain at every update and margin, and at the update of an arm's
  choice its changes apart by where they start. What the readings measured
  must come out the same there, or it stops.

## `edopro`

```sh
uv run python -m ygo_training edopro runs/example/checkpoints/model-00959.pt --verbose
```

Seats a checkpoint in a room hosted in EDOPro, against a player. Host the
room first, in LAN mode (the default port is 7911), under the rules the
duels of training are played by: Master Rule 1, 8000 Life Points, a hand of
five and one draw a turn. The checkpoint joins as a player with the deck it
trained with (`--learner`, `blackwing` unless said) and plays `--duels`
duels, joining again for each. Ctrl-C ends the session.

- The other player's deck is not named: since format 15 the model's input holds
  no deck list (until then `--opponent` named it, a deck of the pool or a
  `.ydk` file, for the other player's list a room does not tell).
- `--verbose` says each decision as it is taken: the answer, how sure the
  model is of it, and its pilot's answer where that is another. At the end
  comes how often the two agreed.
- Each duel is written to `--log-dir` (`edopro-logs` unless said, nowhere
  with `--no-log`): a `.log` with the duel in words and a `.trace` with
  every message received and every answer sent.
- `--replay FILE.trace` joins nothing: the trace is fed to the seat again
  and the model is asked what the bot was asked then. What it answers is not
  played, so a duel is followed as it went.
- `--host`, `--port`, `--password`, `--room-id`, `--name`, `--go-second`,
  `--wait` (seconds to keep trying to reach the room) and `--version`
  (EDOPro's client version, in case the room refuses the default) are those
  of `edopro-bot`.

The room is joined by the client of `ygo-policies-edopro`, as its own
`edopro-bot` joins one: the lobby, the seat and the logs are its code. What
is seated is the caller of an episode (`crates/ygo-env/src/edopro.rs`): the
model is shown a decision as an episode shows it and answers with its first
choice, as in an evaluation; the deck's pilot places the cards, and a
single answer is taken. `crates/ygo-env/tests/edopro.rs` serves duels of the
trainer through a scripted room and has the seat give the episode's answers
from the episode's arrays.

A room plays EDOPro's own card scripts and card database, not the
trainer's, and the seat is as faithful as the two are alike:

- A script says what an answer is called (an effect's description), and
  the model's input says of every answer whether it names an effect. With
  the scripts of an EDOPro of April 2025, the activations of Mystical Space
  Typhoon and Pot of Duality come without the description the trainer's
  scripts give them: over four duels this changed one first choice in 747.
  `edopro-review --trace` tells it of a duel.
- A deck list of the pool may name a card by an alternate-artwork code that
  an EDOPro does not hold; it then loads without the card. Of the pool's
  lists, with the EDOPro of April 2025: Gorz the Emissary of Darkness in
  the Monarch list (44330100, for 44330098) and Honest in the Lightsworn
  list (37742479, for 37742478). The list a player loads there needs the
  code the room knows, and the model then meets that card under a code
  its training did not show it.

Both go away when the room plays the scripts and the database of
`native/ygo-policies/vendor`, which need an engine of their time: on the
engine of that EDOPro they fail at once (a function they call is not
there), so the three are updated together. EDOPro does that itself: when
it starts it fetches a repository that brings all three (into
`repositories/` of its folder, as `config/configs.json` sets it up), and
from then on it follows the scripts as they change, which the trainer's
pin does not. Whether an installation plays the trainer's duels as it
stands:

```sh
uv run python tools/edopro_files.py ~/EDOPro
```

The folder is read as EDOPro reads it: a repository's scripts and card
rows before the ones it came with, and a repository's engine before its
own. For the cards of the pool's deck lists it is set beside the trainer's:
each card's row as the engine is given it, its script, and the helper
scripts. Then the policies' bench plays the same pilot games on
EDOPro's three and on the trainer's, `--deck` (`blackwing` unless said)
against every other deck of the pool, `--games` each (8), and they are
compared game by game: the same game asks the same questions and takes
the same answers to its end. It exits with 1 when a card of the pool is
missing or told otherwise, its script is another, or a game is another or
fails; a helper that differs is said, and the games decide. `--keep DIR`
leaves EDOPro's files as they were read, and the games, there. It takes a
few seconds.

- With the EDOPro of April 2025 as it came, 16 of the 240 games are the
  trainer's; with its repository fetched on 2026-10-10, all 240.
- A repository that was never fetched is said. EDOPro tells why in
  `error.log` of its folder; `authentication required but no callback set`
  is a git configuration of the account running EDOPro that rewrites
  `https://github.com/` to SSH (`url.<...>.insteadOf`), which EDOPro reads
  too and cannot sign in for.
- It tells of the files, not of a room: `edopro-review --trace` reads a
  duel a room played.

## `edopro-review`

```sh
uv run python -m ygo_training edopro-review ~/EDOPro/replay/_LastReplay.yrp \
  --checkpoint runs/example/checkpoints/model-00959.pt
```

Reads a duel EDOPro kept again and sets each player's moves beside the
pilot's of their deck, and beside a checkpoint's. EDOPro keeps every duel
it plays: `replay/_LastReplay.yrp` in its folder is the last one, and a
replay it saves under its time (a `.yrpX`) holds the same inside. What is
kept is the engine's seed, the rules, both players' cards and every answer
either gave, so the duel played again on this checkout's engine and card
scripts is the same duel, with nothing hidden; the library tells who gave
each answer as it goes. It is the same duel only on the engine and scripts
it was played on: where an answer is not one the engine takes, the command
says so and stops.

- Each player's deck is told from the cards they played: the deck of the
  pool most of their Main Deck is of, where three quarters of it are.
  `--decks FIRST,SECOND` names them, the player who went first, then the
  other. The pilot of that deck is asked at each of the player's decisions
  a network would be asked; a player whose cards are no deck of the pool
  has no pilot.
- `--checkpoint FILE` asks a checkpoint as well, at the moves of whoever
  played its deck (`--learner`, `blackwing` unless said): its first choice,
  how sure it is of each answer, and how its value head rates the player's
  chances there.
- `--player NAME` chooses whose moves are listed, by a part of their name;
  the player whose name does not begin with `[AI]` unless said.
- `--trace FILE` is the bot's own trace of the duel (`edopro-logs/*.trace`).
  It is fed to the seat again, each decision given the answer the duel was
  played with, and what the room showed the bot is set beside the same
  decisions of the duel played again here: how many are the same in every
  field, which fields differ in the others and at which answers, and, with
  `--checkpoint`, at how many the checkpoint's first choice is another or
  its probabilities move by more than a point. It is the check that a room
  shows a model what its training showed it; a trace of another duel is
  said to be one.
- Two copies of a card that lie alike, and two cards the player cannot see,
  are one answer.

It prints how often each player gave the pilot's answer and the
checkpoint's, then the listed player's decisions where either would have
answered otherwise, and writes to `--output` (`analysis/edopro` unless
said), under the duel's time: the review (`.review.txt`), the duel as a
record of the policies' form (`.record.json`), every decision
(`.decisions.json`), and, where the policies' bench is built
(`cargo build --release -p ygo-policies-bench` in `native/ygo-policies`),
the duel in words with both hands shown (`.story.txt`).

A duel a player gave up is read as far as it went. The answers of the other
player come back as EDOPro's own client writes them, which is not how a
policy writes them: `crates/ygo-env/src/seat.rs` reads a selection of cards
in every form the engine reads.

## The named plays

```sh
uv run python tools/named_plays.py \
  copy=runs/learner/once-few/checkpoints/model-00359.pt \
  long=runs/ppo-long-rate/one-pass-long/checkpoints/model-02279.pt \
  --keep analysis/named-plays
```

What a checkpoint wins against pilots does not see a play that costs
nothing against a pilot, and a pilot plays on the same whatever it is
shown. Three such plays of the Blackwing checkpoints are named in
[PPO with a head that predicts](../experiments/2026-10-09-ppo-with-a-head-that-predicts.md),
and this counts them in a checkpoint's own games, for one checkpoint or
for several beside each other:

- **Allure of Darkness**, by the turns it could be activated from the
  hand and what else was in hand when the turn first offered it: a DARK
  monster, nothing, or Spells and Traps alone. Activated with only those
  beside it, how often some could still be set at that decision, and how
  often the hand then went to the Graveyard.
- **Trap Hole**, with Bottomless Trap Hole and Dimensional Prison: of the
  windows that offer the card, how many the checkpoint takes and how many
  its pilot would; for Trap Hole also at a monster of 1,000 to 1,499 ATK
  and at the first window of a game.
- **Threatening Roar**: where it is activated (their Battle Phase with
  monsters in Attack Position, where the pilot plays it; their Main Phase
  1; their Draw Phase), how many of the activations of their Main Phase
  answer a card on the chain, and how many are spent before they have an
  attacker in answer to nothing.
- What it wins, against Machina and in the mirror among it, how often its
  first choice is its pilot's, and how many turns it takes to beat a
  player who does nothing but enter the Battle Phase and leave it.

Each checkpoint plays the same games, on the processor: `--games` (160)
against the pilot of each of `--opponents` (the thirteen decks a Blackwing
learner trains against, and Machina) and `--idle-games` (60) against the
idle player with each of `--idle-opponents`, on the deals of `--seed`.
That is 2,240 games and about seven minutes a checkpoint. With `--keep`
the decisions are kept under each checkpoint's name and read again the
next time; a file of another checkpoint, of other games or of other
revisions of the duels is refused. `--output` writes the counts.

The cards are those of the Blackwing list and the counts are of these
three plays: it is a measure for this experiment's question, not of play
in general.

## A machine's speed

```sh
bash tools/benchmark.sh
```

says how many duels an hour a machine trains in every mode of the trainer,
starting from nothing: a fresh clone is enough, with no checkpoint, no
recorded duels and no Python environment. It sets itself up, does a fixed
exercise, writes a report and stops. Speed only: nothing it writes is a
result.

**The setup**, each step only where it is missing:

- the submodules;
- a Rust toolchain, installed by rustup when `cargo` is not on the PATH;
- the engine, the environment and the bench (`cargo build --release`);
- a Python environment made with `python -m venv` and pip, with the torch
  that `uv.lock` names. uv is not used. Where the system's Python comes
  without pip, as on Debian and Ubuntu until `python3-venv` is installed,
  pip's own installer is fetched for the environment.

All of it goes to one folder, `benchmark/` under the repository root
(`FOLDER`), but for the build, which stays where every command looks for it
(`target/` and `native/ygo-policies/target/`). Nothing is installed in the
home folder: the toolchain, the crates of the build and the packages are
in that folder, 5.4 GB of it (the environment 4.7, Rust 0.7), and the
build is 2.8 GB more. It fetches Rust's installer and toolchain
(rustup.rs), the crates (crates.io), pip's installer (bootstrap.pypa.io)
and the packages (pypi.org): 2.75 GB of them, torch with its libraries for
the card. What it cannot bring it asks for before it makes anything: git,
curl, a C++ compiler, and a Python from 3.11 to 3.13.
`PYTHON=.venv/bin/python` uses an environment that is there, and `TORCH`
names another version of torch than the lock's, for a card it does not
know.

**The exercise** is the same games on every machine, 520 duels an update
on the deals of seed 2012. A network first imitates the Blackwing pilot
from nothing for 40 updates: that is the first mode's measure, and it gives
the others a player whose games are of ordinary length (after 40 such
updates a copy wins about half of the suite in games of 12 to 13 turns; a
new network's games are not games anyone trains on). Every other mode then
makes five updates on from that copy, a run of its own, and the last four
are timed.

| Mode | What makes it |
| --- | --- |
| Imitation of the pilot | `--imitation-only-updates` |
| of the pilots of two decks | `--learner blackwing,lightsworn` |
| of a checkpoint | `--imitation-teacher`, the copy itself |
| of recorded duels | `--imitation-records`; the duels are recorded there and then, the pilot with the cheapest search |
| the same with the lead head taught | `--imitation-lead-weight` |
| PPO against the pilots | the model plays, one pass |
| with the pilots' answers an anchor | `--teacher-imitation-weight` |
| half of the duels against itself | `--self-play-duels`, half of an update's |
| every duel against itself | `--self-play-duels`, all of them |
| every duel against a held checkpoint | `--opponent-checkpoint`, the copy held |

The copy is also measured by its first choice: the suite against the pilots
(1,664 games) and the mirror against itself held (2,048). A learner of
several decks that plays its own games goes through the same pool as one
deck's and is not timed apart.

**The report** is `benchmark/report.txt` and `report.json`, and
`benchmark/STAMP/` holds what it was made from: the runs' figures, the logs
of every part and how the exercise was set. For each mode, the mean of its
timed updates: what an update takes, all of it; duels an hour at that; its
games, with the engines' and the model's parts where the run tells them;
the update; the decisions it learns from, and those a second. A mode that
fails is named as not measured and the others go on. The checkpoints of the
exercise are removed at the end (`KEEP=1` keeps them).

`THREADS` (the machine's, or the share of them a container is allowed,
unless named) and `SLOTS` (16) are the pool's;
`DEVICE` is the card where torch can compute on one, and the processor
otherwise.

Tried from nothing on the development machine on 2026-10-11, in a fresh
clone with no Rust, no uv and a Python without pip: the setup took three
and a half minutes (the build two and a third, the packages one) and a
second run made none of it again. That was the exercise at a tenth of its
size on the processor.

**The present CUDA machine**, at the full size on 2026-10-12, with its
environment as it was (`PYTHON=.venv/bin/python`): 21 minutes and 13
seconds, every mode measured. This is what another machine's report is
set beside, **at the same revision of the trainer**: the report names it.
The three changes of 2026-10-13 (above) take 1.6 to 1.9 s off an update
whose games the model plays, so a machine measured after them is set
beside this one measured again. The two modes by the workers
went with them on 2026-10-16, and recorded duels have been imitated
through the pool since.

```
Training speed, 20261012-175630
  Intel(R) Core(TM) i5-14600K, 20 threads, 92 GiB
  NVIDIA GeForce RTX 4070 SUPER, 12 GiB, CUDA 13.0
  Linux 7.0.0-38-generic, Python 3.13.14, torch 2.13.0+cu130, cargo 1.98.1 (797e8a9bc 2026-08-05)
  the trainer at 936ffd18, the pilots at 1b85b633
  device cuda, 520 duels an update, a pool of 20 threads with 16 duels each, 24 workers
  on from a copy of 40 imitating updates, 4 updates timed in each mode after one that warms up

Updates that imitate (the one imitated plays the learner's seat)
                                                 An update      Duels  Its games  engines   model      The  Decisions  Decisions
                                                         s    an hour          s        s       s update s              a second
  the pilot                                            7.4    253,719        1.7        -     0.0      5.3     41,841      5,671
  the pilot, by the workers                           10.3    181,866        3.2        -     0.0      6.6     41,408      4,023
  the pilots of two decks, one learner                 7.0    267,068        1.7        -     0.0      5.0     39,178      5,589
  a checkpoint                                        14.3    131,161        5.9      3.0     2.1      8.1     39,592      2,774
  recorded duels, by the workers                      10.5    177,712        3.3        -     0.0      6.6     40,617      3,856
  recorded duels, the lead head taught too            10.6    176,806        3.3        -     0.0      6.6     40,617      3,836
Updates whose games the model plays (PPO, one pass)
                                                 An update      Duels  Its games  engines   model      The  Decisions  Decisions
                                                         s    an hour          s        s       s update s              a second
  against the pilots                                  12.1    154,357        5.8      2.9     2.2      4.2     37,633      3,103
  against the pilots, by the workers                  27.9     67,067       20.5        -    11.1      4.8     38,571      1,382
  against the pilots, their answers an anchor         14.0    133,492        5.8      3.0     2.0      4.4     37,104      2,646
  half of the duels against itself                    15.0    124,810        6.4      2.8     2.6      5.8     51,767      3,451
  every duel against itself                           19.2     97,330        7.7      2.8     3.5      7.8     70,320      3,656
  every duel against a held checkpoint                14.5    129,366        8.7      2.7     4.6      3.9     35,089      2,425
The copy measured by its first choice
  the suite against the pilots: 1,664 games in 29 s, 207,514 an hour
  the mirror against a held checkpoint: 2,048 games in 32 s, 227,843 an hour
All of it 21 min 13 s, of which the setup 2 s (submodules 0 s, toolchain 0 s, build 0 s, python 2 s)
```

The engines' part of the games is told only where the model plays some of
them: where a pilot plays every seat, the games are the engines'.

**The exercise is for setting machines beside each other, and is not what
a long run takes.** Its player is a copy of 40 updates, whose games are
shorter than a trained network's. What the same machine does in runs of
its own:

| | An update in the exercise | In a run of its own | Duels an hour there |
| --- | ---: | ---: | ---: |
| Imitation of the pilot | 7.4 s | 6.6 to 7.4 s | 253,000 to 284,000 |
| PPO against the pilots | 12.1 s | 14.0 to 14.6 s | 128,000 to 134,000 |
| PPO, half of the duels against itself | 15.0 s | 25.3 s | 74,000 |
| PPO, every duel against a held checkpoint | 14.5 s | 21 to 28 s | 66,000 to 90,000 |

Imitation is the same, since a pilot's games do not depend on the
learner. Where the model plays, a trained network's games have more
decisions (45,000 to 47,000 an update against the pilots where the copy
has 37,600), and against itself or a network many more.

With half of the duels against itself both seats of those are learned
from: an update has 79,800 decisions where one against the pilots has
47,300, its games take 11.9 s for 6.7 s and its pass 9.2 s for 5.3 s
(the two arms of the experiment on self-play, 2026-10-12). Against a held
checkpoint an update is as long as the games are: 22 s for the exploiters
of that experiment's start and 28 s for one of the arm that had played
itself, whose games against it have a tenth more decisions.

### Several runs at once

```sh
uv run python tools/side_by_side.py \
  runs/self-play/with-itself/checkpoints/model-03239.pt --itself 260
```

says what a machine does with several runs of the trainer at once. A run
alone leaves most of a machine idle: its threads wait while the model
answers, and the card while the engines play. The same short run of PPO,
on from the checkpoint, is made alone and then two and three times side by
side (`--counts`), and for each count it tells what an update took in
every run and where its seconds went, how much longer that is than alone,
the duels an hour the machine played in all, and the load while the timed
updates ran: the processor threads the runs themselves kept busy, the
card's share of the time and the memory in use on it (and why, where the
card could not be read). It also tells whether every run played the
games of the run alone, and what all of a count took from its first start
to its last end. Five updates a run, the first not timed (`--updates`):
nine minutes on the CUDA machine.

```sh
bash experiments/commands/side-by-side.sh
```

makes it twice, as processes have a card, by turns (`plain`), and through
NVIDIA's MPS, which lets them use it at the same moment (`mps`), and sets
the two beside each other (`tools/side_by_side.py --beside`): for each
count the duels an hour in all and how many times the run alone that is,
and whether the run alone played the same games both ways. For the second
part it starts the MPS control daemon that comes with the card's driver
(`nvidia-cuda-mps-control`), with its pipes and logs in a folder of its
own under `/tmp`, so that only what the script starts goes through it,
and tells it to quit when the part ends, however it ends. The reports go
to `analysis/side-by-side/`, with a line on how many clients the daemon's
log names. `PARTS=plain` goes without MPS, `COUNTS` names other counts
(`"1 2 3 4 6"`). About twenty minutes. Tried on the development machine on
2026-10-13 at a small size on the processor, with a stand-in for the
daemon: MPS itself has not been run by this script anywhere yet.

`--itself N` has N duels of an update played against itself and `--held
CHECKPOINT` every duel against that checkpoint held; with neither the
pilots play the other seat. A run keeps the pool threads it has alone
(`--threads`, 20), so its games are the games it plays alone, and the
copies are one run: the same seeds, the same games. Speed only: nothing it
writes is a result. The report and the runs' logs are kept in
`runs/side-by-side/STAMP/`, and the runs' checkpoints are removed
(`--keep` keeps them).

On the CUDA machine on 2026-10-13, half of an update's 520 duels against
itself, on from a checkpoint of the experiment on self-play, with the
trainer as it was before that day's changes:

| Runs at once | An update in each | Duels an hour in all | Its games: engines, model | The pass | Threads the runs keep busy |
|---:|---:|---:|---:|---:|---:|
| 1 | 24.1 s | 77,700 | 3.6 s, 5.4 s | 9.0 s | 2.2 |
| 2 | 44.6 s | 84,000 | 4.1 s, 8.1 s | 23.2 s | 3.4 |
| 3 | 62.9 to 63.6 s | 88,600 | 4.5 s, 18.6 s | 30.1 s | 4.6 |

- **Two runs at once do 1.08 times the work of one, and three 1.14
  times.** All of it took 122 s for one run, 224 s for two and 324 s for
  three.
- **The runs wait for the card.** The engines' part hardly grows and the
  machine's 20 threads are a quarter busy with three runs, while the pass
  takes 2.6 and 3.3 times as long and the model's turns up to 3.4 times.
  Two processes use a card by turns. The card's own share of the time
  was not read: the tool left that column empty on this machine.
- Every run played the games of the run alone, update for update.
- Not tried: NVIDIA's MPS, which lets processes share a card at once, and
  runs against the pilots, where the card has less to do.

**The same after that day's changes, as processes have the card and
through MPS** (`experiments/commands/side-by-side.sh`, the same checkpoint
and duels):

| Runs at once | An update in each | Duels an hour in all | Times the run alone | Through MPS: an update in each | Duels an hour in all | Times the run alone |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 18.0 s | 103,800 | 1.00 | 18.0 s | 103,900 | 1.00 |
| 2 | 27.3 s | 137,200 | 1.32 | 21.4 s | 174,900 | 1.68 |
| 3 | 42.5 s | 132,100 | 1.27 | 25.6 s | 219,300 | 2.11 |

- **Through MPS three runs do 2.11 times the work of one, and two 1.68
  times.** Three arms at once take 1.4 times as long as one alone. All
  of it took 91 s, 108 s and 131 s for one, two and three runs, where
  without MPS it took 91 s, 136 s and 209 s.
- **The games are the same**: the run alone played the same games with
  MPS as without, and every run at once played the games of the run alone.
- **A run alone is a quarter faster than that morning**: 18.0 s an update
  for 24.1 s, the games 9.8 s for 10.7 s and the pass 5.7 s for 9.0 s.
  Without MPS two runs now do 1.32 times the work of one, where they did
  1.08 times: the card has less to do for each.
- The card's own load could not be read on that machine that day:
  `nvidia-smi` ended with
  `Failed to initialize NVML: Driver/library version mismatch`.
- To use it for runs of one's own: start the daemon as the script does
  (`nvidia-cuda-mps-control -d`, with `CUDA_MPS_PIPE_DIRECTORY` and
  `CUDA_MPS_LOG_DIRECTORY` set to folders of one's own), start the runs
  from that shell, and tell it to quit when they have ended (`echo quit |
  nvidia-cuda-mps-control`).

**An experiment's trainings at once** (`experiments/commands/at-once.sh`,
which a script sources). `AT_ONCE=N` has so many of a script's trainings
that do not wait for each other go at the same time, through an MPS
server on a card which the script starts and tells to quit:

```sh
AT_ONCE=2 PARTS=arms bash experiments/commands/self-play.sh
```

- Each training writes to a log beside its run (`RUNS/NAME.log`, which
  the script names as it starts one), and the script goes on to its
  measurements when all of them are over: one at a time, as processes
  have the card, as ever.
- **Each run is as long as it would be alone and half as long again**
  (three at once, above); what is gained is the runs made beside it.
  Three runs that would take nine hours one after the other take about
  four and a half at once.
- **Ctrl-C stops them all**, each where it stands (started again, the
  script takes its runs on from their last checkpoints), and a second
  Ctrl-C at once; so does the script's end while they go. Until
  2026-10-16 it stopped the script alone: under bash 5.3, the CUDA
  machine's, what a script puts in the background does not hear Ctrl-C,
  and the runs went on.
- A training that fails does not stop the others. The script stops when
  they are over and names it; started again, it takes its runs on from
  where they stand. Under MPS a process that the card faults can take the
  server and the other runs with it: the same remedy.
- Without the daemon's control program, or with `MPS=0`, the runs go at
  once and take the card by turns: three then do 1.4 times the work of
  one, where through MPS they do twice. `AT_ONCE=1`, the default, is one
  after the other in the terminal.
- Every run keeps its own pool threads and workers: three runs of 20
  threads on the CUDA machine's 20 are what was measured.

What a rented machine did with runs side by side is told below: there
three runs against the pilots did 2.1 times the work of one, without MPS
and before these changes.

### A rented machine

```sh
export RENTED=root@HOST:PORT
bash tools/rented.sh machine
bash tools/rented.sh send
bash tools/rented.sh start benchmark bash tools/benchmark.sh
bash tools/rented.sh watch benchmark
INTO=benchmark/rented bash tools/rented.sh fetch benchmark/report.txt
```

`tools/rented.sh` drives a machine rented by the hour from this checkout,
over SSH. It knows no provider: the machine is whatever answers at `RENTED`
with a login by key, bash, git and tar. Renting it, stopping it and
deleting it are done where it was rented.

- `machine` tells what answers there: the processor, the threads a program
  sees and how many of them the machine may run at once, the memory, the
  card and its driver, the room on its disks, and which of the tools the
  scripts ask for are there.
- `send` takes this checkout's HEAD there (`AT`, `/workspace/ygo` unless
  named). It is pushed over SSH into a bare repository beside that
  checkout, so a private repository needs no key of GitHub's on someone
  else's machine; the submodules, which are public, the machine fetches
  itself. What is not committed is not sent.
- `put PATH...` copies files or folders of this checkout that git does
  not hold, a checkpoint say, to the same paths in the checkout there.
- `start NAME COMMAND...` leaves COMMAND running in the checkout there when
  the connection closes, with its output in `runs/rented/NAME.log` and its
  exit status in `NAME.exit` beside it. `watch NAME` shows the end of the
  output and whether NAME still runs; `wait NAME` stays until it has ended
  and ends with its status; `state NAME` is one word of it for a script
  (`running`, `ended 0`, `gone`, `none`).
- `fetch PATH...` copies files or folders of the checkout there to the same
  paths here, under `INTO` where one is named. Folders are merged.
- `follow NAME COMMAND...` stays until NAME has ended and brings what it
  writes here as it goes. COMMAND is run in the checkout there and names
  the files to bring, one a line: every `EVERY` seconds (600) while NAME
  runs, and once more when it has ended, `rsync` brings them to the same
  paths here, only what is new, and takes a file half brought on. Then
  what is here is held against what is there, file by file by its
  checksum. It ends with 0 when every file named is here as it is there,
  whatever came of NAME, and with 3 when not. The machine not answering
  for a while is waited out.
- `run COMMAND...` is a command there with its output here.

`tests/test_rented.py` runs every command of it against this machine, with
an `ssh` that runs the command here standing in for the connection.

```sh
bash tools/rented.sh send
bash tools/rented.sh put runs/model-02279-rate.pt
bash tools/rented.sh start capacity bash experiments/commands/rented-capacity.sh
bash tools/rented.sh watch capacity
INTO=analysis/rented bash tools/rented.sh fetch analysis/rented-capacity
```

`experiments/commands/rented-capacity.sh`, run on the machine itself, says
how many runs of the trainer it carries at once. It makes the benchmark's
setup where that is missing (`ONLY=setup bash tools/benchmark.sh`: the
toolchain, the build and a Python environment, and no exercise; what
another command needs of it is left in `benchmark/setup.sh`), and then
`experiments/commands/side-by-side.sh` with what the machine has: its
share of the processor's threads for a run's pool, as many duels a thread
as keep 320 in flight, as the CUDA machine has, one to six runs at once
(`COUNTS`, `"1 2 3 4 6"`), and the runs through MPS as well where its
daemon is on the PATH. The checkpoint is one sent there with `put`
(`CHECKPOINT`). It makes those runs for each of `WAYS`, as
`experiments/commands/call-graphs.sh` makes them on the CUDA machine:
`one-group` (the trainer as it was before 2026-10-13, every call as it
is), `graphs` (calls of up to half the duels in flight from CUDA graphs)
and, where named, `graphs-two-groups` (what a run on a card takes where
it names nothing); a thread's duels are then an even count, and
`analysis/rented-capacity/plain.txt` and `mps.txt` set the ways beside
each other. Tried on the development machine on 2026-10-13 at a small
size on the processor, with a stand-in for the MPS daemon, and run to its
end on a rented machine the same day (the part below on how many runs a
rented machine carries).

**What a rented machine must have.** A driver the lock's torch can use:
torch 2.13.0 is built for CUDA 13.0, which a driver of the 580 series or
newer serves. With an older driver nothing fails: the trainer, asked for
the card and finding none it can use, computes on the processor, and
`device` in a run's `run.json`, or the lines of the machine in the
benchmark's report, say which it was. Then git, curl, a C++ compiler and a
Python from 3.11 to 3.13, which the benchmark asks for. The experiments'
scripts also call `uv` and `cargo`, and have not been run on a rented
machine; `experiments/commands/side-by-side.sh` takes `PYTHON` for an
interpreter where uv is not there.

**A card that does not start.** A second rented machine (2026-10-13, an
RTX 5080 of the provider's community cloud) named its card to `nvidia-smi`
and let no program compute on it: `CUDA unknown error`, for the PyTorch of
the provider's own image as for ours, before and after a restart. The
capacity script went on the processor without a word until its load was
seen. `tools/rented.sh machine` now asks the driver itself (`cuda: starts`,
or that it does not), and `experiments/commands/rented-capacity.sh` asks
before it sets anything up and stops where the card does not start. That
machine also had no address of its own: only the provider's proxy answered,
which gives a terminal and takes no command, so `tools/rented.sh` could
not reach it and the files went by the provider's own tool (`runpodctl
send` here, `runpodctl receive` there). Nothing was measured on it.

**A container's threads.** A rented machine is often a container that sees
every thread of its host and may run a few of them at once. The first one
saw 128 and was held to 13.6, where it was sold as 16 vCPUs, and `nproc`
tells the 128. The benchmark takes the share for `THREADS` and gives torch
and cargo the same number; an experiment's script would be given `THREADS`
and `OMP_NUM_THREADS` by hand there.

**The first one**, on 2026-10-13: a pod of Runpod's Secure Cloud at $0.89
an hour, an RTX 4090 with 16 vCPUs of an AMD EPYC 75F3 and 62 GB, the
checkout on its volume, which is a network disk there. Sending the
repository and fetching the submodules took 2 minutes 43 seconds, the
setup from nothing 8 minutes 36, and the exercise 51 minutes where the
CUDA machine's takes 21.

```
Training speed, 20261012-224243
  AMD EPYC 75F3 32-Core Processor, 128 threads (13.6 of them allowed at once), 504 GiB
  NVIDIA GeForce RTX 4090, 24 GiB, CUDA 13.0
  Linux 6.8.0-134-generic, Python 3.13.8, torch 2.13.0+cu130, cargo 1.99.0 (5f94df478 2026-08-27)
  the trainer at 900a1c33, the pilots at 1b85b633
  device cuda, 520 duels an update, a pool of 14 threads with 16 duels each, 24 workers
  on from a copy of 40 imitating updates, 4 updates timed in each mode after one that warms up

Updates that imitate (the one imitated plays the learner's seat)
                                                 An update      Duels  Its games  engines   model      The  Decisions  Decisions
                                                         s    an hour          s        s       s update s              a second
  the pilot                                           11.6    160,771        2.6        -     0.0      8.2     41,841      3,593
  the pilot, by the workers                           17.0    110,442        5.2        -     0.0     10.6     41,408      2,443
  the pilots of two decks, one learner                11.5    162,490        2.6        -     0.0      8.0     39,178      3,401
  a checkpoint                                        22.5     83,044       10.7      5.7     3.6     11.0     39,789      1,765
  recorded duels, by the workers                      17.2    108,927        4.8        -     0.0     10.9     40,617      2,363
  recorded duels, the lead head taught too            17.6    106,219        5.2        -     0.0     10.9     40,617      2,305
Updates whose games the model plays (PPO, one pass)
                                                 An update      Duels  Its games  engines   model      The  Decisions  Decisions
                                                         s    an hour          s        s       s update s              a second
  against the pilots                                  20.8     90,029       11.0      5.9     3.9      6.8     38,038      1,829
  against the pilots, by the workers                  49.7     37,702       37.2        -    24.7      8.1     36,528        736
  against the pilots, their answers an anchor         23.2     80,680       10.5      5.4     3.8      7.4     36,275      1,563
  half of the duels against itself                    26.0     71,913       13.1      6.5     4.9      8.9     53,832      2,068
  every duel against itself                           31.2     60,032       14.0      6.1     5.8     12.0     69,549      2,230
  every duel against a held checkpoint                26.2     71,394       17.4      6.0     9.4      6.1     34,515      1,316
The copy measured by its first choice
  the suite against the pilots: 1,664 games in 90 s, 66,674 an hour
  the mirror against a held checkpoint: 2,048 games in 95 s, 77,769 an hour
All of it 59 min 46 s, of which the setup 8 min 36 s (submodules 0 s, toolchain 10 s, build 4 min 55 s, python 3 min 31 s)
```

| An update, or a suite | The rented machine | The CUDA machine | Times as long |
| --- | ---: | ---: | ---: |
| Imitation of the pilot | 11.6 s | 7.4 s | 1.6 |
| PPO against the pilots | 20.8 s | 12.1 s | 1.7 |
| PPO, half of the duels against itself | 26.0 s | 15.0 s | 1.7 |
| PPO, every duel against a held checkpoint | 26.2 s | 14.5 s | 1.8 |
| The suite against the pilots | 90 s | 29 s | 3.1 |

- **The machine with the larger card was the slower one, the pass of an
  update included**: 6.8 s on its RTX 4090 where the machine of the
  RTX 4070 SUPER takes 4.2 s. Why is not measured. The processor, its
  quota, the disk and the host all differ, and each machine trained a copy
  of its own, so this is no comparison of the two cards on one input
  (corrected on 2026-10-13: this said that one thread of Python drives the
  pass and that the thread was the slower processor's, which was a
  reading and not a measure). Sampled every ten seconds, 3.6 to 3.9
  threads were busy while an update's games were played and the card 10%
  to 16%; during the pass 1.1 to 1.6 threads and the card about 30%.
- **The network disk costs a fifth of a suite**: 89 s with the checkout on
  the volume, and 69 s and 73 s with the same checkout copied to the
  container's own disk. The part that records duels took 529 s with 0.4
  threads busy.
- **Runs side by side use what one run leaves idle.** PPO against the
  pilots, five updates on from the exercise's copy, the last four timed:

  | Runs at once | Pool threads each | An update, each | Duels an hour, together |
  | ---: | ---: | ---: | ---: |
  | 1 | 14 | 20.8 s | 90,000 |
  | 1 | 4 | 36.6 s | 51,000 |
  | 2 | 7 | 28.6 and 30.4 s | 127,000 |
  | 3 | 4 | 37.8 to 40.4 s | 144,000 |
  | 2 | 14 | 24.6 and 24.9 s | 151,000 |
  | 3 | 14 | 28.2 to 30.4 s | 193,000 |

  Three runs that each keep every pool thread take 1.4 times as long an
  update and do 2.1 times the work together, and sharing the threads out
  between them is worse. One measure each, of four updates. The runs side
  by side were of three seeds and so of other games: 2.14 times the duels
  and 2.07 times the decisions of the one alone. With fewer threads a run
  also has fewer duels in flight, so the rows of 4 and 7 threads do not
  tell the threads' part from the batches'.

**How many runs a rented machine carries** (2026-10-13,
`experiments/commands/rented-capacity.sh` to its end): a pod of Runpod's
Secure Cloud at $0.50 an hour, an RTX 3090 with 32 threads of an AMD EPYC
7C13 allowed to it, the checkout on the container's own disk. The arm's
checkpoint, half of the duels against itself, a pool of 32 threads with 10
duels each in every run, four timed updates a run. An update in each run,
and the duels an hour of all of them:

| Runs at once | As it was | As it was, through MPS | Calls from graphs | Calls from graphs, through MPS |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 36.9 s, 50,800 | 41.7 s, 44,900 | 31.9 s, 58,600 | 31.8 s, 58,900 |
| 2 | 48.7 s, 76,900 | 43.2 s, 86,700 | 39.2 s, 95,500 | 35.9 s, 104,400 |
| 3 | 59.6 s, 94,200 | 41.5 s, 135,400 | 45.2 s, 124,200 | 34.9 s, 161,000 |
| 4 | 72.5 s, 103,200 | 41.4 s, 181,100 | 52.6 s, 142,600 | 35.5 s, 210,900 |
| 6 | 99.5 s, 112,800 | 44.6 s, 251,700 | 72.9 s, 154,100 | 39.4 s, 285,000 |

- **Through MPS a run more costs next to nothing there.** Six runs at
  once take 39.4 s an update each where one alone takes 31.8 s, and play
  285,000 duels an hour, 4.8 times the one. With six the card was busy
  86% of the time and 14 of the 32 threads; more than six were not tried,
  and the duels an hour were still rising.
- **As processes have the card, six runs do 2.2 times the work of one**,
  2.6 times with the graphs, and the card is busy 96% to 98% of the time:
  through MPS it does 2.2 and 1.8 times that work at 79% to 86%.
- **One run there is half a run on the CUDA machine**: 31.8 s an update
  with the graphs where the CUDA machine takes 16.3 s. Its games take
  13.6 s (the engines 5.3 s, the model 5.1 s) where the CUDA machine's
  take 7.8 s, and its pass 12.4 s where the CUDA machine's took 5.7 s.
  The graphs take the model's part of the games from 10.7 s to 5.0 s
  there, an update from 36.9 s to 31.9 s.
- **The same games**: every run at once played the games of the run
  alone, and the run alone the same games as it was and with the graphs,
  in each of five updates.
- **A run alone through MPS took 41.7 s as it was where it took 36.9 s
  without**, and the same either way with the graphs. One measure each of
  four updates: not looked into.
- **By the hour**: 285,000 duels for $0.50 with six runs through MPS,
  where the CUDA machine's three runs through MPS play 244,000. The pod
  was up an hour and 35 minutes: six and a half minutes of setup from
  nothing, and an hour and 24 minutes of runs.
- Not made there: the duels in two groups on top of the graphs
  (`WAYS="graphs-two-groups"`), and runs against the pilots alone.

**An experiment's arms on a pod, to their end**
(`experiments/commands/more-decks-pod.sh`, run on the machine the
checkpoints are to be on):

```sh
RENTED=root@HOST:PORT POD=ID bash experiments/commands/more-decks-pod.sh
```

- It sends the checkout's HEAD, starts the arms of
  `experiments/commands/more-decks.sh` there on the benchmark's setup
  (`experiments/commands/rented-setup.sh`, which asks the driver for the
  card first, makes the setup where it is missing and hands the command
  the setup's interpreter, device and threads), six arms at once through
  MPS, and follows them: every ten minutes what the arms keep is brought
  to the same paths here (`runs/more-decks`, and the arms' own output in
  `runs/rented`). A run passes through a checkpoint after every update
  and keeps one in `STEP`; `PARTS=kept` of the arms' script names the
  files that stay, and only those are brought.
- **When the arms have ended, or failed, and every file kept there is
  here as it is there, the pod is stopped and deleted** (`runpodctl pod
  stop`, `runpodctl pod delete`; `POD` is its id at Runpod), and
  `runpodctl` is asked whether it still names the pod: where it does, the
  script ends with 4 and says that the pod is to be deleted where it was
  rented. Before anything is started `runpodctl` must name that pod: it
  is set up with the account's key (`runpodctl doctor`), which the script
  neither reads nor is given.
- **The pod's runs are on its volume** (`VOLUME`, `/workspace`): the
  script makes the checkout's `runs` folder there a link to a folder of
  the volume before anything is started. A pod's own disk is cleared when
  the pod stops, and its volume is not, so what the arms wrote is still
  there after a stop or a restart of the pod. The checkout, the build and
  the setup stay on the pod's own disk, where a suite was a fifth faster.
- **A pod is deleted with what it holds.** Where not everything could be
  brought, the pod is stopped and not deleted: what the arms wrote stays
  on its volume, which is paid for until the pod is deleted, and the
  script ends with 3 and says so. The pod is then started again where it
  was rented and the script run again with the address it has then: it
  brings what is there and deletes the pod. (A pod's line in
  `runs/rented-hosts` is removed first where its address is the same and
  its key another.) A pod without a volume is left running in that case,
  since a stop would clear what it holds. `LEAVE=1` leaves the pod as it
  is at the end.
- It is started again at any time, a terminal that closed or the machine
  restarted: the arms run on there, and it goes on following them. Arms
  that stopped with their pod are started again and go on from what the
  volume holds, the setup made again on the pod's new disk. A pod that no
  longer holds what was brought from it, on its volume either, has lost
  it, and nothing is started over what is here.
- This machine's SSH key must be among the account's before the pod is
  made: a pod takes the keys the account has when it is created.
- `tests/test_more_decks_pod.py` runs it against this machine, with
  stand-ins for the connection, for `runpodctl` and for the arms: what a
  run keeps is here and what it passes through is not; the pod is deleted
  after arms that failed too; where the copy is not whole it is stopped
  and not deleted, and what its volume holds is brought when it is back;
  a pod without a volume is left running then; a pod still named after
  its deletion is told. It has not been run against a pod yet.

## `status`

```sh
uv run python -m ygo_training status runs/example/metrics.jsonl
uv run python -m ygo_training status runs/example/metrics.jsonl --ppo
uv run python -m ygo_training status runs/example/metrics.jsonl --evaluation-suite
```

The default table shows samples, training and held-out win rates, cutoffs and
timings per version. `--ppo` shows the optimization: value explained
variance, KL, clipping, entropy, imitation loss and agreement with the
teacher. The explained variance is that of the value head before the update,
on the update's own games, at the first step of each transaction, where the
head is trained. In imitation-only updates it was taken over every step until
2026-10-07, 8% of which are later steps; a record with
`imitation_transactions` holds the newer figure. `--evaluation-suite` shows
the pooled and per-matchup held-out results. `--eval-only` keeps the
versions that were evaluated.

## The search

`policy-bench search` (in `native/ygo-policies`) measures what a one-step
search on top of a pilot gains, and with `--record true` writes its duels as
records the environment can play again. `tools/policy_search_*.py` summarize
its rows. Build it with `cargo build --release -p ygo-policies-bench -p
ygo-policies-ffi` in that directory: a bare `cargo build` there leaves the
bench as it was.

```sh
cd native/ygo-policies
target/release/policy-bench search --policies blackwing \
  --opponents burn,crystal,dragunity,gishki,gladiator,heroes,infernity,lightsworn,monarch,morphtronic,rock-block,spellcaster \
  --games 64 --seed 900000 --workers 22 --record true --log true \
  --output ../../analysis/search/heldout.jsonl
```

The same command at the same revisions writes the same games.

With `--log true` the rows also hold how every alternative did in every
world it was tried in. The search takes an alternative when it is ahead of
the pilot's answer by a paired z over the same worlds, in three stages.
`tools/policy_search_firmness.py ROWS` reads from the log how firmly each
deviation was decided: how many decisions ended with an alternative ahead
that was not taken, how a lead holds in the worlds that had no part in
choosing it, and how often the same test takes the same alternative when its
worlds are drawn again. It checks first that it reads the test as the bench
ran it, by the choice logged at every decision.

Where the pilot's answer wins every world of the first stage nothing else
is tried, since nothing can be ahead, and the log says nothing of the
alternatives there. `--log full` has them tried there as well, through the
first stage: the games are the same, with a tenth to a fifth more playouts,
and the log then holds a first-stage result for every alternative at every
decision examined.

The bench numbers the games of a matchup from the run's `--seed` on, so two
runs share deals unless their seeds lie further apart than their games per
matchup. Give the run to train on and the run to measure on seeds that do.

## Changing the model's input

The model's input has one definition: the crates of `native/ygo-model-input`
(`input/src` for the arrays, `seat/src` for the view and the seat's trail),
which build every array the model reads, here through the library and
wherever else a checkpoint is seated. That was decided on 2026-10-16;
until then a Python reference here built the same arrays, and the two were
held together byte for byte. That submodule is a public repository of its
own, so a change of the input is:

1. the change there, with the decisions it checks itself against written
   again from the library, the library built first, and both committed
   there:

   ```sh
   cargo build --release
   uv run python tools/model_input_vectors.py \
     native/ygo-model-input/input/vectors/decisions.json
   cargo test --release -p ygo-model-input
   ```

2. what the model makes of the arrays, where their shapes change
   (`ygo_training/schema.py`, `model.py`, the checkpoint format), and the
   new revision of the submodule pinned here, in one commit.

`tests/test_model_input_vectors.py` and that repository's own tests fail
while the file is not what the library gives, so an input changed by
mistake is caught both here and there; `tests/test_packed.py` holds what
the trainer says of the arrays to what the library builds. That repository
is pushed before this one, and a machine that pulls takes it with
`git submodule update --init --recursive`; the run scripts refuse until it
has.

A checkpoint needs the revision of the input it was trained with. A project
that seats one pins that revision, whatever this repository has moved to
since.

The latest such change is format 15 (2026-10-16): both deck lists dropped,
which a copy was found to play as well without
([the experiment](../experiments/2026-10-15-without-deck-lists.md)), and
with them the encoder of a deck's cards and their two places in the
state's first layer. Checkpoints of formats 13 and 14 read the lists and do
not load at it or after; the revision before it (`05d27141`) runs them, as
it runs the scripts of the experiments that made them. Format 16
(2026-10-17) changes no array of the input: what it adds, a card's
description and identities hidden, is looked up by the codes the input
already holds (`train --card-table`, above).

## A checkpoint of format 14, played by today's code

```sh
uv run python tools/fold_deck_lists.py runs/more-decks/decks-01/checkpoints/model-*.pt \
  --own blackwing --opponent blackwing --old-checkout CHECKOUT \
  --output-dir runs/more-decks/decks-01/blackwing-mirror
```

For the duels of one pairing of decks a network of format 14 has an equal of
today's format. It reads the two deck lists, its seat's own and the other
seat's, each as one vector; a deck's arrays are built once and shared by its
decisions, so in every duel of the same two decks the two vectors are the
same at every decision, and what they add to the state's first layer is a
constant. The tool folds it into that layer's bias, drops the columns that
read the lists and the lists' encoder, and writes the network in format 15,
the pairing in its metadata (`folded_deck_lists`, with the checkpoint it came
from, that checkpoint's fingerprint and the old revision). The vectors are
the old network's own, worked out by the old code: `CHECKOUT` is a checkout
at `05d27141` or before, with its submodules at its pins, built, and with an
environment of its own (`uv sync --frozen --offline` there):
`bash tools/old_checkout.sh 05d27141 ../ygo-05d27141` makes one, a worktree
of that revision beside this checkout, its submodules filled from this one's
without the network. The code of `a7523f68`, which played the record's
mirror, is that of `05d27141`.

A folded checkpoint loads only where everything it plays is its pairing,
its own seat's deck first (`checkpoints.folded_pairing`): `evaluate-suite`
with every case of that pairing and its decks' own lists, or held for cases
that are; `train`, the other commands and the tools refuse it. It plays as
it did, the other seat's list included, which no network of today's reads:
it is for measuring the networks of before against those of now, not for
training.

Checked on 2026-10-20 with `decks-01` and `decks-13` of
[more decks](../experiments/2026-10-14-more-decks.md) in the Blackwing
mirror. At `a7523f68`, copies of the two in format 14 with the lists'
columns at zero and the vectors in the bias played the 4,096 games of the
mirror the two had played there, game for game. Folded and played by
today's code, `decks-13`'s last checkpoint against `decks-01` at each of
its seventeen checkpoints ended 69,617 of the same 69,632 games as at
`a7523f68` (the others' cause not told: the rules since, which offer no
activation at a chain window that brought the seat back, or the rounding of
the bias). A network of one deck has little use for the lists: `decks-01`'s
vector of Blackwing's list has 8 of its 96 numbers above zero at its copy's
end (a length of 0.04) and at most three at its sixteen checkpoints of PPO
(under 0.032), none at nine of them, where `decks-13`'s has four, of length
0.95.

```sh
NAME=all71-10560 \
  CHECKPOINTS=runs/seventy-one-decks/all71/checkpoints/model-27359.pt,runs/seventy-one-decks/all71/checkpoints/model-27479.pt,runs/seventy-one-decks/all71/checkpoints/model-27599.pt \
  bash experiments/commands/blackwing-mirror-curve.sh
```

Where a network stands in the Blackwing mirror on the curve of `decks-01`,
the network of Blackwing alone: each checkpoint named against `decks-01` at
its copy's end and after each 120 updates of its PPO, folded for the mirror,
every deal of `eval_suites/blackwing-against-checkpoint.json` from either
seat, into `analysis/blackwing-mirror-curve/NAME`. The report
(`tools/blackwing_mirror_curve.py`) gives the network's share against each,
over or under `decks-01` at two standard errors, and where it stands: between
the last point it is over and the first it is under, with the games of the
mirror `decks-01` had played by each (280 an update). On the development
machine's processor 2,048 games take some 45 s: a checkpoint at all
seventeen points, 25 minutes; `POINTS` names fewer. `NAME=decks-13` with
`decks-13`'s last checkpoint folded is the curve the record measured
(`analysis/blackwing-mirror-curve/decks-13`): it stands between `decks-01`
after 0 and after 120 updates.

## Exporting a checkpoint

```sh
uv run python tools/export_checkpoint.py \
  runs/example/checkpoints/model-02279.pt exported/model-02279
```

For a project that seats a checkpoint without PyTorch. It writes three files
to the folder:

- `model.safetensors`: the network's weights alone, in the safetensors
  format (a JSON header, then the numbers as they are), under the names the
  network has them by.
- `model.json`: which checkpoint it is (its updates, format and
  fingerprint), how the network reads the board (`board_representation`:
  with `mean-pool` the four `board_*` weights are not read), the revisions
  of this checkout, and every weight's shape.
- `outputs.json`: what the network answers to each decision of
  `native/ygo-model-input/input/vectors/decisions.json`, as a batch of one:
  a score for each answer, the place of the highest, how far it is ahead of
  the next (`margin`), and the value.

A forward pass written elsewhere is checked with them: the shared crates
make the arrays of each decision of that file, the pass reads them with
these weights, and what comes out is `outputs.json`. The reference is
`PolicyValueNet.evaluate` (`ygo_training/model.py`).

- The scores are 32-bit floats computed on the processor and reach the
  hundreds: a port agrees with them to a few parts in a million, not to the
  bit, and a decision with a `margin` under that is no test of the answer
  taken. Copies of a card that lie alike score the same to the bit, and the
  first of them is the answer.
- `outputs.json` names the revision of the shared repository and a digest
  of the file it answered: a port reads that revision's decisions.
- The revisions in `model.json` are those of the checkout the export ran
  in. They are those of the checkpoint's training when nothing has moved
  since; a checkpoint is shown what it was trained on at the engine, the
  scripts, the card database, the policies and the input of its training.
- Of the 29 MB of weights, 25 are the two tables a card's code is looked up
  in (65,536 rows each for the low and the high half of the code); a
  checkpoint reads the rows of the cards it meets.

## The web client's seat

```sh
uv run python tools/web_client_duels.py runs/model-02279.pt saved-duel.json
```

Whether the web client (`beat-claudio-yu-gi-oh`, a repository of its own)
seats a checkpoint as the trainer does. It has the seat and the input of
`native/ygo-model-input`, and around them its own build of the engine, its
own room and its own forward pass. A file is a duel as it saves one while
the duel is played: the engine's seed, both players' cards as they were
dealt, every answer either player gave, and the seed its pilot was made
with. The page keeps it in the tab's session storage (`duelInProgress`,
copied whole into a file or only its `saved`), and its engine gives it as
`LocalGame::save`. Several files are read in one call.

Each duel is played again on this checkout's engine, scripts and card data
with the checkpoint in the trainer's own seat, answering with its first
choice, against the other player's answers as they were kept. The page
seats the checkpoint as the trainer does when every answer of that seat is
the one this seat gives, byte for byte, and the other player's answers fit
to the last. It ends with 1 when a duel is not that.

- **The checkpoint is the one the page played with**, and the checkout at
  the revisions the page was built from: another checkpoint, engine, script
  or card leaves the duel after a few decisions, and the output says after
  how many.
- **The pilot is made with the page's seed.** It places the cards and
  breaks its ties by its seed. With another seed a card goes to another
  zone and nothing else differs; zones are no part of the network's input.
- **A duel is kept up to the last question its player was asked**, and is
  compared that far.
- A deck of the player's that is no list of the pool is told to the network
  here in the order it was dealt, and on the page in the order of its list.
  A deck is read as the mean of its cards, so a score differs in its last
  digit at most.
- Measured on 2026-10-11 with the long run's last checkpoint: 32 duels,
  all 4,956 answers of the network's seat the same (the record of the
  experiment on PPO with a head that predicts has it).

## Moving the pin

Before the `native/ygo-policies` pin moves, play a few plain and searched
games with the bench; after it has moved and everything is rebuilt, play
them again and compare:

```sh
cd native/ygo-policies && cargo build --release -p ygo-policies-bench -p ygo-policies-ffi
OPPONENTS=burn,crystal,dragunity,gishki,gladiator,heroes,infernity,lightsworn,monarch,morphtronic,rock-block,spellcaster
target/release/policy-bench matchup --policies blackwing --opponents $OPPONENTS \
  --games 16 --record true --output ../../analysis/pins/NAME/blackwing-matchup-16.jsonl
target/release/policy-bench search --policies blackwing \
  --opponents monarch,heroes,lightsworn,gladiator --games 4 --record true \
  --output ../../analysis/pins/NAME/blackwing-search-4x4.jsonl
cd ../.. && uv run python tools/compare_bench_rows.py \
  analysis/pins/OLD/blackwing-matchup-16.jsonl analysis/pins/NEW/blackwing-matchup-16.jsonl
```

The comparison says which games ended otherwise, were answered otherwise or
only asked otherwise, and exits with 1 when anything differs. A model
trained before the move still sees what it was trained on only if the plain
games are identical.

The policies can play a card script mended, in place of CardScripts' own
(`crates/ygo-policies-duel/scripts/`, with why in `src/script_fixes.rs`):
CardScripts is ProjectIgnis's repository itself, which the policies do not
change. None is played now. The only one so far was Batteryman Fuel
Cell's: its cost can destroy the only card it could target, and the script
then failed on the target it did not have, where EDOPro logs the error and
the effect does nothing. Mended at first to do nothing there (2026-10-21),
which changed no duel that ran to its end; from 2026-10-22 played as
CardScripts mended it (ProjectIgnis/CardScripts#1281, `188c4b93`; the
policies' `0ed8ee1`): the effect is not offered where its cost would take
its only target, which changes a duel where that comes about even when the
effect is not activated. Batteryman's pilot against every other deck (128
games a pairing, 8,960 games) and Blackwing's matchup above (192 games) are
the same games with it, digest for digest. On 2026-10-24 the pin of
`vendor/CardScripts` moved to `188c4b93` itself and the copy went: the
pilots' games of the forty and the eighteen decks' suites and of the
thirteen decks' screens (33,696) and `all71`'s on the decks whose scripts
changed (15,392) end as they did, game for game, and only Double Spell's
stored effect profile, a card no deck plays, changed. A test holds each
mend to the script it mends, and fails when CardScripts changes that
script: look at the mend again, and drop one that the pin of
`vendor/CardScripts` holds. `tools/edopro_files.py` does not see a mend:
the bench applies it on either side.

A move that is meant to change the pilots changes the duels of their decks,
and the comparison then says which matchups. Recorded games of those
matchups no longer play again, and no win rate against them is comparable
across the move: measure the pilot's reference again (`evaluate-suite
--pilot`), and a seed against it before anything starts from the seed.
