# fourteen-decks: the thirteen decks of training and Cyber in one network

A network of fourteen decks made from their champions: each champion's games
imitated for 2,240 updates of 520 duels (160 a deck), the champion against
itself in its mirror and against the pilots of the thirteen decks of
training; then 4,320 updates of PPO at 1e-4, each of 520 duels, every one
of them against itself on the ordered pairs of its fourteen decks, a
quarter of them in the mirrors, and none against a pilot.

- **The checkpoint:** format 15, the board read as `mean-pool`, after 6,560
  updates.
- **Measured** by its first choice
  ([its record](../experiments/2026-10-26-fourteen-decks.md), the arm
  `quarter`), every deal played from both seats. Head to head in each
  mirror against the deck's champion ([the champions](champions.md)), 2,048
  deals: a mean of 49.1% ± 0.2%. Against `all71` on every pairing of its
  decks, by its last three checkpoints, 1,024 deals each: a mean of 56.8%
  over the 105 pairings, the weakest Lightsworn against Burn at 49.9% ±
  0.5%. Against the pilots of the thirteen decks, each deck against each,
  128 games: 64.1%, where `all71` wins 58.3% of the same games.

| Deck | `--learner` | Head to head against its champion | Its mirror against `all71` | Its pairings against `all71` | Against the pilots (`all71`'s) |
|---|---|---|---|---|---|
| Blackwing | `blackwing` | 49.6% | 67.0% | 56.6% | 75.5% (70.9%) |
| Cyber | `cyber` | 48.7% | 78.7% | 59.3% | 66.9% (56.4%) |
| Lightsworn | `lightsworn` | 48.6% | 57.3% | 54.0% | 79.6% (76.7%) |
| Infernity | `infernity` | 48.3% | 62.5% | 56.8% | 73.8% (66.7%) |
| Gishki | `gishki` | 50.7% | 67.8% | 57.5% | 60.9% (54.9%) |
| Burn | `burn` | 46.1% | 63.1% | 53.2% | 29.7% (26.0%) |
| Crystal Beasts | `crystal` | 51.5% | 65.2% | 56.4% | 43.1% (40.1%) |
| Dragunity | `dragunity` | 47.1% | 62.8% | 55.7% | 76.3% (72.9%) |
| Gladiator Beasts | `gladiator` | 50.3% | 65.9% | 56.6% | 76.1% (66.0%) |
| HEROes | `heroes` | 49.6% | 71.1% | 56.2% | 81.4% (75.8%) |
| Monarchs | `monarch` | 46.7% | 69.7% | 59.0% | 79.6% (74.0%) |
| Morphtronics | `morphtronic` | 49.4% | 59.1% | 54.6% | 48.2% (41.2%) |
| Rock-Block | `rock-block` | 49.1% | 62.3% | 55.9% | 72.0% (65.6%) |
| Spellcasters | `spellcaster` | 51.8% | 68.0% | 53.9% | 34.0% (29.7%) |

A deck's pairings are the fourteen it is in, its mirror among them. On the
pairings of Blackwing, Cyber, Lightsworn and Infernity the network of
[four-decks](four-decks.md) is the stronger: head to head there this one
wins 47.1% ± 0.2%. Its decks against others are not measured, and against
another deck [all71](all71.md) may still be the better opponent.

## The files

- `fourteen-decks.pt`: the network for this repository's tools, its weights
  and what describes them, without the optimizer's state. Played with
  `python -m ygo_training evaluate-suite fourteen-decks.pt --suite ...`, or
  seated in a room of EDOPro with
  `python -m ygo_training edopro fourteen-decks.pt --learner DECK`, the deck
  one of the table's ([the commands](../docs/commands.md)).
- `fourteen-decks.safetensors`, `fourteen-decks.json` and
  `fourteen-decks.outputs.json`: the network for a project that seats it
  without PyTorch: its weights, how it reads its input, and what it answers
  to the decisions of
  [ygo-model-input](https://github.com/fourlastor/ygo-model-input)'s
  vectors, to check a port against (`tools/export_checkpoint.py`).
- `SHA256SUMS`, for `sha256sum --check`.

## License

The weights are licensed under the Apache License 2.0, as the code is.
Playing them needs the duel engine and the card scripts, which are licensed
under the GNU Affero General Public License v3 ([the README](../README.md)).
