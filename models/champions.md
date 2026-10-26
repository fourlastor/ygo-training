# champions: each deck's champion of its mirror

One network for each deck of the table, each made by the recipe that makes
a champion of its mirror: the deck's pilot imitated for 240 updates, then
720 updates of PPO at 1e-4, each of 520 duels, half of them its mirror
against itself and the others against the pilots of the thirteen decks of
training. Each plays its own mirror better than `all71`, the network of
every deck, and is released to be played against in that mirror.

- **The checkpoints:** format 15, the board read as `mean-pool`, each after
  960 updates.
- **Measured** by their first choice: each champion's last three
  checkpoints (after 720, 840 and 960 updates, the last of them released)
  against `all71` in its deck's mirror, 2,048 deals each played from both
  seats. A champion wins at least 52% of the games by two standard errors.

| Deck | Its list | Files | Its mirror against `all71` | Record |
|---|---|---|---|---|
| Blackwing | `Blackwing Assassin.ydk` | `blackwing` | 64.8% ± 0.5% | [Two decks](../experiments/2026-10-24-two-decks.md) |
| Cyber | `Power Surge.ydk` | `cyber` | 76.4% ± 0.5% | [The Cyber champion](../experiments/2026-10-23-cyber-champion.md) |
| Lightsworn | `Lightsworn Judgment.ydk` | `lightsworn` | 56.5% ± 0.5% | [Four decks](../experiments/2026-10-24-four-decks.md) |
| Infernity | `Infernity Infinity.ydk` | `infernity` | 63.0% ± 0.5% | [Four decks](../experiments/2026-10-24-four-decks.md) |
| Gishki | `Undersea Ceremony.ydk` | `gishki` | 65.5% ± 0.5% | [Four decks](../experiments/2026-10-24-four-decks.md) |
| Burn | `Burn Princess.ydk` | `burn` | 63.8% ± 0.4% | [Four decks](../experiments/2026-10-24-four-decks.md) |
| Crystal Beasts | `12 Crystal Beast - Rainbow.ydk` | `crystal` | 63.6% ± 0.6% | [Four decks](../experiments/2026-10-24-four-decks.md) |
| Dragunity | `Dragunity Flight.ydk` | `dragunity` | 62.8% ± 0.5% | [Four decks](../experiments/2026-10-24-four-decks.md) |
| Gladiator Beasts | `Fight, Gladiators!.ydk` | `gladiator` | 67.7% ± 0.5% | [Four decks](../experiments/2026-10-24-four-decks.md) |
| HEROes | `Fusion Heroes.ydk` | `heroes` | 68.3% ± 0.5% | [Four decks](../experiments/2026-10-24-four-decks.md) |
| Monarchs | `Emperor, Arise!.ydk` | `monarch` | 70.1% ± 0.5% | [Four decks](../experiments/2026-10-24-four-decks.md) |
| Morphtronics | `01 Morphtronic - Straight Up.ydk` | `morphtronic` | 59.2% ± 0.5% | [Four decks](../experiments/2026-10-24-four-decks.md) |
| Rock-Block | `06 Koaki Meiru - Rock Block.ydk` | `rock-block` | 63.3% ± 0.5% | [Four decks](../experiments/2026-10-24-four-decks.md) |
| Spellcasters | `14 - Spellcaster's Command.ydk` | `spellcaster` | 65.6% ± 0.5% | [Four decks](../experiments/2026-10-24-four-decks.md) |

Blackwing's and Cyber's are the networks of their own releases,
[blackwing](blackwing.md) and [cyber](cyber.md). Champions of more decks
join this release as they are made.

Outside its mirror a champion is weaker than `all71` playing its deck: of
the two measured, Blackwing's and Cyber's, playing their deck against
`all71` playing each deck outside the thirteen, win 7.0 and 5.1 points
less than `all71` playing it
([the record](../experiments/2026-10-24-two-decks.md)). Against another
deck, [all71](all71.md) plays it better.

## Playing one

A champion joins a room hosted in EDOPro, in LAN mode, as a player with its
deck, which `--learner` names as the files do:
`python -m ygo_training edopro burn.pt --learner burn`. The room is played
by the rules of training's duels: Master Rule 1, 8000 Life Points, a hand
of five and one draw a turn. Its opponent plays the
same deck, the list the table names among
[the policies' lists](https://github.com/fourlastor/ygo-policies/tree/eebace8d3a2777c4e759dc2d2d563a5e05535961/decks).
A room plays EDOPro's card scripts and database, and the seat is as
faithful as they are alike to training's: how to check an installation, and
the two lists that name a card by an artwork's code an EDOPro may not hold
(Lightsworn's Honest and the Monarchs' Gorz), are in
[the commands](../docs/commands.md).

## The files

- `DECK.pt`, for each deck of the table: the network for this repository's
  tools, its weights and what describes them, without the optimizer's
  state. Played with `python -m ygo_training evaluate-suite DECK.pt --suite ...`,
  or seated in a room of EDOPro as above
  ([the commands](../docs/commands.md)).
- `DECK.safetensors`, `DECK.json` and `DECK.outputs.json`: the network for
  a project that seats it without PyTorch: its weights, how it reads its
  input, and what it answers to the decisions of
  [ygo-model-input](https://github.com/fourlastor/ygo-model-input)'s
  vectors, to check a port against (`tools/export_checkpoint.py`).
- `SHA256SUMS`, one for every file, for `sha256sum --check`.

## License

The weights are licensed under the Apache License 2.0, as the code is.
Playing them needs the duel engine and the card scripts, which are licensed
under the GNU Affero General Public License v3 ([the README](../README.md)).
