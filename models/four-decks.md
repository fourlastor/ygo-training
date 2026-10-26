# four-decks: Blackwing, Cyber, Lightsworn and Infernity in one network

A network of four decks made from their champions: each champion's games
imitated for 960 updates of 520 duels (240 a deck), the champion against
itself in its mirror and against the pilots of the thirteen decks of
training; then 2,880 updates of PPO at 1e-4, each of 520 duels, half of them
against itself on every ordered pair of its four decks as often as any
other, and the others against the thirteen decks' pilots.

- **The checkpoint:** format 15, the board read as `mean-pool`, after 3,840
  updates.
- **Measured** by its first choice
  ([its record](../experiments/2026-10-24-four-decks.md)), every deal
  played from both seats. Head to head in each mirror against the deck's
  champion, 2,048 deals: 50.1% against [Blackwing alone](blackwing.md),
  52.7% against [Cyber alone](cyber.md), 49.5% against Lightsworn's and
  48.1% against Infernity's ([the champions](champions.md)). Against
  `all71` on every pairing of its decks, by its last three checkpoints:

| Pairing | Against `all71` |
|---|---|
| Blackwing mirror | 65.4% |
| Blackwing and Cyber | 63.2% |
| Blackwing and Lightsworn | 58.1% |
| Blackwing and Infernity | 61.6% |
| Cyber mirror | 79.7% |
| Cyber and Lightsworn | 57.2% |
| Cyber and Infernity | 62.8% |
| Lightsworn mirror | 56.4% |
| Lightsworn and Infernity | 60.7% |
| Infernity mirror | 65.5% |

Its decks against others are not measured. Single-deck champions were weaker
than `all71` there ([the champions](champions.md)), and against another deck
[all71](all71.md) may still be the better opponent.

## The files

- `four-decks.pt`: the network for this repository's tools, its weights and
  what describes them, without the optimizer's state. Played with
  `python -m ygo_training evaluate-suite four-decks.pt --suite ...`, or
  seated in a room of EDOPro with
  `python -m ygo_training edopro four-decks.pt --learner DECK`, the deck
  `blackwing`, `cyber`, `lightsworn` or `infernity`
  ([the commands](../docs/commands.md)).
- `four-decks.safetensors`, `four-decks.json` and `four-decks.outputs.json`:
  the network for a project that seats it without PyTorch: its weights, how
  it reads its input, and what it answers to the decisions of
  [ygo-model-input](https://github.com/fourlastor/ygo-model-input)'s
  vectors, to check a port against (`tools/export_checkpoint.py`).
- `SHA256SUMS`, for `sha256sum --check`.

## License

The weights are licensed under the Apache License 2.0, as the code is.
Playing them needs the duel engine and the card scripts, which are licensed
under the GNU Affero General Public License v3 ([the README](../README.md)).
