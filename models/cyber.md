# cyber: Cyber alone

A network of one deck, Cyber (Power Surge), made by the recipe that makes a
champion of its mirror: the deck's pilot imitated for 240 updates, then 720
updates of PPO at 1e-4, each of 520 duels, half of them its mirror against
itself and the others against the pilots of the thirteen decks of training.

- **The checkpoint:** format 15, the board read as `mean-pool`, after 960
  updates.
- **Measured** by its first choice
  ([its record](../experiments/2026-10-23-cyber-champion.md)): 76.4% of the
  Cyber mirror against `all71`, 2,048 deals each played from both seats;
  62.5% of Cyber's 208 games against the thirteen decks' pilots, where
  `all71` wins 57.5%. Outside its mirror it is weaker: playing Cyber against
  `all71` playing each of the 57 other decks outside the thirteen, it wins
  38.3% of the games where `all71` playing Cyber wins 43.4%
  ([the record](../experiments/2026-10-24-two-decks.md)).

## The files

- `cyber.pt`: the network for this repository's tools, its weights and what
  describes them, without the optimizer's state. Played with
  `python -m ygo_training evaluate-suite cyber.pt --suite ...`, or seated in
  a room of EDOPro with `python -m ygo_training edopro cyber.pt --learner cyber`
  ([the commands](../docs/commands.md)).
- `cyber.safetensors`, `cyber.json` and `cyber.outputs.json`: the network for
  a project that seats it without PyTorch: its weights, how it reads its
  input, and what it answers to the decisions of
  [ygo-model-input](https://github.com/fourlastor/ygo-model-input)'s
  vectors, to check a port against (`tools/export_checkpoint.py`).
- `SHA256SUMS`, for `sha256sum --check`.

## License

The weights are licensed under the Apache License 2.0, as the code is.
Playing them needs the duel engine and the card scripts, which are licensed
under the GNU Affero General Public License v3 ([the README](../README.md)).
