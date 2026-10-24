# blackwing-cyber: Blackwing and Cyber in one network

A network of two decks, Blackwing and Cyber, made by the recipe of a
single-deck champion for both at once: the two pilots imitated for 480
updates (240 a deck), then 2,880 updates of PPO at 1e-4, each of 520 duels,
half of them against itself on every ordered pair of its two decks, so that
each mirror has as many games as a champion's, and the others against the
pilots of the thirteen decks of training.

- **The checkpoint:** format 15, the board read as `mean-pool`, after 3,360
  updates.
- **Measured** by its first choice, 2,048 deals each played from both seats
  ([its record](../experiments/2026-10-24-two-decks.md)): head to head in
  each mirror against the deck's network of its own, 52.3% against
  [Blackwing alone](blackwing.md) and 57.2% against [Cyber alone](cyber.md);
  against `all71`, 67.8% of the Blackwing mirror and 81.3% of the Cyber
  mirror, 3.0 and 4.9 points over each of those; and 63.5% of Blackwing
  against Cyber, playing either deck.

## The files

- `blackwing-cyber.pt`: the network for this repository's tools, its weights
  and what describes them, without the optimizer's state. Played with
  `python -m ygo_training evaluate-suite blackwing-cyber.pt --suite ...`, or
  seated in a room of EDOPro with
  `python -m ygo_training edopro blackwing-cyber.pt --learner DECK`
  ([the commands](../docs/commands.md)).
- `blackwing-cyber.safetensors`, `blackwing-cyber.json` and
  `blackwing-cyber.outputs.json`: the network for a project that seats it
  without PyTorch: its weights, how it reads its input, and what it answers
  to the decisions of
  [ygo-model-input](https://github.com/fourlastor/ygo-model-input)'s
  vectors, to check a port against (`tools/export_checkpoint.py`).
- `SHA256SUMS`, for `sha256sum --check`.

## License

The weights are licensed under the Apache License 2.0, as the code is.
Playing them needs the duel engine and the card scripts, which are licensed
under the GNU Affero General Public License v3 ([the README](../README.md)).
