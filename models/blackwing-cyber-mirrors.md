# blackwing-cyber-mirrors: Blackwing and Cyber on their mirrors alone

A network of two decks, Blackwing and Cyber, given exactly the games of the
two single-deck networks together and no more: the two pilots imitated for
480 updates (240 a deck), then 1,440 updates of PPO at 1e-4, each of 520
duels, half of them against itself, each deck in its own mirror only, and
the others against the pilots of the thirteen decks of training. It never
played Blackwing against Cyber.

- **The checkpoint:** format 15, the board read as `mean-pool`, after 1,920
  updates.
- **Measured** by its first choice, 2,048 deals each played from both seats
  ([its record](../experiments/2026-10-24-two-decks.md)): head to head in
  each mirror against the deck's network of its own, 52.4% against
  [Blackwing alone](blackwing.md) and 52.2% against [Cyber alone](cyber.md);
  against `all71`, 67.6% of the Blackwing mirror and 78.4% of the Cyber
  mirror, 2.8 and 2.0 points over each of those. In Blackwing against Cyber,
  which it never played, it wins 54.0% against `all71`, where the network
  that played that matchup ([blackwing-cyber](blackwing-cyber.md)) wins
  63.5%.

## The files

- `blackwing-cyber-mirrors.pt`: the network for this repository's tools, its
  weights and what describes them, without the optimizer's state. Played
  with `python -m ygo_training evaluate-suite blackwing-cyber-mirrors.pt --suite ...`,
  or seated in a room of EDOPro with
  `python -m ygo_training edopro blackwing-cyber-mirrors.pt --learner DECK`
  ([the commands](../docs/commands.md)).
- `blackwing-cyber-mirrors.safetensors`, `blackwing-cyber-mirrors.json` and
  `blackwing-cyber-mirrors.outputs.json`: the network for a project that
  seats it without PyTorch: its weights, how it reads its input, and what it
  answers to the decisions of
  [ygo-model-input](https://github.com/fourlastor/ygo-model-input)'s
  vectors, to check a port against (`tools/export_checkpoint.py`).
- `SHA256SUMS`, for `sha256sum --check`.

## License

The weights are licensed under the Apache License 2.0, as the code is.
Playing them needs the duel engine and the card scripts, which are licensed
under the GNU Affero General Public License v3 ([the README](../README.md)).
