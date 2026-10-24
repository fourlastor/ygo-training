# all71: every deck of the pool

A network that plays all 71 decks of the policies' pool, made from scratch:
the decks' pilots imitated, 240 updates a deck (17,040 updates), then 10,560
updates of PPO at 1e-4, each of 520 duels, half of them against itself on
any two of its decks and the others against the pilots of all 71 decks.

- **The checkpoint:** format 15, the board read as `mean-pool`, after
  27,600 updates.
- **Measured** by its first choice on deals held out of training, against
  the pilots of the thirteen decks of training
  ([its record](../experiments/2026-10-19-seventy-one-decks.md)): 58.8% of
  the thirteen decks' own games; 54.4% with the eighteen decks of the
  earlier network of thirty-one, 7.8 points over their own pilots; and
  61.9% with the forty decks new to the pool, where their own pilots win
  45.9%, over its pilot with every one but Countdown.
- **Beside the single-deck networks:** each of them beats it in its own
  mirror (Blackwing alone 64.8%, Cyber alone 76.4%), and loses to it with
  its deck against the decks it never played
  ([the record](../experiments/2026-10-24-two-decks.md)).

## The files

- `all71.pt`: the network for this repository's tools, its weights and what
  describes them, without the optimizer's state. Played with
  `python -m ygo_training evaluate-suite all71.pt --suite ...`, or seated in
  a room of EDOPro with `python -m ygo_training edopro all71.pt --learner DECK`
  ([the commands](../docs/commands.md)).
- `all71.safetensors`, `all71.json` and `all71.outputs.json`: the network
  for a project that seats it without PyTorch: its weights, how it reads its
  input, and what it answers to the decisions of
  [ygo-model-input](https://github.com/fourlastor/ygo-model-input)'s
  vectors, to check a port against (`tools/export_checkpoint.py`).
- `SHA256SUMS`, for `sha256sum --check`.

## License

The weights are licensed under the Apache License 2.0, as the code is.
Playing them needs the duel engine and the card scripts, which are licensed
under the GNU Affero General Public License v3 ([the README](../README.md)).
