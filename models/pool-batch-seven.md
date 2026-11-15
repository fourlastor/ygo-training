# pool-batch-seven: the seventy-one and 475 game Decks in one network

A network that plays 546 decks: the 71 decks of the policies' pool and 475 Decks of
the games of the era (`data/decks/games/`), taken on by the seven batches of the
[expanded pool](../experiments/2026-11-02-expanded-pool.md). It starts from
[seventy-one-champions](seventy-one-champions.md) and goes on with 49,680 updates of
PPO at 1e-4, each of 520 duels against itself: sixty-two new Decks in the first batch,
fifty in each of the next four, a hundred in the sixth and 113 in the last. Most duels
of a batch have one of its new Decks in a seat and the other drawn from all the decks.
At the decisions of the decks it already played, in their duels with each other, each
batch is held to the network it began from, and from batch four the duels of two of
the seventy-one to `seventy-one-champions`.

- **The checkpoint:** format 15, the board read as `mean-pool`. It is the mean of the
  weights of the last six checkpoints batch seven kept, at its 3,840th update (step
  82,800).
- **Trained on** some 43 million duels: 25.8 million in the seven batches, after the
  17.2 million that made `seventy-one-champions` (5.9 million imitated from the
  champions' recordings and 11.3 million of PPO).
- **Its cards:** its 546 decks hold 3,269 of the 4,065 cards of the World Championship
  2011 list, 80.4%, its banned cards counted in, where the seventy-one decks hold 956.
- **Measured** by its first choice at batch seven's end
  ([its record](../experiments/2026-11-02-expanded-pool.md)), every deal played from
  both seats. Head to head in each of the seventy-one's mirrors against the deck's
  champion, 128 deals: a mean of 49.2% ± 0.3%, 1.5 ± 0.4 points under
  `seventy-one-champions` on the same deals, the lowest `deckout` 37.1%, `dragunity`
  41.0%, `gladiator` 43.0%, `reptilianne` 43.4% and `blackwing` 44.1%. The 433 decks
  taken on before batch seven, every ordered pairing of them, head to head against
  batch six's network: 50.08% ± 0.04% of 1,499,912 games. The 113 Decks batch seven
  took on, on their pairings with the 433, head to head against batch six's network:
  52.9%, where that network stood at 50.1% against itself; against the pilots of the
  thirteen decks of training, 46.2%, where batch six's network won 43.8%.

Its mirrors have slid against `seventy-one-champions` batch by batch, −0.9, −1.1, −1.3
and −1.5 points since batch four, not decided by the record's mark. It has not been
measured against `all71` or `seventy-one-champions` across the pool, nor against decks
it never met: nine lists of the games among them (`game-0037`, `game-0065`,
`game-0093`, `game-0167`, `game-0171`, `game-0233`, `game-0335`, `game-0379` and
`game-0397`), kept out of its training for faults of the engine or of the seat, since
mended.

## The files

- `pool-batch-seven.pt`: the network for this repository's tools, its weights and what
  describes them. Played with `python -m ygo_training evaluate-suite
  pool-batch-seven.pt --suite ...`, or seated in a room of EDOPro with
  `python -m ygo_training edopro pool-batch-seven.pt --learner DECK`, the deck one of
  the table's ([the commands](../docs/commands.md)).
- `pool-batch-seven.safetensors`, `pool-batch-seven.json` and
  `pool-batch-seven.outputs.json`: the network for a project that seats it without
  PyTorch: its weights, how it reads its input, and what it answers to the decisions of
  [ygo-model-input](https://github.com/fourlastor/ygo-model-input)'s vectors, to check
  a port against (`tools/export_checkpoint.py`).
- `SHA256SUMS`, for `sha256sum --check`.

## License

The weights are licensed under the Apache License 2.0, as the code is. Playing them
needs the duel engine and the card scripts, which are licensed under the GNU Affero
General Public License v3 ([the README](../README.md)).
