# seventy-one-champions: every deck of the pool, taught by its champion

A network that plays all 71 decks of the policies' pool, made from their
champions: each deck's champion recorded once and imitated for 11,360 updates of
520 duels (160 a deck), the champion against itself in its mirror and against
the pilots of the thirteen decks of training; then 21,900 updates of PPO at 1e-4,
each of 520 duels, every one of them against itself on the ordered pairs of its
71 decks and none against a pilot, a quarter of them in the mirrors for its first
5,760 updates and half after that.

- **The checkpoint:** format 15, the board read as `mean-pool`. It is the mean of
  the weights of the last thirteen checkpoints PPO kept, 240 updates apart, from
  18,880 to 21,760 updates: a single checkpoint carries the noise of its last
  updates, and where the run measured both, after 5,760 updates, such a mean stood
  2.0 ± 0.4 points over the last checkpoint in the mirrors.
- **Measured** by its first choice
  ([its record](../experiments/2026-10-28-seventy-one-champions.md)), every deal
  played from both seats. Head to head in each mirror against the deck's champion,
  2,048 deals: a mean of 50.4% ± 0.1%. Against `all71` on every ordered pairing, 64
  games each, a deck read from both seats: a mean of 53.4%, sixty-six decks over
  50% by two standard errors, five level with it and none under. Against the pilots
  of the thirteen decks, each deck against each, 128 games: 62.0%, where `all71`
  wins 59.7% of the same games.

| `--learner` | Head to head against its champion | Against `all71`, from both seats | Against the pilots (`all71`'s) |
|---|---|---|---|
| `agents` | 51.8% | 55.7% | 79.7% (72.7%) |
| `alien` | 51.6% | 53.5% | 72.2% (72.2%) |
| `amazoness` | 50.6% | 52.2% | 65.2% (66.2%) |
| `arcana` | 41.7% | 52.4% | 33.3% (31.1%) |
| `batteryman` | 49.7% | 52.5% | 57.0% (57.0%) |
| `benkei` | 52.2% | 53.5% | 57.2% (54.5%) |
| `blackwing` | 49.7% | 54.1% | 72.2% (70.9%) |
| `burn` | 46.9% | 51.4% | 27.7% (26.0%) |
| `chain-burn` | 52.2% | 50.4% | 58.3% (54.7%) |
| `chaos` | 52.9% | 53.0% | 82.0% (81.2%) |
| `cloudian` | 47.8% | 51.5% | 57.3% (56.6%) |
| `countdown` | 50.3% | 54.1% | 75.7% (67.2%) |
| `counter-fairy` | 49.0% | 51.9% | 63.2% (61.2%) |
| `crystal` | 50.1% | 54.3% | 42.8% (40.1%) |
| `cyber` | 56.8% | 54.7% | 59.7% (56.4%) |
| `dark-world` | 50.9% | 53.3% | 65.0% (62.6%) |
| `deckout` | 40.6% | 51.1% | 60.3% (57.3%) |
| `demise` | 49.3% | 52.5% | 65.2% (61.2%) |
| `destiny-board` | 51.1% | 51.8% | 51.1% (47.1%) |
| `destiny-hero` | 46.1% | 52.2% | 24.0% (21.8%) |
| `draconic-might` | 51.1% | 52.2% | 60.0% (58.5%) |
| `dragunity` | 47.9% | 53.7% | 75.3% (72.9%) |
| `exodia` | 50.9% | 50.4% | 77.4% (78.5%) |
| `fabled` | 49.5% | 56.1% | 60.5% (51.2%) |
| `fish` | 51.4% | 53.3% | 69.4% (69.4%) |
| `flamvell` | 51.7% | 54.1% | 67.3% (64.2%) |
| `fortune-lady` | 49.9% | 52.1% | 39.4% (38.9%) |
| `garden` | 52.1% | 55.4% | 62.6% (60.3%) |
| `gem-knight` | 57.4% | 54.2% | 61.6% (58.7%) |
| `gemini` | 51.4% | 54.9% | 74.8% (70.3%) |
| `genex` | 52.4% | 54.9% | 67.5% (64.0%) |
| `gishki` | 48.1% | 54.0% | 58.4% (54.9%) |
| `gladiator` | 51.0% | 53.8% | 73.9% (66.0%) |
| `gravekeeper` | 52.1% | 51.9% | 72.5% (73.5%) |
| `gusto` | 52.7% | 58.4% | 73.8% (70.1%) |
| `harpie` | 48.4% | 52.7% | 64.9% (63.3%) |
| `herald` | 48.0% | 53.8% | 60.5% (55.6%) |
| `heroes` | 49.8% | 53.8% | 78.7% (75.8%) |
| `ice-barrier` | 55.8% | 53.6% | 56.3% (55.6%) |
| `infernity` | 46.6% | 54.6% | 72.1% (66.7%) |
| `iron-chain` | 52.6% | 54.7% | 66.4% (63.8%) |
| `jurrac` | 54.1% | 55.7% | 72.1% (67.7%) |
| `karakuri` | 55.6% | 55.1% | 68.1% (63.5%) |
| `lightsworn` | 51.9% | 53.8% | 77.5% (76.7%) |
| `machina` | 52.5% | 53.7% | 89.7% (88.6%) |
| `macro-dd` | 46.8% | 50.0% | 81.2% (82.2%) |
| `malefic` | 47.8% | 50.0% | 69.1% (70.3%) |
| `monarch` | 47.6% | 54.0% | 78.0% (74.0%) |
| `morphtronic` | 49.3% | 52.0% | 44.4% (41.2%) |
| `naturia` | 52.3% | 55.0% | 71.3% (68.6%) |
| `nordic` | 51.1% | 53.9% | 79.4% (76.8%) |
| `nurse` | 51.7% | 51.3% | 47.2% (44.8%) |
| `ojama` | 48.3% | 52.1% | 17.7% (16.2%) |
| `psychic` | 51.9% | 53.9% | 63.0% (59.3%) |
| `pyramid` | 45.6% | 52.0% | 34.4% (35.2%) |
| `quickdraw-plant` | 50.9% | 53.9% | 76.6% (74.8%) |
| `reptilianne` | 49.4% | 54.1% | 59.0% (57.0%) |
| `rock-block` | 46.1% | 53.3% | 69.4% (65.6%) |
| `scrap` | 49.6% | 53.8% | 83.1% (79.9%) |
| `six-samurai` | 48.9% | 53.3% | 79.2% (77.3%) |
| `spellcaster` | 50.3% | 52.2% | 32.5% (29.7%) |
| `spirit` | 54.1% | 54.1% | 51.3% (46.3%) |
| `tele-dad` | 56.4% | 54.0% | 70.0% (69.7%) |
| `toon` | 49.8% | 51.1% | 21.0% (19.5%) |
| `venom` | 47.0% | 52.3% | 25.7% (24.8%) |
| `verdict` | 47.9% | 52.8% | 84.3% (83.8%) |
| `volcanic` | 57.1% | 54.4% | 62.0% (58.2%) |
| `watt` | 47.4% | 50.8% | 21.9% (29.1%) |
| `worm` | 51.6% | 53.3% | 61.1% (58.9%) |
| `x-saber` | 51.8% | 54.1% | 71.5% (69.8%) |
| `zombie` | 51.1% | 55.5% | 80.3% (77.2%) |

It falls short of what its record asked of it in two places. Two mirrors stand
under its floor of 45% against their champions, `deckout` at 40.6% and `arcana` at
41.7%, each ± 0.7%. Five decks are level with `all71` and not over it: `malefic`,
`macro-dd`, `chain-burn`, `exodia` and `watt`, each ± 0.5%.

`arcana`'s mirror stalls behind The Fool: with one in Defense on its side the
network stops summoning, against any deck, and in its mirror that leaves seat 1 to
win most games by deck-out. Holding that habit out gains nothing anywhere
([its record](../experiments/2026-10-28-seventy-one-champions.md)). Against decks
it never met, this network has not been measured.

## The files

- `seventy-one-champions.pt`: the network for this repository's tools, its weights
  and what describes them. Played with `python -m ygo_training evaluate-suite
  seventy-one-champions.pt --suite ...`, or seated in a room of EDOPro with
  `python -m ygo_training edopro seventy-one-champions.pt --learner DECK`, the deck
  one of the table's ([the commands](../docs/commands.md)).
- `seventy-one-champions.safetensors`, `seventy-one-champions.json` and
  `seventy-one-champions.outputs.json`: the network for a project that seats it
  without PyTorch: its weights, how it reads its input, and what it answers to the
  decisions of [ygo-model-input](https://github.com/fourlastor/ygo-model-input)'s
  vectors, to check a port against (`tools/export_checkpoint.py`).
- `SHA256SUMS`, for `sha256sum --check`.

## License

The weights are licensed under the Apache License 2.0, as the code is. Playing them
needs the duel engine and the card scripts, which are licensed under the GNU Affero
General Public License v3 ([the README](../README.md)).
