#!/usr/bin/env bash
# Whether vectors made from the cards' texts put together the cards that do
# the same things. A measure, not an experiment: nothing is trained. About
# three minutes on a processor, from the repository root; with SCRIPTS=1 the
# cards' scripts are measured too, as texts are (about twenty minutes, most of
# it voyage on the scripts):
#
# 1. tools/card_embedding_check.py texts: the effect texts of the policies'
#    pool with what each card's script declares;
# 2. tools/card_text_embeddings.py: the texts (and with SCRIPTS=1 the
#    scripts) turned into vectors by voyage-4-nano, fetched at its commit into Hugging Face's cache
#    where it is not there (0.70 GB, no sign-in), with transformers beside
#    the trainer's packages for the one command (uv run --with);
# 3. tools/card_embedding_check.py check: those vectors, as they are, cut
#    down to 128 numbers and beside the words', against random vectors, the
#    words of the texts and of the scripts, and the scripts' key; each by
#    what the scripts declare and by how the texts read, the judge made from
#    what the vectors were made from in brackets.
#
# Into OUT: texts.json, voyage-4-nano-text.npy (and -lua.npy), report.txt and
# report.json. The scripts were measured on 2026-10-17 and found no better
# than the texts, which are faster and not the engine's own: off unless asked.
set -euo pipefail
OUT=${OUT:-analysis/card-embeddings}
SCRIPTS=${SCRIPTS:-0}
pinned=$(git submodule status --recursive)
if grep -q '^[^ ]' <<< "$pinned"; then
  echo "A submodule is not at its pinned revision: git submodule update --init --recursive" >&2
  exit 1
fi
mkdir -p "$OUT"
uv run python tools/card_embedding_check.py texts "$OUT/texts.json"
fields=(text)
check=(--embeddings "voyage-4-nano=$OUT/voyage-4-nano-text.npy")
if [ "$SCRIPTS" = 1 ]; then
  fields+=(lua)
  check+=(--scripts --from-scripts "voyage-4-nano on scripts=$OUT/voyage-4-nano-lua.npy")
fi
for field in "${fields[@]}"; do
  uv run --with transformers==5.20.1 --with huggingface-hub==2.2.0 \
    python tools/card_text_embeddings.py "$OUT/texts.json" --field "$field" --max-length 2048 \
    --output "$OUT/voyage-4-nano-$field.npy"
done
uv run python tools/card_embedding_check.py check "$OUT/texts.json" "${check[@]}" \
  --output "$OUT/report.txt"
