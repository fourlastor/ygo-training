"""The cards' texts turned into vectors by voyage-4-nano, for
``tools/card_embedding_check.py``: one row a text, in the order of its
``texts`` file, as the model's card on Hugging Face says to make them (the
document prompt before the text, the mean over every token, the length made
one).

    uv run --with transformers==5.20.1 --with huggingface-hub==2.2.0 \\
        python tools/card_text_embeddings.py analysis/card-embeddings/texts.json \\
        --output analysis/card-embeddings/voyage-4-nano.npy

``uv run --with`` adds transformers and Hugging Face's client beside the
trainer's packages for the one command, and leaves its environment as it is.
The model is ``voyageai/voyage-4-nano`` at the commit named below, fetched
into Hugging Face's own cache when it is not there (0.70 GB; it asks for no
sign-in).  Its ``modeling_qwen3_bidirectional.py`` is the model, Qwen3 with
every token seeing every other and a last linear layer to 2,048 numbers: it
was read before it was first run, and runs only where it is still that file
(its SHA-256 below).  ``--model`` names a copy already at hand instead, and
``--packages`` a folder of the packages (``uv pip install --target``) to take
after this interpreter's own.  ``--field lua`` turns the cards' scripts into
numbers in the same way where it is their texts otherwise.  On the processor:
about two minutes for the pool's texts.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

PROMPT = "Represent the document for retrieval: "
MODEL = "voyageai/voyage-4-nano"
REVISION = "433ece5e27a1611e9360dd7134ecaaf79fb3681d"
# modeling_qwen3_bidirectional.py as it was read, 2026-10-17.
CODE = "425304da2571d17bf04add3b70fab46d4b7d4466abbe118ef553ccb6fe8fadfe"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("texts", type=Path)
    parser.add_argument("--model", type=Path, help=f"a copy of {MODEL}; fetched if left out")
    parser.add_argument("--packages", type=Path, help="a folder of packages to take after these")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--field", default="text", help="what of each card to turn into numbers: text or lua"
    )
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=512)
    args = parser.parse_args()
    if args.packages:
        sys.path.append(str(args.packages))
    import numpy as np
    import torch
    from transformers import AutoTokenizer

    if args.model is None:
        from huggingface_hub import snapshot_download

        args.model = Path(snapshot_download(MODEL, revision=REVISION))
    code = args.model / "modeling_qwen3_bidirectional.py"
    if hashlib.sha256(code.read_bytes()).hexdigest() != CODE:
        raise SystemExit(f"{code} is not the model's code that was read: not run")
    spec = importlib.util.spec_from_file_location("modeling_qwen3_bidirectional", code)
    assert spec is not None and spec.loader is not None
    modeling = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modeling)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = modeling.Qwen3BidirectionalModel.from_pretrained(
        args.model, dtype=torch.float32, local_files_only=True
    )
    model.eval()
    texts = [card[args.field] for card in json.loads(args.texts.read_text())]
    # Texts of a length together, so that a batch is little padding.
    order = sorted(range(len(texts)), key=lambda index: len(texts[index]))
    made = np.zeros((len(texts), model.config.num_labels), dtype=np.float32)
    started = time.perf_counter()
    with torch.inference_mode():
        for first in range(0, len(order), args.batch):
            places = order[first : first + args.batch]
            tokens = tokenizer(
                [PROMPT + texts[place] for place in places],
                padding=True,
                truncation=True,
                max_length=args.max_length,
                return_tensors="pt",
            )
            hidden = model(**tokens).last_hidden_state
            mask = tokens["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
            pooled = torch.nn.functional.normalize(pooled, dim=-1)
            made[places] = pooled.numpy()
            if first // args.batch % 50 == 0:
                print(
                    f"{first + len(places)} of {len(texts)}, {time.perf_counter() - started:.0f} s",
                    flush=True,
                )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, made)
    print(f"{len(texts)} texts in {time.perf_counter() - started:.0f} s: {args.output}")


if __name__ == "__main__":
    main()
