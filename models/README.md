# Released networks

One card for each network released, `NAME.md`, which is the release's title
and notes. The release's tag is `model/NAME`, on the commit whose code plays
the network.

To release one:

1. Write its files, from its checkpoint, with
   `uv run python tools/model_release.py NAME CHECKPOINT`: they are written
   to `release/NAME/`.
2. Run the workflow *Model release* (`.github/workflows/model-release.yml`)
   with the name, and the commit if not `main`: it drafts the release from
   the card, its links into the repository pinned to that commit.
3. Attach the files of `release/NAME/` to the draft, and publish it.
