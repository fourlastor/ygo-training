# Released networks

One card for each network released, `NAME.md`, which is the release's title
and notes. The release's tag is `model/NAME`, on the commit whose code plays
the network.

To release one, in one step:

    uv run python tools/model_release.py NAME CHECKPOINT --publish

It writes the release's files to `release/NAME/`, then makes the release
with `gh` (signed in, the commit pushed): the tag on `HEAD`, or the commit
`--ref` names, the card as its title and notes, the files attached, and
publishes it. The card, the commit on GitHub and a tag not taken yet are
checked before anything is written.

Or in three, from the browser:

1. Write the files with `uv run python tools/model_release.py NAME CHECKPOINT`.
2. Run the workflow *Model release* (`.github/workflows/model-release.yml`)
   with the name, and the commit if not `main`: it drafts the release from
   the card.
3. Attach the files of `release/NAME/` to the draft, and publish it.

Either way the notes are the card without its title, its links into the
repository pinned to the tagged commit, a paragraph to a line
(`tools/release_notes.py`).

A release may hold several networks under one card and one tag: each is
named by `--network NAME=CHECKPOINT` in place of the checkpoint, its files
are named by it, and one `SHA256SUMS` covers them all
(`champions.md`, the champions of their mirrors):

    uv run python tools/model_release.py champions \
        --network burn=CHECKPOINT --network gishki=CHECKPOINT ... --publish
