"""Whether an EDOPro plays the duels the trainer plays.

    uv run python tools/edopro_files.py ~/EDOPro

A room plays EDOPro's own engine, card scripts and card database, and a
model seated there is shown what they make of a duel.  This reads an
installation's as EDOPro reads them: the scripts and the card rows of its
repositories (`config/configs.json`) before the ones it came with, and the
engine a repository brings before its own.  For the cards of the pool's
deck lists it sets them beside the trainer's (`native/ygo-policies/vendor`):
each card's row as the engine is given it, its script, and the helper
scripts.  Then the policies' bench plays the same pilot games on
EDOPro's three and on the trainer's (`--deck` against every other deck of
the pool, `--games` each) and they are compared game by game: the same game
asks the same questions and takes the same answers to its end.

The exit status is 1 when a card of the pool is missing or told otherwise,
its script is another, or a game is another or fails.  A helper that
differs is said and decides nothing by itself: the games do.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import shutil
import sqlite3
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import NamedTuple

sys.path.insert(0, str(Path(__file__).resolve().parent))

from compare_bench_rows import games as bench_games

ROOT = Path(__file__).resolve().parents[1]
POLICIES = ROOT / "native/ygo-policies"
# What the engine is given of a card (`Core::open` of ygo-policies-duel).
COLUMNS = "id,alias,setcode,type,atk,def,level,race,attribute"
ENGINE = "libocgcore.so"
# The deck lists a card is said to be in, before the rest are counted.
LISTS = 3
ENDED = ("winner", "reason", "turns", "decisions", "digest")


class Installation(NamedTuple):
    """What an EDOPro reads, in the order it reads it."""

    root: Path
    # Folders of scripts: a name is taken from the first that has it.
    scripts: list[Path]
    # Card databases: a card is as the last that holds it tells it.
    databases: list[Path]
    engine: Path
    # Repositories it is set up with and never fetched.
    unfetched: list[str]


def installation(root: Path) -> Installation:
    """An EDOPro's folder as EDOPro reads it.  A repository's scripts go
    before `expansions/script` and `script`, its card rows are loaded after
    those of `expansions`, and the engine of the first that brings one
    replaces the installation's own."""
    if not (root / "config/configs.json").is_file():
        raise SystemExit(f"{root} is no EDOPro folder: it has no config/configs.json")
    repositories = []
    for name in ("configs.json", "user_configs.json"):
        path = root / "config" / name
        if path.is_file():
            repositories += json.loads(path.read_text(encoding="utf8")).get("repos") or []
    scripts, databases, engines, unfetched = [], [], [], []
    for repository in repositories:
        if not repository.get("should_read", True) or "repo_path" not in repository:
            continue
        folder = root / repository["repo_path"]
        told = [key for key in ("script_path", "data_path", "core_path") if key in repository]
        if not told:
            continue
        if not (folder / ".git").exists():
            unfetched.append(repository.get("url") or repository["repo_path"])
            continue
        if "script_path" in repository and (folder / repository["script_path"]).is_dir():
            scripts.append(folder / repository["script_path"])
        if "data_path" in repository:
            databases += sorted((folder / repository["data_path"]).glob("*.cdb"))
        if repository.get("has_core"):
            engine = folder / repository.get("core_path", "") / ENGINE
            if engine.is_file():
                engines.append(engine)
    scripts += [path for path in (root / "expansions/script", root / "script") if path.is_dir()]
    databases = sorted((root / "expansions").glob("*.cdb")) + databases
    if not databases:
        raise SystemExit(f"{root} has no card database in expansions or in a repository")
    return Installation(
        root=root,
        scripts=scripts,
        databases=databases,
        engine=engines[0] if engines else root / ENGINE,
        unfetched=unfetched,
    )


def named(scripts: list[Path]) -> dict[str, Path]:
    """Every script by its name: from the first folder that has it, in the
    folder itself or one of those inside it."""
    found: dict[str, Path] = {}
    for folder in scripts:
        if not folder.is_dir():
            continue
        inside = [folder, *sorted(path for path in folder.iterdir() if path.is_dir())]
        here: dict[str, Path] = {}
        for place in inside:
            for path in sorted(place.glob("*.lua")):
                here.setdefault(path.name, path)
        for name, path in here.items():
            found.setdefault(name, path)
    return found


def helpers(scripts: list[Path]) -> set[str]:
    """The scripts that are no card's: those that lie in a folder itself."""
    return {path.name for folder in scripts for path in folder.glob("*.lua")}


def rows(databases: list[Path]) -> dict[int, tuple]:
    """Every card as the engine is given it: the last database holding it."""
    found: dict[int, tuple] = {}
    for path in databases:
        database = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        try:
            found.update({row[0]: row for row in database.execute(f"select {COLUMNS} from datas")})
        finally:
            database.close()
    return found


def pool(decks: Path) -> dict[int, list[str]]:
    """The cards of the pool's deck lists, and the lists each is in."""
    cards: dict[int, list[str]] = {}
    for deck in sorted(decks.glob("*.ydk")):
        for line in deck.read_text(encoding="utf8").splitlines():
            if line.strip().isdigit() and deck.stem not in cards.setdefault(int(line), []):
                cards[int(line)].append(deck.stem)
    return cards


def interface(engine: Path) -> str:
    """The version of its interface an engine tells."""
    major, minor = ctypes.c_int(), ctypes.c_int()
    ctypes.CDLL(str(engine)).OCG_GetVersion(ctypes.byref(major), ctypes.byref(minor))
    return f"{major.value}.{minor.value}"


def files(found: Installation, vendor: Path, decks: Path) -> tuple[list[str], int]:
    """The pool's cards in EDOPro's files beside the trainer's, in words, and
    how many of them EDOPro lacks or has otherwise."""
    cards = pool(decks)
    ours = rows([vendor / "BabelCdb/cards.cdb"])
    theirs = rows(found.databases)
    names = sqlite3.connect(f"file:{vendor / 'BabelCdb/cards.cdb'}?mode=ro", uri=True)
    name = dict(names.execute("select id, name from texts"))
    names.close()

    def card(code: int) -> str:
        lists = cards.get(code, [])
        said = ", ".join(lists[:LISTS]) + (
            f" and {len(lists) - LISTS} more" if lists[LISTS:] else ""
        )
        return (
            f"{code} ({name.get(code, 'no name')}; {said or 'the card another is an artwork of'})"
        )

    lines = [f"{len(cards)} cards in {len(list(decks.glob('*.ydk')))} deck lists"]
    missing = sorted(code for code in cards if code not in theirs)
    other = sorted(code for code in cards if code in theirs and theirs[code] != ours[code])
    lines.append(
        f"  rows     {len(cards) - len(missing) - len(other)} as the trainer's, "
        f"{len(other)} told otherwise, {len(missing)} missing"
    )
    lines += [f"    missing: {card(code)}" for code in missing]
    for code in other:
        apart = [
            f"{column} {mine} here and {its} there"
            for column, mine, its in zip(COLUMNS.split(","), ours[code], theirs[code], strict=True)
            if mine != its
        ]
        lines.append(f"    told otherwise: {card(code)}: {'; '.join(apart)}")

    # A card's script, and its original's when it is another artwork.
    wanted = set(cards) | {ours[code][1] for code in cards if ours[code][1]}
    mine = named([vendor / "CardScripts/official", vendor / "CardScripts/pre-errata"])
    held = named(found.scripts)
    scripted = sorted(code for code in wanted if f"c{code}.lua" in mine)
    absent = [code for code in scripted if f"c{code}.lua" not in held]
    changed = [
        code
        for code in scripted
        if f"c{code}.lua" in held
        and held[f"c{code}.lua"].read_bytes() != mine[f"c{code}.lua"].read_bytes()
    ]
    lines.append(
        f"  scripts  {len(scripted) - len(absent) - len(changed)} as the trainer's, "
        f"{len(changed)} another, {len(absent)} missing"
    )
    lines += [f"    missing: {card(code)}" for code in absent]
    lines += [
        f"    another: {card(code)}: {held[f'c{code}.lua'].relative_to(found.root)}"
        for code in changed
    ]
    extra = sorted(code for code in wanted if f"c{code}.lua" not in mine and f"c{code}.lua" in held)
    lines += [f"    a script the trainer has none for: {card(code)}" for code in extra]

    common = sorted(path.name for path in (vendor / "CardScripts").glob("*.lua"))
    apart = [
        helper
        for helper in common
        if helper not in held
        or held[helper].read_bytes() != (vendor / "CardScripts" / helper).read_bytes()
    ]
    more = sorted(helpers(found.scripts) - set(common))
    lines.append(
        f"  helpers  {len(common) - len(apart)} of {len(common)} as the trainer's"
        + (f"; another: {', '.join(apart)}" if apart else "")
        + (f"; EDOPro's alone: {', '.join(more)}" if more else "")
    )
    return lines, len(missing) + len(other) + len(absent) + len(changed) + len(extra)


def assembled(found: Installation, vendor: Path, target: Path) -> tuple[Path, Path]:
    """EDOPro's card rows as one database and its scripts as one folder of
    the trainer's layout, for the bench to open: the names the trainer's
    folder holds and EDOPro's helpers, each as EDOPro would read it."""
    target.mkdir(parents=True, exist_ok=True)
    cards = target / "cards.cdb"
    first, *others = found.databases
    shutil.copyfile(first, cards)
    database = sqlite3.connect(f"file:{cards}", uri=True)
    try:
        for other in others:
            database.execute("attach database ? as other", (f"file:{other}?mode=ro",))
            for table in ("datas", "texts"):
                columns = [
                    [row[1] for row in database.execute(f"pragma {where}.table_info({table})")]
                    for where in ("main", "other")
                ]
                if columns[0] != columns[1]:
                    raise SystemExit(f"{other} tells {table} with other columns than {first}")
                database.execute(f"insert or replace into main.{table} select * from other.{table}")
            database.commit()
            database.execute("detach database other")
    finally:
        database.close()

    scripts = target / "scripts"
    held = named(found.scripts)
    (scripts / "official").mkdir(parents=True)
    trainer = vendor / "CardScripts"
    root = {path.name for path in trainer.glob("*.lua")} | helpers(found.scripts)
    inside = {
        path.name
        for folder in ("official", "pre-errata")
        for path in (trainer / folder).glob("*.lua")
    }
    for name in root:
        if name in held:
            (scripts / name).symlink_to(held[name])
    for name in inside - root:
        if name in held:
            (scripts / "official" / name).symlink_to(held[name])
    return cards, scripts


def played(output: Path, deck: str, games: int, seed: int, workers: int, *others: str) -> dict:
    """The pilots' games of a deck against every other deck of the pool, as
    the bench plays them: each by its deal and seat."""
    bench = POLICIES / "target/release/policy-bench"
    if not (bench.is_file() and (bench.parent / "libygo_policies.so").is_file()):
        raise SystemExit(
            "policy-bench is not built: cargo build --release -p ygo-policies-bench "
            "-p ygo-policies-ffi in native/ygo-policies"
        )
    # The bench refuses to rank a run with a failed game and still tells each.
    subprocess.run(
        [
            str(bench),
            "matchup",
            *("--policies", deck, "--opponents", "all", "--games", str(games)),
            *("--seed", str(seed), "--workers", str(workers), "--output", str(output)),
            *others,
        ],
        cwd=POLICIES,
        check=False,
        capture_output=True,
    )
    if not output.is_file():
        raise SystemExit(f"the bench wrote no games to {output}")
    return bench_games(str(output))


def beside(ours: dict, theirs: dict, deck: str) -> tuple[list[str], int]:
    """EDOPro's games beside the trainer's, in words, and how many are not
    the trainer's.  `deck` is the one every game is played by."""
    broken = sorted(key for key, game in ours.items() if "digest" not in game)
    if broken:
        raise SystemExit(f"the trainer's own game {broken[0]} failed: {ours[broken[0]]}")
    if ours.keys() != theirs.keys():
        raise SystemExit("the two runs are not of the same games")
    failed = sorted(key for key, game in theirs.items() if "digest" not in game)
    another = sorted(
        key
        for key, game in theirs.items()
        if key not in failed and any(game.get(field) != ours[key].get(field) for field in ENDED)
    )
    same = len(ours) - len(failed) - len(another)
    lines = [
        (
            f"  {same} of {len(ours)} the same question for question, "
            f"{len(another)} another, {len(failed)} failed"
        )
    ]
    for label, keys in (("another", another), ("failed", failed)):
        decks = sorted({name for key in keys for name in key[:2]} - {deck})
        if keys:
            lines.append(f"    {label} against: {', '.join(decks)}")
    if failed:
        lines.append(f"    the first failure: {theirs[failed[0]].get('error', theirs[failed[0]])}")
    return lines, len(another) + len(failed)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("edopro", type=Path, help="EDOPro's folder, the one holding its config")
    parser.add_argument("--deck", default="blackwing", help="the deck every other is played by")
    parser.add_argument("--games", type=int, default=8, help="games against each other deck")
    parser.add_argument("--seed", type=int, default=730000)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument(
        "--keep", type=Path, help="leave EDOPro's files as read, and the games, here"
    )
    args = parser.parse_args()

    vendor = POLICIES / "vendor"
    found = installation(args.edopro.expanduser().resolve())

    def place(path: Path) -> str:
        return str(path.relative_to(found.root)) if path.is_relative_to(found.root) else str(path)

    print(f"EDOPro at {found.root}")
    for url in found.unfetched:
        print(f"  never fetched: {url}")
    print(f"  engine   {place(found.engine)} (interface {interface(found.engine)})")
    revisions = [
        f"{place(folder)}"
        + (
            f" ({(folder / 'VERSION').read_text().strip()[:9]})"
            if (folder / "VERSION").is_file()
            else ""
        )
        for folder in found.scripts
    ]
    print(f"  scripts  {', then '.join(revisions)}")
    folders: dict[str, int] = {}
    for path in found.databases:
        folders[place(path.parent)] = folders.get(place(path.parent), 0) + 1
    print(
        f"  cards    {', then '.join(f'{count} databases of {name}' for name, count in folders.items())}"
    )

    lines, differences = files(found, vendor, POLICIES / "decks")
    print("\n".join(lines))

    with tempfile.TemporaryDirectory() as temporary:
        folder = args.keep or Path(temporary)
        if args.keep is not None and (folder / "scripts").exists():
            shutil.rmtree(folder / "scripts")
        cards, scripts = assembled(found, vendor, folder)
        ours = played(folder / "trainer.jsonl", args.deck, args.games, args.seed, args.workers)
        theirs = played(
            folder / "edopro.jsonl",
            args.deck,
            args.games,
            args.seed,
            args.workers,
            *("--core", str(found.engine), "--cards", str(cards), "--scripts", str(scripts)),
        )
    decks = len({name for key in ours for name in key[:2]}) - 1
    print(f"{len(ours)} pilot games: {args.deck} against each of {decks} decks, {args.games} each")
    lines, apart = beside(ours, theirs, args.deck)
    print("\n".join(lines))
    differences += apart
    print(
        "it plays the trainer's duels"
        if not differences
        else "it does not play the trainer's duels: a model there is shown what it was not trained on"
    )
    return 1 if differences else 0


if __name__ == "__main__":
    raise SystemExit(main())
