import json
import sys
from collections import defaultdict


def get_sample_id(sample: dict):
    return sample.get("sampleid") or sample.get("sample_id") or "unknown"


def normalise_state(state: list[str]) -> str:
    return "\n".join(line.rstrip() for line in state)


def find_state_duplicates(games: list[dict]) -> dict:
    """
    Index: canonical_state -> list of (game_name, sample_id, step)
    """
    index: dict[str, list[tuple]] = defaultdict(list)

    for game in games:
        game_name = game.get("name", "unknown")
        for sample in game.get("samples", []):
            sid = get_sample_id(sample)
            for entry in sample.get("rollout", []):
                state = entry.get("state")
                if state is None:
                    continue
                key = normalise_state(state)
                index[key].append((game_name, sid, entry.get("step")))

    return {k: v for k, v in index.items() if len(v) > 1}


def print_report(duplicates: dict) -> None:
    if not duplicates:
        print("✅  Žiadne duplicitné stavy nenájdené.")
        return

    print(f"⚠️  Nájdené duplicitné stavy: {len(duplicates)}\n")

    for idx, (canonical_state, occurrences) in enumerate(duplicates.items(), 1):
        print(f"{'─' * 56}")
        print(f"  Duplicita #{idx}  –  vyskytuje sa {len(occurrences)}×")
        print(f"{'─' * 56}")
        print("  Výskyty:")
        for game_name, sample_id, step in occurrences:
            print(f"    • game={game_name}, sampleid={sample_id}, step={step}")
        print("\n  Stav (board):")
        for line in canonical_state.splitlines():
            print(f"    {line}")
        print()

    print("=" * 56)
    print(f"  {'#':<5} {'Počet':>6}   Výskyty (game / sampleid)")
    print(f"  {'─'*5} {'─'*6}   {'─'*30}")
    for idx, (_, occ) in enumerate(duplicates.items(), 1):
        detail = ", ".join(f"{g}/{s}" for g, s, _ in occ)
        print(f"  #{idx:<4} {len(occ):>6}   {detail}")


def main():
    if len(sys.argv) != 2:
        print(f"Použitie: python {sys.argv[0]} <subor.json>")
        sys.exit(1)

    path = sys.argv[1]
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Chyba: súbor '{path}' neexistuje.")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Chyba: neplatný JSON – {e}")
        sys.exit(1)

    games = data.get("games", [])
    total_samples = sum(len(g.get("samples", [])) for g in games)

    print(f"🔍  Načítané hry:    {len(games)}")
    print(f"    Vzorky celkom:  {total_samples}")
    print(f"    Súbor: {path}\n")

    duplicates = find_state_duplicates(games)
    print_report(duplicates)


if __name__ == "__main__":
    main()