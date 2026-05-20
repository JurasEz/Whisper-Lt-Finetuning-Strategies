from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

RECORDING_RE = re.compile(
    r"^(?P<lossy>[LR])_(?P<speech>[RS])(?P<src>[ADPRST])_"
    r"(?P<gender>[FM])(?P<age>[1-5])_(?P<speaker>[A-Z]{2}\d{3})_(?P<session>\d{2})$"
)

AGE_LABELS = {
    "1": "1 amziaus grupe (0--12 m.)",
    "2": "2 amziaus grupe (13--17 m.)",
    "3": "3 amziaus grupe (18--25 m.)",
    "4": "4 amziaus grupe (26--60 m.)",
    "5": "5 amziaus grupe (60+ m.)",
}

GENDER_LABELS = {
    "F": "Moteris",
    "M": "Vyras",
}


def format_age(age: str) -> str:
    return AGE_LABELS.get(age, f"{age} amziaus grupe")


def format_gender_age(key: str) -> str:
    gender = key[0]
    age = key[1:]
    return f"{GENDER_LABELS.get(gender, gender)}, {format_age(age)}"


def recording_id_from_line(line: str) -> str | None:
    row = json.loads(line)
    return row.get("recording_id")


def count_from_jsonl(paths: list[Path]) -> tuple[Counter, Counter, set[str]]:
    age_segments = Counter()
    gender_age_segments = Counter()
    recordings = set()

    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"Nerastas failas: {path}")

        with path.open("r", encoding="utf-8") as f:
            for line in f:
                rid = recording_id_from_line(line)
                if not rid:
                    continue

                m = RECORDING_RE.match(rid)
                if not m:
                    continue

                age = m.group("age")
                gender = m.group("gender")

                age_segments[age] += 1
                gender_age_segments[f"{gender}{age}"] += 1
                recordings.add(rid)

    return age_segments, gender_age_segments, recordings


def count_recordings(recording_ids: set[str]) -> tuple[Counter, Counter]:
    age_recordings = Counter()
    gender_age_recordings = Counter()

    for rid in recording_ids:
        m = RECORDING_RE.match(rid)
        if not m:
            continue

        age = m.group("age")
        gender = m.group("gender")

        age_recordings[age] += 1
        gender_age_recordings[f"{gender}{age}"] += 1

    return age_recordings, gender_age_recordings


def print_counter(title: str, counter: Counter, formatter) -> None:
    print(title)
    if not counter:
        print("Nerasta duomenu.")
        return

    total = sum(counter.values())
    for key in sorted(counter):
        value = counter[key]
        percent = value / total * 100 if total else 0.0
        print(f"{formatter(key)}: {value} ({percent:.2f} %)")

    print(f"Is viso: {total}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prepared-dir",
        default="prepared_data/ls_rs_spontaneous_norm_80_10_10",
        help="Aplankas su train.jsonl, valid.jsonl ir test.jsonl",
    )
    args = parser.parse_args()

    prepared_dir = Path(args.prepared_dir)
    paths = [prepared_dir / name for name in ["train.jsonl", "valid.jsonl", "test.jsonl"]]

    age_segments, gender_age_segments, recordings = count_from_jsonl(paths)
    age_recordings, gender_age_recordings = count_recordings(recordings)

    print_counter(
        "Segmentai pagal amziaus grupe:",
        age_segments,
        format_age,
    )

    print()
    print_counter(
        "Irasai pagal amziaus grupe:",
        age_recordings,
        format_age,
    )

    print()
    print_counter(
        "Segmentai pagal lyti ir amziu:",
        gender_age_segments,
        format_gender_age,
    )

    print()
    print_counter(
        "Irasai pagal lyti ir amziu:",
        gender_age_recordings,
        format_gender_age,
    )


if __name__ == "__main__":
    main()
