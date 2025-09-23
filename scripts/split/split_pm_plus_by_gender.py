#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
Stratify speakers by gender from label.xlsx and do an 8:2 train/test split.

This script supports two input modes for speaker gender:
1) Excel mode (recommended): read an Excel file (label.xlsx) that contains
   utterance-level rows with at least columns [speaker, gender]. The script
   consolidates to speaker-level gender, performs a gender-stratified 8:2
   split by speaker, and writes a new column 'split_set' back to Excel
   with values 'Train' or 'Test' for each row (by speaker grouping).
2) CSV mode (legacy): read a CSV 'speaker,gender' mapping and split speakers.

In both modes, the script also optionally materializes the split into
directories with symlinks/copies and writes manifests.

Example:
    >>> python scripts/split/split_pm_plus_by_gender.py \\
    ...   --src_root data/raw/PM+ \\
    ...   --out_root data/processed \\
    ...   --label_xlsx data/annotations/label.xlsx \\
    ...   --sheet Sheet1 \\
    ...   --speaker_col speaker \\
    ...   --gender_col gender \\
    ...   --train_ratio 0.8 \\
    ...   --op link \\
    ...     \\
    ...   --seed 42

"""

__author__ = "Liu Yang"
__copyright__ = "Copyright 2025, AIMSL"
__license__ = "MIT"
__maintainer__ = "Liu Yang"
__email__ = "yang.liu6@siat.ac.cn"
__last_updated__ = "2025-09-20"

import argparse
import csv
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Set, Tuple

AUDIO_EXTS = {".wav", ".flac", ".mp3", ".m4a", ".aac", ".ogg"}


@dataclass(frozen=True)
class FileEntry:
    """
    Small struct to hold file metadata.

    Attributes:
        path: Absolute path to the file.
        speaker: Speaker ID (e.g., 'spk119').
        gender: Gender label ('F'|'M'|'Other').

    """

    path: Path
    speaker: str
    gender: str


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Stratified 8:2 split by speaker gender (PM+)."
    )
    parser.add_argument("--src_root", type=str, required=True)
    parser.add_argument("--out_root", type=str, required=True)
    # Input source: either --label_xlsx (Excel) or --gender_csv (legacy)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--label_xlsx",
        type=str,
        help="Excel file with at least [speaker, gender] columns.",
    )
    group.add_argument(
        "--gender_csv",
        type=str,
        help="Legacy CSV with headers 'speaker,gender'.",
    )
    # Excel-related arguments
    parser.add_argument(
        "--sheet",
        type=str,
        default="",
        help="Sheet name or index (0-based). Default: first sheet.",
    )
    parser.add_argument(
        "--speaker_col",
        type=str,
        default="speaker",
        help="Column name containing speaker IDs in Excel.",
    )
    parser.add_argument(
        "--gender_col",
        type=str,
        default="gender",
        help="Column name containing gender labels in Excel.",
    )
    parser.add_argument(
        "--out_xlsx",
        type=str,
        default="",
        help="Output Excel path. If empty and --label_xlsx is given, a file "
        "with suffix '.with_split.xlsx' is written unless --inplace is set.",
    )
    parser.add_argument(
        "--inplace",
        action="store_true",
        help="Overwrite the input Excel by adding/overwriting 'split_set'.",
    )
    # Split/materialization arguments
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--op",
        type=str,
        default="link",
        choices=["link", "copy", "manifest"],
        help="link: create symlinks; copy: copy files; manifest: only CSV.",
    )
    parser.add_argument(
        "--exts",
        type=str,
        default="wav",
        help="Comma-separated audio extensions, e.g., 'wav,flac'.",
    )
    parser.add_argument(
        "--split_names",
        type=str,
        default="train,test",
        help="Comma-separated split names corresponding to "
        "'train_ratio, 1-train_ratio'.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Only print the plan and write nothing.",
    )
    return parser.parse_args()


def ensure_excel_deps() -> None:
    """Ensure pandas and openpyxl are available."""
    try:
        import pandas as _  # noqa: F401
    except Exception as exc:
        raise RuntimeError(
            "Excel mode requires 'pandas' and 'openpyxl'. Install via:\n"
            "  pip install pandas openpyxl"
        ) from exc


def parse_sheet_arg(sheet_arg: str) -> Optional[object]:
    """
    Parse a sheet specifier into a pandas-compatible value.

    Args:
        sheet_arg: Sheet name or numeric index as string.

    Returns:
        Sheet name, integer index, or None for default first sheet.

    """
    if not sheet_arg:
        return None
    s = sheet_arg.strip()
    if s.isdigit():
        return int(s)
    return s


def normalize_gender(x: str) -> str:
    """Normalize gender string to one of {'F','M','Other'}."""
    s = x.strip().lower()
    if s in {"f", "female", "女", "女性"}:
        return "F"
    if s in {"m", "male", "男", "男性"}:
        return "M"
    return "Other"


def parse_speaker(folder_name: str) -> str:
    """Parse speaker id from a folder name like 'spk119-1-1' -> 'spk119'."""
    if "-" in folder_name:
        return folder_name.split("-")[0]
    return folder_name


def read_gender_csv(csv_path: Path) -> Dict[str, str]:
    """
    Read 'speaker,gender' CSV into a mapping.

    Args:
        csv_path: Path to CSV file.

    Returns:
        Mapping from speaker id to normalized gender.

    Raises:
        FileNotFoundError: If the CSV does not exist.
        ValueError: If CSV rows are malformed.

    """
    if not csv_path.exists():
        raise FileNotFoundError(f"Not found: {csv_path}")

    mapping: Dict[str, str] = {}
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if (
            not reader.fieldnames
            or "speaker" not in reader.fieldnames
            or "gender" not in reader.fieldnames
        ):
            raise ValueError("CSV must have headers: speaker,gender")

        for i, row in enumerate(reader, 1):
            spk = str(row["speaker"]).strip()
            gen = normalize_gender(str(row["gender"]))
            if not spk:
                raise ValueError(f"Row {i} has empty speaker")
            mapping[spk] = gen
    return mapping


def collect_files(src_root: Path, exts: Set[str]) -> Dict[str, List[Path]]:
    """
    Collect audio files for each speaker under src_root.

    Args:
        src_root: Root directory that contains speaker folders.
        exts: Allowed audio extensions (lowercase, with dot).

    Returns:
        Mapping from speaker id to a list of audio file paths.

    Raises:
        FileNotFoundError: If src_root does not exist or has no audio.

    """
    if not src_root.exists():
        raise FileNotFoundError(f"Not found: {src_root}")

    spk_to_files: Dict[str, List[Path]] = {}
    for spk_dir in sorted(p for p in src_root.iterdir() if p.is_dir()):
        spk_id = parse_speaker(spk_dir.name)
        files = [
            p for p in spk_dir.rglob("*") if p.is_file() and p.suffix.lower() in exts
        ]
        if files:
            spk_to_files.setdefault(spk_id, []).extend(sorted(files))

    if not spk_to_files:
        raise FileNotFoundError(f"No audio found under: {src_root}")
    return spk_to_files


def check_gender_coverage(
    spk_to_files: Mapping[str, List[Path]], spk_to_gender: Mapping[str, str]
) -> None:
    """
    Validate all speakers have a gender label.

    Args:
        spk_to_files: Mapping speaker -> files.
        spk_to_gender: Mapping speaker -> gender.

    Raises:
        ValueError: If any speaker is missing a gender label.

    """
    missing = [spk for spk in spk_to_files.keys() if spk not in spk_to_gender]
    if missing:
        sample = ", ".join(missing[:10])
        raise ValueError(
            f"Missing gender for {len(missing)} speakers. Examples: {sample}"
        )


def stratified_split_by_speaker(
    spk_to_gender: Mapping[str, str], train_ratio: float, seed: int
) -> Tuple[Set[str], Set[str]]:
    """
    Split speakers into train/test with gender stratification.

    The split is performed per-gender bucket. For each gender bucket with
    n speakers and ratio r, we pick:
    n_test = max(1_{[n>1]}, round(n * (1 - r))) and put the rest into train.

    Args:
        spk_to_gender: Mapping from speaker to gender.
        train_ratio: Fraction for train split (0 < r < 1).
        seed: Random seed.

    Returns:
        A tuple (train_speakers, test_speakers).

    Raises:
        ValueError: If train_ratio is not in (0,1).

    """
    if not (0.0 < train_ratio < 1.0):
        raise ValueError("train_ratio must be in (0, 1)")

    rng = random.Random(seed)
    by_gender: MutableMapping[str, List[str]] = {"F": [], "M": [], "Other": []}
    for spk, g in spk_to_gender.items():
        by_gender.setdefault(g, []).append(spk)

    train: Set[str] = set()
    test: Set[str] = set()

    for _, speakers in by_gender.items():
        if not speakers:
            continue
        rng.shuffle(speakers)
        n = len(speakers)
        n_test = round(n * (1.0 - train_ratio))
        if n > 1:
            n_test = max(n_test, 1)
        else:
            n_test = 0
        n_test = min(n_test, n)
        test_speakers = set(speakers[:n_test])
        train_speakers = set(speakers[n_test:])
        test |= test_speakers
        train |= train_speakers

    return train, test


def materialize_split(
    out_root: Path,
    split_name: str,
    entries: Iterable[FileEntry],
    op: str,
    dry_run: bool,
) -> None:
    """
    Create split directories (symlink/copy) or only manifests.

    Args:
        out_root: Base output directory.
        split_name: Split name ('train' or 'test').
        entries: Iterable of FileEntry to place into the split.
        op: One of {'link','copy','manifest'}.
        dry_run: If True, do not write anything.

    Returns:
        None

    """
    split_dir = out_root / split_name
    man_dir = out_root / "splits"
    man_dir.mkdir(parents=True, exist_ok=True)
    manifest = man_dir / f"manifest_{split_name}.csv"

    if op in {"link", "copy"} and not dry_run:
        split_dir.mkdir(parents=True, exist_ok=True)

    with manifest.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["filepath", "speaker", "gender", "split"])

        for e in entries:
            rel_spk = e.speaker
            dst_dir = split_dir / rel_spk
            dst_dir_rel = Path(rel_spk)
            dst_path = dst_dir / e.path.name

            if op == "link" and not dry_run:
                dst_dir.mkdir(parents=True, exist_ok=True)
                if dst_path.exists():
                    dst_path.unlink()
                dst_path.symlink_to(e.path.resolve())
            elif op == "copy" and not dry_run:
                dst_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(e.path, dst_path)

            writer.writerow(
                [str(dst_dir_rel / e.path.name), e.speaker, e.gender, split_name]
            )


def build_entries(
    spk_to_files: Mapping[str, List[Path]], spk_to_gender: Mapping[str, str]
) -> List[FileEntry]:
    """
    Build per-file entries with speaker and gender.

    Args:
        spk_to_files: Mapping speaker -> list of file paths.
        spk_to_gender: Mapping speaker -> gender.

    Returns:
        List of FileEntry objects.

    """
    out: List[FileEntry] = []
    for spk, paths in spk_to_files.items():
        if spk not in spk_to_gender:
            # Skip speakers not present in Excel/CSV mapping.
            continue
        gender = spk_to_gender[spk]
        for p in paths:
            out.append(FileEntry(path=p, speaker=spk, gender=gender))
    return out


def read_gender_from_excel(
    xlsx_path: Path,
    sheet: Optional[object],
    speaker_col: str,
    gender_col: str,
) -> Tuple["pd.DataFrame", Dict[str, str]]:
    """
    Read Excel and consolidate to a speaker->gender mapping.

    Args:
        xlsx_path: Path to Excel file.
        sheet: Sheet name or index; None for first sheet.
        speaker_col: Column name with speaker IDs.
        gender_col: Column name with gender labels.

    Returns:
        A tuple of (dataframe, speaker_to_gender).

    Raises:
        FileNotFoundError: If Excel not found.
        ValueError: If required columns missing.

    """
    ensure_excel_deps()
    import pandas as pd

    if not xlsx_path.exists():
        raise FileNotFoundError(f"Not found: {xlsx_path}")

    df = pd.read_excel(xlsx_path, sheet_name=sheet, engine="openpyxl")
    cols = {c.strip(): c for c in df.columns.astype(str)}
    if speaker_col not in cols or gender_col not in cols:
        raise ValueError(
            "Missing required columns. Found columns: "
            f"{', '.join(df.columns.astype(str))}"
        )

    s_col = cols[speaker_col]
    g_col = cols[gender_col]

    # Normalize gender and speaker columns.
    df["_speaker_norm"] = df[s_col].astype(str).str.strip()
    df["_gender_norm"] = df[g_col].astype(str).str.strip().map(normalize_gender)

    # Consolidate by majority vote per speaker.
    from collections import Counter, defaultdict

    votes: Dict[str, Counter] = defaultdict(Counter)
    order: Dict[str, int] = {}
    for i, (spk, gen) in enumerate(zip(df["_speaker_norm"], df["_gender_norm"])):
        if spk not in order:
            order[spk] = i
        votes[spk][gen] += 1

    spk_to_gender: Dict[str, str] = {}
    for spk, counter in votes.items():
        most_common = counter.most_common()
        top_count = most_common[0][1]
        tops = [g for g, c in most_common if c == top_count]
        if len(tops) == 1:
            spk_to_gender[spk] = tops[0]
        else:
            # Tie: pick the first occurrence gender.
            first_idx = order[spk]
            spk_to_gender[spk] = df["_gender_norm"].iloc[first_idx]

    return df, spk_to_gender


def annotate_split_in_dataframe(
    df: "pd.DataFrame",
    speaker_col: str,
    train_speakers: Set[str],
    test_speakers: Set[str],
) -> "pd.DataFrame":
    """
    Add or overwrite a 'split_set' column in the DataFrame.

    Args:
        df: Input dataframe read from Excel.
        speaker_col: Column name containing speaker IDs.
        train_speakers: Set of training speakers.
        test_speakers: Set of testing speakers.

    Returns:
        DataFrame with a new/updated 'split_set' column.

    """
    mapping: Dict[str, str] = {}
    mapping.update({s: "Train" for s in train_speakers})
    mapping.update({s: "Test" for s in test_speakers})

    # Use normalized speaker column we created earlier to map values.
    if "_speaker_norm" in df.columns:
        key_series = df["_speaker_norm"]
    else:
        key_series = df[speaker_col].astype(str).str.strip()

    split_vals = key_series.map(mapping).fillna("Train")
    # Default to "Train" for any unmatched speaker (shouldn't happen normally).
    df_out = df.copy()
    df_out["split_set"] = split_vals
    return df_out


def write_excel(df: "pd.DataFrame", out_path: Path, sheet_name: Optional[str]) -> None:
    """
    Write a DataFrame to an Excel file.

    Args:
        df: DataFrame to write.
        out_path: Output Excel file path.
        sheet_name: Sheet name to write (None -> default 'Sheet1').

    Returns:
        None

    """
    import pandas as pd

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name=sheet_name or "Sheet1", index=False)


def main() -> None:
    """Run the gender-stratified split and materialization."""
    args = parse_args()
    src_root = Path(args.src_root)
    out_root = Path(args.out_root)
    exts = {"." + e.strip().lower().lstrip(".") for e in args.exts.split(",")}
    if not exts:
        exts = AUDIO_EXTS

    # Build speaker->gender mapping from the selected source.
    df_excel = None
    spk_to_gender: Dict[str, str]

    if args.label_xlsx:
        sheet = parse_sheet_arg(args.sheet)
        df_excel, spk_to_gender = read_gender_from_excel(
            xlsx_path=Path(args.label_xlsx),
            sheet=sheet,
            speaker_col=args.speaker_col,
            gender_col=args.gender_col,
        )
    else:
        spk_to_gender = read_gender_csv(Path(args.gender_csv))

    # Gather files and restrict to speakers present in mapping.
    spk_to_files_all = collect_files(src_root, exts)
    # Keep only mapped speakers to avoid coverage errors.
    spk_to_files = {
        spk: files for spk, files in spk_to_files_all.items() if spk in spk_to_gender
    }
    if not spk_to_files:
        raise FileNotFoundError(
            "No overlapping speakers between filesystem and provided labels."
        )
    check_gender_coverage(spk_to_files, spk_to_gender)

    # Split speakers
    train_speakers, test_speakers = stratified_split_by_speaker(
        spk_to_gender=spk_to_gender, train_ratio=args.train_ratio, seed=args.seed
    )

    # Report speaker counts (only those present on disk and in mapping).
    spk_present = set(spk_to_files.keys())
    train_speakers &= spk_present
    test_speakers &= spk_present

    n_spk_total = len(spk_present)
    n_spk_train = len(train_speakers)
    n_spk_test = len(test_speakers)
    print(f"Speakers -> total: {n_spk_total}, train: {n_spk_train}, test: {n_spk_test}")

    # Build file entries
    all_entries = build_entries(spk_to_files, spk_to_gender)
    train_entries = [e for e in all_entries if e.speaker in train_speakers]
    test_entries = [e for e in all_entries if e.speaker in test_speakers]

    n_files_total = len(all_entries)
    n_files_train = len(train_entries)
    n_files_test = len(test_entries)
    print(
        f"Files -> total: {n_files_total}, train: {n_files_train}, test: {n_files_test}"
    )

    if args.dry_run:
        print("[Dry-run] No files or Excel will be written.")
        return

    split_names = [s.strip() for s in args.split_names.split(",")]
    if len(split_names) != 2:
        raise ValueError("--split_names must contain exactly two names")
    train_name, test_name = split_names

    # Materialize train/test directories and manifests.
    materialize_split(
        out_root=out_root,
        split_name=train_name,
        entries=train_entries,
        op=args.op,
        dry_run=args.dry_run,
    )
    materialize_split(
        out_root=out_root,
        split_name=test_name,
        entries=test_entries,
        op=args.op,
        dry_run=args.dry_run,
    )
    print("Done. Manifests are saved under:", out_root / "splits")

    # If Excel input was used, add/overwrite 'split_set' and write back.
    if df_excel is not None:
        # Write 'Train'/'Test' split labels at the utterance level (by speaker).
        df_with_split = annotate_split_in_dataframe(
            df=df_excel,
            speaker_col=args.speaker_col,
            train_speakers=train_speakers,
            test_speakers=test_speakers,
        )

        if args.inplace and args.label_xlsx:
            out_xlsx = Path(args.label_xlsx)
        else:
            if args.out_xlsx:
                out_xlsx = Path(args.out_xlsx)
            else:
                inp = Path(args.label_xlsx)
                out_xlsx = inp.with_name(inp.stem + ".with_split.xlsx")

        write_excel(df_with_split, out_xlsx, sheet_name=args.sheet or None)
        print(f"Wrote Excel with 'split_set' -> {out_xlsx}")


if __name__ == "__main__":
    main()
