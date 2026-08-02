import argparse
import hashlib
import json
import os
import re
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import h5py
import nibabel as nib
import numpy as np
import pandas as pd


def parse_csv_list(raw: str) -> List[str]:
    if not raw:
        return []
    return [item.strip() for item in raw.split(",") if item.strip()]


def extract_id(filename: str) -> Optional[str]:
    match = re.match(r"(\d+)_", filename)
    return match.group(1) if match else None


def build_file_map(folder: str) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for root, _, files in os.walk(folder):
        for name in files:
            sid = extract_id(name)
            if sid is None:
                continue
            path = os.path.join(root, name)
            if sid not in mapping:
                mapping[sid] = path
    return mapping


def deduplicate_by_patient(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["PATNO"] = out["PATNO"].astype(str).str.extract(r"(\d+)", expand=False)
    out = out.dropna(subset=["PATNO"]) 
    out["nan_count"] = out.isna().sum(axis=1)
    out = out.sort_values(["PATNO", "nan_count"]).drop_duplicates(subset="PATNO", keep="first")
    out = out.drop(columns=["nan_count"])
    return out.reset_index(drop=True)


def infer_id_column(df: pd.DataFrame, explicit_id_col: Optional[str]) -> str:
    if explicit_id_col:
        if explicit_id_col not in df.columns:
            raise ValueError(f"Requested id column '{explicit_id_col}' was not found in CSV")
        return explicit_id_col

    candidates = ["PATNO", "Subject", "UPDRS_PATNO", "RID"]
    for col in candidates:
        if col in df.columns:
            return col

    raise ValueError(
        "Could not infer an id column. Provide --id-column explicitly (e.g., PATNO or Subject)."
    )


def split_ids(ids: List[str], train_ratio: float, valid_ratio: float, seed: int) -> Tuple[List[str], List[str], List[str]]:
    train_ids: List[str] = []
    valid_ids: List[str] = []
    test_ids: List[str] = []

    for sid in sorted(ids):
        digest = hashlib.sha1(f"{seed}:{sid}".encode("utf-8")).hexdigest()
        bucket = int(digest[:8], 16) / 0xFFFFFFFF

        if bucket < train_ratio:
            train_ids.append(sid)
        elif bucket < train_ratio + valid_ratio:
            valid_ids.append(sid)
        else:
            test_ids.append(sid)

    if len(ids) >= 3:
        splits = {"train": train_ids, "valid": valid_ids, "test": test_ids}
        while not splits["valid"] or not splits["test"]:
            donor_name = max(splits, key=lambda key: len(splits[key]))
            donor = splits[donor_name]
            if len(donor) <= 1:
                break
            moved_id = donor.pop()
            if not splits["valid"]:
                splits["valid"].append(moved_id)
            elif not splits["test"]:
                splits["test"].append(moved_id)

        train_ids, valid_ids, test_ids = splits["train"], splits["valid"], splits["test"]

    return train_ids, valid_ids, test_ids


def stratified_split(
    ids: List[str],
    labels_by_id: Dict[str, str],
    train_ratio: float,
    valid_ratio: float,
    seed: int,
) -> Tuple[List[str], List[str], List[str]]:
    by_label: Dict[str, List[str]] = defaultdict(list)
    for sid in ids:
        by_label[labels_by_id[sid]].append(sid)

    train_all: List[str] = []
    valid_all: List[str] = []
    test_all: List[str] = []

    for label, group_ids in sorted(by_label.items()):
        tr, va, te = split_ids(group_ids, train_ratio, valid_ratio, seed)
        train_all.extend(tr)
        valid_all.extend(va)
        test_all.extend(te)
        print(f"[{label}] split -> train: {len(tr)}, valid: {len(va)}, test: {len(te)}")

    return sorted(train_all), sorted(valid_all), sorted(test_all)


def coerce_numeric(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def remove_modality_duplicates(
    df: pd.DataFrame,
    keep_prefix: str = "MRI_",
    other_prefix: str = "SPECT_",
) -> pd.DataFrame:
    """Drop duplicated modality columns (e.g., keep MRI_Age, drop SPECT_Age).

    This is deterministic: if a column exists in both modalities with the same
    suffix, the secondary modality column is removed regardless of formatting
    differences in the CSV representation.
    """
    out = df.copy()
    to_drop: List[str] = []

    for col in out.columns:
        if not col.startswith(keep_prefix):
            continue
        suffix = col[len(keep_prefix) :]
        other_col = f"{other_prefix}{suffix}"
        if other_col not in out.columns:
            continue

        to_drop.append(other_col)

    if to_drop:
        out = out.drop(columns=to_drop)
        print(f"Dropped duplicated modality columns: {to_drop}")

    return out


def choose_tabular_columns(df: pd.DataFrame, explicit_keep: List[str], explicit_drop: List[str]) -> List[str]:
    excluded = {
        "PATNO",
        "DX",
        "VISCODE",
        "Group",
        "group",
        "Subject",
        "UPDRS_PATNO",
        "RID",
    }

    if explicit_keep:
        missing = [c for c in explicit_keep if c not in df.columns]
        if missing:
            raise ValueError(f"Columns listed in --tabular-columns not found in CSV: {missing}")
        selected = explicit_keep
    else:
        selected = [
            c
            for c in df.columns
            if c not in excluded and np.issubdtype(df[c].dtype, np.number)
        ]

    selected = [c for c in selected if c not in set(explicit_drop)]

    if not selected:
        raise RuntimeError("No tabular columns selected after applying keep/drop rules")

    return selected


def write_split_h5(
    output_path: str,
    ids: List[str],
    mri_map: Dict[str, str],
    spect_map: Dict[str, str],
    labels_by_id: Dict[str, str],
    clinical_df: pd.DataFrame,
    tabular_columns: List[str],
    tabular_mean: np.ndarray,
    tabular_std: np.ndarray,
) -> None:
    with h5py.File(output_path, "w") as hf:
        stats = hf.create_group("stats")
        tab_group = stats.create_group("tabular")
        tab_group.create_dataset("columns", data=np.array(tabular_columns, dtype="S"))
        tab_group.create_dataset("mean", data=tabular_mean.astype(np.float32))
        tab_group.create_dataset("stddev", data=tabular_std.astype(np.float32))

        for sid in ids:
            row = clinical_df.loc[clinical_df["PATNO"] == sid]
            if row.empty:
                continue

            row0 = row.iloc[0]
            mri_path = mri_map[sid]
            spect_path = spect_map[sid]

            mri_data = nib.load(mri_path).get_fdata().astype(np.float32)
            spect_data = nib.load(spect_path).get_fdata().astype(np.float32)
            tabular = row0[tabular_columns].to_numpy(dtype=np.float32)

            g = hf.create_group(sid)
            g.create_dataset("MRI/T1/data", data=mri_data)
            g.create_dataset("SPECT/data", data=spect_data)
            g.create_dataset("tabular", data=tabular)

            viscode = str(row0["VISCODE"]) if "VISCODE" in row0.index and pd.notna(row0["VISCODE"]) else "baseline"
            g.attrs["DX"] = labels_by_id[sid]
            g.attrs["RID"] = int(sid)
            g.attrs["VISCODE"] = viscode


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build PASTA-compatible train/valid/test HDF5 files using separate Control and Parkinson folders."
        )
    )
    parser.add_argument("--control-mri-dir", required=True)
    parser.add_argument("--control-spect-dir", required=True)
    parser.add_argument("--pd-mri-dir", required=True)
    parser.add_argument("--pd-spect-dir", required=True)
    parser.add_argument("--clinical-csv", required=True)
    parser.add_argument("--output-dir", required=True)

    parser.add_argument("--id-column", default=None, help="Column in CSV that contains subject id (PATNO/Subject/UPDRS_PATNO)")
    parser.add_argument(
        "--drop-modality-duplicates",
        action="store_true",
        help="Drop duplicated modality pairs (e.g., keep MRI_* and remove SPECT_* when values are equal).",
    )
    parser.add_argument(
        "--preferred-prefix",
        default="MRI_",
        help="Preferred prefix when dropping modality duplicates (default: MRI_).",
    )
    parser.add_argument(
        "--secondary-prefix",
        default="SPECT_",
        help="Secondary prefix considered duplicate of preferred prefix (default: SPECT_).",
    )
    parser.add_argument(
        "--tabular-columns",
        default="",
        help="Comma-separated list of tabular columns to keep. If omitted, all numeric columns are used.",
    )
    parser.add_argument(
        "--drop-columns",
        default="",
        help="Comma-separated list of columns to drop from tabular features.",
    )

    parser.add_argument("--control-label", default="CN", help="DX label written for control subjects")
    parser.add_argument("--pd-label", default="Parkinson", help="DX label written for Parkinson subjects")
    parser.add_argument("--max-subjects-per-class", type=int, default=0)

    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--valid-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    if args.train_ratio <= 0 or args.valid_ratio <= 0 or args.train_ratio + args.valid_ratio >= 1:
        raise ValueError("train_ratio and valid_ratio must be >0 and their sum must be <1")

    os.makedirs(args.output_dir, exist_ok=True)

    control_mri = build_file_map(args.control_mri_dir)
    control_spect = build_file_map(args.control_spect_dir)
    pd_mri = build_file_map(args.pd_mri_dir)
    pd_spect = build_file_map(args.pd_spect_dir)

    control_ids = sorted(set(control_mri.keys()) & set(control_spect.keys()))
    pd_ids = sorted(set(pd_mri.keys()) & set(pd_spect.keys()))

    if args.max_subjects_per_class and args.max_subjects_per_class > 0:
        control_ids = control_ids[: args.max_subjects_per_class]
        pd_ids = pd_ids[: args.max_subjects_per_class]

    overlap = set(control_ids) & set(pd_ids)
    if overlap:
        raise RuntimeError(f"Found subject ids in both classes, please fix folder assignment. Example: {sorted(list(overlap))[:5]}")

    labels_by_id: Dict[str, str] = {}
    for sid in control_ids:
        labels_by_id[sid] = args.control_label
    for sid in pd_ids:
        labels_by_id[sid] = args.pd_label

    mri_map: Dict[str, str] = {**{k: control_mri[k] for k in control_ids}, **{k: pd_mri[k] for k in pd_ids}}
    spect_map: Dict[str, str] = {**{k: control_spect[k] for k in control_ids}, **{k: pd_spect[k] for k in pd_ids}}

    all_ids = sorted(labels_by_id.keys())
    if len(all_ids) < 3:
        raise RuntimeError("Need at least 3 matched subjects to create train/valid/test splits")

    clinical_df = pd.read_csv(args.clinical_csv)

    if args.drop_modality_duplicates:
        clinical_df = remove_modality_duplicates(
            clinical_df,
            keep_prefix=args.preferred_prefix,
            other_prefix=args.secondary_prefix,
        )

    id_col = infer_id_column(clinical_df, args.id_column)
    clinical_df["PATNO"] = clinical_df[id_col]
    clinical_df = deduplicate_by_patient(clinical_df)
    clinical_df = clinical_df[clinical_df["PATNO"].isin(all_ids)].copy()

    if clinical_df.empty:
        raise RuntimeError("No clinical rows matched the ids found in image folders")

    matched_ids = sorted(set(clinical_df["PATNO"].tolist()) & set(all_ids))
    if len(matched_ids) < 3:
        raise RuntimeError("After matching with CSV, fewer than 3 subjects remain")

    mri_map = {k: mri_map[k] for k in matched_ids}
    spect_map = {k: spect_map[k] for k in matched_ids}
    labels_by_id = {k: labels_by_id[k] for k in matched_ids}

    keep_cols = parse_csv_list(args.tabular_columns)
    drop_cols = parse_csv_list(args.drop_columns)

    clinical_df = coerce_numeric(clinical_df, keep_cols) if keep_cols else clinical_df
    tabular_columns = choose_tabular_columns(clinical_df, keep_cols, drop_cols)
    clinical_df = coerce_numeric(clinical_df, tabular_columns)
    clinical_df[tabular_columns] = clinical_df[tabular_columns].fillna(clinical_df[tabular_columns].mean())

    clean_csv_path = os.path.join(args.output_dir, "clinical_clean.csv")
    clinical_df.to_csv(clean_csv_path, index=False)
    print(f"Saved cleaned clinical CSV to: {clean_csv_path}")

    train_ids, valid_ids, test_ids = stratified_split(
        matched_ids,
        labels_by_id,
        args.train_ratio,
        args.valid_ratio,
        args.seed,
    )

    train_rows = clinical_df[clinical_df["PATNO"].isin(train_ids)]
    tabular_values = train_rows[tabular_columns].to_numpy(dtype=np.float32)
    tabular_mean = tabular_values.mean(axis=0)
    tabular_std = tabular_values.std(axis=0)
    tabular_std[tabular_std < 1e-8] = 1.0

    write_split_h5(
        os.path.join(args.output_dir, "train.h5"),
        train_ids,
        mri_map,
        spect_map,
        labels_by_id,
        clinical_df,
        tabular_columns,
        tabular_mean,
        tabular_std,
    )
    write_split_h5(
        os.path.join(args.output_dir, "valid.h5"),
        valid_ids,
        mri_map,
        spect_map,
        labels_by_id,
        clinical_df,
        tabular_columns,
        tabular_mean,
        tabular_std,
    )
    write_split_h5(
        os.path.join(args.output_dir, "test.h5"),
        test_ids,
        mri_map,
        spect_map,
        labels_by_id,
        clinical_df,
        tabular_columns,
        tabular_mean,
        tabular_std,
    )

    counts = defaultdict(int)
    for sid in matched_ids:
        counts[labels_by_id[sid]] += 1

    split_manifest = {
        "seed": args.seed,
        "train_ratio": args.train_ratio,
        "valid_ratio": args.valid_ratio,
        "class_counts": dict(counts),
        "id_column": id_col,
        "control_label": args.control_label,
        "pd_label": args.pd_label,
        "tabular_columns": tabular_columns,
        "splits": {
            "train": train_ids,
            "valid": valid_ids,
            "test": test_ids,
        },
    }

    with open(os.path.join(args.output_dir, "split_manifest.json"), "w", encoding="utf-8") as handle:
        json.dump(split_manifest, handle, indent=2, ensure_ascii=False)

    print("PASTA H5 files created (Control + Parkinson mode):")
    print(f"  total subjects matched: {len(matched_ids)}")
    print(f"  class counts: {dict(counts)}")
    print(f"  train: {len(train_ids)} subjects")
    print(f"  valid: {len(valid_ids)} subjects")
    print(f"  test : {len(test_ids)} subjects")
    print(f"  tabular columns ({len(tabular_columns)}): {tabular_columns}")


if __name__ == "__main__":
    main()
