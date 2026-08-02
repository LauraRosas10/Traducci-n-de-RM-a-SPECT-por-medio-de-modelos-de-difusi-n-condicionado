
import argparse
import json
import os
import re
from collections import defaultdict
from typing import Dict, List, Optional, Tuple


def extract_id(filename: str) -> Optional[str]:
    # Mantiene tu método original de extraer el ID antes del primer guion bajo
    match = re.match(r"(\d+)_", filename)
    return match.group(1) if match else None


def build_file_map(folder: str) -> Dict[str, str]:
    # Mantiene tu escaneo y mapeo original de directorios
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


def load_manual_splits_from_csv(csv_path: str) -> Tuple[List[str], List[str], List[str]]:
    """Lee únicamente el ID y la Partición del CSV ignorando cualquier otra ruta interna."""
    import pandas as pd

    df = pd.read_csv(csv_path)
    
    # Limpieza de strings para evitar errores por espacios en blanco
    df["subject_id"] = df["subject_id"].astype(str).str.strip()
    df["split"] = df["split"].astype(str).str.strip().str.lower()

    train_ids = df[df["split"] == "train"]["subject_id"].tolist()
    valid_ids = df[df["split"] == "valid"]["subject_id"].tolist()
    test_ids = df[df["split"].isin(["test", "testing"])]["subject_id"].tolist()

    return train_ids, valid_ids, test_ids


def write_split_h5(
    output_path: str,
    ids: List[str],
    mri_map: Dict[str, str],
    spect_map: Dict[str, str],
    labels_by_id: Dict[str, str],
) -> None:
    import h5py
    import nibabel as nib
    import numpy as np

    with h5py.File(output_path, "w") as hf:
        hf.create_group("stats")

        for sid in ids:
            # Si el ID del CSV no se encuentra mapeado en tus carpetas locales, lo salta de forma segura
            if sid not in mri_map or sid not in spect_map:
                print(f" Advertencia: El ID '{sid}' del CSV no se encontró en tus carpetas de imágenes. Omitiendo...")
                continue

            mri_path = mri_map[sid]
            spect_path = spect_map[sid]

            mri_data = nib.load(mri_path).get_fdata().astype(np.float32)
            spect_data = nib.load(spect_path).get_fdata().astype(np.float32)

            g = hf.create_group(sid)
            g.create_dataset("MRI/T1/data", data=mri_data)
            g.create_dataset("SPECT/data", data=spect_data)

            g.attrs["DX"] = labels_by_id[sid]
            g.attrs["RID"] = int(sid)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build image-only HDF5 files using original directories and a custom split CSV definition."
    )
    # Tus 4 rutas originales de imágenes
    parser.add_argument("--control-mri-dir", required=True)
    parser.add_argument("--control-spect-dir", required=True)
    parser.add_argument("--pd-mri-dir", required=True)
    parser.add_argument("--pd-spect-dir", required=True)
    
    # Parámetro para inyectar tu CSV de particiones
    parser.add_argument("--split-csv", required=True, help="Ruta al archivo cocolit_dataset.csv")
    parser.add_argument("--output-dir", required=True)

    parser.add_argument("--control-label", default="CN", help="DX label written for control subjects")
    parser.add_argument("--pd-label", default="Parkinson", help="DX label written for Parkinson subjects")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Cargar particiones (solo nos interesan las columnas id y split)
    print(f"Leyendo mapeo de particiones desde: {args.split_csv}")
    train_ids, valid_ids, test_ids = load_manual_splits_from_csv(args.split_csv)

    all_csv_ids = train_ids + valid_ids + test_ids
    if not all_csv_ids:
        raise RuntimeError("El archivo CSV no contiene IDs válidos o la columna 'split' está vacía.")

    # 2. Mapear tus rutas e imágenes locales anteriores
    control_mri = build_file_map(args.control_mri_dir)
    control_spect = build_file_map(args.control_spect_dir)
    pd_mri = build_file_map(args.pd_mri_dir)
    pd_spect = build_file_map(args.pd_spect_dir)

    control_ids_dir = set(control_mri.keys()) & set(control_spect.keys())
    pd_ids_dir = set(pd_mri.keys()) & set(pd_spect.keys())

    labels_by_id: Dict[str, str] = {}
    for sid in control_ids_dir:
        labels_by_id[sid] = args.control_label
    for sid in pd_ids_dir:
        labels_by_id[sid] = args.pd_label

    mri_map: Dict[str, str] = {**{k: control_mri[k] for k in control_ids_dir}, **{k: pd_mri[k] for k in pd_ids_dir}}
    spect_map: Dict[str, str] = {**{k: control_spect[k] for k in control_ids_dir}, **{k: pd_spect[k] for k in pd_ids_dir}}

    # Filtrar intersección real entre lo que pide el CSV y lo que tienes físicamente en las carpetas
    matched_ids = sorted([sid for sid in all_csv_ids if sid in labels_by_id])

    # 3. Generación de los archivos .h5
    print("Generando archivos H5 utilizando las rutas locales...")
    write_split_h5(os.path.join(args.output_dir, "train.h5"), train_ids, mri_map, spect_map, labels_by_id)
    write_split_h5(os.path.join(args.output_dir, "valid.h5"), valid_ids, mri_map, spect_map, labels_by_id)
    write_split_h5(os.path.join(args.output_dir, "test.h5"), test_ids, mri_map, spect_map, labels_by_id)

    # Conteo final para el informe
    counts = defaultdict(int)
    for sid in matched_ids:
        counts[labels_by_id[sid]] += 1

    # Manifestación de salida
    split_manifest = {
        "split_source_csv": os.path.basename(args.split_csv),
        "class_counts": dict(counts),
        "control_label": args.control_label,
        "pd_label": args.pd_label,
        "splits": {
            "train": train_ids,
            "valid": valid_ids,
            "test": test_ids,
        },
    }

    with open(os.path.join(args.output_dir, "split_manifest.json"), "w", encoding="utf-8") as handle:
        json.dump(split_manifest, handle, indent=2, ensure_ascii=False)

    print("\n ¡Archivos PASTA H5 creados!")
    print(f"   Sujetos totales coincidentes con imágenes en carpetas: {len(matched_ids)}")
    print(f"   Distribución por clase: {dict(counts)}")
    print(f"   Resumen -> Train: {len(train_ids)} | Valid: {len(valid_ids)} | Test: {len(test_ids)}")


if __name__ == "__main__":
    main()
    