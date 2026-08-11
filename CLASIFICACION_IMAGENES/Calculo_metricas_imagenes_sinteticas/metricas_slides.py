import os
import glob
import numpy as np
import nibabel as nib
import pandas as pd

from scipy.ndimage import zoom
from skimage.metrics import (
    mean_squared_error,
    peak_signal_noise_ratio,
    structural_similarity
)

# CONFIGURACIÓN

SYN_FOLDER = '../syn_spect'

ORIGINAL_FOLDERS = [
    '../../../../data_filtered/control/spect',
    '../../../../data_filtered/parkinson/spect'
]

OUTPUT_FILE = 'resultados_metricas_slides_82_94.txt'
SUMMARY_FILE = 'resumen_metricas_slides_82_94_general.csv'

# Rango de slices axiales
SLICE_START = 82
SLICE_END = 94


# FUNCIONES AUXILIARES

def find_original(uid):
    """
    Busca el archivo original correspondiente al UID.
    """
    for folder in ORIGINAL_FOLDERS:
        candidates = glob.glob(
            os.path.join(folder, f'*{uid}*.nii.gz')
        )
        if candidates:
            return candidates[0]
    return None


def adjust_shape(image, target_shape, order=1):
    """
    Redimensiona un volumen al tamaño objetivo.
    """
    factors = [
        t / s for t, s in zip(target_shape, image.shape)
    ]
    return zoom(image, factors, order=order)


def normalize_volume(vol):
    """
    Normalización Min-Max a [0,1].
    """
    return (vol - vol.min()) / (vol.max() - vol.min() + 1e-8)


# MÉTRICAS

def calculate_metricas_slides():

    syn_files = sorted(
        [f for f in os.listdir(SYN_FOLDER) if f.endswith('.nii.gz')]
    )

    resultados = []

    with open(OUTPUT_FILE, 'w') as f:

        f.write(
            "UID,MAE_S82-94,MSE_S82-94,PSNR_S82-94,SSIM_S82-94\n"
        )

        for syn_name in syn_files:

            uid = syn_name.split('_')[0]

            orig_path = find_original(uid)

            if orig_path is None:
                print(f"- No se encontró original para {uid}")
                continue

            try:

                syn_path = os.path.join(SYN_FOLDER, syn_name)

                syn_vol = nib.load(syn_path).get_fdata()
                orig_vol = nib.load(orig_path).get_fdata()

                # Ajustar dimensiones si es necesario
                if syn_vol.shape != orig_vol.shape:
                    orig_vol = adjust_shape(
                        orig_vol,
                        syn_vol.shape
                    )

                # Seleccionar rango de slices
                s_start = max(0, SLICE_START)
                s_end = min(
                    syn_vol.shape[2],
                    SLICE_END + 1
                )

                syn_slice = syn_vol[:, :, s_start:s_end]
                orig_slice = orig_vol[:, :, s_start:s_end]

                # Normalización
                syn_slice = normalize_volume(syn_slice)
                orig_slice = normalize_volume(orig_slice)

                # Para MAE, MSE y PSNR usamos todos los voxels
                s_flat = syn_slice.flatten()
                o_flat = orig_slice.flatten()

                mae = np.mean(np.abs(o_flat - s_flat))

                mse = mean_squared_error(
                    o_flat,
                    s_flat
                )

                psnr = peak_signal_noise_ratio(
                    o_flat,
                    s_flat,
                    data_range=1.0
                )

                # SSIM promedio slice por slice
                ssim_values = []

                for i in range(syn_slice.shape[2]):

                    ssim_i = structural_similarity(
                        orig_slice[:, :, i],
                        syn_slice[:, :, i],
                        data_range=1.0
                    )

                    ssim_values.append(ssim_i)

                ssim = np.mean(ssim_values)

                resultados.append({
                    'UID': uid,
                    'MAE': mae,
                    'MSE': mse,
                    'PSNR': psnr,
                    'SSIM': ssim
                })

                f.write(
                    f"{uid},"
                    f"{mae:.6f},"
                    f"{mse:.6f},"
                    f"{psnr:.6f},"
                    f"{ssim:.6f}\n"
                )

                print(
                    f"✅ Procesado: {uid} | "
                    f"MAE={mae:.4f} "
                    f"MSE={mse:.4f} "
                    f"PSNR={psnr:.2f} "
                    f"SSIM={ssim:.4f}"
                )

            except Exception as e:
                print(f"- Error en {uid}: {e}")

    # RESUMEN GENERAL

    if len(resultados) == 0:
        print("⚠️ No se generaron resultados.")
        return

    df = pd.DataFrame(resultados)

    resumen = (
        df[['MAE', 'MSE', 'PSNR', 'SSIM']]
        .agg(['mean', 'std'])
        .transpose()
    )

    resumen.to_csv(SUMMARY_FILE)


    print("RESUMEN GENERAL")
    print(resumen)

    print(f"\n- Resultados: {OUTPUT_FILE}")
    print(f"- Resumen: {SUMMARY_FILE}")



if __name__ == "__main__":
    calculate_metricas_slides()