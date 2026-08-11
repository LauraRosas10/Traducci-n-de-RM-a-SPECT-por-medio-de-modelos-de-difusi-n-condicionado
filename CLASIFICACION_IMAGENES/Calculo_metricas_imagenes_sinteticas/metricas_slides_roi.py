import os
import numpy as np
import nibabel as nib
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio, structural_similarity
from scipy.ndimage import zoom
import glob
import pandas as pd

# --- CONFIGURACIÓN ---
SYN_FOLDER = '../sinteticas_sin_cy_sin_clinicas/syn_spect_test'
ORIGINAL_FOLDERS = [
    '../data_filtered/control/spect',
    '../data_filtered/parkinson/spect'
]
MASK_PATH = 'parkinson_ROI_mask.nii.gz'
OUTPUT_FILE = 'metricas_sin_cycle_sin_tabular/resultados_metricas_slides_roi_82_94.txt'
SUMMARY_FILE = 'metricas_sin_cycle_sin_tabular/resumen_metricas_slides_roi_82_94_general.csv'

# Rango de slices deseado (eje axial: 2)
SLICE_START = 82
SLICE_END = 94

def find_original(uid):
    for folder in ORIGINAL_FOLDERS:
        candidates = glob.glob(os.path.join(folder, f'*{uid}*.nii.gz'))
        if candidates: return candidates[0]
    return None

def adjust_shape(image, target_shape, order=1):
    factors = [t / s for t, s in zip(target_shape, image.shape)]
    return zoom(image, factors, order=order)

def calculate_metricas_slides_roi():
    # 1. Cargar la máscara ROI
    mask_vol = nib.load(MASK_PATH).get_fdata()
    
    syn_files = [f for f in os.listdir(SYN_FOLDER) if f.endswith('.nii.gz')]
    resultados = []
    
    with open(OUTPUT_FILE, 'w') as f:
        f.write("UID,MAE_S82-94_ROI,MSE_S82-94_ROI,PSNR_S82-94_ROI,SSIM_S82-94_ROI\n")
        
        for syn_name in syn_files:
            uid = syn_name.split('_')[0]
            orig_path = find_original(uid)
            if not orig_path: continue
                
            syn_vol = nib.load(os.path.join(SYN_FOLDER, syn_name)).get_fdata()
            orig_vol = nib.load(orig_path).get_fdata()
            
            # Ajustar volumen original al tamaño del sintético
            if syn_vol.shape != orig_vol.shape:
                orig_vol = adjust_shape(orig_vol, syn_vol.shape)
            
            # Aplicar el rango de slices (82-94)
            s_start, s_end = max(0, SLICE_START), min(syn_vol.shape[2], SLICE_END + 1)
            
            syn_slice = syn_vol[:, :, s_start:s_end]
            orig_slice = orig_vol[:, :, s_start:s_end]
            
            # Ajustar máscara al tamaño del volumen y recortar al mismo rango de slices
            mask_resized = adjust_shape(mask_vol, syn_vol.shape, order=0)
            roi_slice = (mask_resized[:, :, s_start:s_end] > 0)
            
            # Normalizar
            syn_slice = (syn_slice - syn_slice.min()) / (syn_slice.max() - syn_slice.min() + 1e-8)
            orig_slice = (orig_slice - orig_slice.min()) / (orig_slice.max() - orig_slice.min() + 1e-8)
            
            # Aplicar doble filtro (Rango de slices AND máscara ROI)
            # Solo comparamos voxels que están en los cortes 82-94 Y dentro del ROI
            s_final = syn_slice[roi_slice]
            o_final = orig_slice[roi_slice]
            
            if s_final.size == 0:
                print(f"No hay voxels de ROI en los slices 82-94 para: {uid}")
                continue

            mae = np.mean(np.abs(o_final - s_final))
            mse = mean_squared_error(o_final, s_final)
            psnr = peak_signal_noise_ratio(o_final, s_final, data_range=1.0)
            ssim = structural_similarity(o_final, s_final, data_range=1.0)
            
            resultados.append({'UID': uid, 'MAE': mae, 'MSE': mse, 'PSNR': psnr, 'SSIM': ssim})
            f.write(f"{uid},{mae:.4f},{mse:.4f},{psnr:.4f},{ssim:.4f}\n")
            print(f"ROI+Slice 82-94 Procesado: {uid}")

    df = pd.DataFrame(resultados)
    resumen = df[['MAE', 'MSE', 'PSNR', 'SSIM']].agg(['mean', 'std']).transpose()
    resumen.to_csv(SUMMARY_FILE)
    print(f"\nResumen ROI+Slice calculado en: {SUMMARY_FILE}\n", resumen)

if __name__ == "__main__":
    calculate_metricas_slides_roi()