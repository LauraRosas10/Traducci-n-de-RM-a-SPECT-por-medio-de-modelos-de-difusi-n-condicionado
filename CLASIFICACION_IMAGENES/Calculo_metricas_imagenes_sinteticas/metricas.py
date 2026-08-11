import os
import numpy as np
import nibabel as nib
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio, structural_similarity
from scipy.ndimage import zoom
import glob
import pandas as pd

# --- CONFIGURACIÓN ---
SYN_FOLDER = '../sinteticas_sin_cy_sin_clinicas/syn_spect_test'
# Asegúrate de que estas rutas sean correctas desde la carpeta 'graficas/'
ORIGINAL_FOLDERS = [
    '../data_filtered/control/spect',
    '../data_filtered/parkinson/spect'
]
OUTPUT_FILE = 'metricas_sin_cycle_sin_tabular/resultados_metricas_3D_planares.txt'
SUMMARY_FILE = 'metricas_sin_cycle_sin_tabular/resumen_metricas_general.csv'

def find_original(uid):
    """Busca el archivo original basándose en el ID (uid)."""
    for folder in ORIGINAL_FOLDERS:
        candidates = glob.glob(os.path.join(folder, f'*{uid}*.nii.gz'))
        if candidates:
            return candidates[0]
    return None

def adjust_shape(image, target_shape):
    """Ajusta el volumen al tamaño del sintético."""
    factors = [t / s for t, s in zip(target_shape, image.shape)]
    return zoom(image, factors, order=1)

def calcular_metricas_por_plano(vol1, vol2):
    """Calcula MAE, MSE, PSNR y SSIM promediando sobre cortes de los 3 planos."""
    metricas_totales = {'MAE': [], 'MSE': [], 'PSNR': [], 'SSIM': []}
    
    # Recorrer los 3 planos anatómicos: 0 (Sagital), 1 (Coronal), 2 (Axial)
    for eje in [0, 1, 2]:
        for i in range(vol1.shape[eje]):
            if eje == 0:   s, o = vol1[i,:,:], vol2[i,:,:]
            elif eje == 1: s, o = vol1[:,i,:], vol2[:,i,:]
            else:          s, o = vol1[:,:,i], vol2[:,:,i]
            
            # Filtro: evaluar solo cortes con tejido cerebral
            if np.sum(o) > 0:
                metricas_totales['MAE'].append(np.mean(np.abs(o - s)))
                metricas_totales['MSE'].append(mean_squared_error(o, s))
                metricas_totales['PSNR'].append(peak_signal_noise_ratio(o, s, data_range=1.0))
                metricas_totales['SSIM'].append(structural_similarity(o, s, data_range=1.0))
                
    return {k: np.mean(v) for k, v in metricas_totales.items()}

def calculate_metrics():
    syn_files = [f for f in os.listdir(SYN_FOLDER) if f.endswith('.nii.gz')]
    resultados = []
    
    with open(OUTPUT_FILE, 'w') as f:
        f.write("UID,MAE,MSE,PSNR,SSIM\n")
        
        for syn_name in syn_files:
            uid = syn_name.split('_')[0]
            orig_path = find_original(uid)
            if not orig_path:
                print(f"No encontrado original para: {uid}")
                continue
                
            syn_vol = nib.load(os.path.join(SYN_FOLDER, syn_name)).get_fdata()
            orig_vol = nib.load(orig_path).get_fdata()
            
            # Ajustar tamaño
            if syn_vol.shape != orig_vol.shape:
                orig_vol = adjust_shape(orig_vol, syn_vol.shape)
            
            # Normalizar a [0, 1]
            syn_vol = (syn_vol - syn_vol.min()) / (syn_vol.max() - syn_vol.min() + 1e-8)
            orig_vol = (orig_vol - orig_vol.min()) / (orig_vol.max() - orig_vol.min() + 1e-8)

            # Calcular métricas
            res = calcular_metricas_por_plano(syn_vol, orig_vol)
            res['UID'] = uid
            resultados.append(res)
            
            f.write(f"{uid},{res['MAE']:.4f},{res['MSE']:.4f},{res['PSNR']:.4f},{res['SSIM']:.4f}\n")
            print(f"Procesado: {uid}")

    # Generar tabla resumen
    df = pd.DataFrame(resultados)
    resumen = df[['MAE', 'MSE', 'PSNR', 'SSIM']].agg(['mean', 'std']).transpose()
    resumen.to_csv(SUMMARY_FILE)
    print(f"\nResumen general calculado en: {SUMMARY_FILE}")
    print(resumen)

if __name__ == "__main__":
    calculate_metrics()