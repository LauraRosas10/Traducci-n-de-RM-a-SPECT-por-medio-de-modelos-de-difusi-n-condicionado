import torch
import numpy as np
import logging
from tqdm import tqdm
# Asegúrate de que el path de importación sea correcto según tu estructura
from src.datasets.dataset import MRI2PET_2_5D_Dataset 

# Configuración de logging para ver los mensajes del Dataset
logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger(__name__)

def test_dataset_thoroughly():
    # 1. Configuración idéntica a tu archivo YAML/Entrenamiento
    params = {
        "resolution": [192, 224],
        "data_path": "/home/Data/jefelitman_pupils/angel/difusion/PASTA/data/test.h5",
        "output_dim": 15,
        "direction": 'axial',
        "num_slices": 16,
        "resample_mri": False,
        "dx_labels": ['CN', 'Parkinson'],
        "ROI_mask": "/home/Data/jefelitman_pupils/angel/difusion/ROI_mask/parkinson_ROI_mask.nii.gz",
        "classes": 'binary' 
    }

    print("\n--- Iniciando Prueba de Estrés de Datos ---")
    
    try:
        dataset = MRI2PET_2_5D_Dataset(**params)
        print(f"✅ Dataset cargado correctamente. Total de muestras: {len(dataset)}")
    except Exception as e:
        print(f"❌ Error al inicializar el Dataset: {e}")
        return

    # 2. Iteración sobre todas las muestras para encontrar el fallo de dimensiones o archivos
    for i in range(len(dataset)):
        try:
            # Usamos *extra para capturar cualquier número de variables devueltas
            output = dataset[i]
            
            # Intentamos desempaquetar lo básico
            mri, pet, label, tabular, uid = output[0], output[1], output[2], output[3], output[4]
            
            # Verificación de dimensiones (Aquí es donde solía saltar el broadcast error)
            # Si el código llegó aquí es porque el broadcast interno de numpy pasó, 
            # pero verificamos que coincida con lo que el modelo espera.
            expected_shape = (params["output_dim"], params["resolution"][0], params["resolution"][1])
            
            if mri.shape != expected_shape:
                print(f"\n⚠️ Advertencia en Índice {i} (UID: {uid})")
                print(f"   Dimensión obtenida: {mri.shape} vs Esperada: {expected_shape}")

            if i % 50 == 0:
                print(f"Progreso: {i}/{len(dataset)} muestras verificadas...")

        except ValueError as ve:
            print(f"\n❌ ERROR DE VALOR (Posible Broadcast) en Índice {i}")
            print(f"   UID del paciente: {dataset._mri_uid[i]}")
            print(f"   Detalle técnico: {ve}")
            break
        except Exception as e:
            print(f"\n❌ ERROR INESPERADO en Índice {i}")
            print(f"   Tipo de error: {type(e).__name__}")
            print(f"   Mensaje: {e}")
            break

    print("\n--- Prueba Finalizada ---")

if __name__ == "__main__":
    test_dataset_thoroughly()