# PASTA-PD: Pathology-Aware MRI to DAT-SPECT Cross-modal Translation with Diffusion Models

> **Nota:** Este README es una adaptación del repositorio original de [PASTA](https://arxiv.org/abs/2405.16942)
> (pensado originalmente para MRI → PET/FDG en Alzheimer, dataset ADNI) al dominio de **Parkinson**,
> usando el dataset `cocolit_dataset.csv`.
>
> Se asume que la segunda modalidad de imagen (target de la traducción) es
> **SPECT**, por ser el estándar de imagen funcional en el diagnóstico de Parkinson,
> análogo al PET/FDG del repo original. Si la modalidad real es otra (PET con otro trazador, otra
> secuencia de MRI, etc.), reemplaza las referencias a `SPECT/DAT` a continuación.

Repositorio base:
[PASTA: Pathology-Aware MRI to PET Cross-modal Translation with Diffusion Models](https://arxiv.org/abs/2405.16942) (MICCAI 2024) /
[Translating MRI to PET through Conditional Diffusion Models with Enhanced Pathology Awareness](https://doi.org/10.1016/j.media.2026.104035) (Medical Image Analysis)

Esta variante conserva la arquitectura de difusión condicional del repositorio original, pero se
reentrena desde cero sobre datos de Parkinson (no hay transferencia de pesos desde ADNI, ya que el
dominio clínico y las modalidades de imagen son distintos).

## Instalación

1. Crear entorno: `conda env create -n pasta --file requirements.yaml`
2. Activar entorno: `conda activate pasta`

## Datos

Fuente: `cocolit_dataset.csv`

### Cohorte

| Clase | Pacientes |
|---|---|
| Parkinson | 111 |
| CN (control) | 73 |

Etiqueta de diagnóstico: `PD` (control_label=`CN`, pd_label=`Parkinson`).

### Splits (predefinidos por ID de sujeto)

Los splits ya vienen fijados en `split_config.json` (train / valid / test), a nivel de paciente, para
evitar fuga de datos entre imágenes del mismo sujeto:

- **Train:** 139 sujetos
- **Valid:** 18 sujetos
- **Test:** 18 sujetos

### Formato HDF5

Al igual que en el repositorio original, los datos de entrenamiento, validación y prueba deben
almacenarse en archivos [HDF5](https://en.wikipedia.org/wiki/Hierarchical_Data_Format) separados,
con la siguiente jerarquía:

1. **Primer nivel:** ID único del sujeto (coincide con los IDs listados en `split_config.json`).
2. **Segundo nivel**, con las siguientes entradas:
    1. Un grupo `MRI/T1`, con el volumen 3D de resonancia magnética.
    2. Un grupo `SPECT/DAT`, con el volumen 3D de DAT-SPECT. *(ajustar nombre si la modalidad real es otra)*
    3. Un dataset `tabular` de tamaño **16**, con las variables clínicas (ver lista abajo).
    4. Un atributo de cadena `DX` con la etiqueta diagnóstica: `CN` o `Parkinson`.
    5. Un atributo escalar `RID` con el ID del paciente.

3. Grupo `stats`, con la meta-información de normalización de las variables tabulares:
```bash
/stats/tabular           Group
/stats/tabular/columns   Dataset {16}
/stats/tabular/mean      Dataset {16}
/stats/tabular/stddev    Dataset {16}
```
`mean` y `stddev` deben calcularse **únicamente sobre el split de entrenamiento** y reutilizarse
para normalizar val/test (evita fuga de información del conjunto de evaluación).

### Variables tabulares (16)

| # | Variable | Descripción |
|---|---|---|
| 1 | `MRI_Age` | Edad del paciente al momento del estudio de imagen |
| 2 | `MRI_Sex` | Sexo biológico |
| 3 | `UPDRS_NP3TOT` | Puntaje motor total (UPDRS Parte III) |
| 4 | `UPDRS_NHY` | Estadio de Hoehn y Yahr |
| 5 | `UPDRS_NP3BRADY` | Bradicinesia |
| 6 | `UPDRS_NP3GAIT` | Marcha |
| 7 | `UPDRS_NP3PSTBL` | Estabilidad postural |
| 8 | `UPDRS_NP3PTRMR` | Temblor postural, lado derecho |
| 9 | `UPDRS_NP3PTRML` | Temblor postural, lado izquierdo |
| 10 | `UPDRS_NP3RIGN` | Rigidez, cuello |
| 11 | `UPDRS_NP3RIGRU` | Rigidez, extremidad superior derecha |
| 12 | `UPDRS_NP3RIGLU` | Rigidez, extremidad superior izquierda |
| 13 | `UPDRS_NP3RIGRL` | Rigidez, extremidad inferior derecha |
| 14 | `UPDRS_NP3RIGLL` | Rigidez, extremidad inferior izquierda |
| 15 | `UPDRS_DYSKPRES` | Presencia de discinesias |
| 16 | `UPDRS_DYSKIRAT` | Grado de interferencia de las discinesias |

> **Nota sobre datos faltantes:** las escalas UPDRS suelen registrarse solo (o con mayor completitud)
> en pacientes con Parkinson; verifica si los controles (`CN`) tienen valores nulos en estas columnas
> y define una estrategia de imputación (ej. 0 para ausencia de síntomas motores, o exclusión de esas
> filas) antes de calcular `mean`/`stddev`.

## Uso

El paquete usa [PyTorch](https://pytorch.org). Para entrenar y evaluar el modelo, ejecutar el script
`train_mri2pet.py` (renombrar a `train_mri2dat.py` si se desea reflejar la nueva tarea). La
configuración de argumentos se encuentra en `src/config/pasta_pd.yaml`.

Argumentos principales:

  - `--data_dir`: Ruta a los archivos HDF5 de train/valid/test.
  - `--results_folder`: Ruta de salida de entrenamiento/evaluación.
  - `--tab_cond_dim`: **16** (en vez de 6, por las variables UPDRS adicionales).
  - `--model_cycling`: *True* para consistencia de ciclo (cycle exchange).
  - `--eval_mode`: *False* para entrenamiento, *True* para evaluación.
  - `--synthesis`: *True* para guardar las imágenes sintetizadas durante evaluación.

```bash
python train_mri2pet.py --config src/config/pasta_pd.yaml
```

## Diferencias clave respecto al repositorio original (ADNI)

| Aspecto | PASTA original (ADNI) | PASTA-PD (COCOLIT) |
|---|---|---|
| Modalidad de entrada | MRI T1 | MRI T1 |
| Modalidad de salida | PET/FDG | DAT-SPECT |
| Diagnóstico | CN / MCI / Dementia | CN / Parkinson |
| Variables tabulares | 6 (edad, género, educación, MMSE, ADAS-Cog-13, ApoE4) | 16 (edad, género, 14 subescalas UPDRS) |
| Pesos preentrenados | N/A | No aplica (entrenar desde cero, dominio distinto) |
| Balance de clases | — | Desbalanceado (111 PD vs 73 CN) — considerar ponderación de pérdida o muestreo estratificado |


<br/>

Basado en [PASTA](https://arxiv.org/abs/2405.16942), desarrollado a su vez sobre
[lucidrains/denoising-diffusion-pytorch](https://github.com/lucidrains/denoising-diffusion-pytorch) y
[openai/guided-diffusion](https://github.com/openai/guided-diffusion).
