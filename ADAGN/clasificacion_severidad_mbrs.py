import warnings
warnings.filterwarnings("ignore")

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from sklearn.manifold import TSNE
import umap

from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_score,
    cross_val_predict
)

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression


# ====================================================
# CONFIG
# ====================================================

CSV_PATH = "/home/brain/Data/Users/jeison/difusion/infClinica/MDS-UPDRS.csv"


output_dir = Path("results_parkinson/res_192/severidad_mbrs_clasificacion")
base_dir = Path("results_parkinson")
output_dir.mkdir(parents=True, exist_ok=True)


# ====================================================
# MBRS -> CLASE
# ====================================================

def clase_pd(m):

    if pd.isna(m):
        return -1

    if m <= 5:
        return 1

    if 6 <= m <= 12:
        return 2

    if 13 <= m <= 24:
        return 3

    return -1


# ====================================================
# CSV CLINICO
# ====================================================

df = pd.read_csv(CSV_PATH)

df["PATNO"] = df["PATNO"].astype(str)

df["severity"] = df["MBRS"].apply(clase_pd)

df = df[df["severity"] > 0]

print("\nDistribucion clases:")
print(df["severity"].value_counts().sort_index())


# ====================================================
# CARGAR ADAGN
# ====================================================

X = []
y = []
uids = []

missing = 0

feature_folders = sorted(base_dir.glob("*/adagn_features_192"))

print("\nCarpetas encontradas:")
for f in feature_folders:
    print(f)

for feature_dir in feature_folders:

    print(f"\nLeyendo {feature_dir}")

    for f in sorted(feature_dir.glob("*.pt")):

        d = torch.load(f, map_location="cpu")

        uid = str(d["uid"])

        row = df[df["PATNO"] == uid]

        if len(row) == 0:
            missing += 1
            continue

        severity = int(row.iloc[0]["severity"])

        X.append(d["adagn_feature"].numpy())
        y.append(severity)
        uids.append(uid)

print("\nPacientes encontrados:", len(X))
print("Pacientes sin match:", missing)

if len(X) == 0:
    raise RuntimeError(
        "No se encontró ningún paciente emparejado entre los .pt y el CSV"
    )


X = np.stack(X)
y = np.array(y)

print("\n" + "="*60)
print("Pacientes usados:", len(X))
print("Pacientes sin match:", missing)
print("Shape:", X.shape)
print("="*60)

print("\nDistribucion final:")
for c in sorted(np.unique(y)):
    print(f"Clase {c}: {(y==c).sum()}")


# ====================================================
# REDUCCION DE DIMENSIONALIDAD
# ====================================================

print("\nCalculando PCA / t-SNE / UMAP...")

# estandarizacion
X_scaled = StandardScaler().fit_transform(X)

# ---------------------------
# PCA
# ---------------------------

pca = PCA(
    n_components=2,
    random_state=42
)

X_pca = pca.fit_transform(X_scaled)

# ---------------------------
# t-SNE
# ---------------------------

n_samples = len(X)

perplexity = min(30, max(5, n_samples // 10))

X_tsne = TSNE(
    n_components=2,
    perplexity=perplexity,
    learning_rate="auto",
    init="pca",
    random_state=42
).fit_transform(X_scaled)

# ---------------------------
# UMAP
# ---------------------------

X_umap = umap.UMAP(
    n_neighbors=min(15, max(5, n_samples // 10)),
    min_dist=0.1,
    metric="euclidean",
    random_state=42
).fit_transform(X_scaled)

# ====================================================
# FIGURA UNICA
# ====================================================

fig, axes = plt.subplots(
    1,
    3,
    figsize=(18, 6)
)

classes = sorted(np.unique(y))

cmap = plt.cm.tab10

# ---------------------------
# PCA
# ---------------------------

for i, c in enumerate(classes):

    idx = y == c

    axes[0].scatter(
        X_pca[idx, 0],
        X_pca[idx, 1],
        s=40,
        alpha=0.8,
        color=cmap(i),
        label=f"Clase {c}"
    )

axes[0].set_title(
    f"PCA\nVar Exp={pca.explained_variance_ratio_.sum():.2f}"
)

axes[0].set_xlabel("PC1")
axes[0].set_ylabel("PC2")

# ---------------------------
# t-SNE
# ---------------------------

for i, c in enumerate(classes):

    idx = y == c

    axes[1].scatter(
        X_tsne[idx, 0],
        X_tsne[idx, 1],
        s=40,
        alpha=0.8,
        color=cmap(i)
    )

axes[1].set_title(
    f"t-SNE\nPerplexity={perplexity}"
)

axes[1].set_xlabel("Dim 1")
axes[1].set_ylabel("Dim 2")

# ---------------------------
# UMAP
# ---------------------------

for i, c in enumerate(classes):

    idx = y == c

    axes[2].scatter(
        X_umap[idx, 0],
        X_umap[idx, 1],
        s=40,
        alpha=0.8,
        color=cmap(i)
    )

axes[2].set_title("UMAP")

axes[2].set_xlabel("Dim 1")
axes[2].set_ylabel("Dim 2")

# ---------------------------
# Leyenda unica
# ---------------------------

handles, labels = axes[0].get_legend_handles_labels()

fig.legend(
    handles,
    labels,
    loc="upper center",
    ncol=len(classes),
    bbox_to_anchor=(0.5, 1.02)
)

plt.suptitle(
    "Representaciones Latentes ADAGN - Severidad MBRS",
    fontsize=14
)

plt.tight_layout()

plt.savefig(
    output_dir / "PCA_TSNE_UMAP_severidad.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    "Figura guardada:",
    output_dir / "PCA_TSNE_UMAP_severidad.png"
)

# ====================================================
# MODELOS
# ====================================================

models = {

    "LogReg": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            max_iter=5000,
            multi_class="multinomial"
        ))
    ]),

    "SVM": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(
            kernel="rbf"
        ))
    ]),

    "LDA": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LinearDiscriminantAnalysis())
    ]),

    "KNN": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", KNeighborsClassifier(5))
    ]),

    "RF": RandomForestClassifier(
        n_estimators=500,
        random_state=42
    )
}


# ====================================================
# CV
# ====================================================

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

print("\n" + "="*60)
print("CLASIFICACION SEVERIDAD MBRS")
print("="*60)

results = []

best_model_name = None
best_score = -1

for name, model in models.items():

    acc = cross_val_score(
        model,
        X,
        y,
        cv=cv,
        scoring="accuracy"
    )

    f1 = cross_val_score(
        model,
        X,
        y,
        cv=cv,
        scoring="f1_macro"
    )

    print(
        f"{name:10s}"
        f" ACC={acc.mean():.4f}"
        f" F1={f1.mean():.4f}"
    )

    results.append([
        name,
        acc.mean(),
        f1.mean()
    ])

    if f1.mean() > best_score:
        best_score = f1.mean()
        best_model_name = name


# ====================================================
# MEJOR MODELO
# ====================================================

best_model = models[best_model_name]

pred = cross_val_predict(
    best_model,
    X,
    y,
    cv=cv
)

print("\nMejor modelo:", best_model_name)

print("\nAccuracy:",
      accuracy_score(y, pred))

print("Balanced Accuracy:",
      balanced_accuracy_score(y, pred))

print("Macro F1:",
      f1_score(y, pred, average="macro"))

print("\n")
print(classification_report(y, pred))


# ====================================================
# MATRIZ CONFUSION
# ====================================================

cm = confusion_matrix(y, pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=[
        "Clase1",
        "Clase2",
        "Clase3"
    ]
)

disp.plot()

plt.title(
    f"{best_model_name} - Confusion Matrix"
)

plt.tight_layout()

plt.savefig(
    output_dir /
    f"confusion_matrix_{best_model_name}.png"
)

plt.close()


# ====================================================
# CSV RESULTADOS
# ====================================================

results_df = pd.DataFrame(
    results,
    columns=[
        "Modelo",
        "Accuracy",
        "MacroF1"
    ]
)

results_df.to_csv(
    output_dir / "metricas.csv",
    index=False
)

print("\nTodo guardado en:")
print(output_dir)