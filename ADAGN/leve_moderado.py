import warnings
warnings.filterwarnings("ignore")

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
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
    roc_auc_score,
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


output_dir = Path("results_parkinson/middle/leve_moderado_boxplot")
base_dir = Path("results_parkinson")
output_dir.mkdir(parents=True, exist_ok=True)


# ====================================================
# MBRS -> CLASE
# ====================================================

def clase_pd(m):

    if pd.isna(m):
        return -1

    # Leve
    if m <= 5:
        return 0

    # Moderado
    if 6 <= m <= 12:
        return 1

    #se agrego para obtener y lograr evaluar despues no entrenar
    if 13 <= m <= 24:
        return 2      # Severe
    
    
    # Eliminar severos
    return -1


# ====================================================
# CSV CLINICO
# ====================================================

df = pd.read_csv(CSV_PATH)

df["PATNO"] = df["PATNO"].astype(str)

df["severity"] = df["MBRS"].apply(clase_pd)

df = df[df["severity"] >= 0]

print("\nDistribucion clases:")
print(df["severity"].value_counts().sort_index())


# ====================================================
# CARGAR ADAGN
# ====================================================

X = []
y = []
uids = []

missing = 0


feature_folders = sorted(base_dir.glob("*/adagn_features_middle"))

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

mask_train = y != 2

X_train = X[mask_train]
y_train = y[mask_train]

mask_severe = y == 2

X_severe = X[mask_severe]
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
    "Representaciones Latentes ADAGN - leve-moderado-severo",
    fontsize=14
)

plt.tight_layout()

plt.savefig(
    output_dir / "PCA_TSNE_UMAPleve-moderado-severo.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    "Figura guardada:",
    output_dir / "PCA_TSNE_UMAP_severidad.png"
)


# ====================================================
# PCA / TSNE / UMAP SOLO MILD Y MODERATE
# ====================================================

print("\nCalculando PCA / t-SNE / UMAP (solo Mild y Moderate)...")

X_scaled_train = StandardScaler().fit_transform(X_train)

pca_train = PCA(
    n_components=2,
    random_state=42
)

X_pca_train = pca_train.fit_transform(X_scaled_train)

perplexity_train = min(
    30,
    max(5, len(X_train)//10)
)

X_tsne_train = TSNE(
    n_components=2,
    perplexity=perplexity_train,
    learning_rate="auto",
    init="pca",
    random_state=42
).fit_transform(X_scaled_train)

X_umap_train = umap.UMAP(
    n_neighbors=min(15, max(5, len(X_train)//10)),
    min_dist=0.1,
    metric="euclidean",
    random_state=42
).fit_transform(X_scaled_train)


fig, axes = plt.subplots(
    1,
    3,
    figsize=(18,6)
)

classes = sorted(np.unique(y_train))

cmap = plt.cm.tab10

# PCA
for i,c in enumerate(classes):

    idx = y_train == c

    axes[0].scatter(
        X_pca_train[idx,0],
        X_pca_train[idx,1],
        s=40,
        alpha=0.8,
        color=cmap(i),
        label=f"Clase {c}"
    )

axes[0].set_title(
    f"PCA\nVar Exp={pca_train.explained_variance_ratio_.sum():.2f}"
)

# t-SNE
for i,c in enumerate(classes):

    idx = y_train == c

    axes[1].scatter(
        X_tsne_train[idx,0],
        X_tsne_train[idx,1],
        s=40,
        alpha=0.8,
        color=cmap(i)
    )

axes[1].set_title(
    f"t-SNE\nPerplexity={perplexity_train}"
)

# UMAP
for i,c in enumerate(classes):

    idx = y_train == c

    axes[2].scatter(
        X_umap_train[idx,0],
        X_umap_train[idx,1],
        s=40,
        alpha=0.8,
        color=cmap(i)
    )

axes[2].set_title("UMAP")

handles, labels = axes[0].get_legend_handles_labels()

fig.legend(
    handles,
    labels,
    loc="upper center",
    ncol=2,
    bbox_to_anchor=(0.5,1.02)
)

plt.suptitle(
    "Representaciones Latentes ADAGN (Mild vs Moderate)"
)

plt.tight_layout()

plt.savefig(
    output_dir/"PCA_TSNE_UMAP_Mild_Moderate.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()
# ====================================================
# MODELOS
# ====================================================

models = {

    "LogReg": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            max_iter=5000
        ))
    ]),

    "SVM": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(
            kernel="rbf"
        ))
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
print("CLASIFICACION MILD vs MODERATE")
print("="*60)

results = []

best_model_name = None
best_score = -1

for name, model in models.items():

    acc = cross_val_score(
        model,
        X_train,
        y_train,
        cv=cv,
        scoring="accuracy"
    )

    f1 = cross_val_score(
        model,
        X_train,
        y_train,
        cv=cv,
        scoring="f1_macro"
    )
    
    auc = cross_val_score(
    model,
    X_train,
    y_train,
    cv=cv,
    scoring="roc_auc"
    )

    print(
        f"{name:10s}"
        f" ACC={acc.mean():.4f}"
        f" F1={f1.mean():.4f}"
        f" AUC={auc.mean():.4f}"
    )

    results.append([
        name,
        acc.mean(),
        f1.mean(),
        auc.mean()
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
    X_train,
    y_train,
    cv=cv
)

probs = cross_val_predict(
    best_model,
    X_train,
    y_train,
    cv=cv,
    method="predict_proba"
)

auc = roc_auc_score(
    y_train,
    probs[:,1]
)

print("\nMejor modelo:", best_model_name)

print("\nAccuracy:",
      accuracy_score(y_train, pred))

print("Balanced Accuracy:",
      balanced_accuracy_score(y_train, pred))

print("Macro F1:",
      f1_score(y_train, pred, average="macro"))

print("AUC:",
      auc)

print("\n")
print(classification_report(y_train, pred))

best_model.fit(X_train, y_train)


joblib.dump(
    best_model,
    output_dir / "modelo_mild_vs_moderate.pkl"
)

prob_df = pd.DataFrame({
    "PATNO": uids,
    "Clase_real": y,
    "Prob_Moderate": best_model.predict_proba(X)[:,1]
})

prob_df.to_csv(
    output_dir / "probabilidades_pacientes_solo_moderada.csv",
    index=False
)

prob_mild = best_model.predict_proba(
    X[y == 0]
)[:, 1]

prob_moderate = best_model.predict_proba(
    X[y == 1]
)[:, 1]

prob_severe = best_model.predict_proba(
    X[y == 2]
)[:, 1]
# ====================================================
# MATRIZ CONFUSION
# ====================================================

cm = confusion_matrix(y_train, pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=[
        "Leve",
        "Moderado"
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
        "MacroF1",
        "AUC"
    ]
)

results_df.to_csv(
    output_dir / "metricas.csv",
    index=False
)

print("\nTodo guardado en:")
print(output_dir)




plt.figure(figsize=(7,5))

# Boxplot
plt.boxplot(
    [
        prob_mild,
        prob_moderate,
        prob_severe
    ],
    labels=[
        "Mild",
        "Moderate",
        "Severe"
    ],
    widths=0.4
)

# ----------------------------
# Puntos individuales (jitter)
# ----------------------------

np.random.seed(42)

for i, probs in enumerate([
    prob_mild,
    prob_moderate,
    prob_severe
], start=1):

    x = np.random.normal(
        loc=i,
        scale=0.05,
        size=len(probs)
    )

    plt.scatter(
        x,
        probs,
        s=18,
        alpha=0.7
    )

plt.ylabel("Probabilidad de Moderate")

plt.xlabel("Clase real")

plt.title("Clasificador entrenado con Mild vs Moderate")

plt.grid(axis="y", alpha=0.3)

plt.savefig(
    output_dir / "probabilidades_moderate.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()