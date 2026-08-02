
# visualizar_adagn_completo.py
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import torch
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    cross_val_score,
    cross_val_predict,
    LeaveOneOut
)
from sklearn.metrics import (
    silhouette_score,
    adjusted_rand_score,
    davies_bouldin_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)

from scipy.stats import ttest_ind
from scipy.spatial.distance import euclidean

import hdbscan
import umap

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    classification_report
)

# ============================
# CARGA
# ============================
output_dir = Path("results_parkinson/middle/visualizacion_adagn_clasificacion")
output_dir.mkdir(exist_ok=True)
X, y, uids = [], [], []

# feature_dir = Path("results_parkinson/adagn_features")

# for f in sorted(feature_dir.glob("*.pt")):
#     d = torch.load(f, map_location="cpu")
#     X.append(d["adagn_feature"].numpy())
#     y.append(int(d["label"]))
#     uids.append(str(d["uid"]))
    
base_dir = Path("results_parkinson")


for split_dir in sorted(base_dir.glob("adagn_*")):

    feature_dir = split_dir / "adagn_features_middle"

    print("Leyendo:", feature_dir)

    for f in sorted(feature_dir.glob("*.pt")):

        d = torch.load(f, map_location="cpu")

        X.append(d["adagn_feature"].numpy())
        y.append(int(d["label"]))
        uids.append(str(d["uid"]))    
    

X = np.stack(X)
y = np.array(y)

print("="*60)
print("Pacientes:", len(X))
print("Shape X:", X.shape)
print("Shape y:", y.shape)
print("="*60)

# ============================
# ESTADISTICAS
# ============================
print("\nMin:", X.min())
print("Max:", X.max())
print("Mean:", X.mean())
print("Std:", X.std())

mean0 = X[y==0].mean(axis=0)
mean1 = X[y==1].mean(axis=0)

print("\nDistancia centroides:", euclidean(mean0, mean1))

# ============================
# TTEST
# ============================
pvals = []
for i in range(X.shape[1]):
    _, p = ttest_ind(X[y==0,i], X[y==1,i], equal_var=False)
    pvals.append(p)

pvals = np.array(pvals)
print("Features significativas:", np.sum(pvals < 0.05), "/", X.shape[1])

# ============================
# DOS PACIENTES
# ============================
idx0 = np.where(y==0)[0][0]
idx1 = np.where(y==1)[0][0]

plt.figure(figsize=(12,5))
plt.plot(X[idx0], label=f"Control {uids[idx0]}")
plt.plot(X[idx1], label=f"Parkinson {uids[idx1]}")
plt.legend()
plt.grid()
plt.title("Comparación AdaGN")
plt.show()
plt.savefig(output_dir / "comparacion_AdaGN_pacientes.png")
plt.close()

# ============================
# COHEN D
# ============================
def cohens_d(a,b):
    n1,n2=len(a),len(b)
    s1=np.var(a,ddof=1)
    s2=np.var(b,ddof=1)
    pooled=np.sqrt(((n1-1)*s1+(n2-1)*s2)/(n1+n2-2))
    return (np.mean(a)-np.mean(b))/pooled

effects=[]
for i in range(X.shape[1]):
    effects.append(abs(cohens_d(X[y==0,i], X[y==1,i])))

effects=np.array(effects)

print("Cohen D mean:", effects.mean())
print("Cohen D max :", effects.max())

# ============================
# PCA
# ============================
pca2 = PCA(n_components=2)
X_pca2 = pca2.fit_transform(X)

pca3 = PCA(n_components=3)
X_pca3 = pca3.fit_transform(X)

# ============================
# tSNE
# ============================
X_tsne = TSNE(
    n_components=2,
    perplexity=10,
    init="pca",
    random_state=42
).fit_transform(X)

# ============================
# UMAP
# ============================
X_umap = umap.UMAP(
    n_neighbors=10,
    min_dist=0.1,
    random_state=42
).fit_transform(X)

# ============================
# PCA / tSNE / UMAP
# ============================
fig, ax = plt.subplots(1,3, figsize=(18,5))

for label in np.unique(y):
    idx = y == label

    ax[0].scatter(X_pca2[idx,0], X_pca2[idx,1], label=f"Clase {label}")
    ax[1].scatter(X_tsne[idx,0], X_tsne[idx,1])
    ax[2].scatter(X_umap[idx,0], X_umap[idx,1])

ax[0].set_title("PCA")
ax[1].set_title("t-SNE")
ax[2].set_title("UMAP")
ax[0].legend()

plt.tight_layout()
plt.show()
plt.savefig(output_dir / "pca_tsne_umap.png")
plt.close()

# ============================
# CLUSTERING
# ============================
sil_real = silhouette_score(X, y)
db = davies_bouldin_score(X, y)

best_k = 2
best_score = -1

for k in range(2,11):
    km = KMeans(n_clusters=k, random_state=42, n_init=20)
    labels = km.fit_predict(X_pca3)

    score = silhouette_score(X_pca3, labels)

    if score > best_score:
        best_score = score
        best_k = k

kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=20)
clusters = kmeans.fit_predict(X_pca3)

hdb = hdbscan.HDBSCAN(min_cluster_size=3)
hdb_labels = hdb.fit_predict(X)

# ============================
# CLASIFICADORES
# ============================
models = {
    "LogReg": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=5000))
    ]),
    "SVM": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(kernel="rbf", probability=True))
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

print("\n" + "="*60)
print("COMPARACION CLASIFICADORES")
print("="*60)

for name, model in models.items():

    acc = cross_val_score(
        model, X, y,
        cv=5,
        scoring="accuracy"
    )

    auc = cross_val_score(
        model, X, y,
        cv=5,
        scoring="roc_auc"
    )

    print(
        f"{name:10s} "
        f"ACC={acc.mean():.4f} "
        f"AUC={auc.mean():.4f}"
    )

# ============================
# MATRIZ CONFUSION SVM
# ============================
svm = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", SVC(kernel="rbf", probability=True))
])

pred = cross_val_predict(
    svm,
    X,
    y,
    cv=5
)



print("\n" + "="*60)
print("METRICAS SVM")
print("="*60)

print("Accuracy:",
      accuracy_score(y, pred))

print("Balanced Accuracy:",
      balanced_accuracy_score(y, pred))

print("Precision:",
      precision_score(y, pred))

print("Recall:",
      recall_score(y, pred))

print("F1:",
      f1_score(y, pred))

print("\nClassification Report")
print(classification_report(y, pred))

cm = confusion_matrix(y, pred)

ConfusionMatrixDisplay(cm).plot()
plt.title("SVM Confusion Matrix")
plt.show()
plt.savefig(output_dir / "svm_confusion_matrix.png")
plt.close()

# ============================
# RF IMPORTANCE
# ============================
rf = RandomForestClassifier(
    n_estimators=500,
    random_state=42
)

rf.fit(X,y)

importance = rf.feature_importances_

top_idx = np.argsort(importance)[::-1][:20]

plt.figure(figsize=(12,5))
plt.bar(range(20), importance[top_idx])
plt.xticks(range(20), top_idx, rotation=90)
plt.title("Top 20 RF Features")
plt.tight_layout()
plt.show()
plt.savefig(output_dir / "feature_importance.png")
plt.close()


# ============================
# LOOCV RF
# ============================
loo = LeaveOneOut()

acc_loo = cross_val_score(
    rf,
    X,
    y,
    cv=loo,
    scoring="accuracy"
)

print("\nLOOCV RF Accuracy:", acc_loo.mean())

# ============================
# RESUMEN
# ============================
print("\n" + "="*60)
print("RESUMEN FINAL")
print("="*60)

print("Silhouette:", sil_real)
print("Davies Bouldin:", db)
print("ARI KMeans:", adjusted_rand_score(y, clusters))
print("ARI HDBSCAN:", adjusted_rand_score(y, hdb_labels))
print("Cohen D mean:", effects.mean())

from sklearn.model_selection import learning_curve

best_model = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", SVC(kernel="rbf", probability=True))
])

train_sizes, train_scores, val_scores = learning_curve(
    best_model,
    X,
    y,
    cv=5,
    scoring="accuracy",
    train_sizes=np.linspace(0.1, 1.0, 10),
    n_jobs=-1
)

# Media y desviación estándar
train_mean = train_scores.mean(axis=1)
train_std = train_scores.std(axis=1)

val_mean = val_scores.mean(axis=1)
val_std = val_scores.std(axis=1)

plt.figure(figsize=(8,5))

# Train curve
plt.plot(train_sizes, train_mean, label="Train Accuracy", color="blue")
plt.fill_between(
    train_sizes,
    train_mean - train_std,
    train_mean + train_std,
    alpha=0.2,
    color="blue"
)

# Validation curve
plt.plot(train_sizes, val_mean, label="Validation Accuracy", color="orange")
plt.fill_between(
    train_sizes,
    val_mean - val_std,
    val_mean + val_std,
    alpha=0.2,
    color="orange"
)

plt.xlabel("Tamaño del entrenamiento")
plt.ylabel("Accuracy")
plt.title("Curva de aprendizaje con varianza (SVM)")
plt.legend()
plt.grid()

plt.savefig(output_dir / "learning_curve_variance.png")
plt.show()