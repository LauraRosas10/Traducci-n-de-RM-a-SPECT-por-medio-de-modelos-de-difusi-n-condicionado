import warnings
warnings.filterwarnings("ignore")

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_predict
)

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    classification_report
)

from sklearn.base import clone


CSV_PATH = "/home/brain/Data/Users/jeison/difusion/infClinica/MDS-UPDRS.csv"

base_dir = Path("results_parkinson")

output_dir = Path(
    "results_parkinson/middle/probabilidades_mbrs"
)

output_dir.mkdir(
    parents=True,
    exist_ok=True
)

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


df = pd.read_csv(CSV_PATH)

df["PATNO"] = df["PATNO"].astype(str)

df["severity"] = df["MBRS"].apply(clase_pd)

df = df[df["severity"] > 0]

print(df["severity"].value_counts())


X = []
y = []
uids = []
mbrs = []

missing = 0

feature_folders = sorted(
    base_dir.glob("*/adagn_features_middle")
)

for feature_dir in feature_folders:

    for f in sorted(feature_dir.glob("*.pt")):

        d = torch.load(f, map_location="cpu")

        uid = str(d["uid"])

        row = df[df["PATNO"] == uid]


        if len(row) == 0:
            missing += 1
            continue

        X.append(
            d["adagn_feature"].numpy()
        )

        y.append(
            int(row.iloc[0]["severity"])
        )

        uids.append(uid)
        
        mbrs.append(
                row.iloc[0]["MBRS"]
            )

X = np.stack(X)
y = np.array(y)

print("Pacientes:", len(X))
print("Shape:", X.shape)


model = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(
        max_iter=5000,
        multi_class="multinomial",
        random_state=42
    ))
])


cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


pred = cross_val_predict(
    model,
    X,
    y,
    cv=cv,
    method="predict"
)

probs = cross_val_predict(
    model,
    X,
    y,
    cv=cv,
    method="predict_proba"
)


# =====================================================
# CSV CON PROBABILIDADES
# =====================================================

prob_correcta = []

for i in range(len(y)):

    # y contiene clases 1,2,3
    clase_real = y[i]

    # probs usa índices 0,1,2
    prob_correcta.append(
        probs[i, clase_real - 1]
    )

prob_df = pd.DataFrame({

    "UID": uids,
    
    "MBRS": mbrs,

    "Clase_real": y,

    "Clase_pred": pred,

    "Prob_Clase1": probs[:,0],

    "Prob_Clase2": probs[:,1],

    "Prob_Clase3": probs[:,2],

    "Probabilidad_correcta": prob_correcta

})

prob_df.to_csv(
    output_dir/"probabilidades_por_paciente.csv",
    index=False
)

print("CSV guardado.")




print()

print("Accuracy:",
      accuracy_score(y,pred))

print("Balanced Accuracy:",
      balanced_accuracy_score(y,pred))

print("Macro F1:",
      f1_score(
          y,
          pred,
          average="macro"
      ))

print()

print(classification_report(
    y,
    pred
))




# =====================================================
# EVALUACIÓN POR FOLD
# =====================================================

print("\n==============================")
print("RESULTADOS POR FOLD")
print("==============================")

fold_results = []

for fold, (train_idx, test_idx) in enumerate(cv.split(X, y), start=1):

    model_fold = clone(model)

    model_fold.fit(
        X[train_idx],
        y[train_idx]
    )

    pred_fold = model_fold.predict(
        X[test_idx]
    )

    probs_fold = model_fold.predict_proba(
        X[test_idx]
    )

    acc = accuracy_score(
        y[test_idx],
        pred_fold
    )

    bal_acc = balanced_accuracy_score(
        y[test_idx],
        pred_fold
    )

    macro_f1 = f1_score(
        y[test_idx],
        pred_fold,
        average="macro"
    )

    print(f"\nFold {fold}")

    print(f"Accuracy          : {acc:.4f}")

    print(f"Balanced Accuracy : {bal_acc:.4f}")

    print(f"Macro F1          : {macro_f1:.4f}")

    fold_results.append({

        "fold": fold,

        "accuracy": acc,

        "balanced_accuracy": bal_acc,

        "macro_f1": macro_f1,

        "idx": test_idx,

        "pred": pred_fold,

        "probs": probs_fold,
        
        "y_true": y[test_idx]

    })
    
    
    
    
    
# =====================================================
# MEJOR FOLD
# =====================================================

best_fold = max(
    fold_results,
    key=lambda x: x["macro_f1"]   # puedes cambiar a accuracy
)

print("\n==============================")
print("MEJOR FOLD")
print("==============================")

print("Fold:", best_fold["fold"])

print("Accuracy:",
      best_fold["accuracy"])

print("Balanced Accuracy:",
      best_fold["balanced_accuracy"])

print("Macro F1:",
      best_fold["macro_f1"])




idx = best_fold["idx"]

best_probs = best_fold["probs"]

best_pred = best_fold["pred"]

prob_correcta_fold = []

for i, clase_real in enumerate(y[idx]):

    prob_correcta_fold.append(

        best_probs[i, clase_real-1]

    )

best_df = pd.DataFrame({

    "UID": np.array(uids)[idx],

    "MBRS": np.array(mbrs)[idx],

    "Clase_real": y[idx],

    "Clase_pred": best_pred,

    "Prob_Clase1": best_probs[:,0],

    "Prob_Clase2": best_probs[:,1],

    "Prob_Clase3": best_probs[:,2],

    "Probabilidad_correcta": prob_correcta_fold

})

best_df.to_csv(

    output_dir /
    "mejor_fold_probabilidades.csv",

    index=False

)

# =====================================================
# BOXPLOT
# =====================================================

datos = [

    prob_df.loc[
        prob_df["Clase_real"] == 1,
        "Probabilidad_correcta"
    ],

    prob_df.loc[
        prob_df["Clase_real"] == 2,
        "Probabilidad_correcta"
    ],

    prob_df.loc[
        prob_df["Clase_real"] == 3,
        "Probabilidad_correcta"
    ]

]

plt.figure(figsize=(7,6))

plt.boxplot(
    datos,
    labels=[
        "Clase 1",
        "Clase 2",
        "Clase 3"
    ],
    showmeans=True
)

plt.ylabel("Probabilidad")

plt.xlabel("Clase real")

plt.ylim(0,1)

plt.grid(axis="y", alpha=0.3)

plt.title(
    "Distribución de la probabilidad predicha por clase"
)

plt.tight_layout()

plt.savefig(
    output_dir/"boxplot_probabilidades.png",
    dpi=300
)

plt.show()

import seaborn as sns
import matplotlib.pyplot as plt

# Crear un DataFrame solo con lo que se va a graficar
plot_df = prob_df[["Clase_real", "Probabilidad_correcta"]].copy()

# Cambiar etiquetas
plot_df["Clase_real"] = plot_df["Clase_real"].map({
    1: "Mild",
    2: "Moderate",
    3: "Severe"
})

plt.figure(figsize=(7,6))

# Boxplot
sns.boxplot(
    data=plot_df,
    x="Clase_real",
    y="Probabilidad_correcta",
    color="white",
    order=["Mild", "Moderate", "Severe"],
    width=0.55,
    showfliers=False
)

# Puntos individuales
sns.stripplot(
    data=plot_df,
    x="Clase_real",
    y="Probabilidad_correcta",
    order=["Mild", "Moderate", "Severe"],
    color="#dd8452",
    size=6,
    jitter=0.15,
    alpha=0.9
)

plt.xlabel("Real class", fontsize=12)
plt.ylabel("Probability", fontsize=12)
plt.ylim(0,1.05)

plt.tight_layout()

plt.savefig(
    output_dir/"boxplot_probabilidades_estilo_paper.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


plot_df = best_df[[
    "Clase_real",
    "Probabilidad_correcta"
]].copy()

plot_df["Clase_real"] = plot_df["Clase_real"].map({

    1:"Mild",

    2:"Moderate",

    3:"Severe"

})

plt.figure(figsize=(7,6))

sns.boxplot(

    data=plot_df,

    x="Clase_real",

    y="Probabilidad_correcta",

    color="white",
    order=["Mild", "Moderate", "Severe"],

    width=0.55,

    showfliers=False

)

sns.stripplot(

    data=plot_df,

    x="Clase_real",

    y="Probabilidad_correcta",
    order=["Mild", "Moderate", "Severe"],

    color="#dd8452",

    jitter=0.15,

    size=6

)

plt.xlabel("Real class")

plt.ylabel("Probability")

plt.ylim(0,1.05)

plt.tight_layout()

plt.savefig(

    output_dir /
    "boxplot_mejor_fold.png",

    dpi=300,

    bbox_inches="tight"

)

plt.show()