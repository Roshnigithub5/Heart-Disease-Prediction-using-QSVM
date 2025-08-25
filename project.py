"""
Heart Disease Prediction with QSVM vs Classical SVM
- Dataset: Heart Disease (Cleveland-style columns) from a public CSV mirror.
- Goal: Compare Quantum SVM (QSVM, with fidelity-based quantum kernel) vs Classical SVM.
- Fast run: PCA -> 4 features to keep kernel simulation quick.

Quick start (local):
    pip install qiskit qiskit-machine-learning scikit-learn pandas matplotlib

If CSV download fails (no internet in your runtime), set LOCAL_CSV_PATH to your local file.
"""

# ====== Imports ======
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Qiskit ML
from qiskit.circuit.library import ZZFeatureMap
from qiskit_machine_learning.kernels import FidelityStatevectorKernel

# ====== Config ======
RANDOM_STATE = 42
TEST_SIZE = 0.30
USE_PCA = True
PCA_DIM = 4
ZZ_REPS = 2
np.random.seed(RANDOM_STATE)

# Public mirror of heart disease CSV (same columns as common Kaggle dataset)
CSV_URL = None
LOCAL_CSV_PATH = "C:/Users/DELL/Desktop/Project/heart.csv"


# ====== Helpers ======
def banner(text: str):
    print("\n" + "=" * 12 + f" {text} " + "=" * 12)


def load_heart_csv(url: str, local_path: str | None = None) -> pd.DataFrame:
    if local_path:
        return pd.read_csv(local_path)
    try:
        return pd.read_csv(url)
    except Exception as e:
        print("Remote CSV download failed:", e)
        raise RuntimeError(
            "Provide a local CSV path by setting LOCAL_CSV_PATH to your file."
        )


def plot_confusion_matrix(cm, title, target_names):
    fig = plt.figure(figsize=(4, 3))
    plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks([0, 1], target_names, rotation=45)
    plt.yticks([0, 1], target_names)
    for (i, j), val in np.ndenumerate(cm):
        plt.text(j, i, int(val), ha="center", va="center")
    plt.tight_layout()
    plt.savefig("svm_confusion.png")
    plt.savefig("qsvm_confusion.png")
    plt.show()
    

def plot_kernel_heatmap(K_train):
    fig = plt.figure(figsize=(4, 3))
    plt.imshow(K_train, aspect="auto", interpolation="nearest")
    plt.title("Quantum Kernel Gram Matrix (Train)")
    plt.xlabel("Samples")
    plt.ylabel("Samples")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig("kernel_heatmap.png") 
    plt.show()
      


def plot_accuracy_bar(acc_classical, acc_qsvm):
    labels = ["Classical SVM", "QSVM"]
    values = [acc_classical, acc_qsvm]
    fig = plt.figure(figsize=(4, 3))
    plt.bar(labels, values)
    plt.ylim(0.6, 1.0)
    plt.title("Accuracy Comparison")
    for i, v in enumerate(values):
        plt.text(i, v + 0.01, f"{v:.3f}", ha="center")
    plt.tight_layout()
    plt.savefig("accuracy_comparison.png") 
    plt.show()
    


# ====== Pipeline ======
def main():
    banner("Load Dataset")
    df = load_heart_csv(CSV_URL, LOCAL_CSV_PATH)

    # Expect a column named "target" with 0/1 labels.
    if "target" not in df.columns:
        raise ValueError("Expected a 'target' column in the dataset.")
    X = df.drop("target", axis=1).values
    y = df["target"].values
    target_names = ["No Disease", "Heart Disease"]

    print("Shape:", X.shape, " | Classes distribution:", dict(zip([0, 1], np.bincount(y))))
    print("Columns:", list(df.columns))

    banner("Preprocessing")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    if USE_PCA:
        pca = PCA(n_components=PCA_DIM, random_state=RANDOM_STATE)
        X_proc = pca.fit_transform(X_scaled)
        print(
            f"Original dim={X.shape[1]}  -> PCA dim={X_proc.shape[1]} | "
            f"ExplainedVarSum={pca.explained_variance_ratio_.sum():.4f}"
        )
    else:
        X_proc = X_scaled
        print("PCA disabled; using scaled features with dim:", X_proc.shape[1])

    X_train, X_test, y_train, y_test = train_test_split(
        X_proc, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print("Train:", X_train.shape, " Test:", X_test.shape)

    # ===== Classical SVM =====
    banner("Classical SVM (RBF)")
    svm = SVC(kernel="rbf", random_state=RANDOM_STATE)
    svm.fit(X_train, y_train)
    y_pred_svm = svm.predict(X_test)

    acc_svm = accuracy_score(y_test, y_pred_svm)
    print(f"Accuracy: {acc_svm:.4f}\n")
    print("Classification Report (Classical SVM):")
    print(classification_report(y_test, y_pred_svm, target_names=target_names))
    cm_svm = confusion_matrix(y_test, y_pred_svm)
    plot_confusion_matrix(cm_svm, "Classical SVM - Confusion Matrix", target_names)

    # ===== QSVM (Quantum Kernel) =====
    banner("QSVM (FidelityStatevectorKernel + ZZFeatureMap)")
    feature_dim = X_train.shape[1]
    fmap = ZZFeatureMap(feature_dimension=feature_dim, reps=ZZ_REPS)
    qkernel = FidelityStatevectorKernel(feature_map=fmap)

    print("Evaluating quantum kernel (train Gram matrix)...")
    K_train = qkernel.evaluate(x_vec=X_train)
    print("Evaluating quantum kernel (test-vs-train Gram matrix)...")
    K_test = qkernel.evaluate(x_vec=X_test, y_vec=X_train)

    qsvm = SVC(kernel="precomputed", random_state=RANDOM_STATE)
    qsvm.fit(K_train, y_train)
    y_pred_qsvm = qsvm.predict(K_test)

    acc_qsvm = accuracy_score(y_test, y_pred_qsvm)
    print(f"\nQSVM Accuracy: {acc_qsvm:.4f}\n")
    print("Classification Report (QSVM):")
    print(classification_report(y_test, y_pred_qsvm, target_names=target_names))
    cm_qsvm = confusion_matrix(y_test, y_pred_qsvm)
    plot_confusion_matrix(cm_qsvm, "QSVM - Confusion Matrix", target_names)

    # Visuals
    plot_kernel_heatmap(K_train)
    plot_accuracy_bar(acc_svm, acc_qsvm)

    # Summary
    banner("Summary")
    print(f"Classical SVM Accuracy: {acc_svm:.4f}")
    print(f"QSVM Accuracy:         {acc_qsvm:.4f}")
    if acc_qsvm >= acc_svm:
        print("\nQSVM performed on par or better than classical SVM for this setup.")
    else:
        print("\nClassical SVM outperformed QSVM in this run.")
        print("Try tweaks:")
        print("- Change ZZFeatureMap reps (1, 2, 3)")
        print("- Change PCA dim (2–6 or disable PCA)")
        print("- Tune SVM hyperparams (C, gamma)")
        print("- Try noisy/shot-based backends for realism")

if __name__ == "__main__":
    main()
