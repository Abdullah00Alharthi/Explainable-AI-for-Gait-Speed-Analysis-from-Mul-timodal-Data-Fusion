# =====================================
# Gait Classification using Traditional Machine Learning Models
# =====================================
# Evaluates SVM, LDA, and QDA classifiers across multiple gait datasets (MARKERS, GRF, EMG, F_M)
# using subject-wise train/test splits, 10 test subjects per fold.

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.impute import SimpleImputer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, roc_curve, auc, confusion_matrix
)

# =====================================
# 1. Load + Preprocess
# =====================================
def load_and_preprocess(data_file, label_file='SPEED_L.npy', id_file='ID_L.npy'):
    """
    Load and preprocess gait data:
    - Load data and labels
    - Standardize features
    - Flatten to 2D
    - Impute missing values
    """
    X = np.load(data_file)
    y_class = np.load(label_file)
    y_subject = np.load(id_file)

    # Relabel classes if needed (to fix label consistency)
    y_class[y_class == 3] = 2
    y_class[y_class == 4] = 3

    # Handle NaN values
    X = np.nan_to_num(X)

    # Standardization (z-score)
    X = (X - np.mean(X, axis=0, keepdims=True)) / (np.std(X, axis=0, keepdims=True) + 1e-12)

    # Flatten (samples × features)
    X = X.reshape(X.shape[0], -1)

    # Impute missing values (if any remain)
    X = SimpleImputer(strategy="mean").fit_transform(X)

    return X, y_class, y_subject


# =====================================
# 2. Subject-wise Split
# =====================================
def subject_split(X, y_class, y_subject, n_test=10, random_state=42):
    """
    Split dataset by subject ID into training and testing sets.
    n_test: number of subjects to leave out for testing.
    """
    np.random.seed(random_state)
    subjects = np.unique(y_subject)
    test_subjects = np.random.choice(subjects, size=n_test, replace=False)
    train_subjects = np.setdiff1d(subjects, test_subjects)

    idx_train = np.isin(y_subject, train_subjects)
    idx_test = np.isin(y_subject, test_subjects)

    return X[idx_train], y_class[idx_train], X[idx_test], y_class[idx_test], test_subjects


# =====================================
# 3. Train & Evaluate
# =====================================
def train_and_evaluate(X_train, y_train, X_test, y_test, classifiers):
    """
    Train and evaluate multiple classifiers on test data.
    Returns a dictionary containing accuracy, F1-score, ROC, and confusion matrices.
    """
    results = {"accuracy": {}, "f1": {}, "roc": {}, "conf": {}}
    n_classes = len(np.unique(y_train))

    for clf_name, clf in classifiers.items():
        print(f"\nTraining {clf_name}...")
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        # Compute probabilities or decision function
        if hasattr(clf, "decision_function"):
            y_score = clf.decision_function(X_test)
        else:
            y_score = clf.predict_proba(X_test)

        # Compute metrics
        results["accuracy"][clf_name] = accuracy_score(y_test, y_pred)
        results["f1"][clf_name] = f1_score(y_test, y_pred, average="weighted")

        # ROC per class
        fpr, tpr, roc_auc = {}, {}, {}
        for i in range(n_classes):
            y_test_bin = (y_test == i).astype(int)
            if y_score.ndim == 1 or y_score.shape[1] == 1:
                y_score_bin = y_score
            else:
                y_score_bin = y_score[:, i]
            fpr[i], tpr[i], _ = roc_curve(y_test_bin, y_score_bin)
            roc_auc[i] = auc(fpr[i], tpr[i])
        results["roc"][clf_name] = (fpr, tpr, roc_auc)

        # Normalized confusion matrix
        cm = confusion_matrix(y_test, y_pred, normalize="true")
        results["conf"][clf_name] = cm

    return results


# =====================================
# 4. Visualization Functions
# =====================================
def plot_radar(results, datasets, classifiers):
    """Radar plot showing accuracy comparison across datasets."""
    N = len(datasets)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    plt.figure(figsize=(8, 8))
    for clf_name in classifiers.keys():
        values = [results[d]["accuracy"][clf_name] for d in datasets]
        values += values[:1]
        plt.polar(angles, values, marker="o", label=clf_name)

    plt.xticks(angles[:-1], datasets)
    plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0], color="grey", size=10)
    plt.ylim(0, 1)
    plt.title("Radar Plot (Accuracy on 10 Test Subjects)", size=14, weight="bold")
    plt.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    plt.show()


def plot_combined_roc(all_results, datasets, classifiers):
    """Combined ROC curves for all classifiers and datasets."""
    plt.figure(figsize=(10, 8))
    for d in datasets:
        for clf_name in classifiers.keys():
            fpr, tpr, roc_auc = all_results[d]["roc"][clf_name]
            for i in roc_auc.keys():
                plt.plot(fpr[i], tpr[i], label=f'{clf_name} - {d} - class {i} (AUC={roc_auc[i]:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Combined ROC Curves (10 Test Subjects)")
    plt.legend(fontsize=8, loc="lower right")
    plt.show()


def plot_confusion_matrices(all_results, datasets, classifiers):
    """Plot confusion matrices for each dataset and classifier."""
    fig, axes = plt.subplots(len(datasets), len(classifiers), figsize=(15, 12))
    for i, d in enumerate(datasets):
        for j, clf_name in enumerate(classifiers.keys()):
            cm = all_results[d]["conf"][clf_name]
            sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues", ax=axes[i, j],
                        xticklabels=np.unique(list(range(cm.shape[0]))),
                        yticklabels=np.unique(list(range(cm.shape[0]))))
            axes[i, j].set_title(f"{clf_name} - {d}")
            axes[i, j].set_xlabel("Predicted")
            axes[i, j].set_ylabel("True")
    plt.tight_layout()
    plt.show()


# =====================================
# 5. Main Execution
# =====================================
if __name__ == "__main__":
    datasets = {
        "MARKERS": "MARKERS_DATA.npy",
        "GRF": "GRF_DATA.npy",
        "EMG": "EMG_DATA.npy",
        "F_M": "F_M_DATA.npy"
    }

    classifiers = {
        "SVM": OneVsRestClassifier(SVC(kernel="linear", probability=True)),
        "LDA": OneVsRestClassifier(LinearDiscriminantAnalysis()),
        "QDA": OneVsRestClassifier(QuadraticDiscriminantAnalysis())
    }

    all_results = {}

    for d_name, d_file in datasets.items():
        print(f"\n=== Processing {d_name} dataset ===")
        X, y_class, y_subject = load_and_preprocess(d_file)

        # Split data subject-wise
        X_train, y_train, X_test, y_test, test_subjects = subject_split(X, y_class, y_subject, n_test=10)
        print(f"Testing on subjects: {test_subjects}")

        # Train and evaluate models
        results = train_and_evaluate(X_train, y_train, X_test, y_test, classifiers)
        all_results[d_name] = results

        # Print summary
        for clf_name in classifiers.keys():
            print(f"{d_name} - {clf_name}: "
                  f"Accuracy={results['accuracy'][clf_name]:.3f}, "
                  f"F1-score={results['f1'][clf_name]:.3f}")

    # =====================================
    # 6. Visualization
    # =====================================
    plot_radar(all_results, datasets.keys(), classifiers)
    plot_combined_roc(all_results, datasets.keys(), classifiers)
    plot_confusion_matrices(all_results, datasets.keys(), classifiers)
