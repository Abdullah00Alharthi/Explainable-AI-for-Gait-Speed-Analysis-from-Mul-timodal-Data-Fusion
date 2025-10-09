# =====================================
# 1. Import Required Libraries
# =====================================
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.metrics import accuracy_score, f1_score, roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay
from sklearn.impute import SimpleImputer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler
import itertools

# =====================================
# 2. Helper Functions
# =====================================
def load_and_preprocess(data_file, label_file='SPEED_L.npy'):
    """Load, clean, standardize, and flatten dataset."""
    X = np.load(data_file)
    y = np.load(label_file)

    # Adjust labels (3→2, 4→3)
    y[y == 3] = 2
    y[y == 4] = 3

    # Replace NaN with 0
    X = np.nan_to_num(X)

    # Standardize
    X = (X - np.mean(X, axis=0)[None, :, :]) / np.std(X, axis=0)[None, :, :]

    # Flatten
    X = X.reshape(X.shape[0], -1)

    # Impute missing values
    imputer = SimpleImputer(strategy='mean')
    X = imputer.fit_transform(X)

    return X, y, len(np.unique(y))


def cross_validate_and_evaluate(X, y, classifiers, n_splits=10):
    """Perform K-Fold CV and compute mean/std for accuracy and F1."""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    results = {"accuracy": {}, "f1": {}, "roc": {}, "conf": {}}
    n_classes = len(np.unique(y))

    for clf_name, clf in classifiers.items():
        acc_scores, f1_scores = [], []

        print(f"\nRunning {n_splits}-Fold CV for {clf_name}...")
        for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            y_score = clf.decision_function(X_test) if hasattr(clf, "decision_function") else clf.predict_proba(X_test)

            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')
            acc_scores.append(acc)
            f1_scores.append(f1)

        # NEW: compute mean and std
        results["accuracy"][clf_name] = (np.mean(acc_scores), np.std(acc_scores))
        results["f1"][clf_name] = (np.mean(f1_scores), np.std(f1_scores))

        print(f"{clf_name} → Accuracy: {np.mean(acc_scores):.3f} ± {np.std(acc_scores):.3f} | "
              f"F1: {np.mean(f1_scores):.3f} ± {np.std(f1_scores):.3f}")

    return results


def plot_radar(results, datasets, classifiers):
    """Radar plot of mean accuracies across datasets."""
    N = len(datasets)
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    plt.figure(figsize=(8, 8))
    for clf_name in classifiers.keys():
        values = [results[d]["accuracy"][clf_name][0] for d in datasets]
        values += values[:1]
        plt.polar(angles, values, marker='o', label=clf_name)

    plt.xticks(angles[:-1], datasets)
    plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0], color="grey", size=10)
    plt.ylim(0, 1)
    plt.title("Radar Plot of Mean Accuracy Across Datasets", size=14, weight="bold")
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.show()


# =====================================
# 3. Main Script
# =====================================
datasets = {
    "MARKERS": "MARKERS_DATA.npy",
    "GRF": "GRF_DATA.npy",
    "EMG": "EMG_DATA.npy",
    "F_M": "F_M_DATA.npy"
}

classifiers = {
    'SVM': OneVsRestClassifier(SVC(kernel='linear', probability=True)),
    'LDA': OneVsRestClassifier(LinearDiscriminantAnalysis()),
    'QDA': OneVsRestClassifier(QuadraticDiscriminantAnalysis())
}

all_results = {}

for d_name, d_file in datasets.items():
    print(f"\n=== Processing {d_name} dataset ===")
    X, y, n_classes = load_and_preprocess(d_file)
    results = cross_validate_and_evaluate(X, y, classifiers, n_splits=10)
    all_results[d_name] = results

# =====================================
# 4. Display Summary Results
# =====================================
print("\n=== Final Cross-Validation Summary ===")
for d_name in datasets.keys():
    print(f"\nDataset: {d_name}")
    for clf_name in classifiers.keys():
        mean_acc, std_acc = all_results[d_name]["accuracy"][clf_name]
        mean_f1, std_f1 = all_results[d_name]["f1"][clf_name]
        print(f"  {clf_name}: Accuracy = {mean_acc:.3f} ± {std_acc:.3f}, "
              f"F1 = {mean_f1:.3f} ± {std_f1:.3f}")

# =====================================
# 5. Plot Mean Accuracy Radar
# =====================================
plot_radar(all_results, datasets.keys(), classifiers)
