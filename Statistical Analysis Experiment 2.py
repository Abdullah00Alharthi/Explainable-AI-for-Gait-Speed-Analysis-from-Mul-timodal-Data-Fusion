import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.metrics import accuracy_score, f1_score, roc_curve, auc, confusion_matrix
from sklearn.impute import SimpleImputer
from sklearn.multiclass import OneVsRestClassifier
import seaborn as sns

# =====================================
# 1. Load + Preprocess
# =====================================
def load_and_preprocess(data_file, label_file='SPEED_L.npy', id_file='ID_L.npy'):
    X = np.load(data_file)
    y_class = np.load(label_file)
    y_subject = np.load(id_file)

    # Relabel classes (merge 3→2, 4→3)
    y_class[y_class == 3] = 2
    y_class[y_class == 4] = 3

    # Replace NaN values with zeros
    X = np.nan_to_num(X)

    # Standardize features
    X = (X - np.mean(X, axis=0, keepdims=True)) / (np.std(X, axis=0, keepdims=True) + 1e-12)

    # Flatten each sample
    X = X.reshape(X.shape[0], -1)

    # Impute missing values (mean)
    X = SimpleImputer(strategy="mean").fit_transform(X)

    return X, y_class, y_subject


# =====================================
# 2. Subject-wise Split
# =====================================
def subject_split(X, y_class, y_subject, n_test=5, random_state=None):
    np.random.seed(random_state)
    subjects = np.unique(y_subject)
    test_subjects = np.random.choice(subjects, size=n_test, replace=False)
    train_subjects = np.setdiff1d(subjects, test_subjects)

    idx_train = np.isin(y_subject, train_subjects)
    idx_test = np.isin(y_subject, test_subjects)

    return X[idx_train], y_class[idx_train], X[idx_test], y_class[idx_test], test_subjects


# =====================================
# 3. Train & Evaluate (Single Fold)
# =====================================
def train_and_evaluate(X_train, y_train, X_test, y_test, classifiers):
    results = {"accuracy": {}, "f1": {}}
    for clf_name, clf in classifiers.items():
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        results["accuracy"][clf_name] = accuracy_score(y_test, y_pred)
        results["f1"][clf_name] = f1_score(y_test, y_pred, average="weighted")
    return results


# =====================================
# 4. Repeated Cross-Validation (Mean ± STD)
# =====================================
def repeated_subject_cv(X, y_class, y_subject, classifiers, n_repeats=10, n_test=5):
    metrics_summary = {clf_name: {"acc": [], "f1": []} for clf_name in classifiers.keys()}

    for repeat in range(n_repeats):
        random_state = 42 + repeat
        X_train, y_train, X_test, y_test, test_subjects = subject_split(X, y_class, y_subject, n_test, random_state)
        fold_results = train_and_evaluate(X_train, y_train, X_test, y_test, classifiers)

        for clf_name in classifiers.keys():
            metrics_summary[clf_name]["acc"].append(fold_results["accuracy"][clf_name])
            metrics_summary[clf_name]["f1"].append(fold_results["f1"][clf_name])

    # Compute mean ± std
    summary = {}
    for clf_name in classifiers.keys():
        acc_mean, acc_std = np.mean(metrics_summary[clf_name]["acc"]), np.std(metrics_summary[clf_name]["acc"])
        f1_mean, f1_std = np.mean(metrics_summary[clf_name]["f1"]), np.std(metrics_summary[clf_name]["f1"])
        summary[clf_name] = {"acc_mean": acc_mean, "acc_std": acc_std, "f1_mean": f1_mean, "f1_std": f1_std}

    return summary


# =====================================
# 5. Visualization Functions
# =====================================
def plot_radar(results, datasets, classifiers):
    N = len(datasets)
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    plt.figure(figsize=(8,8))
    for clf_name in classifiers.keys():
        values = [results[d][clf_name]["acc_mean"] for d in datasets]
        values += values[:1]
        plt.polar(angles, values, marker="o", label=clf_name)

    plt.xticks(angles[:-1], datasets)
    plt.yticks([0.2,0.4,0.6,0.8,1.0], color="grey", size=10)
    plt.ylim(0,1)
    plt.title("Radar Plot (Mean Accuracy ± STD over 10 Random Splits)", size=14, weight="bold")
    plt.legend(loc="upper right", bbox_to_anchor=(1.3,1.1))
    plt.show()


# =====================================
# 6. Main Execution
# =====================================
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
    summary = repeated_subject_cv(X, y_class, y_subject, classifiers, n_repeats=10, n_test=5)
    all_results[d_name] = summary

    for clf_name, stats in summary.items():
        print(f"{d_name} - {clf_name}: "
              f"Acc = {stats['acc_mean']:.3f} ± {stats['acc_std']:.3f}, "
              f"F1 = {stats['f1_mean']:.3f} ± {stats['f1_std']:.3f}")

# =====================================
# 7. Plot Mean Accuracy Radar
# =====================================
plot_radar(all_results, datasets.keys(), classifiers)
