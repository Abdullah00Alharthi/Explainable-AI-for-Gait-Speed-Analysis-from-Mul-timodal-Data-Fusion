import warnings
warnings.simplefilter('ignore')

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import keras
from keras import backend as K
from keras.models import load_model
from keras.optimizers import Adam
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import innvestigate
import innvestigate.utils as iutils
from innvestigate.tools import Perturbation, PerturbationAnalysis
from matplotlib import rcParams
from scipy.ndimage import gaussian_filter1d

# ==========================
# Configuration
# ==========================
MODEL_DIR = "."   # change if your .h5 files are in a subfolder, e.g. "models"
STEPS = 100
REGION_SHAPE = (14, 14)
BATCH_SIZE = 1
ADAM_LR = 0.002
SAVE_HEATMAP = "lrp_heatmap.png"
SAVE_LINEFIG = "lrp_lines.png"

# ==========================
# Utility functions
# ==========================
def safe_model_path(template: str, modality: str) -> str:
    """Return full model file path given template and modality name.
       Keeps a couple of common replacements stable (F&M -> FM, remove '&')."""
    # Normalize modality string to match filenames saved on disk
    norm = modality.replace("&", "").replace(" ", "").replace("/", "_")
    filename = template.format(norm)
    return os.path.join(MODEL_DIR, filename)


def load_and_prepare_data(file_path: str, shape: tuple, labels: np.ndarray):
    """Load, reshape, split, and standardize dataset."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found: {file_path}")

    X = np.load(file_path).reshape(shape)

    # copy and remap labels as in your original code
    y = labels.copy()
    y[y == 3] = 2
    y[y == 4] = 3

    # train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    # standardize by train mean/std
    mean = np.mean(X_train, axis=0, keepdims=True)
    std = np.std(X_train, axis=0, keepdims=True)
    # avoid division by zero
    std[std == 0] = 1.0
    X_train = (X_train - mean) / std
    X_test  = (X_test  - mean) / std

    return np.nan_to_num(X_train), np.nan_to_num(X_test), y_train, y_test


def perform_lrp_analysis(model_path: str, X_test: np.ndarray, y_test: np.ndarray,
                         region_shape=REGION_SHAPE, steps=STEPS):
    """Perform LRP perturbation analysis on a single model and return accuracy scores array.
       Returns None on failure."""
    # safety: check file exists
    if not os.path.exists(model_path):
        print(f"⚠️ Model file not found: {model_path}")
        return None

    # clear session to avoid duplicate layer-name errors carried over from previous models
    K.clear_session()

    try:
        model = load_model(model_path)
    except ValueError as ex:
        # Common cause: duplicate layer names or corrupted model file
        print(f"⚠️ Failed loading {os.path.basename(model_path)} — ValueError: {ex}")
        print("   -> Common causes: duplicate layer names inside the saved model or incompatible Keras version.")
        print("   Suggestion: re-save the model with unique layer names (or rebuild and save with model.save(...)).")
        return None
    except OSError as ex:
        print(f"⚠️ Failed opening {os.path.basename(model_path)} — OSError: {ex}")
        return None
    except Exception as ex:
        print(f"⚠️ Unexpected error loading {os.path.basename(model_path)}: {ex}")
        return None

    # compile with older-compatible Adam signature
    try:
        opt = Adam(lr=ADAM_LR)
    except TypeError:
        opt = Adam(**({'learning_rate': ADAM_LR} if 'learning_rate' in Adam.__init__.__code__.co_varnames else {'lr': ADAM_LR}))

    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    # Prepare generator and analyzer
    generator = iutils.BatchSequence([X_test, y_test], batch_size=BATCH_SIZE)
    try:
        model_wo_softmax = iutils.keras.graph.model_wo_softmax(model)
    except Exception as e:
        print(f"⚠️ Could not create model_wo_softmax for {os.path.basename(model_path)}: {e}")
        return None

    try:
        analyzer = innvestigate.create_analyzer("lrp.sequential_preset_a_flat", model_wo_softmax)
    except Exception as e:
        print(f"⚠️ innvestigate analyzer creation failed for {os.path.basename(model_path)}: {e}")
        return None

    perturbation = Perturbation("gaussian", region_shape=region_shape, in_place=False)

    try:
        analysis = PerturbationAnalysis(
            analyzer, model, generator, perturbation,
            recompute_analysis=False, steps=steps, regions_per_step=1, verbose=False
        )
        scores = np.array(analysis.compute_perturbation_analysis())
    except Exception as e:
        print(f"⚠️ Perturbation analysis failed for {os.path.basename(model_path)}: {e}")
        return None

    # validate scores shape and return accuracy column if available
    if scores is None or scores.size == 0:
        print(f"⚠️ Perturbation returned empty scores for {os.path.basename(model_path)}")
        return None

    # If the second column is accuracy per your previous structure (index 1)
    if scores.ndim == 2 and scores.shape[1] > 1:
        return scores[:, 1]
    # otherwise return flattened or first column as fallback
    return scores.ravel()


# ==========================
# Main workflow
# ==========================
def main():
    # load labels
    labels_path = "SPEED_L.npy"
    if not os.path.exists(labels_path):
        raise FileNotFoundError(f"Labels file not found: {labels_path}")
    y = np.load(labels_path)

    datasets = {
        'Markers': ('MARKERS_DATA.npy', (1142, 585, 156, 1)),
        'EMG':     ('EMG_DATA.npy',     (1142, 8985, 8, 1)),
        'GRF':     ('GRF_DATA.npy',     (1142, 585, 24, 1)),
        'F&M':     ('F_M_DATA.npy',     (1142, 8985, 12, 1))
    }

    # preprocess datasets (store train/test but we'll use test in LRP)
    processed_data = {}
    for name, (path, shape) in datasets.items():
        full_path = os.path.join(".", path)
        try:
            processed_data[name] = load_and_prepare_data(full_path, shape, y)
        except Exception as e:
            print(f"⚠️ Skipping dataset {name} because data loading failed: {e}")

    # model filename templates (adjust if your files are in a subfolder or have different naming)
    model_variants = {
        'Single CNN': '3DCNN_{}.h5',
        'Dual CNN':   '3DCNN+3DCNN_{}.h5',
        'Quads CNN':  '4_3DCNN_{}.h5'
    }

    # Run analysis
    results = {}
    for modality, data_tuple in processed_data.items():
        X_train, X_test, y_train, y_test = data_tuple
        results[modality] = {}
        for variant, template in model_variants.items():
            model_path = safe_model_path(template, modality)
            print(f"Analyzing {variant} for {modality} -> {os.path.basename(model_path)}")
            scores = perform_lrp_analysis(model_path, X_test, y_test, region_shape=REGION_SHAPE, steps=STEPS)
            if scores is None:
                print(f"⚠️ Skipping {variant} {modality} (no valid scores).")
            else:
                results[modality][variant] = scores

    # ==========================
    # Build heatmap data (skip empty entries)
    # ==========================
    model_names = []
    heatmap_rows = []
    for modality, variants in results.items():
        for variant, scores in variants.items():
            if scores is None or len(scores) == 0:
                continue
            model_names.append(f"{variant} {modality}")
            heatmap_rows.append(scores)

    if len(heatmap_rows) == 0:
        raise RuntimeError("No valid perturbation results available for plotting. Check earlier warnings.")

    heatmap_data = np.array(heatmap_rows)

    # Normalize each row for comparability (preserve sign)
    row_max = np.max(np.abs(heatmap_data), axis=1, keepdims=True)
    row_max[row_max == 0] = 1.0
    norm_heatmap = heatmap_data / (row_max + 1e-12)

    # Plot heatmap
    plt.figure(figsize=(14, 7), dpi=150)
    rcParams.update({'font.family': 'Arial', 'font.size': 14, 'font.weight': 'bold'})
    vmax = np.max(np.abs(norm_heatmap))
    vmin = -vmax

    ax = sns.heatmap(
        norm_heatmap,
        cmap="RdYlBu_r",
        vmin=vmin, vmax=vmax,
        center=0,
        cbar_kws={'label': 'Normalized Relevance Impact', 'shrink': 0.8, 'aspect': 20},
        linewidths=1.2, linecolor='white', annot=False
    )
    ax.set_title("Normalized LRP Perturbation Impact by CNN Model", fontsize=18, pad=20)
    ax.set_xlabel("Perturbation Steps", fontsize=14, labelpad=15)
    ax.set_ylabel("CNN Models", fontsize=14, labelpad=15)

    # x ticks every 5 steps
    x_steps = np.arange(0, norm_heatmap.shape[1], 5)
    ax.set_xticks(x_steps + 0.5)
    ax.set_xticklabels([str(i) for i in x_steps], rotation=0)
    ax.set_yticks(np.arange(len(model_names)) + 0.5)
    ax.set_yticklabels(model_names, rotation=0)

    plt.tight_layout()
    plt.savefig(SAVE_HEATMAP, dpi=300)
    print(f"Saved heatmap to {SAVE_HEATMAP}")
    plt.show()

    # ==========================
    # Smoothed line plot of accuracy (one figure)
    # ==========================
    plt.figure(figsize=(14, 8))
    line_styles = ['-', '--', '-.', ':']
    colors = ['#E74C3C', '#27AE60', '#2980B9', '#8E44AD', '#E67E22', '#16A085', '#34495E', '#7F8C8D']

    for d_index, (modality, variants) in enumerate(results.items()):
        for m_index, (variant, scores) in enumerate(variants.items()):
            if scores is None or len(scores) == 0:
                continue
            smoothed = gaussian_filter1d(scores, sigma=2)
            label = f"{modality} - {variant}"
            color = colors[(d_index * len(variants) + m_index) % len(colors)]
            style = line_styles[m_index % len(line_styles)]
            plt.plot(smoothed, label=label, color=color, linestyle=style, linewidth=2)

    plt.title("Smoothed Perturbation Accuracy Curves", fontsize=20, fontname="Arial")
    plt.xlabel("Perturbation Steps", fontsize=18, fontname="Arial")
    plt.ylabel("Accuracy (smoothed)", fontsize=18, fontname="Arial")
    plt.legend(title='CNN Models:', fontsize=10, loc='lower right')
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(SAVE_LINEFIG, dpi=300)
    print(f"Saved line figure to {SAVE_LINEFIG}")
    plt.show()


if __name__ == "__main__":
    main()
