"""
Gait Classification using 1D CNN with 5-Fold Subject-Wise Cross-Validation
------------------------------------------------------------------------
This script loads gait-related data, performs subject-wise cross-validation,
trains a CNN model with early stopping, evaluates performance, and visualizes
results including confusion matrices, precision-recall curves, radar charts,
and t-SNE embeddings of classified samples.
"""

# ===============================
# 1. Import Required Libraries
# ===============================
import numpy as np
import matplotlib.pyplot as plt
import itertools
import time
from math import pi

from keras.utils import to_categorical
from keras.models import Sequential, Model
from keras.layers import Dense, Flatten, Dropout, BatchNormalization
from keras.layers import Convolution1D, MaxPooling1D
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping

from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.manifold import TSNE

# ===============================
# 2. Load Dataset
# ===============================
# X = np.load('MARKERS_DATA.npy')
# X = np.load('GRF_DATA.npy')
# X = np.load('EMG_DATA.npy')
X = np.load('F_M_DATA.npy')
y1 = np.load('ID_L.npy')         # subject IDs
y2 = np.load('SPEED_L.npy')      # sample class labels

# Replace NaN with zeros
X = np.nan_to_num(X)

# Optional: relabel classes
for i in range(len(y2)):
    if np.all(y2[i] == 3):
        y2[i] = 2
    elif np.all(y2[i] == 4):
        y2[i] = 3

n_classes = 4
y = np.column_stack((y1, y2))

# ===============================
# 3. Cross-Validation Setup
# ===============================
from sklearn.model_selection import KFold

n_folds = 5
seed = 42
np.random.seed(seed)

all_subjects = np.unique(y[:, 0])
np.random.shuffle(all_subjects)

# Define early stopping callback
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=20,
    restore_best_weights=True,
    verbose=1
)

# Store fold accuracies
fold_accuracies = []

# Set plotting style
import matplotlib as mpl
mpl.rcParams['font.family'] = 'Arial'
mpl.rcParams['font.size'] = 12
mpl.rcParams['axes.titlesize'] = 16
mpl.rcParams['axes.labelsize'] = 14
mpl.rcParams['legend.fontsize'] = 12
mpl.rcParams['xtick.labelsize'] = 12
mpl.rcParams['ytick.labelsize'] = 12

# ===============================
# 4. Cross-Validation Loop
# ===============================
fold_size = len(all_subjects) // n_folds

for fold in range(n_folds):
    print(f"\n=== Fold {fold+1} ===")
    
    # Define test subjects for this fold
    test_subjects = all_subjects[fold*fold_size:(fold+1)*fold_size]
    
    # Remaining subjects for train + val
    remaining_subjects = np.setdiff1d(all_subjects, test_subjects)
    np.random.shuffle(remaining_subjects)
    
    # Split remaining into training and validation (approx 80/20)
    n_val = max(1, len(remaining_subjects)//5)
    val_subjects = remaining_subjects[:n_val]
    train_subjects = remaining_subjects[n_val:]
    
    # Function to get subject-specific data
    def get_subject_data(X, y, subjects):
        idx = np.isin(y[:,0], subjects)
        X_subj = X[idx]
        y_subj = y[idx,1]  # class labels
        return X_subj, y_subj
    
    X_train, y_train = get_subject_data(X, y, train_subjects)
    X_val, y_val     = get_subject_data(X, y, val_subjects)
    X_test, y_test   = get_subject_data(X, y, test_subjects)
    
    # Shuffle
    X_train, y_train = shuffle(X_train, y_train, random_state=seed)
    X_val, y_val     = shuffle(X_val, y_val, random_state=seed)
    X_test, y_test   = shuffle(X_test, y_test, random_state=seed)
    
    # One-hot encoding
    y_train = to_categorical(y_train, num_classes=n_classes)
    y_val   = to_categorical(y_val, num_classes=n_classes)
    y_test  = to_categorical(y_test, num_classes=n_classes)
    
    # Standardization
    def standardize(train, test):
        X_train = (train - np.mean(train, axis=0)[None,:,:]) / np.std(train, axis=0)[None,:,:]
        X_test  = (test - np.mean(test, axis=0)[None,:,:]) / np.std(test, axis=0)[None,:,:]
        return np.nan_to_num(X_train), np.nan_to_num(X_test)
    
    X_train, X_val = standardize(X_train, X_val)
    X_train, X_test = standardize(X_train, X_test)
    
    # ===============================
    # Build CNN Model
    # ===============================
    model = Sequential()
    model.add(Convolution1D(filters=12, kernel_size=5, strides=1, padding='same',
                            activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(MaxPooling1D(pool_size=2, strides=2, padding='same'))
    model.add(Convolution1D(filters=24, kernel_size=5, strides=1, activation='relu', padding='same'))
    model.add(MaxPooling1D(pool_size=2, strides=2, padding='same'))
    model.add(Convolution1D(filters=48, kernel_size=5, strides=1, activation='relu', padding='same'))
    model.add(MaxPooling1D(pool_size=2, strides=2, padding='same'))
    model.add(Convolution1D(filters=96, kernel_size=5, strides=1, activation='relu', padding='same'))
    model.add(MaxPooling1D(pool_size=2, strides=2, padding='same'))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(50, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(n_classes, activation='softmax'))
    
    opt = Adam(learning_rate=0.002)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    
    # ===============================
    # Train Model with Early Stopping
    # ===============================
    start_time = time.time()
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=1000,
        batch_size=100,
        callbacks=[early_stop],
        verbose=1
    )
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    
    # ===============================
    # Evaluate on Test Set
    # ===============================
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y_test, axis=1)
    
    acc = np.mean(y_pred_classes == y_true)
    print(f"Test Accuracy: {acc:.4f}")
    fold_accuracies.append(acc)
    
# ===============================
# 5. Report Cross-Validation Results
# ===============================
mean_acc = np.mean(fold_accuracies)
std_acc = np.std(fold_accuracies)
print("\n=== Cross-Validation Results ===")
for i, acc in enumerate(fold_accuracies):
    print(f"Fold {i+1} Accuracy: {acc:.4f}")
print(f"Mean Accuracy: {mean_acc:.4f}, Std: {std_acc:.4f}")


# ===============================
# Additions: Visualization Utils
# ===============================
def plot_training_history(history, fold):
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.title(f'Fold {fold+1} - Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title(f'Fold {fold+1} - Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(cm, classes, fold,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    plt.figure(figsize=(6,6))
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(f'{title} - Fold {fold+1}')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()


def plot_tsne(features, labels, fold):
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    reduced = tsne.fit_transform(features)

    plt.figure(figsize=(8,6))
    scatter = plt.scatter(reduced[:,0], reduced[:,1], c=labels, cmap='tab10', alpha=0.7)
    plt.legend(*scatter.legend_elements(), title="Classes")
    plt.title(f't-SNE of Predicted Classes - Fold {fold+1}')
    plt.show()


