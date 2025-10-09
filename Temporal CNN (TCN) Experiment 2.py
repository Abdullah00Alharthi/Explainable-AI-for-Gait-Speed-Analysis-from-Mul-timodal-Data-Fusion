"""
Gait Classification using Multi-Layer Temporal CNN (TCN) 
with 5-Fold Subject-Wise Cross-Validation
----------------------------------------------------------
This script loads gait-related data, performs subject-wise cross-validation,
trains a multi-layer TCN model with early stopping, evaluates performance,
and reports results.
"""

# ===============================
# 1. Import Required Libraries
# ===============================
import numpy as np
import matplotlib.pyplot as plt
import time

from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping

from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix, classification_report

# Import TCN layer
from tcn import TCN  

# ===============================
# 2. Load Dataset
# ===============================
# X = np.load('MARKERS_DATA.npy')
X = np.load('GRF_DATA.npy')
# X = np.load('EMG_DATA.npy')
# X = np.load('F_M_DATA.npy')
y1 = np.load('ID_L.npy')         # subject IDs
y2 = np.load('SPEED_L.npy')      # sample class labels

# Replace NaN with zeros
X = np.nan_to_num(X)

# Optional: relabel classes (adjust as needed)
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
n_folds = 5
n_test_subjects = 5   # fixed number of test subjects per fold
seed = 42
np.random.seed(seed)

all_subjects = np.unique(y[:, 0])
np.random.shuffle(all_subjects)

# Early stopping
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=20,
    restore_best_weights=True,
    verbose=1
)

# Store fold accuracies
fold_accuracies = []

# ===============================
# 4. Cross-Validation Loop
# ===============================
for fold in range(n_folds):
    print(f"\n=== Fold {fold+1} ===")

    # Randomly select 5 test subjects per fold
    if len(all_subjects) >= n_test_subjects:
        test_subjects = np.random.choice(all_subjects, n_test_subjects, replace=False)
    else:
        test_subjects = all_subjects

    # Remaining subjects for train + val
    remaining_subjects = np.setdiff1d(all_subjects, test_subjects)
    np.random.shuffle(remaining_subjects)

    # Split remaining into training and validation (approx 80/20)
    n_val = max(1, len(remaining_subjects) // 5)
    val_subjects = remaining_subjects[:n_val]
    train_subjects = remaining_subjects[n_val:]

    print(f"Test Subjects (Fold {fold+1}): {test_subjects}")

    # Function to get subject-specific data
    def get_subject_data(X, y, subjects):
        idx = np.isin(y[:, 0], subjects)
        X_subj = X[idx]
        y_subj = y[idx, 1]  # class labels
        return X_subj, y_subj

    X_train, y_train = get_subject_data(X, y, train_subjects)
    X_val, y_val = get_subject_data(X, y, val_subjects)
    X_test, y_test = get_subject_data(X, y, test_subjects)

    # Shuffle
    X_train, y_train = shuffle(X_train, y_train, random_state=seed)
    X_val, y_val = shuffle(X_val, y_val, random_state=seed)
    X_test, y_test = shuffle(X_test, y_test, random_state=seed)

    # One-hot encoding
    y_train = to_categorical(y_train, num_classes=n_classes)
    y_val = to_categorical(y_val, num_classes=n_classes)
    y_test = to_categorical(y_test, num_classes=n_classes)

    # Standardization
    def standardize(train, test):
        X_train = (train - np.mean(train, axis=0)[None, :, :]) / np.std(train, axis=0)[None, :, :]
        X_test = (test - np.mean(test, axis=0)[None, :, :]) / np.std(test, axis=0)[None, :, :]
        return np.nan_to_num(X_train), np.nan_to_num(X_test)

    X_train, X_val = standardize(X_train, X_val)
    X_train, X_test = standardize(X_train, X_test)

    # ===============================
    # Build Multi-Layer Temporal CNN (TCN) Model
    # ===============================
    model = Sequential()

    # First TCN block
    model.add(TCN(nb_filters=12, kernel_size=5, dilations=[1, 2, 4],
                  padding='causal', activation='relu', dropout_rate=0.1,
                  return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(BatchNormalization())

    # Second TCN block
    model.add(TCN(nb_filters=24, kernel_size=5, dilations=[1, 2, 4, 8],
                  padding='causal', activation='relu', dropout_rate=0.1,
                  return_sequences=True))
    model.add(BatchNormalization())

    # Third TCN block
    model.add(TCN(nb_filters=48, kernel_size=5, dilations=[1, 2, 4, 8, 16],
                  padding='causal', activation='relu', dropout_rate=0.1,
                  return_sequences=True))
    model.add(BatchNormalization())

    # Fourth TCN block (final, no sequences returned)
    model.add(TCN(nb_filters=96, kernel_size=5, dilations=[1, 2, 4, 8, 16, 32],
                  padding='causal', activation='relu', dropout_rate=0.2,
                  return_sequences=False))
    model.add(BatchNormalization())

    # Fully connected layers
    model.add(Dropout(0.5))
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(n_classes, activation='softmax'))

    # Compile model
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
    print(f"Test Accuracy (Fold {fold+1}): {acc:.4f}")
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
