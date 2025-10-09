import numpy as np
from sklearn.utils import shuffle
import matplotlib as mpl
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization, GRU, Bidirectional
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from keras.utils import to_categorical

# ===============================
# 1. Load Data
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
# 2. Model Definition
# ===============================
def build_complex_gru_model(input_shape, n_classes):
    model = Sequential()

    # GRU layers
    model.add(Bidirectional(GRU(units=128, return_sequences=True), input_shape=input_shape))
    model.add(Dropout(0.3))
    model.add(BatchNormalization())

    model.add(Bidirectional(GRU(units=256, return_sequences=True)))
    model.add(Dropout(0.4))
    model.add(BatchNormalization())

    model.add(Bidirectional(GRU(units=256)))
    model.add(Dropout(0.4))
    model.add(BatchNormalization())

    # Dense layers
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.4))
    model.add(BatchNormalization())

    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.3))
    model.add(BatchNormalization())

    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.3))

    # Output layer
    model.add(Dense(units=n_classes, activation='softmax'))

    # Compile
    opt = Adam(learning_rate=0.001)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# ===============================
# 3. Cross-Validation Setup
# ===============================
n_folds = 10
all_subjects = np.unique(y[:, 0])
n_subjects = len(all_subjects)

# Early stopping
early_stop = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, verbose=1)

fold_accuracies = []

# Plot settings
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
for fold in range(n_folds):
    seed = np.random.randint(0, 10000)  # random seed per fold
    np.random.seed(seed)
    tf.random.set_seed(seed)

    np.random.shuffle(all_subjects)

    n_test = int(0.2 * n_subjects)
    n_val = int(0.1 * n_subjects)

    test_subjects = all_subjects[:n_test]
    val_subjects = all_subjects[n_test:n_test + n_val]
    train_subjects = all_subjects[n_test + n_val:]

    print(f"\n=== Fold {fold+1} === (Seed={seed})")
    print(f"Train subjects: {len(train_subjects)}, Val: {len(val_subjects)}, Test: {len(test_subjects)}")

    # Function to extract subject data
    def get_subject_data(X, y, subjects):
        idx = np.isin(y[:, 0], subjects)
        X_subj = X[idx]
        y_subj = y[idx, 1]
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
    def standardize(train, val, test):
        mean = np.mean(train, axis=0, keepdims=True)
        std = np.std(train, axis=0, keepdims=True)
        std[std == 0] = 1e-8
        X_train = (train - mean) / std
        X_val = (val - mean) / std
        X_test = (test - mean) / std
        return np.nan_to_num(X_train), np.nan_to_num(X_val), np.nan_to_num(X_test)

    X_train, X_val, X_test = standardize(X_train, X_val, X_test)

    # ===============================
    # 5. Build & Train Model
    # ===============================
    model = build_complex_gru_model((X_train.shape[1], X_train.shape[2]), n_classes)

    history = model.fit(
        X_train, y_train,
        epochs=200,
        batch_size=32,
        validation_data=(X_val, y_val),
        callbacks=[early_stop],
        verbose=1
    )

    # Evaluate on test set
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Fold {fold+1} Test Accuracy: {test_acc:.4f}")
    fold_accuracies.append(test_acc)

# ===============================
# 6. Results
# ===============================
print("\nCross-validation results:")
for i, acc in enumerate(fold_accuracies):
    print(f"Fold {i+1}: {acc:.4f}")
print(f"Mean Accuracy: {np.mean(fold_accuracies):.4f} ± {np.std(fold_accuracies):.4f}")
