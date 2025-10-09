"""
Gait Classification using Multi-Input CNN with 5-Fold Subject-Wise Cross-Validation
-----------------------------------------------------------------------------------
This script loads gait-related data, performs subject-wise cross-validation
with fixed split sizes (35 train, 5 val, 10 test per fold),
trains a multi-input CNN model with early stopping, evaluates performance,
and reports results.
"""

# ===============================
# 1. Import Required Libraries
# ===============================
import numpy as np
import matplotlib.pyplot as plt
import time

from keras.utils import to_categorical
from keras.models import Model
from keras.layers import Dense, Dropout, Flatten, Convolution1D, AveragePooling1D, MaxPooling1D, Input, concatenate
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from sklearn.utils import shuffle

# ===============================
# 2. Load Dataset
# ===============================
# ===================================
# 1. Import Libraries
# ===================================
import numpy as np
import time
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from sklearn.utils import shuffle

# ===================================
# 2. Load Data
# ===================================
X1 = np.load('MARKERS_DATA.npy')
X2 = np.load('GRF_DATA.npy')
X3 = np.load('EMG_DATA.npy')
X4 = np.load('F_M_DATA.npy')

y1 = np.load("ID_L.npy")     # subject IDs
y2 = np.load("SPEED_L.npy")  # class labels

# Replace NaNs with zeros
X1, X2, X3, X4 = map(np.nan_to_num, [X1, X2, X3, X4])

# Optional: relabel classes 3->2, 4->3
for i in range(len(y2)):
    if y2[i] == 3:
        y2[i] = 2
    elif y2[i] == 4:
        y2[i] = 3

n_classes = 4
y = np.column_stack((y1, y2))  # y[:,0]=subject ID, y[:,1]=class

# ===================================
# 3. Cross-Validation Setup
# ===================================
n_folds = 5
seed = 42
np.random.seed(seed)

all_subjects = np.unique(y[:, 0])
np.random.shuffle(all_subjects)
test_folds = np.array_split(all_subjects, n_folds)

early_stop = EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True, verbose=1)
fold_accuracies = []

# ===================================
# 4. Standardization Function
# ===================================
def standardize(train, val, test):
    mean = np.mean(train, axis=0, keepdims=True)
    std  = np.std(train, axis=0, keepdims=True) + 1e-12
    return (train - mean)/std, (val - mean)/std, (test - mean)/std

# ===================================
# 5. Cross-Validation Loop
# ===================================
for fold in range(n_folds):
    print(f"\n=== Fold {fold+1} ===")

    test_subjects = test_folds[fold]
    remaining_subjects = np.setdiff1d(all_subjects, test_subjects)
    np.random.shuffle(remaining_subjects)

    # Example: 5 val subjects, rest train
    val_subjects = remaining_subjects[:5]
    train_subjects = remaining_subjects[5:]

    # Function to get subject-specific data
    def get_subject_data(X, y, subjects):
        idx = np.isin(y[:, 0], subjects)
        return X[idx], y[idx, 1]

    # Extract data for all feature sets
    X1_train, y_train = get_subject_data(X1, y, train_subjects)
    X2_train, _       = get_subject_data(X2, y, train_subjects)
    X3_train, _       = get_subject_data(X3, y, train_subjects)
    X4_train, _       = get_subject_data(X4, y, train_subjects)

    X1_val, y_val     = get_subject_data(X1, y, val_subjects)
    X2_val, _         = get_subject_data(X2, y, val_subjects)
    X3_val, _         = get_subject_data(X3, y, val_subjects)
    X4_val, _         = get_subject_data(X4, y, val_subjects)

    X1_test, y_test   = get_subject_data(X1, y, test_subjects)
    X2_test, _        = get_subject_data(X2, y, test_subjects)
    X3_test, _        = get_subject_data(X3, y, test_subjects)
    X4_test, _        = get_subject_data(X4, y, test_subjects)

    # Shuffle **all together** to keep alignment
    idx = np.arange(len(X1_train))
    np.random.shuffle(idx)
    X1_train, X2_train, X3_train, X4_train, y_train = \
        X1_train[idx], X2_train[idx], X3_train[idx], X4_train[idx], y_train[idx]

    # One-hot encode labels
    y_train = to_categorical(y_train, num_classes=n_classes)
    y_val   = to_categorical(y_val, num_classes=n_classes)
    y_test  = to_categorical(y_test, num_classes=n_classes)

    # Standardize using train stats
    X1_train, X1_val, X1_test = standardize(X1_train, X1_val, X1_test)
    X2_train, X2_val, X2_test = standardize(X2_train, X2_val, X2_test)
    X3_train, X3_val, X3_test = standardize(X3_train, X3_val, X3_test)
    X4_train, X4_val, X4_test = standardize(X4_train, X4_val, X4_test)

    print(f"Fold {fold+1} shapes:")
    print(f"X1_train: {X1_train.shape}, y_train: {y_train.shape}")
    print(f"X1_val:   {X1_val.shape},   y_val:   {y_val.shape}")
    print(f"X1_test:  {X1_test.shape},  y_test:  {y_test.shape}")

    # Here you can build your model (CNN, LSTM, or multi-input) using X1_train,X2_train,X3_train,X4_train
    # and y_train. The same for validation and test.



    # ===============================
    # 5. Build CNN Model
    # ===============================
    def build_model(input_shapes, n_classes):
        inp1 = Input(shape=input_shapes[0])
        inp2 = Input(shape=input_shapes[1])
        inp3 = Input(shape=input_shapes[2])
        inp4 = Input(shape=input_shapes[3])

        # Branch 1
        t1 = Convolution1D(6, 2, activation='relu', padding='same')(inp1)
        t1 = AveragePooling1D(2)(t1)
        t1 = Convolution1D(12, 2, activation='relu', padding='same')(t1)
        t1 = AveragePooling1D(2)(t1)

        # Branch 2
        t2 = Convolution1D(6, 2, activation='relu', padding='same')(inp2)
        t2 = AveragePooling1D(2)(t2)
        t2 = Convolution1D(12, 2, activation='relu', padding='same')(t2)
        t2 = AveragePooling1D(2)(t2)

        # Branch 3
        t3 = Convolution1D(6, 2, activation='relu', padding='same')(inp3)
        t3 = AveragePooling1D(2)(t3)
        t3 = Convolution1D(12, 2, activation='relu', padding='same')(t3)
        t3 = AveragePooling1D(2)(t3)

        # Branch 4
        t4 = Convolution1D(12, 2, activation='relu', padding='same')(inp4)
        t4 = MaxPooling1D(2)(t4)
        t4 = Convolution1D(12, 2, activation='relu', padding='same')(t4)
        t4 = AveragePooling1D(2)(t4)

        # Merge branches
        t123 = concatenate([t1, t2, t3], axis=1)

        t5 = Convolution1D(24, 2, activation='relu', padding='same')(t123)
        t5 = AveragePooling1D(2)(t5)
        t5 = Convolution1D(48, 2, activation='relu', padding='same')(t5)
        t5 = AveragePooling1D(2)(t5)

        t4 = Convolution1D(24, 2, activation='relu', padding='same')(t4)
        t4 = MaxPooling1D(2)(t4)
        t4 = Convolution1D(48, 2, activation='relu', padding='same')(t4)
        t4 = MaxPooling1D(2)(t4)

        merged = concatenate([t5, t4], axis=1)

        out = Flatten()(merged)
        out = Dropout(0.5)(out)
        out = Dense(100, activation='relu')(out)
        out = Dropout(0.2)(out)
        out = Dense(n_classes, activation='softmax')(out)

        return Model(inputs=[inp1, inp2, inp3, inp4], outputs=out)

    model = build_model(
        [X1_train.shape[1:], X2_train.shape[1:], X3_train.shape[1:], X4_train.shape[1:]],
        n_classes
    )

    opt = Adam(learning_rate=0.002)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    # ===============================
    # 6. Train Model
    # ===============================
    start_time = time.time()
    history = model.fit(
        [X1_train, X2_train, X3_train, X4_train], y_train,
        validation_data=([X1_val, X2_val, X3_val, X4_val], y_val),
        epochs=1000,
        batch_size=100,
        callbacks=[early_stop],
        verbose=1
    )
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")

    # ===============================
    # 7. Evaluate
    # ===============================
    y_pred = model.predict([X1_test, X2_test, X3_test, X4_test])
    acc = np.mean(np.argmax(y_pred, axis=1) == np.argmax(y_test, axis=1))
    print(f"Fold {fold+1} Test Accuracy: {acc:.4f}")
    fold_accuracies.append(acc)

# ===============================
# 8. Report Results
# ===============================
print("\n=== Cross-Validation Results ===")
for i, acc in enumerate(fold_accuracies):
    print(f"Fold {i+1} Accuracy: {acc:.4f}")
print(f"Mean Accuracy: {np.mean(fold_accuracies):.4f}, Std: {np.std(fold_accuracies):.4f}")
