"""
Gait Classification using Transformer with 5-Fold Subject-Wise Cross-Validation
-------------------------------------------------------------------------------
This script loads gait-related data, performs subject-wise cross-validation,
trains a Transformer model with early stopping, evaluates performance, and reports
fold-wise and mean accuracy.
"""

# ===============================
# 1. Import Required Libraries
# ===============================
import numpy as np
import time

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, LayerNormalization
from tensorflow.keras.layers import MultiHeadAttention, GlobalAveragePooling1D, Add
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.utils import shuffle

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
# 3. Transformer Encoder Block
# ===============================
def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0.1):
    # Multi-Head Attention
    x = MultiHeadAttention(num_heads=num_heads, key_dim=head_size, dropout=dropout)(inputs, inputs)
    x = Dropout(dropout)(x)
    x = Add()([x, inputs])
    x = LayerNormalization(epsilon=1e-6)(x)
    
    # Feed-Forward Network
    x_ff = Dense(ff_dim, activation='relu')(x)
    x_ff = Dense(inputs.shape[-1])(x_ff)
    x_ff = Dropout(dropout)(x_ff)
    x = Add()([x, x_ff])
    x = LayerNormalization(epsilon=1e-6)(x)
    return x

# ===============================
# 4. Build Transformer Model
# ===============================
def build_transformer_model(input_shape, num_classes, head_size=64, num_heads=4, ff_dim=128, num_blocks=2, dropout=0.1):
    inputs = Input(shape=input_shape)
    x = inputs
    
    for _ in range(num_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)
    
    x = GlobalAveragePooling1D()(x)
    x = Dropout(0.5)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.2)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs, outputs)
    return model

# ===============================
# 5. Cross-Validation Setup
# ===============================
from sklearn.model_selection import KFold

n_folds = 5
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

fold_accuracies = []

# ===============================
# 6. Cross-Validation Loop
# ===============================
fold_size = len(all_subjects) // n_folds

for fold in range(n_folds):
    print(f"\n=== Fold {fold+1} ===")
    
    # Define test subjects
    test_subjects = all_subjects[fold*fold_size:(fold+1)*fold_size]
    
    # Remaining subjects for train + val
    remaining_subjects = np.setdiff1d(all_subjects, test_subjects)
    np.random.shuffle(remaining_subjects)
    
    # Split remaining into training and validation (~80/20)
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
    # Build & Train Transformer Model
    # ===============================
    model = build_transformer_model(input_shape=(X_train.shape[1], X_train.shape[2]), num_classes=n_classes)
    model.compile(optimizer=Adam(learning_rate=0.002), loss='categorical_crossentropy', metrics=['accuracy'])
    
    start_time = time.time()
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=1000,
        batch_size=64,
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
# 7. Report Cross-Validation Results
# ===============================
mean_acc = np.mean(fold_accuracies)
std_acc = np.std(fold_accuracies)
print("\n=== Cross-Validation Results ===")
for i, acc in enumerate(fold_accuracies):
    print(f"Fold {i+1} Accuracy: {acc:.4f}")
print(f"Mean Accuracy: {mean_acc:.4f}, Std: {std_acc:.4f}")
