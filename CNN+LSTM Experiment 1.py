"""
Gait Classification using Multi-Layer Temporal CNN (TCN)
----------------------------------------------------------
This script loads gait-related data, randomly splits into
70% training, 10% validation, and 20% testing sets,
trains a multi-layer TCN model with early stopping,
evaluates performance, and reports results.
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
from sklearn.model_selection import train_test_split
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
y = y2.copy()  # only use class labels for training

# ===============================
# 3. Random Data Split (70/10/20)
# ===============================
seed = 42
np.random.seed(seed)

# Shuffle before splitting
X, y = shuffle(X, y, random_state=seed)

# Split 20% test, 80% remain
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.2, random_state=seed, stratify=y
)

# From remaining 80%, split 10% validation (which is 0.1 * total)
val_ratio = 0.1 / 0.8  # relative to the remaining 80%
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=val_ratio, random_state=seed, stratify=y_temp
)

print(f"Data split summary:")
print(f"Train samples: {len(X_train)} ({len(X_train)/len(X)*100:.1f}%)")
print(f"Validation samples: {len(X_val)} ({len(X_val)/len(X)*100:.1f}%)")
print(f"Test samples: {len(X_test)} ({len(X_test)/len(X)*100:.1f}%)")

# ===============================
# 4. Data Preparation
# ===============================
# One-hot encoding
y_train = to_categorical(y_train, num_classes=n_classes)
y_val = to_categorical(y_val, num_classes=n_classes)
y_test = to_categorical(y_test, num_classes=n_classes)

# Standardization function
def standardize(train, test):
    mean = np.mean(train, axis=0)[None, :, :]
    std = np.std(train, axis=0)[None, :, :]
    std[std == 0] = 1e-6  # avoid division by zero
    X_train = (train - mean) / std
    X_test = (test - mean) / std
    return np.nan_to_num(X_train), np.nan_to_num(X_test)

# Standardize
X_train, X_val = standardize(X_train, X_val)
X_train, X_test = standardize(X_train, X_test)

# ===============================
# 5. Build Multi-Layer Temporal CNN (TCN) Model
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
# 6. Train Model with Early Stopping
# ===============================
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=20,
    restore_best_weights=True,
    verbose=1
)

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
print(f"\nTraining completed in {training_time:.2f} seconds")

# ===============================
# 7. Evaluate on Test Set
# ===============================
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

acc = np.mean(y_pred_classes == y_true)
print(f"\nTest Accuracy: {acc:.4f}")

print("\nClassification Report:")
print(classification_report(y_true, y_pred_classes, digits=4))

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred_classes)
plt.figure(figsize=(6, 5))
plt.imshow(cm, cmap='Blues')
plt.title("Confusion Matrix")
plt.colorbar()
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()
