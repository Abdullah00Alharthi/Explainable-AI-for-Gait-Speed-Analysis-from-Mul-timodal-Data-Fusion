"""
Gait Classification using Multi-Layer Temporal CNN (TCN)
with 10-Fold Randomized Data Splits
----------------------------------------------------------
Uses a randomized 70/10/20 split in each fold (train/val/test),
trains a multi-layer TCN, and reports per-fold metrics and visuals.
"""

# ===============================
# 1. Import Required Libraries
# ===============================
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils import shuffle
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from tcn import TCN  # pip install keras-tcn

# ===============================
# 2. Load and Prepare Dataset
# ===============================
# Example: replace file name with your data
# X = np.load('MARKERS_DATA.npy')
# X = np.load('GRF_DATA.npy')
# X = np.load('EMG_DATA.npy')
X = np.load('F_M_DATA.npy')
y1 = np.load('ID_L.npy')         # subject IDs
y2 = np.load('SPEED_L.npy')  # sample class labels

# Replace NaN with zeros
X = np.nan_to_num(X)

# Optional class relabeling
for i in range(len(y2)):
    if np.all(y2[i] == 3):
        y2[i] = 2
    elif np.all(y2[i] == 4):
        y2[i] = 3

n_classes = 4
y = np.column_stack((y1, y2))

# ===============================
# 3. Define Helper Functions
# ===============================
def standardize(train, test):
    X_train = (train - np.mean(train, axis=0)[None, :, :]) / np.std(train, axis=0)[None, :, :]
    X_test = (test - np.mean(test, axis=0)[None, :, :]) / np.std(test, axis=0)[None, :, :]
    return np.nan_to_num(X_train), np.nan_to_num(X_test)

def build_tcn(input_shape, n_classes):
    model = Sequential()
    model.add(TCN(nb_filters=12, kernel_size=5, dilations=[1,2,4],
                  padding='causal', activation='relu', dropout_rate=0.1,
                  return_sequences=True, input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(TCN(nb_filters=24, kernel_size=5, dilations=[1,2,4,8],
                  padding='causal', activation='relu', dropout_rate=0.1,
                  return_sequences=True))
    model.add(BatchNormalization())
    model.add(TCN(nb_filters=48, kernel_size=5, dilations=[1,2,4,8,16],
                  padding='causal', activation='relu', dropout_rate=0.1,
                  return_sequences=True))
    model.add(BatchNormalization())
    model.add(TCN(nb_filters=96, kernel_size=5, dilations=[1,2,4,8,16,32],
                  padding='causal', activation='relu', dropout_rate=0.2,
                  return_sequences=False))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(n_classes, activation='softmax'))
    opt = Adam(learning_rate=0.002)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# ===============================
# 4. 10-Fold Randomized Evaluation
# ===============================
n_folds = 10
seed = 42
np.random.seed(seed)

fold_accuracies, fold_precisions, fold_recalls, fold_f1s = [], [], [], []

early_stop = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, verbose=1)

for fold in range(1, n_folds+1):
    print(f"\n===== Fold {fold} / {n_folds} =====")
    
    # Shuffle data each fold
    X, y = shuffle(X, y, random_state=seed + fold)
    y_labels = y[:, 1]
    
    # Split train/val/test 70/10/20
    X_train, X_temp, y_train, y_temp = train_test_split(X, y_labels, test_size=0.3, random_state=seed + fold)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=2/3, random_state=seed + fold)
    
    # Standardize
    X_train, X_val = standardize(X_train, X_val)
    X_train, X_test = standardize(X_train, X_test)
    
    # One-hot encode
    y_train = to_categorical(y_train, num_classes=n_classes)
    y_val = to_categorical(y_val, num_classes=n_classes)
    y_test = to_categorical(y_test, num_classes=n_classes)
    
    # Build model
    model = build_tcn((X_train.shape[1], X_train.shape[2]), n_classes)
    
    # Train
    start_time = time.time()
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                        epochs=1000, batch_size=100, callbacks=[early_stop], verbose=0)
    train_time = time.time() - start_time
    print(f"Training time: {train_time:.2f}s")
    
    # Evaluate
    y_pred = model.predict(X_test)
    y_pred_cls = np.argmax(y_pred, axis=1)
    y_true_cls = np.argmax(y_test, axis=1)
    
    acc = np.mean(y_pred_cls == y_true_cls)
    report = classification_report(y_true_cls, y_pred_cls, output_dict=True, zero_division=0)
    prec, rec, f1 = report['weighted avg']['precision'], report['weighted avg']['recall'], report['weighted avg']['f1-score']
    
    fold_accuracies.append(acc)
    fold_precisions.append(prec)
    fold_recalls.append(rec)
    fold_f1s.append(f1)
    
    print(f"Fold {fold}  Accuracy: {acc:.4f}  Precision: {prec:.4f}  Recall: {rec:.4f}  F1: {f1:.4f}")

    # Confusion matrix
    cm = confusion_matrix(y_true_cls, y_pred_cls)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(5,4))
    plt.imshow(cm, cmap='Blues')
    plt.title(f'Normalized Confusion Matrix (Fold {fold})')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.colorbar()
    plt.show()
    
    # t-SNE visualization
    tsne = TSNE(n_components=2, random_state=seed + fold)
    X_embedded = tsne.fit_transform(y_pred)
    plt.figure(figsize=(6,5))
    plt.scatter(X_embedded[:,0], X_embedded[:,1], c=y_true_cls, cmap='tab10', alpha=0.6)
    plt.title(f't-SNE of Test Predictions (Fold {fold})')
    plt.show()
    
    # Show running mean/std
    mean_acc, std_acc = np.mean(fold_accuracies), np.std(fold_accuracies)
    print(f"Running Mean Accuracy: {mean_acc:.4f} ± {std_acc:.4f}")

# ===============================
# 5. Final Summary
# ===============================
print("\n=== Final 10-Fold Results ===")
print(f"Mean Accuracy:  {np.mean(fold_accuracies):.4f} ± {np.std(fold_accuracies):.4f}")
print(f"Mean Precision: {np.mean(fold_precisions):.4f} ± {np.std(fold_precisions):.4f}")
print(f"Mean Recall:    {np.mean(fold_recalls):.4f} ± {np.std(fold_recalls):.4f}")
print(f"Mean F1-score:  {np.mean(fold_f1s):.4f} ± {np.std(fold_f1s):.4f}")
