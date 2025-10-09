"""
Gait Classification using Multi-Branch 1D CNN with 5-Fold Subject-Wise Cross-Validation
Includes:
 - Training/validation plots
 - Raw and normalized confusion matrices
 - t-SNE embeddings of predicted classes
"""

# ===============================
# 1. Import Required Libraries
# ===============================
import numpy as np
import matplotlib.pyplot as plt
import itertools
import time
from keras.utils import to_categorical
from keras.models import Model
from keras.layers import Dense, Flatten, Dropout, BatchNormalization
from keras.layers import Convolution1D, MaxPooling1D, AveragePooling1D, concatenate, Input
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix
from sklearn.manifold import TSNE

# ===============================
# 2. Load Dataset
# ===============================
X = np.load('MARKERS_DATA.npy')
# X = np.load('GRF_DATA.npy')
# X = np.load('EMG_DATA.npy')
# X = np.load('F_M_DATA.npy')
y1 = np.load('ID_L.npy')         # subject IDs
y2 = np.load('SPEED_L.npy')      # sample class labels

X = np.nan_to_num(X)  # replace NaN

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

early_stop = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, verbose=1)
fold_accuracies = []

# ===============================
# 4. Helper Functions for Plots
# ===============================
def plot_training_history(history, fold):
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.title(f'Fold {fold+1} - Accuracy')
    plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.legend()
    plt.subplot(1,2,2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title(f'Fold {fold+1} - Loss')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend()
    plt.tight_layout(); plt.show()

def plot_confusion_matrix(cm, classes, fold, normalize=False):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        title = f'Normalized Confusion Matrix - Fold {fold+1}'
    else:
        title = f'Confusion Matrix - Fold {fold+1}'
    plt.figure(figsize=(6,6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title); plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i,j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i,j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i,j] > thresh else "black")
    plt.ylabel('True label'); plt.xlabel('Predicted label'); plt.tight_layout(); plt.show()

def plot_tsne(features, labels, fold):
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    reduced = tsne.fit_transform(features)
    plt.figure(figsize=(8,6))
    scatter = plt.scatter(reduced[:,0], reduced[:,1], c=labels, cmap='tab10', alpha=0.7)
    plt.legend(*scatter.legend_elements(), title="Classes")
    plt.title(f't-SNE of Predicted Classes - Fold {fold+1}'); plt.show()

# ===============================
# 5. Cross-Validation Loop
# ===============================
fold_size = len(all_subjects) // n_folds

for fold in range(n_folds):
    print(f"\n=== Fold {fold+1} ===")
    test_subjects = all_subjects[fold*fold_size:(fold+1)*fold_size]
    remaining_subjects = np.setdiff1d(all_subjects, test_subjects)
    np.random.shuffle(remaining_subjects)
    n_val = max(1, len(remaining_subjects)//5)
    val_subjects = remaining_subjects[:n_val]
    train_subjects = remaining_subjects[n_val:]

    def get_subject_data(X, y, subjects):
        idx = np.isin(y[:,0], subjects)
        return X[idx], y[idx,1].astype(int)

    X_train, y_train = get_subject_data(X, y, train_subjects)
    X_val, y_val     = get_subject_data(X, y, val_subjects)
    X_test, y_test   = get_subject_data(X, y, test_subjects)

    X_train, y_train = shuffle(X_train, y_train, random_state=seed)
    X_val, y_val     = shuffle(X_val, y_val, random_state=seed)
    X_test, y_test   = shuffle(X_test, y_test, random_state=seed)

    y_train_o = to_categorical(y_train, num_classes=n_classes)
    y_val_o   = to_categorical(y_val, num_classes=n_classes)
    y_test_o  = to_categorical(y_test, num_classes=n_classes)

    def standardize(train, test):
        mean = np.mean(train, axis=0)[None,:,:]
        std  = np.std(train, axis=0)[None,:,:] + 1e-12
        return (train - mean)/std, (test - mean)/std

    X_train, X_val = standardize(X_train, X_val)
    X_train, X_test = standardize(X_train, X_test)

    # ===============================
    # Build Multi-Branch CNN Model
    # ===============================
    l = n_classes
    inp = Input(shape=(X_train.shape[1], X_train.shape[2]))

    t0 = Convolution1D(6,2,activation='relu',padding='same')(inp)
    t0 = AveragePooling1D(2,2,padding='same')(t0)

    t1 = Convolution1D(12,2,activation='relu',padding='same')(t0)
    t1 = AveragePooling1D(2,2,padding='same')(t1)

    t2 = Convolution1D(12,2,activation='relu',padding='same')(t0)
    t2 = MaxPooling1D(2,2,padding='same')(t2)

    t3 = Convolution1D(12,2,activation='relu',padding='same')(t0)
    t3 = AveragePooling1D(2,2,padding='same')(t3)

    t4 = Convolution1D(12,2,activation='relu',padding='same')(t0)
    t4 = MaxPooling1D(2,2,padding='same')(t4)

    t00 = concatenate([t1,t2],axis=1)
    t000 = concatenate([t3,t4],axis=1)

    t11 = Convolution1D(24,2,activation='relu',padding='same')(t00)
    t11 = AveragePooling1D(2,2,padding='same')(t11)
    t11 = Convolution1D(48,2,activation='relu',padding='same')(t11)
    t11 = AveragePooling1D(2,2,padding='same')(t11)

    t22 = Convolution1D(24,2,activation='relu',padding='same')(t00)
    t22 = MaxPooling1D(2,2,padding='same')(t22)
    t22 = Convolution1D(48,2,activation='relu',padding='same')(t22)
    t22 = MaxPooling1D(2,2,padding='same')(t22)

    t33 = Convolution1D(24,2,activation='relu',padding='same')(t000)
    t33 = AveragePooling1D(2,2,padding='same')(t33)
    t33 = Convolution1D(48,2,activation='relu',padding='same')(t33)
    t33 = AveragePooling1D(2,2,padding='same')(t33)

    t44 = Convolution1D(24,2,activation='relu',padding='same')(t000)
    t44 = MaxPooling1D(2,2,padding='same')(t44)
    t44 = Convolution1D(48,2,activation='relu',padding='same')(t44)
    t44 = MaxPooling1D(2,2,padding='same')(t44)

    c1 = concatenate([t11,t22],axis=1)
    c2 = concatenate([t33,t44],axis=1)

    t5 = Convolution1D(96,2,activation='relu',padding='same')(c1)
    t5 = MaxPooling1D(2,2,padding='same')(t5)

    t55 = Convolution1D(96,2,activation='relu',padding='same')(c2)
    t55 = MaxPooling1D(2,2,padding='same')(t55)

    output = concatenate([t5,t55],axis=1)
    output = Flatten()(output)
    output = Dropout(0.5)(output)
    output = Dense(50,activation='relu',name='penultimate_dense')(output)
    output = Dropout(0.2)(output)
    output = Dense(l,activation='softmax')(output)

    model = Model(inputs=[inp], outputs=[output])
    model.compile(Adam(0.002),'categorical_crossentropy',['accuracy'])

    # ===============================
    # Train Model
    # ===============================
    history = model.fit(X_train, y_train_o, validation_data=(X_val, y_val_o),
                        epochs=1000, batch_size=100, callbacks=[early_stop], verbose=1)
    plot_training_history(history, fold)

    # ===============================
    # Evaluate
    # ===============================
    y_pred_prob = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred_prob,axis=1)
    y_true = y_test

    acc = np.mean(y_pred_classes == y_true)
    print(f"Test Accuracy: {acc:.4f}")
    fold_accuracies.append(acc)

    cm = confusion_matrix(y_true,y_pred_classes)
    plot_confusion_matrix(cm,[f'Class {i}' for i in range(n_classes)],fold,normalize=False)
    plot_confusion_matrix(cm,[f'Class {i}' for i in range(n_classes)],fold,normalize=True)

    feature_model = Model(inputs=model.input, outputs=model.get_layer('penultimate_dense').output)
    features = feature_model.predict(X_test)
    plot_tsne(features, y_pred_classes, fold)

# ===============================
# Cross-Validation Summary
# ===============================
mean_acc = np.mean(fold_accuracies)
std_acc = np.std(fold_accuracies)
print("\n=== Cross-Validation Results ===")
for i, acc in enumerate(fold_accuracies):
    print(f"Fold {i+1} Accuracy: {acc:.4f}")
print(f"Mean Accuracy: {mean_acc:.4f}, Std: {std_acc:.4f}")
