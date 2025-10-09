import numpy as np
import time
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from keras.models import Model, load_model
from keras.layers import Input, Dense, Flatten, Convolution1D, Dropout, MaxPooling1D, AveragePooling1D, concatenate
from keras.optimizers import Adam
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc, precision_recall_fscore_support, matthews_corrcoef
import itertools
from sklearn.utils import shuffle

# ===============================
# 1. Load Dataset
# ===============================
X = np.load('MARKERS_DATA.npy')
# X = np.load('GRF_DATA.npy')
# X = np.load('EMG_DATA.npy')
# X = np.load('F_M_DATA.npy')

# y = np.load('ID_L.npy')
y = np.load('SPEED_L.npy')
l = 4

# Relabel classes
y[y == 3] = 2
y[y == 4] = 3
y = to_categorical(y)

# ===============================
# 2. Standardize function
# ===============================
def standardize(train, test):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train.reshape(-1, train.shape[-1])).reshape(train.shape)
    X_test = scaler.transform(test.reshape(-1, test.shape[-1])).reshape(test.shape)
    return X_train, X_test

# ===============================
# 3. Model creation function
# ===============================
def create_model(input_shape, l):
    inp = Input(shape=(input_shape))
    t0 = Convolution1D(filters=6, kernel_size=2, activation='relu', padding='same')(inp)
    t0 = AveragePooling1D(pool_size=2, strides=2, padding='same')(t0)

    t1 = Convolution1D(filters=12, kernel_size=2, activation='relu', padding='same')(t0)
    t1 = AveragePooling1D(pool_size=2, strides=2, padding='same')(t1)
    t2 = Convolution1D(filters=12, kernel_size=2, activation='relu', padding='same')(t0)
    t2 = MaxPooling1D(pool_size=2, strides=2, padding='same')(t2)
    t3 = Convolution1D(filters=12, kernel_size=2, activation='relu', padding='same')(t0)
    t3 = AveragePooling1D(pool_size=2, strides=2, padding='same')(t3)
    t4 = Convolution1D(filters=12, kernel_size=2, activation='relu', padding='same')(t0)
    t4 = MaxPooling1D(pool_size=2, strides=2, padding='same')(t4)

    t00 = concatenate([t1, t2], axis=1)
    t000 = concatenate([t3, t4], axis=1)

    t11 = Convolution1D(filters=24, kernel_size=2, activation='relu', padding='same')(t00)
    t11 = AveragePooling1D(pool_size=2, strides=2, padding='same')(t11)
    t11 = Convolution1D(filters=48, kernel_size=2, activation='relu', padding='same')(t11)
    t11 = AveragePooling1D(pool_size=2, strides=2, padding='same')(t11)

    t22 = Convolution1D(filters=24, kernel_size=2, activation='relu', padding='same')(t00)
    t22 = MaxPooling1D(pool_size=2, strides=2, padding='same')(t22)
    t22 = Convolution1D(filters=48, kernel_size=2, activation='relu', padding='same')(t22)
    t22 = MaxPooling1D(pool_size=2, strides=2, padding='same')(t22)

    t33 = Convolution1D(filters=24, kernel_size=2, activation='relu', padding='same')(t000)
    t33 = AveragePooling1D(pool_size=2, strides=2, padding='same')(t33)
    t33 = Convolution1D(filters=48, kernel_size=2, activation='relu', padding='same')(t33)
    t33 = AveragePooling1D(pool_size=2, strides=2, padding='same')(t33)

    t44 = Convolution1D(filters=24, kernel_size=2, activation='relu', padding='same')(t000)
    t44 = MaxPooling1D(pool_size=2, strides=2, padding='same')(t44)
    t44 = Convolution1D(filters=48, kernel_size=2, activation='relu', padding='same')(t44)
    t44 = MaxPooling1D(pool_size=2, strides=2, padding='same')(t44)

    c1 = concatenate([t11, t22], axis=1)
    c2 = concatenate([t33, t44], axis=1)

    t5 = Convolution1D(filters=96, kernel_size=2, activation='relu', padding='same')(c1)
    t5 = MaxPooling1D(pool_size=2, strides=2, padding='same')(t5)

    t55 = Convolution1D(filters=96, kernel_size=2, activation='relu', padding='same')(c2)
    t55 = MaxPooling1D(pool_size=2, strides=2, padding='same')(t55)

    output = concatenate([t5, t55], axis=1)
    output = Flatten()(output)
    output = Dropout(0.5)(output)
    output = Dense(50, activation='relu')(output)
    output = Dropout(0.2)(output)
    output = Dense(l, activation='softmax')(output)

    model = Model(inputs=[inp], outputs=[output])
    model.compile(optimizer=Adam(learning_rate=0.002), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# ===============================
# 4. 10-Fold Cross-Validation Loop (70% train, 10% val, 20% test)
# ===============================
n_samples = X.shape[0]
fold_accuracies = []
history_list = []
mcc_list = []
fprs, tprs, aucs = [], [], []
best_accuracy = 0
best_model_path = '4_2DCNN_EMG_best_model.h5'

for fold in range(10):
    seed = 42 + fold
    np.random.seed(seed)

    # Shuffle indices
    indices = np.arange(n_samples)
    np.random.shuffle(indices)

    n_test = int(0.2 * n_samples)
    n_val = int(0.1 * n_samples)
    n_train = n_samples - n_test - n_val

    test_idx = indices[:n_test]
    val_idx = indices[n_test:n_test+n_val]
    train_idx = indices[n_test+n_val:]

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    # Standardize
    X_train, X_val = standardize(X_train, X_val)
    X_train, X_test = standardize(X_train, X_test)

    # Shuffle training and validation
    X_train, y_train = shuffle(X_train, y_train, random_state=seed)
    X_val, y_val = shuffle(X_val, y_val, random_state=seed)
    X_test, y_test = shuffle(X_test, y_test, random_state=seed)

    # Create model
    model = create_model((X_train.shape[1], X_train.shape[2]), y.shape[1])
    print(f'\nTraining fold {fold+1} ...')
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=200,
        batch_size=100,
        verbose=1
    )

    history_list.append(history)

    # Evaluate test set
    scores = model.evaluate(X_test, y_test, verbose=0)
    print(f'Score for fold {fold+1}: Accuracy = {scores[1]*100:.2f}%')
    fold_accuracies.append(scores[1]*100)

    # MCC
    y_pred = np.argmax(model.predict(X_test), axis=1)
    y_true = np.argmax(y_test, axis=1)
    mcc_list.append(matthews_corrcoef(y_true, y_pred))

    # Save best model
    if scores[1] > best_accuracy:
        best_accuracy = scores[1]
        model.save(best_model_path)

# ===============================
# 5. Report Results
# ===============================
print(f'\nAverage Accuracy: {np.mean(fold_accuracies):.2f}%')
print(f'Standard Deviation: {np.std(fold_accuracies):.2f}%')

# Plot MCC across folds
plt.figure(figsize=(6, 4), dpi=110)
plt.plot(range(1, len(mcc_list)+1), mcc_list, marker='o', linestyle='-', color='b')
plt.title('Matthews Correlation Coefficient (MCC) Across Folds', fontsize=16)
plt.xlabel('Fold Number', fontsize=14)
plt.ylabel('MCC', fontsize=14)
plt.grid(True, linestyle='--', linewidth=0.8)
plt.show()
