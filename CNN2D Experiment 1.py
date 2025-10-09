import numpy as np
import time
from keras.utils import to_categorical
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential, load_model
from keras.layers import Input, Dense, Activation, Flatten, Convolution1D, Dropout, MaxPooling1D, concatenate, BatchNormalization
from keras.optimizers import Adam
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc, precision_recall_fscore_support, matthews_corrcoef
import itertools

# X = np.load('MARKERS_DATA.npy')
# X = np.load('GRF_DATA.npy')
X = np.load('EMG_DATA.npy')

y = np.load('ID_L.npy')
# y = np.load('SPEED_L.npy')
l = 4
for i in range(len(y)):
    if np.all((y[i] == 3)):
        y[i] = 2

for i in range(len(y)):
    if np.all((y[i] == 4)):
        y[i] = 3

y = to_categorical(y)

# Standardize function
def standardize(train, test):
    """ Standardize data """
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train.reshape(-1, train.shape[-1])).reshape(train.shape)
    X_test = scaler.transform(test.reshape(-1, test.shape[-1])).reshape(test.shape)
    return X_train, X_test

# Model creation function
def create_model(input_shape, l):
    model = Sequential()
    model.add(Convolution1D(filters=96, kernel_size=5, strides=1, padding='same', activation='relu', input_shape=input_shape))
    model.add(MaxPooling1D(pool_size=2, strides=2, padding='same'))
    model.add(Convolution1D(filters=48, kernel_size=5, strides=1, activation='relu', padding='same'))
    model.add(MaxPooling1D(pool_size=2, strides=2, padding='same'))
    model.add(Convolution1D(filters=24, kernel_size=5, strides=1, activation='relu', padding='same'))
    model.add(MaxPooling1D(pool_size=2, strides=2, padding='same'))
    model.add(Convolution1D(filters=12, kernel_size=5, strides=1, activation='relu', padding='same'))
    model.add(MaxPooling1D(pool_size=2, strides=2, padding='same'))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(50, kernel_initializer='random_uniform', bias_initializer='zeros', activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(units=l, kernel_initializer='uniform', activation='softmax'))
    
    opt = Adam(learning_rate=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# KFold cross-validation
kf = KFold(n_splits=10, shuffle=True, random_state=2024)
fold_no = 1
results = []
history_list = []
best_accuracy = 0
best_model_path = '2DCNN_EMG_best_model.h5'
mcc_list = []
fprs = []
tprs = []
aucs = []

for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    # Standardize data
    X_train, X_test = standardize(X_train, X_test)
    
    # Create and train model
    model = create_model((X_train.shape[1], X_train.shape[2]), y.shape[1])
    print(f'Training for fold {fold_no} ...')
    history = model.fit(X_train, y_train, validation_split=0.1, epochs=200, batch_size=100, verbose=1)
    
    # Evaluate model
    scores = model.evaluate(X_test, y_test, verbose=0)
    print(f'Score for fold {fold_no}: {model.metrics_names[1]} of {scores[1]*100}%')
    results.append(scores[1] * 100)
    history_list.append(history)
    
    # Calculate MCC for this fold
    y_pred = np.argmax(model.predict(X_test), axis=1)
    y_true = np.argmax(y_test, axis=1)
    mcc = matthews_corrcoef(y_true, y_pred)
    mcc_list.append(mcc)
    
    # Save the best model
    if scores[1] > best_accuracy:
        best_accuracy = scores[1]
        model.save(best_model_path)
    
    # ROC curve calculation
    y_pred_proba = model.predict(X_test)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    n_classes = y.shape[1]
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_pred_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        
    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_pred_proba.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    
    # Compute macro-average ROC curve and ROC area
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= n_classes
    
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    
    fprs.append(fpr)
    tprs.append(tpr)
    aucs.append(roc_auc)
    
    fold_no += 1

# Print average results
print(f'Average accuracy across folds: {np.mean(results)}%')
print(f'Standard deviation across folds: {np.std(results)}%')




# Plot ROC curves
plt.figure(figsize=(8, 6), dpi=110)
lw = 2  # Line width

plt.plot(fpr["micro"], tpr["micro"],
         label=f'micro-average ROC curve (area = {roc_auc["micro"]:0.2f})',
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label=f'macro-average ROC curve (area = {roc_auc["macro"]:0.2f})',
         color='navy', linestyle=':', linewidth=4)

colors = itertools.cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label=f'ROC curve of class {i} (area = {roc_auc[i]:0.2f})')

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=22)
plt.ylabel('True Positive Rate', fontsize=22)
plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=22)
plt.legend(loc="lower right", fontsize=8)
plt.rcParams['font.family'] = 'Arial'
plt.tight_layout()
plt.grid(True, color="blue", linewidth="0.8", linestyle="--")
plt.show()

# Plotting
plt.figure(figsize=(8, 6), dpi=110)
def plot_metrics(history_list, metric):
    for i, history in enumerate(history_list):
        plt.plot(history.history[metric], label=f'Training fold {i+1}')
        plt.plot(history.history[f'val_{metric}'], linestyle='--', label=f'Validation fold {i+1}')
    plt.title('Model loss (Categorical-Cross-Entropy)', fontsize=22)
    plt.ylabel('Loss [Arbitrary Unit]', fontsize=22)
    plt.xlabel('No. of Epoch', fontsize=22)
    plt.legend(loc="upper right", fontsize=8)
    plt.rcParams['font.family'] = 'Arial'
    plt.tight_layout()
    plt.grid(True, color = "blue", linewidth = "0.8", linestyle = "--")
    plt.show()

plot_metrics(history_list, 'loss')

plt.figure(figsize=(8, 6), dpi=110)
def plot_metrics(history_list, metric):
    for i, history in enumerate(history_list):
        plt.plot(history.history[metric], label=f'Training fold {i+1}')
        plt.plot(history.history[f'val_{metric}'], linestyle='--', label=f'Validation fold {i+1}')
    plt.title('Model accuracy CNN', fontsize=22)
    plt.ylabel('Average Accuracy [0-1]', fontsize=22)
    plt.xlabel('No. of Epoch', fontsize=22)
    plt.legend(loc="lower right", fontsize=8)
    plt.rcParams['font.family'] = 'Arial'
    plt.tight_layout()
    plt.grid(True, color = "blue", linewidth = "0.8", linestyle = "--")
    plt.show()

plot_metrics(history_list, 'accuracy')

# Load the best model
best_model = load_model(best_model_path)

# Standardize the entire dataset before prediction
scaler = StandardScaler()
X_standardized = scaler.fit_transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)

# Make predictions on the test data
predictions = best_model.predict(X_standardized)

# Optional: Convert predictions to class labels
predicted_classes = np.argmax(predictions, axis=1)
true_classes = np.argmax(y, axis=1)

# Plot confusion matrix
def customize_conf_matrix(disp, fontsize=20, color='red'):
    for i in range(disp.text_.shape[0]):
        for j in range(disp.text_.shape[1]):
            disp.text_[i, j].set_fontsize(fontsize)
            disp.text_[i, j].set_color(color)

# Plot confusion matrix
conf_matrix = confusion_matrix(true_classes, predicted_classes)
fig, ax = plt.subplots(figsize=(10, 8), dpi=100)  # Adjust the figure size and DPI as needed
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix)
disp.plot(cmap=plt.cm.Blues, ax=ax)
customize_conf_matrix(disp, fontsize=25, color='red')  # Customize the text
ax.set_title('Confusion Matrix', fontsize=25)
ax.set_ylabel('True Class', fontsize=25)
ax.set_xlabel('Predicted Class', fontsize=25)
plt.rcParams['font.family'] = 'Arial'
plt.tight_layout()
plt.show()

# Plot normalized confusion matrix
conf_matrix_normalized = confusion_matrix(true_classes, predicted_classes, normalize='true')
fig, ax = plt.subplots(figsize=(10, 8), dpi=100)  # Adjust the figure size and DPI as needed
disp_normalized = ConfusionMatrixDisplay(confusion_matrix=conf_matrix_normalized)
disp_normalized.plot(cmap=plt.cm.Blues, ax=ax)
customize_conf_matrix(disp_normalized, fontsize=25, color='red')  # Customize the text
ax.set_title('Normalized Confusion Matrix', fontsize=25)
ax.set_ylabel('True Class', fontsize=25)
ax.set_xlabel('Predicted Class', fontsize=25)
plt.rcParams['font.family'] = 'Arial'
plt.tight_layout()
plt.show()

# Calculate additional metrics
precision, recall, f1, _ = precision_recall_fscore_support(true_classes, predicted_classes, average='weighted')
mcc = matthews_corrcoef(true_classes, predicted_classes)

print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1-score: {f1}')
print(f'MCC: {mcc}')

# Plot MCC across folds
plt.figure(figsize=(8, 6), dpi=110)
plt.plot(range(1, len(mcc_list) + 1), mcc_list, marker='o', linestyle='-', color='b')
plt.title('Matthews Correlation Coefficient (MCC) Across Folds', fontsize=22)
plt.xlabel('Fold Number', fontsize=22)
plt.ylabel('MCC', fontsize=22)
plt.rcParams['font.family'] = 'Arial'
plt.tight_layout()
plt.grid(True, color = "blue", linewidth = "0.8", linestyle = "--")
plt.show()

