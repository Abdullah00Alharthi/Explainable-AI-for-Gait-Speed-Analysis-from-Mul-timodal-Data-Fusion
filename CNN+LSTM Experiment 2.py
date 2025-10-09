
import numpy as np
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import time

X1 = np.load('MARKERS_DATA.npy')
X2 = np.load('GRF_DATA.npy')
X3 = np.load('EMG_DATA.npy')
X4 = np.load('F_M_DATA.npy')
# X1=X1[:,:,0:156]

# y=np.load('ID_L.npy')
y=np.load('SPEED_L.npy')
l=5

X1=np.nan_to_num(X1)
X2=np.nan_to_num(X2)
X3=np.nan_to_num(X3)
X4=np.nan_to_num(X4)


def standardize(r1,r2):
	""" Standardize data """
	# Standardize train and test
	X_r1 = (r1 - np.mean(r1, axis=0)[None,:,:]) / np.std(r1, axis=0)[None,:,:]
	X_r2 = (r2 - np.mean(r2, axis=0)[None,:,:]) / np.std(r2, axis=0)[None,:,:]

	return X_r1, X_r2   
X1, X2 = standardize(X1, X2)  

def standardize(r3,r4):
	""" Standardize data """
	# Standardize train and test
	X_r3 = (r3 - np.mean(r3, axis=0)[None,:,:]) / np.std(r3, axis=0)[None,:,:]
	X_r4 = (r4 - np.mean(r4, axis=0)[None,:,:]) / np.std(r4, axis=0)[None,:,:]

	return X_r3, X_r4   
X3, X4 = standardize(X3, X4)  


X1=np.nan_to_num(X1)
X2=np.nan_to_num(X2)
X3=np.nan_to_num(X3)
X4=np.nan_to_num(X4)


x_test1=X1[0:228]
y_test=y[0:228]
X1=X1[228:1142]
y=y[228:1142]


x_test2=X2[0:228]
X2=X2[228:1142]


x_test3=X3[0:228]
X3=X3[228:1142]


x_test4=X4[0:228]
X4=X4[228:1142]


from sklearn.utils import shuffle
x_train1,x_train2,x_train3,x_train4, y_train = shuffle(X1,X2,X3,X4, y, random_state=42)

y_train=to_categorical(y_train)
y_test=to_categorical(y_test)


from keras.layers import Dense, Activation, Flatten, Convolution1D, Dropout, concatenate , Input, LSTM
from keras.layers import  AveragePooling1D, MaxPooling1D
from keras.models import Model

inp1 = Input(shape=(x_train1.shape[1], x_train1.shape[2]))
inp2 = Input(shape=(x_train2.shape[1], x_train2.shape[2]))
inp3 = Input(shape=(x_train3.shape[1], x_train3.shape[2]))
inp4 = Input(shape=(x_train4.shape[1], x_train4.shape[2]))




t1 = Convolution1D(filters=6, kernel_size=2, strides=1, activation='relu', padding='same')(inp1)
t1 = AveragePooling1D(pool_size=2, strides=2, padding='same')(t1)


t2 = Convolution1D(filters=6, kernel_size=2, strides=1, activation='relu', padding='same')(inp2)
t2 = AveragePooling1D(pool_size=2, strides=2, padding='same')(t2)
t00 = concatenate([t1,t2], axis=1)

t00 = LSTM(units = 100, return_sequences = True)(t00)


t3 = Convolution1D(filters=6, kernel_size=2, strides=1, activation='relu', padding='same')(inp3)
t3 = AveragePooling1D(pool_size=2, strides=2, padding='same')(t3)
t3 = LSTM(units = 100, return_sequences = True)(t3)
    
t4= Convolution1D(filters=6, kernel_size=2, strides=1, activation='relu', padding='same')(inp4)
t4 = AveragePooling1D(pool_size=2, strides=2, padding='same')(t4)



t123 = concatenate([t00,t3], axis=1)


t5= Convolution1D(filters=24, kernel_size=2, strides=1, activation='relu', padding='same')(t123)
t5= MaxPooling1D(pool_size=2, strides=2, padding='same')(t5)
t5 = LSTM(units = 50, return_sequences = True)(t5)


t44= Convolution1D(filters=24, kernel_size=2, strides=1, activation='relu', padding='same')(t4)
t44= MaxPooling1D(pool_size=2, strides=2, padding='same')(t44)
t44 = LSTM(units = 50, return_sequences = True)(t4)



t1234 = concatenate([t5,t44], axis=1)


t6= Convolution1D(filters=96, kernel_size=2, strides=1, activation='relu', padding='same')(t1234)
t6= MaxPooling1D(pool_size=2, strides=2, padding='same')(t6)



output = Flatten()(t6)
output = Dropout(0.5)(output)
output = Dense(units = 100,kernel_initializer = 'uniform', activation = 'relu')(output)
output = Dropout(0.2)(output)
output = Dense(units = l,kernel_initializer = 'uniform', activation = 'softmax')(output)

model = Model(inputs=[inp1,inp2,inp3,inp4], outputs=[output])



from keras.optimizers import Adam
opt = Adam(learning_rate=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
model.compile(optimizer =opt, loss = 'categorical_crossentropy', metrics=['accuracy'])
print(model.summary())
Tic=time.time()
history=model.fit([x_train1,x_train2,x_train3,x_train4], y_train,validation_split=0.1, epochs = 100, batch_size = 100)
toc=time.time()-Tic
print(history.history.keys())


acc_Batch2=[ history.history['accuracy'], history.history['val_accuracy'],history.history['loss'],history.history['val_loss']]

plt.figure(figsize=(8, 4), dpi=100)

plt.plot(history.history['accuracy'],linewidth=2)
plt.plot(history.history['val_accuracy'],linewidth=2)
plt.title('Model accuracy CNN',fontsize=18)
plt.ylabel('Average Accuracy [0-1]',fontsize=18)
plt.xlabel('No. of Epoch',fontsize=18)# summarize history for loss
plt.rcParams['font.family'] = 'Times new Roman'


plt.rcParams['font.weight'] = 'bold'
plt.rcParams['legend.fontsize'] = 'medium'
plt.rcParams['figure.titlesize'] = 'large'

plt.legend(['Training Acuracy', 'Validation Accuracy'], loc='')
plt.tight_layout()
plt.show()
# plt.grid(True)

plt.figure(figsize=(8, 4), dpi=100)

plt.plot(history.history['loss'],'red',linewidth=2)
plt.plot(history.history['val_loss'], 'green',linewidth=2)

plt.title('Model loss( Categorical-Cross-Entropy)',fontsize=18)
plt.ylabel('Loss[Arbitrary Unit]',fontsize=18)
plt.xlabel('No. of Epoch',fontsize=18)
plt.legend(['Training Loss', 'Validation Loss'], fontsize=18, loc='')
plt.tight_layout()
plt.show()
# plt.grid(True)


from sklearn.metrics import confusion_matrix,classification_report
from sklearn.metrics import roc_curve, auc
from scipy import interp
from sklearn.metrics import roc_auc_score
from itertools import cycle

y_pred = model.predict([x_test1,x_test2,x_test3,x_test4])
n_classes = 5


plt.figure(figsize=(8, 4), dpi=100)

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve( y_test[:, i], y_pred[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve( y_test.ravel(), y_pred.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
lw = 2
plt.plot(fpr[2], tpr[2], color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[2])
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic For Gait')
plt.legend(loc="")
plt.show()

# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= n_classes

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# Plot all ROC curves
plt.figure()
plt.plot(fpr["micro"], tpr["micro"],
         label='micro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='macro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic to multi-class Gait')
plt.legend(loc="")
plt.show()

y_pred = (y_pred > 0.5).astype(int) #some of the class multiple class assignments

y_out=y_pred.argmax(1)
y_label=y_test.argmax(1)
cm = confusion_matrix(y_label, y_out)
print(classification_report(y_label, y_out))
import itertools
from sklearn.metrics import confusion_matrix,classification_report 
y_pred = model.predict([x_test1,x_test2,x_test3,x_test4])
y_pred = (y_pred > 0.5).astype(int) #some of the class multiple class assignments
y_out=y_pred.argmax(1)
y_label=y_test.argmax(1)
cm = confusion_matrix(y_label, y_out)
print(classification_report(y_label, y_out))


from sklearn.metrics import confusion_matrix
import seaborn as sns


"""
This function prints and plots the confusion matrix.
Normalization can be applied by setting `normalize=True`."""
def plot_confusion_matrix(cm, classes,
                      normalize=False,
                      title='Confusion matrix',
                      cmap=plt.cm.Blues):
                      if normalize:
                          cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
                          print("Normalized confusion matrix")
                      else:
                          print('Confusion matrix')

                      print(cm)
                      plt.imshow(cm, interpolation='nearest', cmap=cmap)
                      plt.title(title)
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
                        
                      plt.tight_layout()
                      plt.ylabel('True Class')
                      plt.xlabel('Predicted Class')
plt.figure(figsize=(8, 6), dpi=100)
font = {'family' : 'Times new Roman',
        'size'   : 18}
plt.rc('font', **font)
class_names =np.array(['C1 Gait','C2 Gait','C3 Gait','C4 Gait','C5 Gait'],dtype='<U10')
plot_confusion_matrix(cm, classes=class_names,
                      title='Confusion matrix(No. Of Test Data Frames)')
plt.figure(figsize=(8, 6), dpi=100)
plot_confusion_matrix(cm, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')
plt.show()
model.save('CNN+LSTM.h5')
