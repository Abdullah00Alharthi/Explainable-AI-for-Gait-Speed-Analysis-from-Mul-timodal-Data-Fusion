import numpy as np
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import time




X=np.load('MARKERS_DATA.npy')
# X=np.load('GRF_DATA.npy')
# X=np.load('EMG_DATA.npy')
# X=np.load('F_M.npy')

y=np.load('ID_L.npy')
#y=np.load('SPEED_L.npy')


a1=y[0:5]
a2=y[25:30]
a3=y[45:50]
a4=y[67:72]

a11=y[0:5]
a12=y[25:30]
a13=y[45:50]
a14=y[67:72]
a15=y[90:95]
a16=y[114:118]
a17=y[136:141]
a18=y[159:164]
a19=y[183:188]
a20=y[206:211]

a21=y[228:231]
a22=y[250:255]
a23=y[273:278]
a24=y[297:302]
a25=y[317:322]
a26=y[342:347]
a27=y[364:369]
a28=y[389:394]
a29=y[412:417]
a30=y[437:442]

a31=y[460:465]
a32=y[483:487]
a33=y[503:508]
a34=y[524:528]
a35=y[548:553]
a36=y[571:576]
a37=y[595:600]
a38=y[619:624]
a39=y[644:648]
a40=y[0:5]

a41=y[666:671]
a42=y[684:689]
a43=y[709:714]
a44=y[733:738]
a45=y[757:762]
a46=y[782:787]
a47=y[807:812]
a48=y[829:834]
a49=y[853:856]
a50=y[875:880]

a5=y[900:905]
a6=y[925:930]
a7=y[948:952]
a8=y[968:972]
a9=y[990:995]
a10=y[1011:1016]

a51=y[1032:1037]
a52=y[1054:1059]
a53=y[1074:1078]
a54=y[1095:1100]
a55=y[1118:1123]

y=np.concatenate((a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15,a16,a17,a18,a19,a20,a21,a22,a23,a24,a25,a26,a27,a28,a29,a30,a31,a32,a33,a34,a35,a36,a37,a38,a39,a40,a41,a42,a43,a44,a45,a46,a47,a48,a49,a50,a51,a52,a53,a54,a55),axis=0)



a1=X[0:5]
a2=X[25:30]
a3=X[45:50]
a4=X[67:72]

a11=X[0:5]
a12=X[25:30]
a13=X[45:50]
a14=X[67:72]
a15=X[90:95]
a16=X[114:118]
a17=X[136:141]
a18=X[159:164]
a19=X[183:188]
a20=X[206:211]

a21=X[228:231]
a22=X[250:255]
a23=X[273:278]
a24=X[297:302]
a25=X[317:322]
a26=X[342:347]
a27=X[364:369]
a28=X[389:394]
a29=X[412:417]
a30=X[437:442]

a31=X[460:465]
a32=X[483:487]
a33=X[503:508]
a34=X[524:528]
a35=X[548:553]
a36=X[571:576]
a37=X[595:600]
a38=X[619:624]
a39=X[644:648]
a40=X[0:5]

a41=X[666:671]
a42=X[684:689]
a43=X[709:714]
a44=X[733:738]
a45=X[757:762]
a46=X[782:787]
a47=X[807:812]
a48=X[829:834]
a49=X[853:856]
a50=X[875:880]

a5=X[900:905]
a6=X[925:930]
a7=X[948:952]
a8=X[968:972]
a9=X[990:995]
a10=X[1011:1016]

a51=X[1032:1037]
a52=X[1054:1059]
a53=X[1074:1078]
a54=X[1095:1100]
a55=X[1118:1123]

X=np.concatenate((a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15,a16,a17,a18,a19,a20,a21,a22,a23,a24,a25,a26,a27,a28,a29,a30,a31,a32,a33,a34,a35,a36,a37,a38,a39,a40,a41,a42,a43,a44,a45,a46,a47,a48,a49,a50,a51,a52,a53,a54,a55),axis=0)




X=np.nan_to_num(X)
l=49
from sklearn.model_selection import train_test_split
X_train, X_test, label_Tr, label_Test = train_test_split(X, y, test_size = 0.3, random_state = 2000)

label_Tr=to_categorical(label_Tr)
label_Test=to_categorical(label_Test)

print ("Training data shape: N = {:d}, steps = {:d}, channels = {:d}".format(X_train.shape[0],
                                                                             X_train.shape[1],
                                                                             X_train.shape[2]))
print ("Test data shape: N = {:d}, steps = {:d}, channels = {:d}".format(X_test.shape[0],
                                                                         X_test.shape[1],
                                                                         X_test.shape[2]))
def standardize(train,test):
	""" Standardize data """
	# Standardize train and test
	X_train = (train - np.mean(train, axis=0)[None,:,:]) / np.std(train, axis=0)[None,:,:]
	X_test = (test - np.mean(test, axis=0)[None,:,:]) / np.std(test, axis=0)[None,:,:]

	return X_train, X_test   
X_train, X_test = standardize(X_train, X_test)                                                                       
X_train=np.nan_to_num(X_train)
X_test=np.nan_to_num(X_test)




from keras.layers import Dense, Activation, Flatten, Convolution1D, Dropout, concatenate , Input
from keras.layers import  AveragePooling1D
from keras.models import Model

inp = Input(shape=(X_train.shape[1], X_train.shape[2]))

t0 = Convolution1D(filters=6, kernel_size=2, strides=1, activation='relu', padding='same')(inp)
t0 = AveragePooling1D(pool_size=2, strides=2, padding='same')(t0)

t1 = Convolution1D(filters=12, kernel_size=2, strides=1, activation='relu', padding='same')(t0)
t1 = AveragePooling1D(pool_size=2, strides=2, padding='same')(t1)
t1 = Convolution1D(filters=24, kernel_size=2, strides=1, activation='relu', padding='same')(t1)
t1 = AveragePooling1D(pool_size=2, strides=2, padding='same')(t1)

    
t2= Convolution1D(filters=12, kernel_size=2, strides=1, activation='relu', padding='same')(t0)
t2= AveragePooling1D(pool_size=2, strides=2, padding='same')(t2)
t2= Convolution1D(filters=24, kernel_size=2, strides=1, activation='relu', padding='same')(t2)
t2= AveragePooling1D(pool_size=2, strides=2, padding='same')(t2)


t00 = concatenate([t1,t2], axis=1)
t3= Convolution1D(filters=48, kernel_size=2, strides=1, activation='relu', padding='same')(t00)
t3= AveragePooling1D(pool_size=2, strides=2, padding='same')(t3)
t3= Convolution1D(filters=96, kernel_size=2, strides=1, activation='relu', padding='same')(t3)
t3= AveragePooling1D(pool_size=2, strides=2, padding='same')(t3)

t4= Convolution1D(filters=48, kernel_size=2, strides=1, activation='relu', padding='same')(t00)
t4= AveragePooling1D(pool_size=2, strides=2, padding='same')(t4)
t4= Convolution1D(filters=96, kernel_size=2, strides=1, activation='relu', padding='same')(t4)
t4= AveragePooling1D(pool_size=2, strides=2, padding='same')(t4)



output = concatenate([t3,t4], axis=1)
output = Flatten()(output)
output = Dropout(0.5)(output)
output = Dense(units = 50,kernel_initializer = 'uniform', activation = 'relu')(output)
output = Dropout(0.2)(output)
output = Dense(units = l,kernel_initializer = 'uniform', activation = 'softmax')(output)

model = Model(inputs=[inp], outputs=[output])




from keras.optimizers import Adam
opt=Adam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
model.compile(optimizer =opt, loss = 'categorical_crossentropy', metrics=['accuracy'])
print(model.summary())
Tic=time.time()
history=model.fit(X_train, label_Tr,validation_split=0.1, epochs = 200, batch_size = 100)
toc=time.time()-Tic
print(history.history.keys())


acc_Batch2=[ history.history['accuracy'], history.history['val_accuracy'],history.history['loss'],history.history['val_loss']]

plt.figure(figsize=(8, 4), dpi=100)

plt.plot(history.history['accuracy'],linewidth=5)
plt.plot(history.history['val_accuracy'],linewidth=5)
plt.title('Model accuracy CNN',fontsize=18)
plt.ylabel('Average Accuracy [0-1]',fontsize=18)
plt.xlabel('No. of Epoch',fontsize=18)# summarize history for loss
plt.rcParams['font.family'] = 'Times new Roman'


plt.rcParams['font.weight'] = 'bold'
plt.rcParams['legend.fontsize'] = 'medium'
plt.rcParams['figure.titlesize'] = 'large'

plt.legend(['Training Acuracy', 'Validation Accuracy'], loc='lower right')
plt.tight_layout()
plt.show()
plt.grid(True)

plt.figure(figsize=(8, 4), dpi=100)

plt.plot(history.history['loss'],'red',linewidth=5)
plt.plot(history.history['val_loss'], 'green',linewidth=5)

plt.title('Model loss( Categorical-Cross-Entropy)',fontsize=18)
plt.ylabel('Loss[Arbitrary Unit]',fontsize=18)
plt.xlabel('No. of Epoch',fontsize=18)
plt.legend(['Training Loss', 'Validation Loss'], fontsize=18, loc='upper right')
plt.tight_layout()
plt.show()
plt.grid(True)

from sklearn.metrics import confusion_matrix,classification_report
from sklearn.metrics import roc_curve, auc
from scipy import interp
from sklearn.metrics import roc_auc_score
from itertools import cycle

y_pred = model.predict(X_test)
n_classes = 5


plt.figure(figsize=(8, 4), dpi=100)

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve( label_Test[:, i], y_pred[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve( label_Test.ravel(), y_pred.ravel())
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
plt.legend(loc="lower right")
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
plt.legend(loc="lower right")
plt.show()

y_pred = (y_pred > 0.5).astype(int) #some of the class multiple class assignments

y_out=y_pred.argmax(1)
y_label=label_Test.argmax(1)
cm = confusion_matrix(y_label, y_out)
print(classification_report(y_label, y_out))
import itertools
from sklearn.metrics import confusion_matrix,classification_report 
y_pred = model.predict(X_test)
y_pred = (y_pred > 0.5).astype(int) #some of the class multiple class assignments
y_out=y_pred.argmax(1)
y_label=label_Test.argmax(1)
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
#model.save('0_vs_2_first_1_sub_95.h5')
