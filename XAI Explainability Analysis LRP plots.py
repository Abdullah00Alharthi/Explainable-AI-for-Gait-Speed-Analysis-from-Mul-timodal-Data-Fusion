import numpy as np
from keras.utils import to_categorical
import matplotlib.pyplot as plt


X=np.load('MARKERS_DATA.npy')
#X=X[:, :-435, :]

X=X.reshape((1142, 585, 156, 1))


y=np.load('SPEED_L.npy')
l=4
for i in range(len(y)):
    if np.all((y[i]==3)):
      y[i]=2 
      
for i in range(len(y)):
    if np.all((y[i]==4)):
      y[i]=3 

X=np.nan_to_num(X)





from sklearn.model_selection import train_test_split
X_train, X_test, label_Tr, label_Test = train_test_split(X, y, test_size = 0.2, random_state = 42)



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


from keras.models import load_model
from keras.optimizers import adam


model = load_model('4_3DCNN_MARKERS.h5')

opt=adam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
model.compile(loss='categorical_crossentropy',optimizer=opt,metrics=['accuracy'])

r=X_test[64]

i = r[np.newaxis]

y_pred = model.predict(i)
y_pred = (y_pred > 0.5).astype(int)

y_out=y_pred.argmax(1)

print (y_out)

import innvestigate
import innvestigate.utils

model = innvestigate.utils.model_wo_softmax(model)

analyzer = innvestigate.create_analyzer("lrp.sequential_preset_a_flat", model)

# Apply analyzer w.r.t. maximum activated output-neuron
a = analyzer.analyze(i)
a = a.sum(axis=np.argmax(np.asarray(a.shape) == 2))
a /= np.max(np.abs(a))

a=a.reshape((585,156)) 
r=r.reshape((585,156)) 

#a=np.mean(a,axis=1)
#r=np.mean(r,axis=1)

a=a[0:120:]
r=r[0:120:]




import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# --- Define 52 anatomical labels ---
feature_labels = [
    "L_IAS", "L_IPS", "R_IPS", "R_IAS", "L_FTC", "L_FLE", "L_FME", "L_FAX",
    "L_TTC", "L_FAL", "L_TAM", "L_FCC", "L_FM1", "L_FM2", "L_FM5",
    "R_FTC", "R_FLE", "R_FME", "R_FAX", "R_TTC", "R_FAL", "R_TAM", "R_FCC",
    "R_FM1", "R_FM2", "R_FM5", "CV7", "TV10", "SXS", "SJN", "L_SIA", "L_SRS",
    "L_SAA", "L_SAE", "L_HLE", "L_HME", "L_UOA", "L_RSP", "L_UHE", "L_HM2",
    "L_HM5", "R_SIA", "R_SRS", "R_SAA", "R_SAE", "R_HLE", "R_HME", "R_UOA",
    "R_RSP", "R_UHE", "R_HM2", "R_HM5"
]

# Repeat each label 3 times (for 3 readings)
expanded_labels = np.repeat(feature_labels, 3)

# --- Line plots ---
plt.figure(figsize=(14, 10), dpi=120)

# Input signal line plot
plt.subplot(2, 1, 1)
plt.plot(r, linewidth=1.5)
plt.title("Input Signal Line Plot", fontsize=14, pad=8, weight='bold')
plt.xlabel("Time Steps", fontsize=11)
plt.ylabel("Signal Value", fontsize=11)
plt.xticks(fontsize=9)
plt.yticks(fontsize=9)
plt.grid(True, linestyle='--', alpha=0.5)

# LRP feature relevance line plot
plt.subplot(2, 1, 2)
plt.plot(a, linewidth=1.5)
plt.title("Feature Relevance Line Plot (LRP)", fontsize=14, pad=8, weight='bold')
plt.xlabel("Time Steps", fontsize=11)
plt.ylabel("Relevance Value", fontsize=11)
plt.xticks(fontsize=9)
plt.yticks(fontsize=9)
plt.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()

# --- Heatmap 1: Input Signal ---
plt.figure(figsize=(14, 6), dpi=120)
ax = sns.heatmap(
    r,
    cmap="viridis",
    cbar_kws={'label': 'Input Signal Value'},
    linewidths=0.2,
    linecolor='black'
)
plt.title("Input Signal Heatmap", fontsize=14, pad=8, weight='bold')
plt.xlabel("Feature (Body Segment)", fontsize=11, labelpad=5)
plt.ylabel("Time Steps", fontsize=11, labelpad=5)
ax.set_xticks(np.arange(1.5, len(expanded_labels), 3))
ax.set_xticklabels(feature_labels, rotation=90, fontsize=8)
ax.set_yticklabels(ax.get_yticks().astype(int), fontsize=8)
plt.tight_layout()
plt.show()

# --- Heatmap 2: LRP Feature Relevance ---
plt.figure(figsize=(14, 6), dpi=120)
ax = sns.heatmap(
    a,
    cmap="coolwarm",
    cbar_kws={'label': 'LRP Feature Relevance'},
    linewidths=0.2,
    linecolor='black'
)
plt.title("Feature Relevance Heatmap (LRP)", fontsize=14, pad=8, weight='bold')
plt.xlabel("Feature (Body Segment)", fontsize=11, labelpad=5)
plt.ylabel("Time Steps", fontsize=11, labelpad=5)
ax.set_xticks(np.arange(1.5, len(expanded_labels), 3))
ax.set_xticklabels(feature_labels, rotation=90, fontsize=8)
ax.set_yticklabels(ax.get_yticks().astype(int), fontsize=8)
plt.tight_layout()
plt.show()
