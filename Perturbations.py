import warnings
warnings.simplefilter('ignore')
import matplotlib.pyplot as plt
import numpy as np
import keras
import keras.backend
import keras.models
import innvestigate
import innvestigate.utils as iutils
from innvestigate.tools import Perturbation, PerturbationAnalysis


X=np.load('MARKERS_DATA.npy')
X=X.reshape((1142, 585, 156, 1))

# X=np.load('GRF_DATA.npy')
# X=X.reshape((1142, 585, 24, 1))

# X=np.load('EMG_DATA.npy')
# X=X.reshape((1142, 8985, 8, 1))


#y=np.load('ID_L.npy')
y=np.load('SPEED_L.npy')
l=4
for i in range(len(y)):
    if np.all((y[i]==3)):
      y[i]=2 
      
for i in range(len(y)):
    if np.all((y[i]==4)):
      y[i]=3 






from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
X_train, X_test, label_Tr, label_Test = train_test_split(X, y, test_size = 0.2, random_state = 62)
label_Tr=to_categorical(label_Tr)
label_Test=to_categorical(label_Test)




def standardize(train,test):
	""" Standardize data """
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


perturbation_function = "gaussian"
region_shape  = (21,21)
steps = 25
regions_per_step = 1 
generator = iutils.BatchSequence([X_test, label_Test], batch_size=100)
input_range=[np.amin(X_test),np.amax(X_test)]
noise_scale = (input_range[1]-input_range[0]) * 0.1
methods = [
    ("input",                 {},                       X_test,      "Input"),
    ("deep_taylor.bounded",   {"low": input_range[0],
                               "high": input_range[1]}, X_test,        "Deep Taylor"),
    
    ("lrp.alpha_2_beta_1",       {},                        X_test,        "lrp alpha2_beta_1",),

    ("lrp.z",                 {},                       X_test,        "LRP-Z"),
    ("lrp.epsilon",           {"epsilon": 1},           X_test,        "LRP-Epsilon"),
  
    ("lrp.sequential_preset_a_flat",{"epsilon": 1},     X_test,       "LRP-PresetAFlat"),
    ("gradient",            {"postprocess": "abs"},     X_test,       "Gradient"),
    ("deconvnet",           {},                         X_test,        "Deconvnet"),
    ("guided_backprop",     {},                         X_test,         "Guided Backprop"),
    
]
selected_methods_indices = [0,1,2,3,4,5,6,7,8]
selected_methods = [methods[i] for i in selected_methods_indices]
print('Using method(s) "{}".'.format([method[0] for method in selected_methods]))
model_wo_softmax = iutils.keras.graph.model_wo_softmax(model)

analyzers = [innvestigate.create_analyzer(method[0],
                                        model_wo_softmax,
                                        **method[1]) for method in selected_methods]


scores_selected_methods = dict()
perturbation_analyses = list()
for method, analyzer in zip(selected_methods, analyzers):
    print("Method: {}".format(method[0]))
    perturbation = Perturbation(perturbation_function, region_shape=region_shape, in_place=False)
    perturbation_analysis = PerturbationAnalysis(analyzer, model, generator, perturbation, recompute_analysis=False,
                                                steps=steps, regions_per_step=regions_per_step, verbose=False)
    scores = perturbation_analysis.compute_perturbation_analysis()
    scores_selected_methods[method[0]] = np.array(scores)
    perturbation_analyses.append(perturbation_analysis)
    print()
  
    
plt.figure(figsize=(12, 6), dpi=110)
plt.rcParams['font.family'] = 'Arils'
plt.rcParams['font.size'] = 18
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['legend.fontsize'] = 'medium'
plt.rcParams['figure.titlesize'] = 'medium'
aopc = list()  # Area over the perturbation curve
baseline_accuracy = scores_selected_methods["input"][:, 1]
for method_name in scores_selected_methods.keys():
    scores = scores_selected_methods[method_name]
    accuracy = scores[:, 1]
#    aopc.append(accuracy[0] - np.mean(accuracy))
    
    label = "{} (accuracy: {:.3f})".format(method_name, accuracy[-1])
    plt.plot(accuracy - baseline_accuracy, label=label,linewidth=2)
plt.title('Compression of XAI Analyzers')    
plt.xlabel("Perturbation steps")
plt.ylabel("Difference of accuracy")
plt.xticks(np.array(range(scores.shape[0])))
plt.legend(prop={'size':8})
plt.grid(True, color = "blue", linestyle = "--")
plt.show()




import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

plt.figure(figsize=(10, 6), dpi=110)
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 16
plt.rcParams['font.weight'] = 'bold'

# Prepare heatmap data
baseline_accuracy = scores_selected_methods["input"][:, 1]

# Collect accuracy differences for all methods
heatmap_data = []
method_names = []

for method_name, scores in scores_selected_methods.items():
    accuracy = scores[:, 1]
    diff = accuracy - baseline_accuracy
    heatmap_data.append(diff)
    method_names.append(method_name)

# Convert to numpy array
heatmap_data = np.array(heatmap_data)

# Plot heatmap
ax = sns.heatmap(
    heatmap_data,
    cmap="coolwarm",
    annot=False,
    cbar_kws={'label': 'Difference of Accuracy'}
)

# Title and labels
plt.title("Compression of XAI Analyzers (Heatmap)")
plt.xlabel("Perturbation Steps")
plt.ylabel("Methods")

# ✅ Show only every 5th step on x-axis
step_interval = 5
x_positions = np.arange(heatmap_data.shape[1])
x_labels = np.arange(heatmap_data.shape[1])

plt.xticks(
    x_positions[::step_interval] + 0.5,  # tick positions centered in cells
    x_labels[::step_interval],
    rotation=0
)
plt.yticks(np.arange(len(method_names)) + 0.5, method_names, rotation=0)

plt.tight_layout()
plt.show()

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

plt.figure(figsize=(12, 6), dpi=120)
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 14
plt.rcParams['font.weight'] = 'bold'

# Prepare heatmap data
baseline_accuracy = scores_selected_methods["input"][:, 1]

heatmap_data = []
method_names = []

for method_name, scores in scores_selected_methods.items():
    accuracy = scores[:, 1]
    diff = accuracy - baseline_accuracy
    heatmap_data.append(diff)
    method_names.append(method_name)

heatmap_data = np.array(heatmap_data)

# ✅ Safe normalization around zero without TwoSlopeNorm
vmax = np.max(np.abs(heatmap_data))
vmin = -vmax

# Plot heatmap
ax = sns.heatmap(
    heatmap_data,
    cmap="RdBu_r",
    vmin=vmin,
    vmax=vmax,
    cbar_kws={'label': 'Δ Accuracy (relative to baseline)'},
    linewidths=0.5,
    linecolor='white'
)

plt.title("Perturbation Sensitivity of XAI Methods", fontsize=16, fontweight='bold')
plt.xlabel("Perturbation Steps")
plt.ylabel("XAI Methods")

# Show only every 5th x label
step_interval = 5
x_positions = np.arange(heatmap_data.shape[1])
x_labels = np.arange(heatmap_data.shape[1])
plt.xticks(x_positions[::step_interval] + 0.5, x_labels[::step_interval], rotation=0)
plt.yticks(np.arange(len(method_names)) + 0.5, method_names, rotation=0)

plt.tight_layout()
plt.show()

import matplotlib.pyplot as plt
import numpy as np

plt.figure(figsize=(10, 5), dpi=120)
for method_name, scores in scores_selected_methods.items():
    accuracy = scores[:, 1]
    diff = accuracy - scores_selected_methods["input"][:, 1]
    plt.plot(diff, label=method_name, linewidth=2)
    # Add smooth trend if needed (optional)
plt.xlabel("Perturbation Steps")
plt.ylabel("Δ Accuracy")
plt.title("Accuracy Degradation Across Perturbation Steps")
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(loc='best', fontsize=9)
plt.tight_layout()
plt.show()
