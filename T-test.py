import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors

# -------------------------------
# Define Models and Data types
# -------------------------------
models = [
    'SVM', 'LDA', 'QDA', 'Single CNN', 'Dual CNN', 'Quads CNN',
    'MS Fusion CNN', 'Hybrid CNN+LSTM', 'TCN', 'Transformer', 'GRU'
]
data_types = ['GRF', 'EMG', 'Markers', 'F&M']

# -------------------------------
# F1 scores (%)
# -------------------------------
f1_scores = np.array([
    [[66.3,31.3,30.1],[35.7,35,34.1],[94.1,89.6,89.1],[87,84.2,87]],
    [[68.6,29.7,33.6],[36.1,35.1,16.5],[67.4,65.2,61.9],[78,79.5,80]],
    [[46.9,24.3,22.7],[30.2,22.3,16],[58.2,46.8,49.7],[77,50.7,55.8]],
    [[91.2,56.5,63.6],[92.1,94,90],[95.4,92.8,91],[91,86.2,87.9]],
    [[88.4,64.8,63.09],[94.3,91.3,90.5],[95.3,92.7,91.3],[93,90.09,89.2]],
    [[82.4,81.6,81.3],[93.3,91.5,92.4],[96.6,96.2,95.9],[93,91,89]],
    [[96,93,90],[96,93,90],[96,93,90],[96,93,90]],
    [[95,90,91.2],[95,90,91.2],[95,90,91.2],[95,90,91.2]],
    [[64.7,63.72,61.1],[37.06,34.1,37.06],[93.3,94,93.5],[91,88.3,80]],
    [[44.08,41.6,39.7],[61.1,62.3,59.1],[93.77,89.8,89.5],[87,89.4,81]],
    [[43.18,44.6,37.6],[51.4,54.7,53.6],[85,87,81],[75,77,79.3]]
])

# -------------------------------
# Standard deviations (%)
# -------------------------------
f1_std = np.array([
    [[3.6,6.6,8.4],[3.1,4.2,7.3],[1.7,3.5,2.7],[2.3,3.7,2.1]],
    [[5,1.9,4.8],[3.5,2.09,10.7],[5.5,3.5,7.8],[6.8,3.6,5.6]],
    [[4.9,3.5,12.6],[3.3,5.9,3.6],[9.6,9.7,3.9],[2.4,8.3,4.4]],
    [[4.4,1,1.4],[3.2,1.8,3.1],[2.6,4.54,4.2],[1.8,1.2,2.8]],
    [[4.4,13.6,10.8],[1.6,6.8,2.1],[1.1,3.52,1.61],[1.75,3.77,1.2]],
    [[1.4,2.3,1.3],[1.0,1.3,1.4],[1.4,1.6,0.9],[1.3,2.0,1.8]],
    [[0.3,1.3,2.8],[0.3,1.3,2.8],[0.3,1.3,2.8],[0.3,1.3,2.8]],
    [[0.6,2.6,1.7],[0.6,2.6,1.7],[0.6,2.6,1.7],[0.6,2.6,1.7]],
    [[4.4,4.4,12.07],[7.4,8.6,7.4],[1.68,1.1,2.1],[2.34,2.6,1.9]],
    [[8.4,3.7,6.5],[5.6,4.1,1],[1.09,2.1,2.6],[2.08,2.65,2.87]],
    [[7.4,4.6,7.4],[8.56,10.01,9.8],[1.35,2.6,1.2],[2.35,3.2,1.9]]
])

# -------------------------------
# Plotting
# -------------------------------
experiments = ['Exp1', 'Exp2', 'Exp3']
x = np.arange(len(experiments))
width = 0.02
plt.figure(figsize=(22,8))

# Generate 132 unique colors
all_colors = list(mcolors.CSS4_COLORS.values())  # big list
unique_colors = all_colors[:len(models)*len(data_types)*len(experiments)]  # pick first 132 colors

color_idx = 0
num_models = len(models)
num_data = len(data_types)
total_bars = num_models * num_data

for exp_idx in range(len(experiments)):
    for i, model in enumerate(models):
        for j, data in enumerate(data_types):
            color = unique_colors[color_idx]
            color_idx += 1
            bar_pos = x[exp_idx] + width*(i*num_data + j - total_bars/2)
            plt.bar(bar_pos, f1_scores[i,j,exp_idx], width=width, yerr=f1_std[i,j,exp_idx],
                    capsize=3, label=f'{model}-{data}' if exp_idx==0 else "", color=color, alpha=0.9)

plt.xticks(x, experiments)
plt.ylabel('F1 Score (%)')
plt.title('F1 Scores for All Models, Data Types, and Experiments')

# Legend
handles, labels = plt.gca().get_legend_handles_labels()
plt.legend(handles, labels,
           loc='upper center',
           bbox_to_anchor=(0.5, -0.15),
           ncol=6,
           fontsize='small')

plt.grid(axis='y')
plt.tight_layout()
plt.show()

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel
import seaborn as sns
import matplotlib.pyplot as plt

# ------------------------------
# 1. Define the data (mean ± std)
# ------------------------------
data = {
    'Model': ['SVM','LDA','QDA','Single CNN','Dual CNN','Quads CNN',
              'Multi-Source Fusion CNN','Hybrid CNN+LSTM','TCN','Transformer','GRU'],
    'GRF': [(66.3,3.6),(68.6,5),(46.9,4.9),(91.2,4.4),(88.4,4.4),(82.4,1.4),(96,0.3),(95,0.6),(64.7,4.4),(44.08,8.4),(43.18,7.4)],
    'EMG': [(35.7,3.1),(36.1,3.5),(30.2,3.3),(92.1,3.2),(94.3,1.6),(93.3,1.0),(96,0.3),(95,0.6),(37.06,7.4),(61.1,5.6),(51.4,8.56)],
    'Markers': [(94.1,1.7),(67.4,5.5),(58.2,9.6),(95.4,2.6),(95.3,1.1),(96.6,1.4),(96,0.3),(95,0.6),(93.3,1.68),(93.77,1.09),(85,1.35)],
    'F&M': [(87,2.3),(78,6.8),(77,2.4),(91,1.8),(93,1.75),(93,1.3),(96,0.3),(95,0.6),(91,2.34),(87,2.08),(75,2.35)]
}

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel
import seaborn as sns
import matplotlib.pyplot as plt

# ------------------------------
# 1. Define the data (mean ± std)
# ------------------------------
data = {
    'Model': list(range(1,12)),  # Replace model names with numbers 1-11
    'GRF': [(66.3,3.6),(68.6,5),(46.9,4.9),(91.2,4.4),(88.4,4.4),(82.4,1.4),(96,0.3),(95,0.6),(64.7,4.4),(44.08,8.4),(43.18,7.4)],
    'EMG': [(35.7,3.1),(36.1,3.5),(30.2,3.3),(92.1,3.2),(94.3,1.6),(93.3,1.0),(96,0.3),(95,0.6),(37.06,7.4),(61.1,5.6),(51.4,8.56)],
    'Markers': [(94.1,1.7),(67.4,5.5),(58.2,9.6),(95.4,2.6),(95.3,1.1),(96.6,1.4),(96,0.3),(95,0.6),(93.3,1.68),(93.77,1.09),(85,1.35)],
    'F&M': [(87,2.3),(78,6.8),(77,2.4),(91,1.8),(93,1.75),(93,1.3),(96,0.3),(95,0.6),(91,2.34),(87,2.08),(75,2.35)]
}

# ------------------------------
# 1. Custom format function for heatmap annotations
# ------------------------------
def format_value(val):
    if np.isnan(val):
        return ""
    elif val < 0.1:
        return "{:.1e}".format(val)  # scientific notation for very small numbers
    else:
        return "{:.3f}".format(val)  # 3 digits for normal numbers

# ------------------------------
# 2. Simulate samples for each model, data type, and experiment
# ------------------------------
experiments = ['Exp1','Exp2','Exp3']
data_types = ['GRF','EMG','Markers','F&M']
n_samples = 10  # simulate 10 samples per mean ± std

simulated = {}
for model_idx, model in enumerate(data['Model']):
    simulated[model] = {}
    for dt in data_types:
        simulated[model][dt] = {}
        mean, std = data[dt][model_idx]
        simulated[model][dt]['Exp1'] = np.random.normal(mean,std,n_samples)
        simulated[model][dt]['Exp2'] = np.random.normal(mean,std,n_samples)
        simulated[model][dt]['Exp3'] = np.random.normal(mean,std,n_samples)

# ------------------------------
# 3. Paired t-test between models for each data type AND each experiment
# ------------------------------
for dt in data_types:
    for exp in experiments:
        models = data['Model']
        p_matrix_models = pd.DataFrame(np.zeros((len(models),len(models))), index=models, columns=models)
        
        for i, m1 in enumerate(models):
            for j, m2 in enumerate(models):
                if i < j:
                    t_stat, p_val = ttest_rel(simulated[m1][dt][exp], simulated[m2][dt][exp])
                    p_matrix_models.loc[m1,m2] = p_val
                    p_matrix_models.loc[m2,m1] = p_val
                elif i==j:
                    p_matrix_models.loc[m1,m2] = np.nan

        print(f"\nPaired t-test P-values between models ({dt}, {exp}):")
        print(p_matrix_models)

        # Format values for heatmap
        annot_data = p_matrix_models.applymap(format_value)
        
        plt.figure(figsize=(16,8))
        sns.heatmap(p_matrix_models, annot=annot_data, fmt="", cmap='coolwarm', 
                    cbar_kws={'label':'p-value'}, annot_kws={"size":20})
        plt.title(f"Paired t-test P-values between Models ({dt}, {exp})", fontsize=20)
        plt.xlabel('Models', fontsize=20)
        plt.ylabel('Models', fontsize=20)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.show()

# ------------------------------
# 4. Paired t-test between experiments for each model and data type
# ------------------------------
for dt in data_types:
    p_matrix_experiments = pd.DataFrame(np.zeros((len(models),3)), index=models, columns=['Exp1_vs_Exp2','Exp1_vs_Exp3','Exp2_vs_Exp3'])
    
    for model in models:
        t_stat, p_val = ttest_rel(simulated[model][dt]['Exp1'], simulated[model][dt]['Exp2'])
        p_matrix_experiments.loc[model,'Exp1_vs_Exp2'] = p_val
        t_stat, p_val = ttest_rel(simulated[model][dt]['Exp1'], simulated[model][dt]['Exp3'])
        p_matrix_experiments.loc[model,'Exp1_vs_Exp3'] = p_val
        t_stat, p_val = ttest_rel(simulated[model][dt]['Exp2'], simulated[model][dt]['Exp3'])
        p_matrix_experiments.loc[model,'Exp2_vs_Exp3'] = p_val

    print(f"\nPaired t-test P-values between experiments for each model ({dt}):")
    print(p_matrix_experiments)

    # Format values for heatmap
    annot_data_exp = p_matrix_experiments.applymap(format_value)
    
    plt.figure(figsize=(10,16))
    sns.heatmap(p_matrix_experiments, annot=annot_data_exp, fmt="", cmap='coolwarm', 
                cbar_kws={'label':'p-value'}, annot_kws={"size":20})
    plt.title(f"Paired t-test P-values between Experiments for each Model ({dt})", fontsize=20)
    plt.xlabel('Comparisons', fontsize=20)
    plt.ylabel('Models', fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.show()