from pathlib import Path
from itertools import combinations
import warnings
print("start import")
warnings.filterwarnings("ignore")
import sys
print(sys.executable)
print('\n'.join(sys.path))
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# import scanpy as sc
# from sklearn.preprocessing import normalize
# from sklearn.metrics import roc_auc_score, pairwise_distances, accuracy_score
import torch
# import torch.optim as optim
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# import scipy.sparse as sp
# from numba import jit
from Annotatability import models
# from Annotatability import metrics
# import scvi
import logging
# import squidpy as sq
# import contextlib
# import io
import random
import numpy as np
# import pandas as pd
import scanpy as sc
import squidpy as sq
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
# from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder, OneHotEncoder  # Added imports
import scipy.sparse as sp
# from sklearn.model_selection import train_test_split
print("finished import")
logging.getLogger("scvi").setLevel(logging.WARNING)

SMALL_SIZE = 16
MEDIUM_SIZE = 20.5
BIGGER_SIZE = 24
#plt.rcParams["font.family"] = "Verdana"
plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
sc.set_figure_params(scanpy=True, fontsize=20.5)

# Define custom color palette
annotation_order = ['Easy-to-learn', 'Ambiguous', 'Hard-to-learn']
annotation_colors = ['green', 'orange', 'red']
palette = dict(zip(annotation_order, annotation_colors))


from tqdm import tqdm
import contextlib
import io

# Initialize StringIO object to suppress outputs
f = io.StringIO()

def train_and_get_prob_list(adata, label_key, epoch_num, device=device, batch_size=128):
    print('Training the model...')
    with contextlib.redirect_stdout(f):
        prob_list = models.follow_training_dyn_neural_net(
            adata,
            label_key=label_key,
            iterNum=epoch_num,
            device=device,
            batch_size=batch_size
        )
    print('Training complete.')
    return prob_list

def calculate_confidence_and_variability(prob_list, n_obs, epoch_num):
    with contextlib.redirect_stdout(f):
        all_conf, all_var = models.probability_list_to_confidence_and_var(
            prob_list,
            n_obs=n_obs,
            epoch_num=epoch_num
        )
    return all_conf, all_var

def find_cutoffs(adata, label_key, device, probability, percentile, epoch_num):
    with contextlib.redirect_stdout(f):
        cutoff_conf, cutoff_var = models.find_cutoff_paramter(
            adata,
            label_key,
            device,
            probability=probability,
            percentile=percentile,
            epoch_num=epoch_num
        )
    return cutoff_conf, cutoff_var

def assign_annotations(adata, all_conf, all_var, cutoff_conf, cutoff_var, annotation_col='Annotation'):
    adata.obs["var"] = all_var.detach().numpy()
    adata.obs["conf"] = all_conf.detach().numpy()
    adata.obs['conf_binaries'] = pd.Categorical(
        (adata.obs['conf'] > cutoff_conf) | (adata.obs['var'] > cutoff_var)
    )

    annotation_list = []
    for i in tqdm(range(adata.n_obs), desc='Assigning annotations'):
        if adata.obs['conf_binaries'].iloc[i]:
            if (adata.obs['conf'].iloc[i] > 0.95) & (adata.obs['var'].iloc[i] < 0.15):
                annotation_list.append('Easy-to-learn')
            else:
                annotation_list.append('Ambiguous')
        else:
            annotation_list.append('Hard-to-learn')

    adata.obs[annotation_col] = annotation_list
    adata.obs['Confidence'] = adata.obs['conf']
    adata.obs['Variability'] = adata.obs['var']
    return adata

import scanpy as sc

# Load the dataset
# PBMC_full is obtained from https://cellxgene.cziscience.com/collections/03f821b4-87be-4ff4-b65a-b5fc00061da7
# Change to your path below
adata_full = sc.read_h5ad('PBMC_full.h5ad')

# Define the list of cell types to keep (updated to match actual cell types in the dataset)
cell_types_to_keep = [
    'B cell',
    'CD4-positive helper T cell',
    'naive thymus-derived CD8-positive, alpha-beta T cell',
    'naive thymus-derived CD4-positive, alpha-beta T cell',
    'classical monocyte',
    'dendritic cell',
    'CD16-negative, CD56-bright natural killer cell',
    'mature NK T cell'
]

# **Filter to include only healthy cells**
if 'COVID_status' in adata_full.obs.columns:
    if 'Healthy' in adata_full.obs['COVID_status'].unique():
        adata_healthy = adata_full[adata_full.obs['COVID_status'] == 'Healthy'].copy()
        print("Filtered healthy cells using 'COVID_status'.")
    else:
        raise ValueError("'Healthy' label not found in 'COVID_status' column.")
else:
    raise KeyError("'COVID_status' column not found in adata_full.obs.")

# Inspect the cell types available in the healthy dataset before filtering
print("Available cell types in the healthy dataset before filtering:")
print(adata_healthy.obs['cell_type'].unique())

# **Filter the data to include only the selected cell types**
adata_healthy = adata_healthy[adata_healthy.obs['cell_type'].isin(cell_types_to_keep)].copy()

# **Throw an error if no healthy cells are found after cell type filtering**
if adata_healthy.n_obs == 0:
    raise ValueError("No healthy cells found after filtering for the selected cell types.")

# Inspect the cell types available after filtering
print("Available cell types in the healthy dataset after filtering:")
print(adata_healthy.obs['cell_type'].unique())

# Normalize and log-transform the data
sc.pp.normalize_total(adata_healthy, target_sum=1e4)
sc.pp.log1p(adata_healthy)

pbmc = adata_healthy

# Final verification
print(f"Final number of cells after all filtering: {pbmc.n_obs}")
print(f"Final number of genes: {pbmc.n_vars}")

print("PBMC")
# Train the model and get probability list for PBMC dataset
prob_list_pbmc = train_and_get_prob_list(pbmc, label_key='cell_type', epoch_num=150, device=device, batch_size=64)

# Calculate confidence and variability for PBMC
all_conf_pbmc, all_var_pbmc = calculate_confidence_and_variability(prob_list_pbmc, n_obs=pbmc.n_obs, epoch_num=150)

# Find cutoffs for PBMC
conf_cutoff_pbmc, var_cutoff_pbmc = find_cutoffs(pbmc, 'cell_type', device, probability=0.1, percentile=90, epoch_num=150)

# Assign annotations
pbmc = assign_annotations(pbmc, all_conf_pbmc, all_var_pbmc, conf_cutoff_pbmc, var_cutoff_pbmc, annotation_col='Annotation')

# Count the number of cells in each group
group_counts_pbmc = pbmc.obs['Annotation'].value_counts()

print(group_counts_pbmc)

subset_size = 100000
pbmc = pbmc[pbmc.obs.sample(n=subset_size, random_state=42).index].copy()

# Assuming 'group_counts_pbmc' is a pandas Series with annotations as indices
group_counts_pbmc = pbmc.obs['Annotation'].value_counts()

# Assign counts to E_pbmc, A_pbmc, H_pbmc
E_pbmc = group_counts_pbmc.get('Easy-to-learn', 0)
A_pbmc = group_counts_pbmc.get('Ambiguous', 0)
H_pbmc = group_counts_pbmc.get('Hard-to-learn', 0)

# Get the indices of each group
easy_indices_pbmc = pbmc.obs.index[pbmc.obs['Annotation'] == 'Easy-to-learn'].tolist()
ambiguous_indices_pbmc = pbmc.obs.index[pbmc.obs['Annotation'] == 'Ambiguous'].tolist()
hard_indices_pbmc = pbmc.obs.index[pbmc.obs['Annotation'] == 'Hard-to-learn'].tolist()

# Fit LabelEncoder on the entire dataset labels
global_label_encoder_pbmc = LabelEncoder()
global_label_encoder_pbmc.fit(pbmc.obs['cell_type'])
num_classes_pbmc = len(global_label_encoder_pbmc.classes_)

# Verify the counts match E_pbmc, A_pbmc, H_pbmc
print("PBMC:")
print(f"Number of Easy-to-learn samples: {len(easy_indices_pbmc)}")
print(f"Number of Ambiguous samples: {len(ambiguous_indices_pbmc)}")
print(f"Number of Hard-to-learn samples: {len(hard_indices_pbmc)}")

# save indicies
np.save('easy_indices_pbmc.npy', easy_indices_pbmc)
np.save('ambiguous_indices_pbmc.npy', ambiguous_indices_pbmc)
np.save('hard_indices_pbmc.npy', hard_indices_pbmc)

#save label encoder
import joblib
joblib.dump(global_label_encoder_pbmc, 'global_label_encoder_pbmc.pkl')

#save group counts
np.save('group_counts_pbmc.npy', group_counts_pbmc)

#save E, A, H counts
np.save('E_pbmc.npy', E_pbmc)
np.save('A_pbmc.npy', A_pbmc)
np.save('H_pbmc.npy', H_pbmc)


