import sys
import torch
from annotability_automations import *
from dataset import *
from merfish import Merfish
from minimal import Minimal
from pbmc import PBMC
from anndata_manager import *

### GLOBAL PARAMETERS ###
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
repeats_per_size = 12
train_sizes = [500]
### END GLOBAL PARAMETERS ###

dataset = PBMC()
dataset.load_data()
adata = dataset.preprocess_data()
adata = dataset.filter_by_health(clear_sick=True)
label_key = 'cell_type'
epoch_num_annot = 3
epoch_num_composition = 2
swap_probability = 0.1
percentile = 90
batch_size = 128

format_manager = AnnDataManager()

# format_manager.general_info(adata)
adata, group_counts = annotate("pbmc_healthy", adata, label_key, epoch_num_annot, device, swap_probability, percentile, batch_size)
comp_opt_subset_to_not(
    "pbmc_healthy", adata, label_key, 
    {'Easy-to-learn':205, 'Ambiguous':295, 'Hard-to-learn':0}, 
    device, epoch_num_composition, epoch_num_annot, batch_size, 
    format_manager
    )

# create_comps_for_workers(
#     "pbmc_healthy", adata,
#     train_sizes=train_sizes, repeats_per_size=repeats_per_size,
#     )