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
train_sizes = [500, 1000]
### END GLOBAL PARAMETERS ###

# Read the data argument from the command line
if len(sys.argv) < 2:
    print("Usage: python script.py <data>")
    sys.exit(1)

dataset_name = sys.argv[1]

if dataset_name == 'merfish':
    dataset = Merfish()
    dataset.load_data()
    adata = dataset.preprocess_data()
    label_key = 'CellType'
    epoch_num_annot = 150
    epoch_num_composition = 30
    swap_probability = 0.1
    percentile = 90
    batch_size = 64

# if dataset_name == 'minimal':
#     dataset = Minimal()
#     dataset.load_data()
#     adata = dataset.preprocess_data()
#     label_key = 'CellType'
#     epoch_num_annot = 1
#     epoch_num_composition = 1
#     swap_probability = 0.1
#     percentile = 90
#     batch_size = 2048

if dataset_name == 'pbmc':
    dataset = PBMC()
    dataset.load_data()
    adata = dataset.preprocess_data()
    label_key = 'cell_type'
    epoch_num_annot = 40
    epoch_num_composition = 20
    swap_probability = 0.1
    percentile = 90
    batch_size = 64

if dataset_name == 'pbmc_healthy':
    dataset = PBMC()
    dataset.load_data()
    adata = dataset.preprocess_data()
    adata = dataset.filter_by_health(clear_sick=True)
    label_key = 'cell_type'
    epoch_num_annot = 50
    epoch_num_composition = 25
    swap_probability = 0.1
    percentile = 90
    batch_size = 64

general_info(adata)
adata, group_counts = annotate(dataset_name, adata, label_key, epoch_num_annot, device, swap_probability, percentile, batch_size)
best_compositions, label_encoder = find_optimal_compositions(dataset_name, adata, label_key, group_counts, train_sizes, 
                        repeats_per_size, device, epoch_num_composition, batch_size)
visualize_optimal_compositions(dataset_name)
highest_confidence_samples(adata, train_sizes, device, label_encoder, dataset_name, label_key)
comp_opt_subset_to_not(dataset_name, adata, label_key, group_counts, device, epoch_num_composition, epoch_num_annot, batch_size, "anndata_manager")
