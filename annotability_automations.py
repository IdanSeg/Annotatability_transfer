import warnings

warnings.filterwarnings("ignore")
import sys
print(sys.executable)
print('\n'.join(sys.path))
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
import scipy.sparse as sp
from Annotatability import models
import logging
import contextlib
import io
import scanpy as sc
import torch.nn as nn
import random
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# Set up logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.getLogger("scvi").setLevel(logging.WARNING)

SMALL_SIZE = 16
MEDIUM_SIZE = 20.5
BIGGER_SIZE = 24

# Define custom color palette
annotation_order = ['Easy-to-learn', 'Ambiguous', 'Hard-to-learn']
annotation_colors = ['green', 'orange', 'red']
palette = dict(zip(annotation_order, annotation_colors))

# Initialize StringIO object to suppress outputs
f = io.StringIO()

def train_and_get_prob_list(adata, label_key, epoch_num, device, batch_size):
    logging.info('Training the model...')
    with contextlib.redirect_stdout(f):
        prob_list = models.follow_training_dyn_neural_net(
            adata,
            label_key=label_key,
            iterNum=epoch_num,
            device=device,
            batch_size=batch_size
        )
    logging.info('Training complete.')
    return prob_list

def calculate_confidence_and_variability(prob_list, n_obs, epoch_num):
    logging.info('Calculating confidence and variability...')
    with contextlib.redirect_stdout(f):
        all_conf, all_var = models.probability_list_to_confidence_and_var(
            prob_list,
            n_obs=n_obs,
            epoch_num=epoch_num
        )
    logging.info('Calculation complete.')
    return all_conf, all_var

def find_cutoffs(adata, label_key, device, probability, percentile, epoch_num):
    logging.info('Finding cutoffs...')
    with contextlib.redirect_stdout(f):
        cutoff_conf, cutoff_var = models.find_cutoff_paramter(
            adata,
            label_key,
            device,
            probability=probability,
            percentile=percentile,
            epoch_num=epoch_num
        )
    logging.info('Cutoffs found: cutoff_conf=%s, cutoff_var=%s', cutoff_conf, cutoff_var)
    return cutoff_conf, cutoff_var

def assign_annotations(adata, all_conf, all_var, cutoff_conf, cutoff_var, annotation_col='Annotation'):
    logging.info('Assigning annotations...')
    adata.obs["var"] = all_var.detach().numpy()
    adata.obs["conf"] = all_conf.detach().numpy()
    adata.obs['conf_binaries'] = pd.Categorical(
        (adata.obs['conf'] > cutoff_conf) | (adata.obs['var'] > cutoff_var)
    )

    annotation_list = []
    # Disable tqdm output by setting disable=True
    for i in tqdm(range(adata.n_obs), desc='Assigning annotations', disable=True):
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
    logging.info('Annotation assignment complete.')
    return adata

class BaseNet(nn.Module):
    def __init__(self, layer_sizes):
        super(BaseNet, self).__init__()
        layers = []
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = torch.relu(layer(x))
        x = self.layers[-1](x)
        return torch.log_softmax(x, dim=1)

class Net(BaseNet):
    def __init__(self, input_size, output_size):
        layer_sizes = [input_size, int(input_size / 2), int(input_size / 4), output_size]
        super(Net, self).__init__(layer_sizes)

def one_hot_encode(labels, label_encoder):
    logging.debug('One-hot encoding labels...')
    values = np.array(labels)
    integer_encoded = label_encoder.transform(values)
    onehot_encoder = OneHotEncoder(sparse_output=False, categories='auto')
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    logging.debug('One-hot encoding complete.')
    return onehot_encoded

def is_scipy_cs_sparse(matrix):
    logging.debug('Checking if matrix is scipy csr sparse...')
    result = sp.issparse(matrix) and matrix.getformat() == 'csr'
    logging.debug('Matrix is scipy csr sparse: %s', result)
    return result

def train_and_evaluate_model(
    adata_train, 
    adata_test, 
    label_key, 
    label_encoder, 
    num_classes,      
    epoch_num, 
    device,         
    batch_size
):
    """
    Trains and evaluates a neural network model on the provided training and testing data.
    
    Parameters:
    - adata_train (AnnData): Training dataset.
    - adata_test (AnnData): Testing dataset.
    - label_key (str): Key in adata.obs that contains the labels.
    - label_encoder (LabelEncoder): Fitted LabelEncoder instance.
    - num_classes (int): Number of unique classes in the dataset.
    - epoch_num (int): Number of training epochs.
    - device (str or torch.device): Device to run the training on ('cpu' or 'cuda').
    - batch_size (int): Batch size for training.
    
    Returns:
    - test_loss (float): Loss on the test dataset.
    """
    logging.info('Starting training and evaluation of the model...')
    # Encode labels using the provided label encoder
    logging.debug('Encoding labels for training and testing datasets...')
    one_hot_label_train = one_hot_encode(adata_train.obs[label_key], label_encoder=label_encoder)
    one_hot_label_test = one_hot_encode(adata_test.obs[label_key], label_encoder=label_encoder)

    # Initialize the neural network
    logging.debug('Initializing the neural network...')
    net = Net(adata_train.X.shape[1], output_size=num_classes)
    net.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    # Prepare training data
    logging.debug('Preparing training data...')
    if is_scipy_cs_sparse(adata_train.X):
        x_train = adata_train.X.toarray()
    else:
        x_train = np.array(adata_train.X)
    tensor_x_train = torch.Tensor(x_train).to(device)
    tensor_y_train = torch.LongTensor(np.argmax(one_hot_label_train, axis=1)).to(device)
    train_dataset = TensorDataset(tensor_x_train, tensor_y_train)
    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Prepare test data
    logging.debug('Preparing test data...')
    if is_scipy_cs_sparse(adata_test.X):
        x_test = adata_test.X.toarray()
    else:
        x_test = np.array(adata_test.X)
    tensor_x_test = torch.Tensor(x_test).to(device)
    tensor_y_test = torch.LongTensor(np.argmax(one_hot_label_test, axis=1)).to(device)

    # Train the network
    logging.info('Starting training for %d epochs...', epoch_num)
    net.train()
    for epoch in range(epoch_num):
        logging.debug('Epoch %d/%d', epoch + 1, epoch_num)
        for inputs, labels in trainloader:
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    logging.info('Training complete.')

    # Evaluate on test set
    logging.info('Evaluating the model on the test dataset...')
    net.eval()
    with torch.no_grad():
        outputs = net(tensor_x_test)
        test_loss = criterion(outputs, tensor_y_test).item()
    logging.info('Evaluation complete. Test loss: %f', test_loss)

    logging.info('Training and evaluation process completed.')
    return test_loss

def annotate(adata, label_key, epoch_num, device, swap_probability, percentile, batch_size):
    logging.info('Starting annotation process...')
    prob_list = train_and_get_prob_list(adata, label_key=label_key, epoch_num=epoch_num, device=device, batch_size=batch_size)
    all_conf, all_var = calculate_confidence_and_variability(prob_list, n_obs=adata.n_obs, epoch_num=epoch_num)
    conf_cutoff, var_cutoff = find_cutoffs(adata, label_key, device, probability=swap_probability, percentile=percentile, epoch_num=epoch_num)
    adata = assign_annotations(adata, all_conf, all_var, conf_cutoff, var_cutoff, annotation_col='Annotation')
    group_counts = adata.obs['Annotation'].value_counts()
    logging.info('Annotation process complete.')
    logging.info('Group counts: %s', group_counts.to_dict())
    return adata, group_counts

def find_optimal_compositions(
    dataset_name,
    adata,
    label_key,
    group_counts,
    train_sizes,
    repeats_per_size,
    csv_file,
    device,
    epoch_num,
    batch_size
):
    """
    Runs the training and evaluation experiment for a given dataset.

    Parameters:
    - dataset_name (str): Name identifier for the dataset (e.g., 'merfish', 'pbmc').
    - adata (AnnData): The dataset to process.
    - label_key (str): The key in adata.obs that contains the labels.
    - group_counts (dict): Dictionary containing counts for 'Easy-to-learn', 'Ambiguous', 'Hard-to-learn'.
    - train_sizes (list of int): List of training set sizes to experiment with.
    - repeats_per_size (int): Number of repeats for each training size.
    - csv_file (str): Filename to save/load the results.
    - device (torch.device): The device to run the training on ('cpu' or 'cuda').
    - epoch_num (int): Number of training epochs.
    - batch_size (int): Batch size for training.

    Returns:
    - best_compositions (dict): Dictionary containing the best compositions and their corresponding test losses.
    """
    logging.info('Starting find_optimal_compositions for dataset: %s', dataset_name)

    # Load existing results from CSV or create an empty DataFrame
    try:
        logging.info('Loading existing results from %s', csv_file)
        results_df = pd.read_csv(csv_file)
    except FileNotFoundError:
        logging.warning('CSV file %s not found. Starting with an empty DataFrame.', csv_file)
        # Include 'Train_Indices' and 'Test_Indices' columns
        results_df = pd.DataFrame(columns=['Train_Size', 'Easy', 'Ambiguous', 'Hard', 'Test_Loss', 'Train_Indices', 'Test_Indices'])

    # Convert the 'Train_Size' column to a dictionary with counts for faster lookup
    existing_counts = results_df['Train_Size'].value_counts().to_dict()
    logging.debug('Existing counts per Train_Size: %s', existing_counts)

    # Assuming 'group_counts' is a pandas Series with annotations as indices
    group_counts = adata.obs['Annotation'].value_counts()
    logging.info('Group counts in the data: %s', group_counts.to_dict())

    # Assign counts to E, A, H
    E = group_counts.get('Easy-to-learn', 0)
    A = group_counts.get('Ambiguous', 0)
    H = group_counts.get('Hard-to-learn', 0)

    # Get the indices of each group
    easy_indices = adata.obs.index[adata.obs['Annotation'] == 'Easy-to-learn'].tolist()
    ambiguous_indices = adata.obs.index[adata.obs['Annotation'] == 'Ambiguous'].tolist()
    hard_indices = adata.obs.index[adata.obs['Annotation'] == 'Hard-to-learn'].tolist()

    # Fit LabelEncoder on the entire dataset labels
    label_encoder = LabelEncoder()
    label_encoder.fit(adata.obs[label_key])
    num_classes = len(label_encoder.classes_)

    best_compositions = {}

    for T in train_sizes:
        current_runs = existing_counts.get(T, 0)
        runs_needed = repeats_per_size - current_runs

        logging.info('Processing Train_Size=%d: current_runs=%d, runs_needed=%d', T, current_runs, runs_needed)

        if runs_needed <= 0:
            # Use existing entries
            existing_rows = results_df[results_df['Train_Size'] == T]
            for idx, row in existing_rows.iterrows():
                easy = row['Easy']
                ambiguous = row['Ambiguous']
                hard = row['Hard']
                test_loss = row['Test_Loss']
                train_indices_str = row.get('Train_Indices', None)
                test_indices_str = row.get('Test_Indices', None)
                logging.info(
                    f"Using cached result for {dataset_name} Train_Size={T}: Easy={easy}, Ambiguous={ambiguous}, Hard={hard}, Test Loss={test_loss}"
                )
                # Store the cached results
                if T not in best_compositions:
                    best_compositions[T] = []
                best_compositions[T].append({
                    'composition': (easy, ambiguous, hard),
                    'Test_Loss': test_loss,
                    'Train_Indices': train_indices_str,
                    'Test_Indices': test_indices_str
                })
            continue  # Skip computation for this T as all repeats are already done

        else:
            logging.info(f"Processing {dataset_name} train dataset size: {T} (Run {current_runs + 1} to {repeats_per_size})")

            # Calculate test size (25% of train size)
            test_size = int(0.25 * T)
            total_size = T + test_size
            logging.info(f"Total dataset size (Train + Test): {total_size} (Train: {T}, Test: {test_size})")

            # Select the test indices once per dataset size
            all_indices = adata.obs.index.tolist()
            # Ensure we have enough samples for test set
            if len(all_indices) < test_size:
                logging.warning(f"Not enough samples for Test Size={test_size} at Train_Size={T}")
                continue  # Skip if not enough samples

            # Randomly sample test_size samples for the test set
            test_indices = random.sample(all_indices, test_size)

            # Define step size as a function of T
            step_size = max(1, T // 100)

            # Generate compositions summing up to T (train size)
            compositions = []
            E = group_counts.get('Easy-to-learn', 0)
            A = group_counts.get('Ambiguous', 0)
            H = group_counts.get('Hard-to-learn', 0)
            logging.debug('Group counts: E=%d, A=%d, H=%d', E, A, H)
            for e in range(0, min(T, E) + 1, step_size):
                for a in range(0, min(T - e, A) + 1, step_size):
                    h = T - e - a
                    if h >= 0 and h <= H:
                        compositions.append((e, a, h))
            if not compositions:
                logging.warning(f"No valid compositions for Train Size={T}")
                # Save an entry indicating no valid compositions
                new_row = {
                    'Train_Size': T,
                    'Easy': None,
                    'Ambiguous': None,
                    'Hard': None,
                    'Test_Loss': None,
                    'Train_Indices': None,
                    'Test_Indices': ','.join(map(str, test_indices))
                }
                new_row_df = pd.DataFrame([new_row])
                results_df = pd.concat([results_df, new_row_df], ignore_index=True)
                results_df.to_csv(csv_file, index=False)
                continue

            logging.info(f"Total compositions for Train Size={T}: {len(compositions)}")

            for run in range(current_runs + 1, repeats_per_size + 1):
                logging.info(f"--- Run {run} for Train_Size={T} ---")

                min_loss = float('inf')
                best_comp = None
                best_train_indices = None

                # For each composition, train and get test loss
                # Disable tqdm output by setting disable=True
                for comp in tqdm(compositions, desc=f"Testing compositions for Train Size={T} - Run {run}", disable=True):
                    e, a, h = comp
                    # Ensure not exceeding group counts
                    if e > E or a > A or h > H:
                        continue  # Invalid composition

                    # Ensure we have enough samples in each group
                    if len(easy_indices) < e or len(ambiguous_indices) < a or len(hard_indices) < h:
                        continue  # Skip if not enough samples

                    # Randomly sample e, a, h samples from each group for training
                    available_easy = list(set(easy_indices) - set(test_indices))
                    available_ambiguous = list(set(ambiguous_indices) - set(test_indices))
                    available_hard = list(set(hard_indices) - set(test_indices))

                    if len(available_easy) < e or len(available_ambiguous) < a or len(available_hard) < h:
                        continue  # Not enough samples after excluding test set

                    train_easy_indices = random.sample(available_easy, e) if e > 0 else []
                    train_ambiguous_indices = random.sample(available_ambiguous, a) if a > 0 else []
                    train_hard_indices = random.sample(available_hard, h) if h > 0 else []
                    train_indices = train_easy_indices + train_ambiguous_indices + train_hard_indices

                    # Ensure total train samples equal T
                    if len(train_indices) != T:
                        continue  # Skip if train size mismatch

                    # Create training and testing datasets
                    adata_train = adata[train_indices].copy()
                    adata_test = adata[test_indices].copy()

                    # Train and get test loss
                    test_loss = train_and_evaluate_model(
                        adata_train=adata_train, 
                        adata_test=adata_test, 
                        label_key=label_key, 
                        label_encoder=label_encoder,
                        num_classes=num_classes,
                        epoch_num=epoch_num, 
                        device=device, 
                        batch_size=batch_size
                    )

                    # Update minimum loss and best composition
                    if test_loss < min_loss:
                        min_loss = test_loss
                        best_comp = comp
                        best_train_indices = train_indices.copy()

                if best_comp is not None:
                    easy, ambiguous, hard = best_comp
                    logging.info(
                        f"Best composition for {dataset_name} Train_Size={T} (Run {run}): Easy={easy}, Ambiguous={ambiguous}, Hard={hard}, Test Loss={min_loss}"
                    )

                    # Append to best_compositions
                    if T not in best_compositions:
                        best_compositions[T] = []
                    best_compositions[T].append({
                        'composition': best_comp,
                        'Test_Loss': min_loss,
                        'Train_Indices': best_train_indices,
                        'Test_Indices': test_indices  # Same test_indices for all runs of this T
                    })

                    # Save the result to the DataFrame and CSV
                    new_row = {
                        'Train_Size': T,
                        'Easy': easy,
                        'Ambiguous': ambiguous,
                        'Hard': hard,
                        'Test_Loss': min_loss,
                        'Train_Indices': ','.join(map(str, best_train_indices)),
                        'Test_Indices': ','.join(map(str, test_indices))
                    }
                    new_row_df = pd.DataFrame([new_row])
                    results_df = pd.concat([results_df, new_row_df], ignore_index=True)
                    results_df.to_csv(csv_file, index=False)
                else:
                    logging.warning(f"No valid compositions found for {dataset_name} Train_Size={T} (Run {run})")
                    # Save an entry indicating no valid compositions
                    new_row = {
                        'Train_Size': T,
                        'Easy': None,
                        'Ambiguous': None,
                        'Hard': None,
                        'Test_Loss': None,
                        'Train_Indices': None,
                        'Test_Indices': ','.join(map(str, test_indices))
                    }
                    new_row_df = pd.DataFrame([new_row])
                    results_df = pd.concat([results_df, new_row_df], ignore_index=True)
                    results_df.to_csv(csv_file, index=False)
    logging.info('find_optimal_compositions completed for dataset: %s', dataset_name)
    return best_compositions, label_encoder

'''
    - **Workflow:**
        1. **Load Existing Results:** Checks if the CSV file exists to load previous results; otherwise, initializes an empty DataFrame.
        2. **Iterate Over Training Sizes:** For each specified training size, it checks how many runs have been completed and determines the remaining runs needed.
        3. **Sampling Test Indices:** Randomly selects test indices ensuring reproducibility and avoiding overlaps with training data.
        4. **Generate Compositions:** Creates possible compositions of 'Easy', 'Ambiguous', and 'Hard' samples that sum up to the training size.
        5. **Training Loop:** For each run and each possible composition, it trains the model and records the test loss.
        6. **Logging and Saving Results:** Updates the `best_compositions` dictionary and saves the results to the CSV file.

'''

def visualize_optimal_compositions(csv_file):
    logging.info('Starting visualization of optimal compositions...')
    # Load the compositions from the CSV file
    try:
        results_df = pd.read_csv(csv_file)
        logging.info('Loaded results from %s', csv_file)
    except FileNotFoundError:
        logging.error(f"CSV file '{csv_file}' not found.")
        results_df = pd.DataFrame(columns=['Train_Size', 'Easy', 'Ambiguous', 'Hard', 'Test_Loss'])

    # Filter out rows with missing compositions
    results_df = results_df.dropna(subset=['Easy', 'Ambiguous', 'Hard'])

    # Convert counts to floats and Train_Size to int
    results_df['Easy'] = results_df['Easy'].astype(float)
    results_df['Ambiguous'] = results_df['Ambiguous'].astype(float)
    results_df['Hard'] = results_df['Hard'].astype(float)
    results_df['Train_Size'] = results_df['Train_Size'].astype(int)

    # Check if all train sizes have the same number of runs
    counts_per_size = results_df['Train_Size'].value_counts()
    if counts_per_size.nunique() != 1:
        logging.warning("Not all train sizes have the same number of rows in the CSV for each train size.")

    # Calculate total and proportions for each row
    results_df['Total'] = results_df['Easy'] + results_df['Ambiguous'] + results_df['Hard']
    results_df['Proportion_Easy'] = results_df['Easy'] / results_df['Total']
    results_df['Proportion_Ambiguous'] = results_df['Ambiguous'] / results_df['Total']
    results_df['Proportion_Hard'] = results_df['Hard'] / results_df['Total']

    # Group by Train_Size and calculate mean and standard deviation of proportions
    grouped = results_df.groupby('Train_Size').agg({
        'Proportion_Easy': ['mean', 'std'],
        'Proportion_Ambiguous': ['mean', 'std'],
        'Proportion_Hard': ['mean', 'std']
    }).reset_index()

    # Flatten MultiIndex columns
    grouped.columns = ['Train_Size',
                    'Proportion_Easy_mean', 'Proportion_Easy_std',
                    'Proportion_Ambiguous_mean', 'Proportion_Ambiguous_std',
                    'Proportion_Hard_mean', 'Proportion_Hard_std']

    # Ensure that the mean proportions sum to 1 (optional assertion)
    if not np.allclose(grouped[['Proportion_Easy_mean', 'Proportion_Ambiguous_mean', 'Proportion_Hard_mean']].sum(axis=1), 1):
        logging.error("Mean proportions do not sum to 1.")
        raise ValueError("Mean proportions do not sum to 1.")

    # Prepare data for plotting
    train_sizes = grouped['Train_Size'].values
    proportion_e_mean = grouped['Proportion_Easy_mean'].values
    proportion_a_mean = grouped['Proportion_Ambiguous_mean'].values
    proportion_h_mean = grouped['Proportion_Hard_mean'].values
    proportion_e_std = grouped['Proportion_Easy_std'].values
    proportion_a_std = grouped['Proportion_Ambiguous_std'].values
    proportion_h_std = grouped['Proportion_Hard_std'].values

    # Verify that all arrays have the same length
    array_lengths = [len(train_sizes), len(proportion_e_mean), len(proportion_a_mean), len(proportion_h_mean),
                    len(proportion_e_std), len(proportion_a_std), len(proportion_h_std)]
    if len(set(array_lengths)) != 1:
        logging.error(f"Array length mismatch: {array_lengths}")
        raise ValueError(f"Array length mismatch: {array_lengths}")

    # Plotting grouped bar chart with error bars (variance) without percentage labels
    logging.info('Creating plot for optimal compositions...')
    # Set up the plot for Grouped Bar Chart
    fig, ax = plt.subplots(figsize=(14, 8))
    index = np.arange(len(train_sizes))

    # Customize the axes
    ax.set_xticks(index)
    ax.set_xticklabels([str(size) for size in train_sizes], rotation=45)
    ax.set_ylabel('Average Proportion')
    ax.set_xlabel('Train Set Size')
    ax.set_title('Optimal Composition of Train Set Samples with Standard Deviation')
    ax.legend(loc='upper right')
    ax.set_ylim(0, 1)
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig('optimal_compositions.png')
    logging.info('Visualization saved as optimal_compositions.png')

def highest_confidense_samples(input_csv, adata, train_sizes, device, global_label_encoder):
    logging.info('Starting processing of highest confidence samples...')
    # Read the optimal_compositions_detailed.csv to get the test indices used previously
    best_comp_df = pd.read_csv(input_csv)

    # Initialize a new DataFrame to store the results
    high_conf_df = pd.DataFrame(columns=['Train_Size', 'Train_Indices', 'Test_Indices', 'Test_Loss'])

    # Sort the samples by confidence scores in descending order
    # Assuming 'conf' is the column in adata.obs that contains confidence scores
    sorted_conf = adata.obs.sort_values(by='conf', ascending=False)

    # For each train size
    for T in train_sizes:
        logging.info(f"Processing high-confidence composition for Train_Size={T}")
        
        # Get the entries for Train_Size=T to retrieve the test indices
        size_df = best_comp_df[best_comp_df['Train_Size'] == T]
        
        if size_df.empty:
            logging.warning(f"No entries found for Train_Size={T} in {input_csv}. Skipping.")
            continue
        
        # Assuming all entries for a given Train_Size use the same Test_Indices
        # Fetch unique Test_Indices for Train_Size=T
        unique_test_indices = size_df['Test_Indices'].unique()
        
        if len(unique_test_indices) != 1:
            logging.warning(f"Multiple test sets found for Train_Size={T}. Using the first one.")
        
        test_indices_str = unique_test_indices[0]
        
        if pd.isnull(test_indices_str):
            logging.warning(f"No test indices found for Train_Size={T}. Skipping.")
            continue  # Skip if no test indices
        
        test_indices = test_indices_str.split(',')
        
        # Select the top T samples with highest confidence as the training set
        # Ensure no overlap with test_indices
        top_conf_indices = sorted_conf.index.difference(test_indices)[:T].tolist()
        
        if len(top_conf_indices) < T:
            logging.warning(f"Not enough available samples to select top {T} without overlapping with test set.")
            logging.warning(f"Available samples: {len(top_conf_indices)}")
            continue  # Skip if not enough samples
        
        # Create training and testing datasets
        adata_train = adata[top_conf_indices].copy()
        adata_test = adata[test_indices].copy()
        
        # Train and get test loss
        test_loss = train_and_evaluate_model(
            adata_train, adata_test, label_key='CellType', label_encoder=global_label_encoder,
            num_classes=len(global_label_encoder.classes_),  # Added num_classes
            epoch_num=30, device=device, batch_size=64
        )
        
        # Save the result
        high_conf_df = high_conf_df.append({
            'Train_Size': T,
            'Train_Indices': ','.join(map(str, top_conf_indices)),
            'Test_Indices': ','.join(map(str, test_indices)),
            'Test_Loss': test_loss
        }, ignore_index=True)
        
        logging.info(f"Train_Size={T}, Test Loss={test_loss}")

    # Save the results to a new CSV file
    high_conf_df.to_csv('high_confidence_compositions.csv', index=False)
    logging.info("High-confidence compositions have been saved to 'high_confidence_compositions.csv'.")

    # Read the results from the CSV files
    optimal_comp_df = pd.read_csv('optimal_compositions_detailed.csv')
    high_conf_df = pd.read_csv('high_confidence_compositions.csv')

    # Calculate the average test loss for each Train_Size in the optimal compositions
    optimal_loss_df = optimal_comp_df.groupby('Train_Size')['Test_Loss'].mean().reset_index()
    optimal_loss_df.rename(columns={'Test_Loss': 'Optimal_Test_Loss'}, inplace=True)

    # Prepare the test loss for the high confidence compositions
    high_conf_loss_df = high_conf_df[['Train_Size', 'Test_Loss']]
    high_conf_loss_df.rename(columns={'Test_Loss': 'High_Conf_Test_Loss'}, inplace=True)

    # Merge the two DataFrames on 'Train_Size'
    comparison_df = pd.merge(optimal_loss_df, high_conf_loss_df, on='Train_Size')

    # Sort by Train_Size
    comparison_df.sort_values('Train_Size', inplace=True)

    # Plotting
    logging.info('Creating comparison plot...')
    plt.figure(figsize=(10, 6))
    plt.plot(comparison_df['Train_Size'], comparison_df['Optimal_Test_Loss'], marker='o', label='Optimal Composition')
    plt.plot(comparison_df['Train_Size'], comparison_df['High_Conf_Test_Loss'], marker='s', label='High Confidence Composition')

    plt.title('Comparison of Test Losses by Train Size')
    plt.xlabel('Train Size')
    plt.ylabel('Test Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('comparison_plot.png')
    logging.info('Comparison plot saved as comparison_plot.png')