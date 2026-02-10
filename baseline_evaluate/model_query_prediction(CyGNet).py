import os
import warnings
import numpy as np
import torch
import scipy.sparse as sp
from tqdm import tqdm
from torch import optim

# Suppress warning messages
warnings.filterwarnings(action='ignore')

# --------------------------
# Configuration & Device Setup
# --------------------------
# Set number of CPU threads for PyTorch
torch.set_num_threads(2)

# Import external modules and configurations
from utils import load_quadruples, get_total_number
from config import args
from link_prediction import link_prediction
from evolution import calc_raw_mrr, calc_filtered_test_mrr

# Device configuration (GPU if available and specified)
use_cuda = args.gpu >= 0 and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

# --------------------------
# Data Loading
# --------------------------
# Load dataset splits (train/valid/test)
dataset_path = f'./data/{args.dataset}'
if args.dataset == 'ICEWS14':
    # Special case for ICEWS14 (uses test.txt for dev set)
    train_data, train_times = load_quadruples(dataset_path, 'train.txt')
    test_data, test_times = load_quadruples(dataset_path, 'test.txt')
    dev_data, dev_times = load_quadruples(dataset_path, 'test.txt')
else:
    # Standard dataset split (train/valid/test)
    train_data, train_times = load_quadruples(dataset_path, 'train.txt')
    test_data, test_times = load_quadruples(dataset_path, 'test.txt')
    dev_data, dev_times = load_quadruples(dataset_path, 'valid.txt')

# Combine all timestamps from all splits
all_times = np.concatenate([train_times, dev_times, test_times])

# Get dataset statistics (number of entities/relations)
num_entities, num_relations = get_total_number(dataset_path, 'stat.txt')

# Calculate number of time bins based on timestamp interval
num_time_bins = int(max(all_times) / args.time_stamp) + 1
print(f'Number of time bins: {num_time_bins}')

# --------------------------
# Model Initialization
# --------------------------
# Initialize link prediction model
model = link_prediction(
    num_entities,
    args.hidden_dim,
    num_relations,
    num_time_bins,
    use_cuda
)
model.to(device)

# --------------------------
# Load Precomputed Copy Sequences
# --------------------------
# Initialize sparse matrices for copy sequences (object/subject)
seq_shape = (num_entities * num_relations, num_entities)
all_tail_seq_obj = sp.csr_matrix(([], ([], [])), shape=seq_shape)
all_tail_seq_sub = sp.csr_matrix(([], ([], [])), shape=seq_shape)

# Aggregate copy sequences across all training timestamps
for timestamp in train_times:
    # Load object copy sequence for current timestamp
    obj_seq_path = f'./data/{args.dataset}/copy_seq/train_h_r_copy_seq_{timestamp}.npz'
    tim_tail_seq_obj = sp.load_npz(obj_seq_path)

    # Load subject copy sequence for current timestamp
    sub_seq_path = f'./data/{args.dataset}/copy_seq_sub/train_h_r_copy_seq_{timestamp}.npz'
    tim_tail_seq_sub = sp.load_npz(sub_seq_path)

    # Accumulate sequences
    all_tail_seq_obj += tim_tail_seq_obj
    all_tail_seq_sub += tim_tail_seq_sub

# --------------------------
# Load Pre-trained Model Weights
# --------------------------
# Define model checkpoint path
model_checkpoint_path = f'./results/bestmodel/{args.dataset}/CyGNet_{args.dataset}_bestmodel_state.pth'
batch_size = args.batch_size

# Load checkpoint (CPU-compatible)
checkpoint = torch.load(model_checkpoint_path, map_location='cpu')

# Set model to training mode and load weights
model.train()
model.load_state_dict(checkpoint['state_dict'])
print(f"Loaded best model from epoch: {checkpoint['epoch']}")

# --------------------------
# Model Inference on Test Set
# --------------------------
# Initialize storage for predictions
quad_list = []
object_score_list = []
subject_score_list = []

# Process test set in batches
num_batches = (test_data.shape[0] + batch_size - 1) // batch_size
for batch_idx in tqdm(range(num_batches), desc="Processing test batches"):
    # Calculate batch boundaries
    batch_start = batch_idx * batch_size
    batch_end = min(test_data.shape[0], (batch_idx + 1) * batch_size)
    batch_data = test_data[batch_start: batch_end]

    # Prepare batch inputs
    test_labels = torch.LongTensor(batch_data[:, 2])
    sequence_indices = batch_data[:, 0] * num_relations + batch_data[:, 1]

    # Convert sparse copy sequence to dense tensor
    tail_sequence = torch.Tensor(all_tail_seq_obj[sequence_indices].todense())
    one_hot_tail_sequence = tail_sequence.masked_fill(tail_sequence != 0, 1)

    # Move tensors to appropriate device (GPU/CPU)
    if use_cuda:
        test_labels = test_labels.to(device)
        one_hot_tail_sequence = one_hot_tail_sequence.to(device)

    # Get model predictions for object and subject
    object_scores = model(batch_data, one_hot_tail_sequence, entity='object')
    subject_scores = model(batch_data, one_hot_tail_sequence, entity='subject')

    # Store predictions (move to CPU for storage)
    quad_list.append(torch.as_tensor(batch_data, dtype=torch.int32).cpu())
    object_score_list.append(object_scores.cpu())
    subject_score_list.append(subject_scores.cpu())

    # Clear GPU memory cache
    if use_cuda:
        torch.cuda.empty_cache()

# --------------------------
# Save Prediction Results
# --------------------------
# Concatenate batch results
all_quadruples = torch.cat(quad_list, dim=0)
all_object_scores = torch.cat(object_score_list, dim=0)
all_subject_scores = torch.cat(subject_score_list, dim=0)

# Save object (tail) predictions
object_pred_path = f'./data/{args.dataset}/entity_predict_answer_CyGNet_tail.pt'
torch.save({
    "quad": all_quadruples,
    "scores": all_object_scores
}, object_pred_path)

# Save subject (head) predictions
subject_pred_path = f'./data/{args.dataset}/entity_predict_answer_CyGNet_head.pt'
torch.save({
    "quad": all_quadruples,
    "scores": all_subject_scores  # Fixed: previously saved object scores for subject
}, subject_pred_path)

print("Prediction processing completed successfully.")