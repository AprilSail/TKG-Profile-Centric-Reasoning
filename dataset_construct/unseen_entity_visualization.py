import os
from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


def load_file(path, e_dict, t_set):
    """
    Load timestamp and entity data from text file
    :param path: Path to the input text file
    :param e_dict: Dictionary mapping timestamp to set of entities
    :param t_set: Set to store timestamps from the file
    """
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 4:
                continue  # Skip invalid lines with insufficient fields
            s, p, o, t = parts[:4]
            e_dict[t].add(s)  # Add subject entity to timestamp set
            e_dict[t].add(o)  # Add object entity to timestamp set
            t_set.add(t)  # Record timestamp from current line


# Formatter function for y-axis (convert to thousands)
def thousands_formatter(x, pos):
    """Format y-axis values to thousands scale"""
    return f'{x / 1000:g}'


# Create subplots (1 row, 6 columns)
fig, ax = plt.subplots(1, 6)
fig.subplots_adjust(wspace=0)  # Remove space between subplots

# List of datasets to process (maintain original order)
dataset_name = "YAGO"
print(f"Processing dataset: {dataset_name}")

# Initialize data structures
train_t = set()
valid_t = set()
test_t = set()
e_dict = defaultdict(set)
data_path = "./dataset/" + dataset_name + "/"

# Load data from train/valid/test files
load_file(data_path + "train.txt", e_dict, train_t)
load_file(data_path + "valid.txt", e_dict, valid_t)
load_file(data_path + "test.txt", e_dict, test_t)

# Sort timestamps and create timestamp-to-index mapping
all_t_sorted = sorted(e_dict, key=lambda x: int(x))
t2idx = {t: i for i, t in enumerate(all_t_sorted)}

# Map train/valid/test timestamps to sorted indices
train_idx = sorted(t2idx[t] for t in train_t)
valid_idx = sorted(t2idx[t] for t in valid_t)
test_idx = sorted(t2idx[t] for t in test_t)

# Calculate cumulative entity count over time
all_entities = set()
present_entity_num = []
for t in all_t_sorted:
    all_entities.update(e_dict[t])  # Add entities from current timestamp
    present_entity_num.append(len(all_entities))  # Record cumulative count

# Print cumulative entity count for current dataset
total_entities = present_entity_num[-1] if present_entity_num else 0
print(f"{dataset_name} total cumulative entities: {total_entities}")

# Configure y-axis formatter (thousands scale)
ax.yaxis.set_major_formatter(FuncFormatter(thousands_formatter))
# Add thousands scale indicator to subplot
ax.text(0.0, 1.02, r'$\times 10^3$', transform=ax.transAxes,
        ha='left', va='bottom', fontname='Arial')

# Plot training data curve
if train_idx:
    train_y = [present_entity_num[i] for i in train_idx]
    ax.plot(train_idx, train_y, linewidth=5, label="Train", color='r')

# Plot validation data curve (connect to last training point)
if valid_idx and train_idx:
    valid_x = [train_idx[-1]] + valid_idx
    valid_y = [present_entity_num[train_idx[-1]]] + [present_entity_num[i] for i in valid_idx]
    ax.plot(valid_x, valid_y, linewidth=5, label="Valid", color='g')

# Plot test data curve (connect to last validation point)
if test_idx and valid_idx:
    test_x = [valid_idx[-1]] + test_idx
    test_y = [present_entity_num[valid_idx[-1]]] + [present_entity_num[i] for i in test_idx]
    ax.plot(test_x, test_y, linewidth=5, label="Test", color='y')

# Set subplot labels
ax.set_xlabel(dataset_name, fontsize=18)
ax.set_ylabel("Existed Entities", fontsize=18)

# Add grid to subplot
ax.grid(True, linestyle='--', linewidth=1.5, alpha=0.9)
# Set subplot background transparency
ax.patch.set_alpha(0.15)

# Add legend to last subplot (using handles from first subplot)
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels, loc='upper left', fontsize=18)

# Adjust layout and save figure
plt.tight_layout()
# Save plot as PDF with high resolution
plt.savefig("entity_curve.pdf", dpi=300)
# Display plot
plt.show()
# Close plot to release memory
plt.close()