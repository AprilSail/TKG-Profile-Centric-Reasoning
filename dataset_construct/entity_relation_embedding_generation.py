import os
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

# --------------------------
# 1. Configuration & Initialization
# --------------------------
# Model configuration (Llama 3.1 8B)
MODEL_PATH = "meta-llama/Meta-Llama-3.1-8B"
DATASET_NAME = "YAGO"

# Path configuration
BASE_DATASET_PATH = f"./dataset/{DATASET_NAME}/"
BASE_EMBEDDING_PATH = f"./embeddings/{DATASET_NAME}/"

# Create embedding output directory if not exists
os.makedirs(BASE_EMBEDDING_PATH, exist_ok=True)


# --------------------------
# 2. Load Pre-trained LLM and Tokenizer
# --------------------------
def load_llm_model(model_path):
    """
    Load pre-trained Llama model and tokenizer with memory-efficient settings
    :param model_path: Path to pre-trained Llama model
    :return: tokenizer, model (eval mode)
    """
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Load model with float16 precision and automatic device mapping
    model = AutoModel.from_pretrained(
        model_path,
        torch_dtype=torch.float16,  # Half precision to save GPU memory
        device_map="auto"  # Auto assign to GPU/CPU
    )

    # Set model to evaluation mode (disable dropout)
    model.eval()
    print("LLM model loaded successfully.")
    return tokenizer, model


# Load model and tokenizer
tokenizer, model = load_llm_model(MODEL_PATH)


# --------------------------
# 3. Load Entity and Relation Data
# --------------------------
def load_id_mapping(file_path, is_entity=True):
    """
    Load entity/relation to ID mapping from text file
    :param file_path: Path to entity2id.txt/relation2id.txt
    :param is_entity: Flag to distinguish entity/relation processing
    :return: list of names, list of corresponding IDs
    """
    names = []
    ids = []

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue  # Skip empty lines

            # Parse entity (4 fields) or relation (2 fields) line
            parts = line.split('\t')
            if is_entity:
                name, id_val = parts[0], parts[1]
            else:
                name, id_val = parts[0], parts[1]

            # Clean special characters (<>) from name
            name = name.replace("<", "").replace(">", "")
            names.append(name)
            ids.append(id_val)

    return names, ids


# Load entity mappings
entity_file = os.path.join(BASE_DATASET_PATH, "entity2id.txt")
entity_names, entity_ids = load_id_mapping(entity_file, is_entity=True)

# Load relation mappings
relation_file = os.path.join(BASE_DATASET_PATH, "relation2id.txt")
relation_names, relation_ids = load_id_mapping(relation_file, is_entity=False)


# --------------------------
# 4. Generate Embeddings
# --------------------------
def generate_embeddings(names, ids, output_path, tokenizer, model, desc):
    """
    Generate embeddings for entities/relations using LLM
    :param names: List of entity/relation names to encode
    :param ids: Corresponding ID list
    :param output_path: Path to save embeddings
    :param tokenizer: Pre-loaded tokenizer
    :param model: Pre-loaded LLM model
    :param desc: Progress bar description
    """
    with open(output_path, "w", encoding="utf-8") as fout:
        # Process each name with progress bar
        for idx, name in enumerate(tqdm(names, desc=desc)):
            # Tokenize input (truncate/pad to fit model context)
            inputs = tokenizer(
                name,
                return_tensors="pt",
                truncation=True,
                padding=True
            )

            # Generate embedding with no gradient computation
            with torch.no_grad():
                outputs = model(**inputs)
                # Get last hidden state: [batch_size, sequence_length, hidden_size]
                hidden_states = outputs.last_hidden_state
                # Calculate mean embedding (exclude special tokens at start/end)
                embedding = hidden_states[:, 1:-1, :].mean(dim=1).squeeze().cpu().numpy()

            # Format embedding to 6 decimal places string
            emb_str = " ".join([f"{x:.6f}" for x in embedding])
            # Write to file: ID \t Name \t Embedding
            fout.write(f"{ids[idx]}\t{name}\t{emb_str}\n")


# Generate and save entity embeddings
entity_emb_path = os.path.join(BASE_EMBEDDING_PATH, "entity_embedding.txt")
generate_embeddings(
    entity_names,
    entity_ids,
    entity_emb_path,
    tokenizer,
    model,
    desc="Generating entity embeddings"
)

# Generate and save relation embeddings
relation_emb_path = os.path.join(BASE_EMBEDDING_PATH, "relation_embedding.txt")
generate_embeddings(
    relation_names,
    relation_ids,
    relation_emb_path,
    tokenizer,
    model,
    desc="Generating relation embeddings"
)

print("All embeddings generated and saved successfully.")