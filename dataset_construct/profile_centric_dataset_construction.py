import os
import numpy as np
from typing import List, Tuple
from tqdm import tqdm


# --------------------------
# Core Data Processing Functions
# --------------------------
def read_quadruples(path: str) -> List[Tuple[str, str, str, str]]:
    """
    Read quadruples (subject, predicate, object, timestamp) from text file
    :param path: Path to input text file with tab-separated quadruples
    :return: List of quadruple tuples (s, p, o, t)
    """
    quadruples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 4:  # Basic validation for valid quadruples
                quadruples.append(tuple([parts[0], parts[1], parts[2], parts[3]]))
    return quadruples


def semantic_similarity(a: str, b: str, embedding_dict: dict) -> float:
    """
    Calculate cosine similarity between two embeddings
    :param a: First entity/relation ID
    :param b: Second entity/relation ID
    :param embedding_dict: Dictionary mapping IDs to embedding vectors
    :return: Cosine similarity score (0.0 if IDs not found or zero norm)
    """
    # Return 0 if either ID is missing from embedding dictionary
    if a not in embedding_dict or b not in embedding_dict:
        return 0.0

    vec_a = embedding_dict[a]
    vec_b = embedding_dict[b]

    # Compute cosine similarity
    dot_product = np.dot(vec_a, vec_b)
    norm_product = np.linalg.norm(vec_a) * np.linalg.norm(vec_b)

    # Avoid division by zero
    if norm_product == 0:
        return 0.0
    return float(dot_product / norm_product)


def quadruple_similarity(quad1: Tuple[str, str, str, str],
                         quad2: Tuple[str, str, str, str],
                         entity_embeddings: dict,
                         relation_embeddings: dict,
                         max_timestamp: float,
                         alpha: float = 0.4,
                         beta: float = 0.3,
                         gamma: float = 0.3) -> float:
    """
    Calculate composite similarity score between two quadruples
    :param quad1: First quadruple (s1, p1, o1, t1)
    :param quad2: Second quadruple (s2, p2, o2, t2)
    :param entity_embeddings: Entity ID to embedding dictionary
    :param relation_embeddings: Relation ID to embedding dictionary
    :param max_timestamp: Maximum timestamp value in dataset (for normalization)
    :param alpha: Weight for relation similarity
    :param beta: Weight for entity similarity (split equally between subject/object)
    :param gamma: Weight for temporal distance penalty
    :return: Composite similarity score
    """
    s1, p1, o1, t1 = quad1
    s2, p2, o2, t2 = quad2

    # Calculate relation similarity
    rel_sim = semantic_similarity(p1, p2, relation_embeddings)

    # Calculate entity similarity (subject + object)
    ent_sim = semantic_similarity(s1, s2, entity_embeddings) + semantic_similarity(o1, o2, entity_embeddings)

    # Calculate normalized temporal distance penalty
    time_diff = abs(int(t1) - int(t2)) / max_timestamp

    # Composite similarity formula
    return alpha * rel_sim + beta * ent_sim - gamma * time_diff


def find_related_quadruples(target_quad: Tuple[str, str, str, str],
                            all_quadruples: List[Tuple[str, str, str, str]],
                            top_k: int,
                            entity_embeddings: dict,
                            relation_embeddings: dict,
                            max_timestamp: float) -> Tuple[List[Tuple], List[Tuple], List[Tuple]]:
    """
    Find top-K related quadruples for target based on similarity
    :param target_quad: Target quadruple (s, p, o, t)
    :param all_quadruples: Full list of dataset quadruples
    :param top_k: Number of top similar quadruples to retrieve
    :param entity_embeddings: Entity ID to embedding dictionary
    :param relation_embeddings: Relation ID to embedding dictionary
    :param max_timestamp: Maximum timestamp value in dataset
    :return: (top-K subject-related, top-K object-related, top-K predicate-related) quadruples
    """
    s, p, o, t = target_quad
    target_time = int(t)

    # Filter candidate quadruples by entity/temporal constraints
    subject_candidates = [q for q in all_quadruples if
                          (q[0] == s or q[2] == s) and q != target_quad and int(q[3]) <= target_time]

    object_candidates = [q for q in all_quadruples if
                         (q[0] == o or q[2] == o) and q != target_quad and int(q[3]) <= target_time]

    predicate_candidates = [q for q in all_quadruples if
                            q[1] == p and q != target_quad and (target_time - 5) <= int(q[3]) <= target_time]

    # Sort candidates by similarity (descending) and take top-K
    top_subject = sorted(subject_candidates,
                         key=lambda q: quadruple_similarity(target_quad, q, entity_embeddings, relation_embeddings,
                                                            max_timestamp),
                         reverse=True)[:top_k]

    top_object = sorted(object_candidates,
                        key=lambda q: quadruple_similarity(target_quad, q, entity_embeddings, relation_embeddings,
                                                           max_timestamp),
                        reverse=True)[:top_k]

    top_predicate = sorted(predicate_candidates,
                           key=lambda q: quadruple_similarity(target_quad, q, entity_embeddings, relation_embeddings,
                                                              max_timestamp),
                           reverse=True)[:top_k]

    return top_subject, top_object, top_predicate


def quadruples_to_string(quadruples: List[Tuple[str, str, str, str]]) -> str:
    """
    Convert list of quadruples to human-readable string format
    :param quadruples: List of quadruple tuples
    :return: Formatted string (e.g., "[(s1,p1,o1,t1); (s2,p2,o2,t2)]")
    """
    if not quadruples:
        return "[]"
    return "[" + "; ".join(["(" + ",".join(q) + ")" for q in quadruples]) + "]"


def process_dataset_split(split_name: str,
                          all_quadruples: List[Tuple[str, str, str, str]],
                          top_k: int,
                          input_dir: str,
                          dataset_name: str,
                          entity_embeddings: dict,
                          relation_embeddings: dict,
                          max_timestamp: float):
    """
    Process a single dataset split (train/valid/test) to generate profile-augmented data
    :param split_name: Name of split (train/valid/test)
    :param all_quadruples: Full list of dataset quadruples
    :param top_k: Number of related quadruples to retrieve
    :param input_dir: Directory containing original split files
    :param dataset_name: Name of target dataset (e.g., YAGO)
    :param entity_embeddings: Entity ID to embedding dictionary
    :param relation_embeddings: Relation ID to embedding dictionary
    :param max_timestamp: Maximum timestamp value in dataset
    """
    # Configure paths
    input_path = os.path.join(input_dir, f"{split_name}.txt")
    profile_version = 6  # Fixed profile version (removed magic number variable)
    output_dir = f"./dataset_profile_{profile_version}/{dataset_name}"
    output_path = os.path.join(output_dir, f"{split_name}_profile_{profile_version}.txt")

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Load split quadruples
    split_quadruples = read_quadruples(input_path)
    print(f"[INFO] Processing {input_path}: {len(split_quadruples)} quadruples")

    # Initialize storage for processed data
    target_quadruples = []
    subject_related = []
    object_related = []

    # Process each quadruple in split
    for quad in tqdm(split_quadruples, desc=f"Processing {split_name}"):
        s, p, o, t = quad

        # Get related quadruples
        rel_s, rel_o, rel_supply = find_related_quadruples(quad, all_quadruples, top_k,
                                                           entity_embeddings, relation_embeddings, max_timestamp)

        # Fill missing candidates with supply candidates
        if len(rel_s) < top_k:
            rel_s += rel_supply[:top_k - len(rel_s)]
        if len(rel_o) < top_k:
            rel_o += rel_supply[:top_k - len(rel_o)]

        # Prepend target quadruple to related lists
        rel_s = [quad] + rel_s
        rel_o = [quad] + rel_o

        # Pad to fixed length (10) if insufficient candidates
        if len(rel_s) < 10:
            rel_s += [rel_s[-1]] * (10 - len(rel_s)) if rel_s else [None] * 10
        if len(rel_o) < 10:
            rel_o += [rel_o[-1]] * (10 - len(rel_o)) if rel_o else [None] * 10

        # Store processed data
        target_quadruples.append(quad)
        subject_related.append(rel_s)
        object_related.append(rel_o)

    # Write processed data to output file
    with open(output_path, "w", encoding="utf-8") as fout:
        for idx in range(len(target_quadruples)):
            s, p, o, t = target_quadruples[idx]
            # Write line format: (s,p,o,t) \t [related_s] \t [related_o]
            line = f"({s},{p},{o},{t})\t{quadruples_to_string(subject_related[idx][:profile_version])}\t{quadruples_to_string(object_related[idx][:profile_version])}\n"
            fout.write(line)

    print(f"[INFO] Saved processed data to {output_path}")


# --------------------------
# Main Execution
# --------------------------
if __name__ == "__main__":
    # Configuration (centralized for easy modification)
    DATASET_NAME = "YAGO"
    INPUT_DATA_DIR = f"./dataset/{DATASET_NAME}"
    EMBEDDING_DIR = f"./dataset_embeddings/{DATASET_NAME}"
    TOP_K = 10  # Number of related quadruples to retrieve

    print(f"\nProcessing dataset: {DATASET_NAME}\n")

    # --------------------------
    # 1. Load Embeddings
    # --------------------------
    # Load entity embeddings (ID -> vector)
    entity_embeddings = {}
    entity_emb_path = os.path.join(EMBEDDING_DIR, "entity_embedding.txt")
    with open(entity_emb_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 3:
                ent_id, ent_name, vec_str = parts[0], parts[1], parts[2]
                entity_embeddings[ent_id] = np.array(vec_str.split(), dtype=float)

    # Load relation embeddings (ID -> vector)
    relation_embeddings = {}
    rel_emb_path = os.path.join(EMBEDDING_DIR, "relation_embedding.txt")
    with open(rel_emb_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 3:
                rel_id, rel_name, vec_str = parts[0], parts[1], parts[2]
                relation_embeddings[rel_id] = np.array(vec_str.split(), dtype=float)

    # --------------------------
    # 2. Load Full Dataset
    # --------------------------
    # Read train/valid/test splits
    train_quadruples = read_quadruples(os.path.join(INPUT_DATA_DIR, "train.txt"))
    valid_quadruples = read_quadruples(os.path.join(INPUT_DATA_DIR, "valid.txt"))
    test_quadruples = read_quadruples(os.path.join(INPUT_DATA_DIR, "test.txt"))

    # Combine and deduplicate all quadruples
    all_quadruples = list(set(train_quadruples + valid_quadruples + test_quadruples))
    print(f"[INFO] Loaded {len(all_quadruples)} unique quadruples from all splits")

    # Calculate maximum timestamp for normalization
    max_timestamp = max(float(q[3]) for q in all_quadruples) if all_quadruples else 0.0

    # --------------------------
    # 3. Process Dataset Split (Test)
    # --------------------------
    process_dataset_split("test", all_quadruples, TOP_K, INPUT_DATA_DIR, DATASET_NAME,
                          entity_embeddings, relation_embeddings, max_timestamp)