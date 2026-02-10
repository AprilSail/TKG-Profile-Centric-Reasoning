import torch
import numpy as np
from scipy.stats import kendalltau
from tqdm import tqdm
import os


# --------------------------
# 1. Load Baseline Prediction Results
# --------------------------
def load_baseline_predictions(dataset: str, baseline: str) -> tuple[dict, dict]:
    """
    Load tail (object) and head (subject) prediction results from PyTorch files
    :param dataset: Name of target dataset (e.g., ICEWS14)
    :param baseline: Name of baseline model (e.g., CyGNet)
    :return: (pred_o_dict, pred_s_dict) - dictionaries mapping quadruple to top-K scores
    """
    # Define prediction file paths
    tail_pred_path = f"./{dataset}_entity_predict_answer_{baseline}_tail.pt"
    head_pred_path = f"./{dataset}_entity_predict_answer_{baseline}_head.pt"

    # Load tail predictions (object prediction)
    tail_data = torch.load(tail_pred_path, map_location=torch.device('cpu'))
    test_quadruples = tail_data["quad"]  # Shape: N*4 (list of quadruple tensors)
    topk_tail_scores = tail_data["scores"]  # Shape: N*10 (top-10 candidate scores for object)

    # Load head predictions (subject prediction)
    head_data = torch.load(head_pred_path, map_location=torch.device('cpu'))
    topk_head_scores = head_data["scores"]  # Shape: N*10 (top-10 candidate scores for subject)

    # Build mapping dictionaries (quadruple tuple -> top-K scores)
    pred_o_dict = {}
    pred_s_dict = {}

    for i in range(len(test_quadruples)):
        # Convert tensor quadruple to integer tuple (key for dictionaries)
        quad_tensor = tuple(test_quadruples[i])
        quad_key = (int(quad_tensor[0]), int(quad_tensor[1]), int(quad_tensor[2]), int(quad_tensor[3]))

        pred_o_dict[quad_key] = topk_tail_scores[i]
        pred_s_dict[quad_key] = topk_head_scores[i]

    return pred_o_dict, pred_s_dict


# --------------------------
# 2. Load Profile Files
# --------------------------
def load_profiles(dataset: str, q_value: int) -> tuple[dict, dict]:
    """
    Load profile-augmented quadruple lists from text file
    :param dataset: Name of target dataset
    :param q_value: Number of profile quadruples per entry (e.g., 6)
    :return: (profile_s_dict, profile_o_dict) - mappings from quadruple to profile lists
    """
    # Define profile file path
    profile_path = f"./datasets/Profile_Centric_Dataset_{q_value}/{dataset}/testset_profile_{q_value}.txt"

    profile_s_dict = {}  # Subject-related profile quadruples
    profile_o_dict = {}  # Object-related profile quadruples

    with open(profile_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue  # Skip invalid lines

            # Parse target quadruple from first field
            quad_str = parts[0].strip("()")
            s, p, o, t = quad_str.split(",")
            quad_key = (int(s), int(p), int(o), int(t))

            # Parse subject-related profile list
            profile_s_list = []
            profile_s_str = parts[1].strip("[]")
            profile_s_entries = profile_s_str.split("; ")

            for idx in range(q_value):
                entry = profile_s_entries[idx].strip("()")
                s_profile, p_profile, o_profile, t_profile = entry.split(",")
                profile_s_list.append((int(s_profile), int(p_profile), int(o_profile), int(t_profile)))

            # Parse object-related profile list
            profile_o_list = []
            profile_o_str = parts[2].strip("[]")
            profile_o_entries = profile_o_str.split("; ")

            for idx in range(q_value):
                entry = profile_o_entries[idx].strip("()")
                s_profile, p_profile, o_profile, t_profile = entry.split(",")
                profile_o_list.append((int(s_profile), int(p_profile), int(o_profile), int(t_profile)))

            profile_s_dict[quad_key] = profile_s_list
            profile_o_dict[quad_key] = profile_o_list

    return profile_s_dict, profile_o_dict


# --------------------------
# 3. Load Entity/Relation Embeddings
# --------------------------
def load_embeddings(dataset: str) -> tuple[dict, dict]:
    """
    Load precomputed entity and relation embeddings from text files
    :param dataset: Name of target dataset
    :return: (entity_embeddings, relation_embeddings) - mappings from ID to embedding vector
    """
    embedding_dir = f"./dataset_embeddings/{dataset}"

    # Load entity embeddings (ID -> vector)
    entity_embeddings = {}
    entity_emb_path = os.path.join(embedding_dir, "entity_embedding.txt")

    with open(entity_emb_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue

            ent_id = int(parts[0])
            vec_str = parts[2]
            entity_embeddings[ent_id] = np.array(vec_str.split(), dtype=float)

    # Load relation embeddings (ID -> vector)
    relation_embeddings = {}
    rel_emb_path = os.path.join(embedding_dir, "relation_embedding.txt")

    with open(rel_emb_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue

            rel_id = int(parts[0])
            vec_str = parts[2]
            relation_embeddings[rel_id] = np.array(vec_str.split(), dtype=float)

    return entity_embeddings, relation_embeddings


# --------------------------
# 4. Quadruple Embedding Calculation
# --------------------------
def get_quad_embedding(quad: tuple[int, int, int, int],
                       entity_embeddings: dict,
                       relation_embeddings: dict) -> np.ndarray:
    """
    Generate combined embedding for a quadruple (s,p,o,t)
    :param quad: Input quadruple (s, p, o, t)
    :param entity_embeddings: Entity ID to embedding dictionary
    :param relation_embeddings: Relation ID to embedding dictionary
    :return: Concatenated embedding vector [s_emb, p_emb, o_emb, t_feature]
    """
    s, p, o, t = quad

    # Get entity embeddings (fallback to ID 0 if missing)
    s_emb = entity_embeddings[s] if s in entity_embeddings else entity_embeddings[0]
    o_emb = entity_embeddings[o] if o in entity_embeddings else entity_embeddings[0]

    # Get relation embedding
    p_emb = relation_embeddings[p]

    # Create time feature (scalar converted to array for concatenation)
    t_feature = np.array([float(t)])

    # Concatenate all features into single embedding vector
    quad_embedding = np.concatenate([s_emb, p_emb, o_emb, t_feature])
    return quad_embedding


def compute_embedding_similarity(emb1: np.ndarray,
                                 emb2: np.ndarray,
                                 max_time: float) -> float:
    """
    Calculate similarity score between two quadruple embeddings (cosine similarity - time penalty)
    :param emb1: First quadruple embedding
    :param emb2: Second quadruple embedding
    :param max_time: Maximum timestamp value (for time penalty normalization)
    :return: Combined similarity score
    """
    # Extract entity/relation embeddings (exclude time feature)
    ent_rel_emb1 = emb1[:-1]
    ent_rel_emb2 = emb2[:-1]

    # Calculate cosine similarity for entity/relation embeddings
    norm1 = np.linalg.norm(ent_rel_emb1)
    norm2 = np.linalg.norm(ent_rel_emb2)

    if norm1 == 0 or norm2 == 0:
        cosine_similarity = 0.0
    else:
        cosine_similarity = np.dot(ent_rel_emb1, ent_rel_emb2) / (norm1 * norm2)

    # Calculate normalized time penalty
    time_difference = abs(emb1[-1] - emb2[-1])
    time_penalty = time_difference / max_time

    # Combined similarity score (cosine similarity minus time penalty)
    return cosine_similarity - time_penalty


# --------------------------
# 5. Core Evaluation Function
# --------------------------
def evaluate_candidate(quad: tuple[int, int, int, int],
                       q_value: int,
                       candidate_id: int,
                       profile_list: list[tuple],
                       entity_embeddings: dict,
                       relation_embeddings: dict,
                       mode: str = 'o',
                       max_time: float = -1) -> float:
    """
    Evaluate candidate entity by calculating Kendall Tau between profile ranking and similarity ranking
    :param quad: Target quadruple (s, p, o, t)
    :param q_value: Number of profile quadruples
    :param candidate_id: ID of candidate entity to evaluate
    :param profile_list: List of profile quadruples
    :param entity_embeddings: Entity ID to embedding dictionary
    :param relation_embeddings: Relation ID to embedding dictionary
    :param mode: Evaluation mode ('o' for object, 's' for subject)
    :param max_time: Maximum timestamp value for time penalty calculation
    :return: Kendall Tau correlation coefficient
    """
    # Generate new quadruples with candidate entity replacing target position
    new_quadruples = []
    for profile_quad in profile_list:
        if mode == 'o':
            # Replace object in profile quadruple with candidate ID
            new_quad = (profile_quad[0], profile_quad[1], candidate_id, profile_quad[3])
        else:
            # Replace subject in profile quadruple with candidate ID
            new_quad = (candidate_id, profile_quad[1], profile_quad[2], profile_quad[3])
        new_quadruples.append(new_quad)

    # Get embedding for original quadruple
    original_emb = get_quad_embedding(quad, entity_embeddings, relation_embeddings)

    # Calculate similarity between original and new quadruples
    similarity_scores = []
    for new_quad in new_quadruples:
        new_emb = get_quad_embedding(new_quad, entity_embeddings, relation_embeddings)
        sim_score = compute_embedding_similarity(original_emb, new_emb, max_time)
        similarity_scores.append(sim_score)

    # Generate ranking from similarity scores (descending order)
    sorted_indices = np.argsort(similarity_scores)[::-1]
    ranking = np.zeros(q_value, dtype=int)

    for rank, idx in enumerate(sorted_indices):
        ranking[idx] = rank

    # Default ranking (0 to Q-1)
    default_ranking = np.arange(q_value)

    # Calculate Kendall Tau correlation
    tau, _ = kendalltau(default_ranking, ranking)
    return tau


# --------------------------
# 6. Main Evaluation Pipeline
# --------------------------
def main(dataset: str, baseline: str, q_value: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Main pipeline for evaluating baseline predictions against profile-augmented data
    :param dataset: Name of target dataset
    :param baseline: Name of baseline model
    :param q_value: Number of profile quadruples per entry
    :return: (final_tail_scores, final_head_scores) - average Kendall Tau scores for top-K candidates
    """
    # Load prediction results
    pred_o_dict, pred_s_dict = load_baseline_predictions(dataset, baseline)

    # Load profile data
    profile_s_dict, profile_o_dict = load_profiles(dataset, q_value)

    # Load embeddings
    entity_embeddings, relation_embeddings = load_embeddings(dataset)

    # Calculate maximum timestamp for time penalty normalization
    all_timestamps = [quad[3] for quad in pred_o_dict.keys()]
    max_time = max(all_timestamps) if all_timestamps else 1.0

    # Evaluate tail (object) predictions
    tail_scores = []
    for quad, topk_candidates in tqdm(pred_o_dict.items(), desc="Evaluating tail predictions"):
        if quad not in profile_o_dict:
            continue

        profile_list = profile_o_dict[quad]
        candidate_scores = []

        for candidate in topk_candidates:
            # Convert tensor candidate ID to integer if needed
            candidate_id = candidate.item() if isinstance(candidate, torch.Tensor) else int(candidate)
            tau = evaluate_candidate(quad, q_value, candidate_id, profile_list,
                                     entity_embeddings, relation_embeddings, mode='o', max_time=max_time)
            candidate_scores.append(tau)

        tail_scores.append(candidate_scores)

    # Calculate average tail scores across all quadruples
    final_tail_scores = np.mean(tail_scores, axis=0) if tail_scores else np.zeros(10)

    # Evaluate head (subject) predictions
    head_scores = []
    for quad, topk_candidates in tqdm(pred_s_dict.items(), desc="Evaluating head predictions"):
        if quad not in profile_s_dict:
            continue

        profile_list = profile_s_dict[quad]
        candidate_scores = []

        for candidate in topk_candidates:
            candidate_id = candidate.item() if isinstance(candidate, torch.Tensor) else int(candidate)
            tau = evaluate_candidate(quad, q_value, candidate_id, profile_list,
                                     entity_embeddings, relation_embeddings, mode='s', max_time=max_time)
            candidate_scores.append(tau)

        head_scores.append(candidate_scores)

    # Calculate average head scores across all quadruples
    final_head_scores = np.mean(head_scores, axis=0) if head_scores else np.zeros(10)

    return final_tail_scores, final_head_scores


# --------------------------
# Execution Entry Point
# --------------------------
if __name__ == "__main__":
    # Configuration parameters
    DATASET = "ICEWS14"
    BASELINE_MODEL = "CyGNet"
    Q_VALUE = 6  # Number of profile quadruples

    # Weight vectors for P@3 and P@10 calculation
    W3 = [0.51020408, 0.30612245, 0.18367347]
    W10 = [0.40243336, 0.24146002, 0.14487601, 0.08692561, 0.05215536,
           0.03129322, 0.01877593, 0.01126556, 0.00675934, 0.0040556]

    # Run main evaluation pipeline
    tail_results, head_results = main(DATASET, BASELINE_MODEL, Q_VALUE)

    # Calculate combined results (average of tail and head)
    combined_results = [(a + b) / 2 for a, b in zip(tail_results, head_results)]

    # Calculate P@1, P@3, P@10 metrics
    p_at_1 = combined_results[0]
    p_at_3 = np.dot(combined_results[:3], W3)
    p_at_10 = np.dot(combined_results[:10], W10)

    # Print evaluation results
    print(
        f"{BASELINE_MODEL}\t{DATASET}\tQ{Q_VALUE}\tP@1: {round(p_at_1, 6)}\tP@3: {round(p_at_3, 6)}\tP@10: {round(p_at_10, 6)}")