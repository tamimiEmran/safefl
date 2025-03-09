import numpy as np
import torch
import torch.nn.functional as F
import numpy as np

isGroupGradientsToBeAveraged = False 
bypass_robust=False # normalizes groups gradients and averages them
simple_average=True # simple average of group gradients

def simulate_groups(heirichal_params, number_of_users, seed):
    """
    Simulates groups by assigning users to groups and initializing necessary parameters.
    Ensures each group has an equal number of users (or as close as possible).
    
    Args:
        heirichal_params: Dictionary containing hierarchical parameters
        number_of_users: Number of users/clients in the system
        seed: Random seed for reproducibility
    """
    # Set random seed for reproducibility
    np.random.seed(seed)
    
    round_num = heirichal_params["round"]
    num_groups = heirichal_params["num groups"]
    
    # Initialize user membership and scores if first round
    if round_num == 1:
        # Initialize user scores with 0.0 for all users
        heirichal_params["user score"] = [0.0] * number_of_users
        
        # Create array of user indices and shuffle it
        user_indices = np.arange(number_of_users)
        np.random.shuffle(user_indices)
        
        # Calculate base number of users per group and remainder
        base_per_group = number_of_users // num_groups
        remainder = number_of_users % num_groups
        
        # Assign users to groups ensuring equal distribution
        user_membership = [0] * number_of_users
        start_idx = 0
        
        for group_id in range(num_groups):
            # Add one extra user to some groups if there's a remainder
            group_size = base_per_group + (1 if group_id < remainder else 0)
            end_idx = start_idx + group_size
            
            # Assign this range of users to the current group
            for idx in range(start_idx, end_idx):
                if idx < number_of_users:
                    user_membership[user_indices[idx]] = group_id
            
            start_idx = end_idx
        
        heirichal_params["user membership"] = user_membership
    
    # Increment round number
    
    
    return heirichal_params


def shuffle_users(heirichal_params, number_of_users, seed):
    """
    Shuffles users diagonally across groups, handling unequal group sizes.
    Uses dummy users for shuffling but removes them from final assignment.
    
    Args:
        heirichal_params: Dictionary containing hierarchical parameters
        number_of_users: Number of users/clients in the system
        seed: Random seed for reproducibility
    """
    # Set random seed
    np.random.seed(seed)
    
    # Get current user membership and number of groups
    user_membership = heirichal_params["user membership"]
    num_groups = heirichal_params["num groups"]
    
    # Create 2D representation of groups and users
    groups_2d = [[] for _ in range(num_groups)]
    for user_id, group_id in enumerate(user_membership):
        if user_id < number_of_users:  # Only consider real users
            groups_2d[group_id].append(user_id)
    
    # Find the maximum group size
    max_group_size = max(len(group) for group in groups_2d)
    
    # Add dummy users to make all groups equal size
    dummy_user_start = number_of_users
    for group_id, group in enumerate(groups_2d):
        while len(group) < max_group_size:
            group.append(dummy_user_start)
            dummy_user_start += 1
    
    # Flatten the 2D list with dummy users
    all_users = [user for group in groups_2d for user in group]
    
    # Create a diagonal pattern
    new_positions = {}  # Will map user_id -> new_group_id
    
    # Calculate total diagonals
    total_diags = num_groups * 2 - 1
    s_center = (total_diags - 1) // 2
    
    # Create diagonal traversal order
    diagonal_order = list(range(s_center, total_diags)) + list(range(s_center - 1, -1, -1))
    
    # Assign users to new groups based on diagonal pattern
    user_index = 0
    for old_group_id in range(num_groups):
        for pos_in_group in range(max_group_size):
            user_id = groups_2d[old_group_id][pos_in_group]
            
            # Get diagonal position
            diag_idx = diagonal_order[user_index % len(diagonal_order)]
            
            # Compute new group_id using diagonal pattern
            new_group = (old_group_id + diag_idx) % num_groups
            
            # Store the new group for this user
            new_positions[user_id] = new_group
            
            user_index += 1
    
    # Create new user membership list, ignoring dummy users
    new_user_membership = [0] * number_of_users
    for user_id in range(number_of_users):
        if user_id in new_positions:
            new_user_membership[user_id] = new_positions[user_id]
        else:
            # Fallback for any user not in the mapping (should not happen)
            new_user_membership[user_id] = np.random.randint(0, num_groups)
    
    # Update the user membership
    heirichal_params["user membership"] = new_user_membership
    
    return heirichal_params


def organize_users_by_group(user_group_assignments, total_group_count, user_trust_scores):
    """
    Organizes users by their group membership and scores.
    
    Args:
        user_group_assignments: List indicating which group each user belongs to
        total_group_count: Total number of groups
        user_trust_scores: List of trust scores for each user
        
    Returns:
        List of groups, where each group contains tuples of (user_id, trust_score)
    """
    groups_with_users = [[] for _ in range(total_group_count)]
    for user_id, group_id in enumerate(user_group_assignments):
        if user_id < len(user_trust_scores):  # Ensure we don't go out of bounds
            groups_with_users[group_id].append((user_id, user_trust_scores[user_id]))
    return groups_with_users

def filter_malicious_users(groups_with_users, all_user_trust_scores, malicious_percentage, total_user_count):
    """
    Filters out assumed malicious users globally based on their scores.
    
    Args:
        groups_with_users: List of groups, where each group contains tuples of (user_id, trust_score)
        all_user_trust_scores: List of trust scores for each user
        malicious_percentage: Percentage of users assumed to be malicious overall
        total_user_count: Total number of users across all groups
        
    Returns:
        List of valid groups after filtering, where each entry contains
        the original group_id and list of trusted user indices
    """
    # Calculate total number of users to exclude
    users_to_exclude_count = max(1, int(malicious_percentage * total_user_count))
    
    # Get indices of users with lowest scores globally
    all_users_with_scores = [(user_id, score) for user_id, score in enumerate(all_user_trust_scores) if user_id < total_user_count]
    users_sorted_by_score = sorted(all_users_with_scores, key=lambda x: x[1])
    malicious_user_ids = set([user[0] for user in users_sorted_by_score[:users_to_exclude_count]])
    
    # Filter malicious users from each group
    filtered_groups_with_ids = []
    for group_id, group_users in enumerate(groups_with_users):
        # Keep only trusted users
        trusted_user_ids = [user_id for user_id, _ in group_users if user_id not in malicious_user_ids]
        
        # If after filtering, we have more than 1 user, keep the group
        if len(trusted_user_ids) > 1:
            filtered_groups_with_ids.append((group_id, trusted_user_ids))
    
    return filtered_groups_with_ids

def compute_group_gradients(filtered_groups_with_ids, user_gradients, gradient_shape, computation_device):
    """
    Computes the average gradient for each filtered group.
    
    Args:
        filtered_groups_with_ids: List of valid groups after filtering with their original indices
        user_gradients: List of parameter gradients for each user
        gradient_shape: Shape of the gradient tensors
        computation_device: Device for tensor computations
        
    Returns:
        Dictionary mapping original group indices to their aggregated gradients
    """
    # Use a dictionary to maintain the original group indices
    group_to_gradient_mapping = {}
    
    for group_id, trusted_user_ids in filtered_groups_with_ids:
        if not trusted_user_ids:  # Skip if no users in group (shouldn't happen)
            continue
            
        # Aggregate gradients for this group
        aggregated_group_gradient = torch.zeros(gradient_shape).to(computation_device)
        for trusted_user_id in trusted_user_ids:
            aggregated_group_gradient += user_gradients[trusted_user_id]
        
        # Average the gradients
        if isGroupGradientsToBeAveraged:
            aggregated_group_gradient /= len(trusted_user_ids)
            
        # Store with original group index
        group_to_gradient_mapping[group_id] = aggregated_group_gradient
    
    return group_to_gradient_mapping



def find_most_trusted_group(groups_with_users, user_scores):
    """
    Identifies the most trusted group based on average user trust scores.
    
    Args:
        groups_with_users: List of groups, each containing tuples of (user_id, score)
        user_scores: List of trust scores for all users
        
    Returns:
        Tuple of (most_trusted_group_id, trusted_user_ids), or (0, []) if no valid groups
    """
    # Calculate average trust score for each group
    group_trust_scores = {}
    for group_id, group_users in enumerate(groups_with_users):
        if len(group_users) > 0:
            group_avg_score = sum(score for _, score in group_users) / len(group_users)
            group_trust_scores[group_id] = group_avg_score
    
    if not group_trust_scores:
        return 0, []  # Return default group ID and empty list instead of None
        
    # Find most trusted group
    most_trusted_group = max(group_trust_scores.items(), key=lambda x: x[1])[0]
    trusted_user_ids = [user_id for user_id, _ in groups_with_users[most_trusted_group]]
    
    return most_trusted_group, trusted_user_ids


def select_top_trusted_users(trusted_user_ids, user_scores, proportion=0.5):
    """
    Selects the top trusted users from a group based on their trust scores.
    
    Args:
        trusted_user_ids: List of user IDs
        user_scores: List of trust scores for all users
        proportion: Proportion of users to select (default 0.5)
        
    Returns:
        List of selected user IDs
    """
    if len(trusted_user_ids) <= 1:
        return trusted_user_ids
        
    user_scores_pairs = [(user_id, user_scores[user_id]) for user_id in trusted_user_ids]
    user_scores_pairs.sort(key=lambda x: x[1], reverse=True)
    
    # Select top proportion of users (at least 1)
    top_count = max(int(len(user_scores_pairs) * proportion), 1)
    return [user_id for user_id, _ in user_scores_pairs[:top_count]]


def handle_filtering_fallback(groups_with_users, user_scores):
    """
    Provides fallback strategies when all groups are filtered out.
    
    Args:
        groups_with_users: List of groups, each containing tuples of (user_id, score)
        user_scores: List of trust scores for all users
        
    Returns:
        List of tuples (group_id, user_ids) for valid groups
    """
    # Try most trusted group first
    most_trusted_group, trusted_user_ids = find_most_trusted_group(groups_with_users, user_scores)
    
    if len(trusted_user_ids) > 1:  # Check length instead of None
        best_user_ids = select_top_trusted_users(trusted_user_ids, user_scores)
        if len(best_user_ids) > 1:
            return [(most_trusted_group, best_user_ids)]
    
    # Try largest group as fallback
    if groups_with_users:
        largest_group_id = max(range(len(groups_with_users)), 
                              key=lambda i: len(groups_with_users[i]))
        largest_group_users = [user_id for user_id, _ in groups_with_users[largest_group_id]]
        
        if len(largest_group_users) > 1:
            return [(largest_group_id, largest_group_users)]
    
    # Ultimate fallback - use all users across all groups
    all_users = []
    for group in groups_with_users:
        all_users.extend([user_id for user_id, _ in group])
    
    if all_users:
        return [(0, all_users)]
    
    return []


#group_gradients_for_scoring = hfl.aggregate_groups(gradients, device, seed, heirichal_params, skip_filtering=True)


def aggregate_groups(all_user_gradients, computation_device, random_seed, hierarchical_parameters, skip_filtering=False):
    """
    Aggregates gradients within each group, optionally excluding assumed malicious users
    globally and removing groups with only one user after filtering.
    
    Args:
        all_user_gradients: List of gradients from all clients
        computation_device: Device used for computation
        random_seed: Random seed for reproducibility
        hierarchical_parameters: Dictionary containing hierarchical parameters
        skip_filtering: If True, bypasses the malicious user filtering step
        
    Returns:
        Dictionary mapping group indices to their aggregated gradients
    """
    np.random.seed(random_seed)
    
    # Extract parameters
    user_group_assignments = hierarchical_parameters["user membership"]
    total_group_count = hierarchical_parameters["num groups"]
    user_trust_scores = hierarchical_parameters["user score"]
    malicious_percentage = hierarchical_parameters["assumed_mal_prct"]
    total_user_count = len(all_user_gradients)
    
    # Convert gradients to parameter lists
    user_gradient_vectors = [torch.cat([param.reshape((-1, 1)) for param in gradient], dim=0) for gradient in all_user_gradients]
    gradient_shape = user_gradient_vectors[0].size()
    
    # Organize users by group
    groups_with_users = organize_users_by_group(user_group_assignments, total_group_count, user_trust_scores)
    
    if skip_filtering:
        # Skip filtering step
        filtered_groups_with_ids = []
        for group_id, group_users in enumerate(groups_with_users):
            user_ids = [user_id for user_id, _ in group_users]
            if len(user_ids) > 1:
                filtered_groups_with_ids.append((group_id, user_ids))
    else:
        # Apply normal filtering
        filtered_groups_with_ids = filter_malicious_users(groups_with_users, user_trust_scores, malicious_percentage, total_user_count)
    
    # Check if we have any valid groups after filtering
    if len(filtered_groups_with_ids) == 0:
        print("Warning: All groups were filtered out. Using fallback strategy.")
        filtered_groups_with_ids = handle_filtering_fallback(groups_with_users, user_trust_scores)
    
    # Compute gradients for each group
    group_to_gradient_mapping = compute_group_gradients(filtered_groups_with_ids, user_gradient_vectors, gradient_shape, computation_device)
    
    return group_to_gradient_mapping



def score_groups(group_to_gradient_mapping, hierarchical_parameters):
    """
    Score groups using a combined adaptive threshold and ensemble voting approach,
    accounting for potentially missing groups.
    
    Args:
        group_to_gradient_mapping: Dictionary mapping group IDs to their aggregated gradients
        hierarchical_parameters: Dictionary containing hierarchical parameters
        
    Returns:
        Dictionary mapping group IDs to their trust scores
    """
    # Get existing group IDs
    existing_group_ids = list(group_to_gradient_mapping.keys())
    number_of_existing_groups = len(existing_group_ids)
    
    # Skip scoring if there are too few groups
    if number_of_existing_groups < 3:
        return {group_id: 1.0 for group_id in existing_group_ids}
    
    # Calculate pairwise cosine similarities between all existing groups
    cos_sim = torch.zeros((number_of_existing_groups, number_of_existing_groups), dtype=torch.float32)
    
    for i in range(number_of_existing_groups):
        group_id_i = existing_group_ids[i]
        for j in range(i+1, number_of_existing_groups):
            group_id_j = existing_group_ids[j]
            # Calculate cosine similarity between group gradients
            similarity = F.cosine_similarity(
                group_to_gradient_mapping[group_id_i], 
                group_to_gradient_mapping[group_id_j], 
                dim=0, 
                eps=1e-9
            )
            # Store similarity (symmetric matrix)
            cos_sim[i, j] = cos_sim[j, i] = similarity
    
    # --- Ensemble Voting ---
    # Each group votes on other groups' trustworthiness
    group_votes = torch.zeros(number_of_existing_groups)
    
    for i in range(number_of_existing_groups):
        # Each group rates others based on their similarity
        similarities = cos_sim[i, :]
        # Don't count self-similarity
        similarities[i] = 0
        # Calculate votes: higher similarity gets more trust
        group_votes += similarities
    
    # Convert votes tensor to dictionary mapping original group IDs to scores
    group_id_to_score = {
        existing_group_ids[i]: float(group_votes[i]) 
        for i in range(number_of_existing_groups)
    }
    
    return group_id_to_score



def update_user_scores(heirichal_params, groups_scores):
    user_scores = heirichal_params["user score"]
    number_of_groups = heirichal_params["num groups"]

    # create a list similar in length to the user scores
    # and initialize it with zeros
    user_scores_adjustments = [0.0] * len(user_scores)
    current_user_scores = [0.0] * len(user_scores)

    # Sort groups by their scores
    group_ranking = sorted(groups_scores, key=groups_scores.get, reverse=True)

    # Calculate the middle point for determining positive/negative adjustments
    mid_point = (number_of_groups - 1) / 2

    # Create mapping of group_id to score adjustment
    # Groups above middle point get positive scores, below get negative
    group_adjustments = {
        group_id: (mid_point - rank) / number_of_groups
        for rank, group_id in enumerate(group_ranking)
    }

    # Update each user's score based on their group's adjustment
    for user_id, group_id in enumerate(heirichal_params["user membership"]):
        user_scores[user_id] += group_adjustments[group_id]
        user_scores_adjustments[user_id] = group_adjustments[group_id]
        current_user_scores[user_id] = user_scores[user_id]

    
    heirichal_params["user score"] = user_scores

    return heirichal_params, user_scores_adjustments, current_user_scores


    
def robust_groups_aggregation(group_gradients, net, lr, device, heirichal_params):
    """
    Implements a robust aggregation mechanism for group gradients.
    
    Args:
        group_gradients: Dictionary mapping group IDs to their aggregated gradients
        net: Model used for training
        lr: Learning rate for the optimizer
        device: Device used for computation
        heirichal_params: Dictionary containing hierarchical parameters
        bypass_robust: If True, bypasses the robust filtering mechanism
        simple_average: If True, uses a simple average of all gradients
        
    Returns:
        global_update: The robustly aggregated update vector
    """
    import torch
    
    # Verify we have gradients to work with
    if not group_gradients:
        raise ValueError("No group gradients available for aggregation")
    
    # Simple averaging of all gradients if requested
    if simple_average:
        first_grad = next(iter(group_gradients.values()))
        global_update = torch.zeros_like(first_grad).to(device)
        for grad in group_gradients.values():
            global_update += grad
        global_update /= len(group_gradients)
        
    # Skip robust aggregation if requested
    elif bypass_robust:
        # Calculate L2 norm for each group gradient
        group_norms = {group_id: torch.norm(grad, p=2) for group_id, grad in group_gradients.items()}
        
        # Normalize all gradients by their norms
        normalized_gradients = {}
        for group_id, grad in group_gradients.items():
            normalized_gradients[group_id] = grad / group_norms[group_id]
        
        # Average the normalized gradients
        first_grad = next(iter(normalized_gradients.values()))
        global_update = torch.zeros_like(first_grad).to(device)
        for grad in normalized_gradients.values():
            global_update += grad
        global_update /= len(normalized_gradients)
        
    # Use robust aggregation
    else:
        # Calculate L2 norm for each group gradient
        group_norms = {group_id: torch.norm(grad, p=2) for group_id, grad in group_gradients.items()}
        
        # Find median L2 norm
        median_norm = torch.median(torch.tensor(list(group_norms.values())))
        
        # Filter groups with L2 norms below or equal to median
        filtered_gradients = {}
        for group_id, grad in group_gradients.items():
            if group_norms[group_id] <= median_norm:
                # Scale down gradients by l2Current/l2Median
                scaling_factor = group_norms[group_id] / median_norm
                filtered_gradients[group_id] = grad * scaling_factor
        
        # If all gradients were filtered out, use median
        if len(filtered_gradients) == 0:
            # Find median gradient by sorting norm values
            sorted_group_ids = [group_id for group_id, _ in 
                              sorted(group_norms.items(), key=lambda item: item[1])]
            median_idx = len(sorted_group_ids) // 2
            median_group_id = sorted_group_ids[median_idx]
            global_update = group_gradients[median_group_id]
        else:
            # Average the filtered and scaled gradients
            first_grad = next(iter(filtered_gradients.values()))
            global_update = torch.zeros_like(first_grad).to(device)
            for grad in filtered_gradients.values():
                global_update += grad
            global_update /= len(filtered_gradients)
    
    # Update the global model
    idx = 0
    for j, param in enumerate(net.parameters()):
        param.add_(global_update[idx:(idx + torch.numel(param))].reshape(tuple(param.size())), alpha=-lr)
        idx += torch.numel(param)
    
    return global_update