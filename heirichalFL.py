import numpy as np
import torch
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
    heirichal_params["round"] += 1
    
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


def aggregate_groups(gradients, net, lr, device, seed, heirichal_params):
    """
    Aggregates gradients within each group using weighted averaging.
    
    Args:
        gradients: List of gradients from all clients
        net: Model parameters
        lr: Learning rate
        device: Device used for computation
        seed: Random seed for reproducibility
        heirichal_params: Dictionary containing hierarchical parameters
        
    Returns:
        List of aggregated gradients for each group
    """

    
    np.random.seed(seed)
    
    # Extract user membership information
    user_membership = heirichal_params["user membership"]
    num_groups = heirichal_params["num groups"]
    
    # Convert gradients to parameter lists for processing
    param_list = [torch.cat([xx.reshape((-1, 1)) for xx in x], dim=0) for x in gradients]
    
    # Get dimensions of gradient vectors
    grad_shape = param_list[0].size()
    
    # Create a list to store aggregated gradients for each group
    group_gradients = [torch.zeros(grad_shape).to(device) for _ in range(num_groups)]
    group_sizes = [0] * num_groups
    
    # Assign gradients to their respective groups and count group sizes
    for client_idx, group_idx in enumerate(user_membership):
        if client_idx < len(param_list):  # Ensure we don't go out of bounds
            group_gradients[group_idx] += param_list[client_idx]
            group_sizes[group_idx] += 1
    
    # Average gradients within each group
    for group_idx in range(num_groups):
        if group_sizes[group_idx] > 0:
            group_gradients[group_idx] /= group_sizes[group_idx]
    
    return group_gradients



def score_groups(group_gradients, heirichal_params):
    """
    Score groups using a combined adaptive threshold and ensemble voting approach
    
    Args:
        group_gradients: List of aggregated gradients for each group
        heirichal_params: Dictionary containing hierarchical parameters
        
    Returns:
        Dictionary of scores for each group
    """
    import torch
    import torch.nn.functional as F
    import numpy as np
    
    num_groups = heirichal_params["num groups"]
    
    # Skip scoring if there are too few groups
    if num_groups < 3:
        return {i: 1.0 for i in range(num_groups)}
    
    # Calculate pairwise cosine similarities between all groups
    cos_sim = torch.zeros((num_groups, num_groups), dtype=torch.float32)
    
    for i in range(num_groups):
        for j in range(i+1, num_groups):
            # Calculate cosine similarity between group gradients
            similarity = F.cosine_similarity(
                group_gradients[i], 
                group_gradients[j], 
                dim=0, 
                eps=1e-9
            )
            # Store similarity (symmetric matrix)
            cos_sim[i, j] = cos_sim[j, i] = similarity
    
    # --- Ensemble Voting ---
    # Each group votes on other groups' trustworthiness
    votes = torch.zeros(num_groups)
    
    for i in range(num_groups):
        # Each group rates others based on their similarity
        similarities = cos_sim[i, :]
        # Don't count self-similarity
        similarities[i] = 0
        # Calculate votes: higher similarity gets more trust
        votes += similarities

    
    return [float(v) for v in votes]
