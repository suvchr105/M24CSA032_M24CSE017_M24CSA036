import torch
import numpy as np
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader
from tqdm import tqdm

def select_prototypes(model, dataset, exemplar_count_per_class, device):
    """
    Selects prototype samples for each class using K-Means clustering in the feature space.
    """
    model.eval()
    data_loader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    # Collect all features and metadata
    all_features = []
    all_labels = []
    all_indices = []
    
    with torch.no_grad():
        for i, (audio, visual, ques, attn_mask, anser, que_id, label) in enumerate(tqdm(data_loader, desc="Extracting features for prototypes")):
            audio = audio.to(device)
            visual = visual.to(device)
            ques = ques.to(device)
            attn_mask = attn_mask.to(device)
            
            # Forward pass to get features
            # We use the final fused feature before the classifier
            _, feat = model(audio=audio, visual=visual, question=ques, attention_mask=attn_mask, out_features=True)
            
            all_features.append(feat.cpu().numpy())
            all_labels.append(label.numpy())
            # We need to map back to the dataset samples
            # indices = np.arange(i * 32, i * 32 + len(label))
            # all_indices.append(indices)

    all_features = np.concatenate(all_features, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    unique_labels = np.unique(all_labels)
    selected_samples = []
    
    for label in unique_labels:
        class_indices = np.where(all_labels == label)[0]
        class_features = all_features[class_indices]
        
        # Determine how many prototypes to select for this class
        num_clusters = min(len(class_indices), exemplar_count_per_class)
        if num_clusters <= 0:
            continue
            
        # Run K-Means
        kmeans = KMeans(n_provinces=num_clusters, n_init=10, random_state=42)
        kmeans.fit(class_features)
        centroids = kmeans.cluster_centers_
        
        # For each centroid, find the nearest neighbor in the class features
        for centroid in centroids:
            distances = np.linalg.norm(class_features - centroid, axis=1)
            nearest_idx_in_class = np.argmin(distances)
            global_idx = class_indices[nearest_idx_in_class]
            
            # Retrieve the sample from the dataset
            sample = dataset.all_current_data_vids[global_idx]
            selected_samples.append(sample)
            
    return selected_samples

def update_exemplar_set_with_prototypes(args, model, train_dataset, exemplar_dataset, device):
    """Updates the exemplar dataset using prototype selection."""
    total_seen_classes = len(train_dataset.label_to_ix)
    exemplar_num_per_class = max(1, args.memory_size // total_seen_classes)
    
    print(f"Selecting prototypes: {exemplar_num_per_class} per class for {total_seen_classes} classes.")
    
    # 1. Trimming old exemplars to make room (re-selecting prototypes for OLD classes might be ideal, 
    # but here we just process the CURRENT task samples and then maybe update the whole buffer if needed)
    
    # For simplicity, we select prototypes for the CURRENTly finished task/step samples
    new_prototypes = select_prototypes(model, train_dataset, exemplar_num_per_class, device)
    
    # 2. Add new prototypes and potentially prune old ones to stay within memory_size
    # Current behavior of baseline is to append and then trim.
    
    if exemplar_dataset.exemplar_class_vids_set:
        # Prune existing to make room
        # existing_per_class = args.memory_size // (total_seen_classes - some_value) ...
        # Simplified: just keep the most recent ones or re-run selection on the whole buffer later if possible.
        # But usually, it's: store prototypes of current step, and keep prototypes of previous steps.
        
        # Trim existing to exemplar_num_per_class * (total_seen_classes - new_classes)
        # We'll just append and then the exemplarLoader logic in baseline trims it.
        pass
        
    exemplar_dataset.exemplar_class_vids_set.extend(new_prototypes)
    # The baseline's update_exemplars also trims the set.
    # We should ensure the buffer doesn't overflow.
    exemplar_dataset.exemplar_vids_set = [v for v in exemplar_dataset.exemplar_class_vids_set if v is not None]
    
    # Optional: ensure overall size limit
    if len(exemplar_dataset.exemplar_vids_set) > args.memory_size:
        # Keep the latest ones or better: keep equal distribution
        # For now, let's just stick to the baseline's trimming logic if I can update it.
        pass

