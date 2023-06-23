import numpy as np
import os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm import tqdm
import PSO 

# Features and Labels Path
data_dir = "SAMPLE"

# Get the list of all .npy files in the directory
file_list = [f for f in os.listdir(data_dir) if f.endswith('.npy')]

# Iterate over all unique patient identifiers
patient_ids = set([f.split('_')[1].split('.')[0] for f in file_list])

# Parameters for the PSO
n_particles = 100
n_iterations = 50
w_max = 1
w_min = 0.2
c1 = 1
c2 = 1

# Store best accuracies and corresponding subject IDs
best_accuracies = []
subject_ids = []

# Store best-performing feature filters and corresponding indices
best_feature_filters = []
kept_feature_indices = []
removed_feature_indices = []

# Counter for skipped iterations
skipped_iterations = 0

# Wrap patient_ids with tqdm for progress bar
for patient_id in tqdm(patient_ids):
    # Load the features and labels
    features = np.load(os.path.join(data_dir, f"Features_{patient_id}.npy"))
    labels = np.load(os.path.join(data_dir, f"Labels_{patient_id}.npy"))

    # Reshape to (epochs, channel x frequency x time)
    features_flat = features.reshape(features.shape[0], -1)

    # Apply PSO
    best_position, best_score = pso_svm.PSO(features_flat, labels, n_particles, n_iterations, w_max, w_min, c1, c2)

    # Apply the best mask to the features
    best_mask = best_position.astype(bool)
    reduced_features = features_flat[:, best_mask]

    if reduced_features.shape[1] == 0:
        print(f"No features selected for {patient_id}, skipping...")
        skipped_iterations += 1
        continue

    print(f"Best accuracy for {patient_id}: {best_score}")

    # Append data to lists
    best_accuracies.append(best_score)
    subject_ids.append(patient_id)

    # Save best performing feature filter and corresponding indices
    best_feature_filters.append(best_mask)
    kept_feature_indices.append(np.where(best_mask)[0])
    removed_feature_indices.append(np.where(~best_mask)[0])

average_accuracy = np.mean(best_accuracies)
print(f"Average accuracy: {average_accuracy}")

# Print number of skipped iterations
print(f"Number of skipped iterations: {skipped_iterations}")

# Bar plot
plt.bar(subject_ids, best_accuracies)
plt.xlabel('Subject ID')
plt.ylabel('Accuracy')
plt.yticks(np.arange(0, 1, 0.05))  # adjust as per your needs
plt.title('SVM Classifier Accuracies per Subject')
plt.savefig('svm_accuracies.png')  # Save the plot
plt.show()

# Save the best performing feature filters and the corresponding indices 
np.save('best_feature_filters.npy', best_feature_filters)
np.save('kept_feature_indices.npy', kept_feature_indices)
np.save('removed_feature_indices.npy', removed_feature_indices)

# Write parameters, skipped iterations, and best filter performances to a text file
with open('pso_svm_info.txt', 'w') as f:
    f.write(f'Number of skipped iterations: {skipped_iterations}\n')
    f.write(f'PSO parameters: n_particles={n_particles}, n_iterations={n_iterations}, w_max={w_max}, w_min={w_min}, c1={c1}, c2={c2}\n')
    for id, acc, kept, removed in zip(subject_ids, best_accuracies, kept_feature_indices, removed_feature_indices):
        percent_kept = len(kept) / (len(kept) + len(removed)) * 100
        print(f'Subject ID: {id}, Accuracy: {acc}, Percent of features kept: {percent_kept}%')
        f.write(f'Subject ID: {id}, Accuracy: {acc}, Percent of features kept: {percent_kept}%, Kept features: {len(kept)}, Removed features: {len(removed)}\n')
