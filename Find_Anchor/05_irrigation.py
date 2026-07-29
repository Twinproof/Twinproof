import pandas as pd
import numpy as np
from pathlib import Path

# Load the refined clustered data.
SCRIPT_DIR = Path(__file__).resolve().parent
refined_file = SCRIPT_DIR / "anchor_cluster" / "anchor_cluster_route_004_refined.csv"
data = pd.read_csv(refined_file)

# Split Anchor_Info into start and end columns.
data[['start', 'end']] = data['Anchor_Info'].str.extract(r'\((\d+),\s*(\d+)\)').astype(int)

# Calculate segment lengths.
data['length'] = data['end'] - data['start']

# Count samples in each cluster.
category_counts = data['Cluster_Label'].value_counts()

# Retain clusters containing more than two samples.
valid_categories = category_counts[category_counts > 2].index.tolist()
data = data[data['Cluster_Label'].isin(valid_categories)]

# Store filtered cluster data.
filtered_dfs = []

# Process each cluster independently.
for label in data['Cluster_Label'].unique():
    subset = data[data['Cluster_Label'] == label].copy()

    # Detect outliers only in clusters with more than two samples.
    if category_counts[label] > 2:
        q1, q3 = subset['start'].quantile([0.2  , 0.8])
        iqr = q3 - q1
        lower_bound = q1 - 0.5 * iqr
        upper_bound = q3 + 0.5 * iqr

        subset = subset[(subset['start'] >= lower_bound) & (subset['start'] <= upper_bound)]

    filtered_dfs.append(subset)

# Combine the filtered clusters.
filtered_data = pd.concat(filtered_dfs).reset_index(drop=True)

# Order clusters by the median start index.
category_order = filtered_data.groupby('Cluster_Label')['start'].median().sort_values().index.tolist()

# Generate consecutive cluster labels in the resulting order.
new_labels = {old_label: new_label for new_label, old_label in enumerate(category_order)}

# Replace Cluster_Label with the new labels.
filtered_data['Cluster_Label'] = filtered_data['Cluster_Label'].map(new_labels)

# Remove temporary columns.
filtered_data = filtered_data.drop(columns=['start', 'end', 'length'])

# Sort by the new Cluster_Label before saving.
filtered_data = filtered_data.sort_values(by='Cluster_Label').reset_index(drop=True)

# Save the filtered data.
filtered_data.to_csv(refined_file, index=False)
print("Processing complete. Filtered data saved.")
