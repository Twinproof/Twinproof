import ast
import os
from collections import defaultdict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path
matplotlib.use('TkAgg')
import PDR_Path
import Same


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
file_path = PROJECT_ROOT / "data" / "collectionData" / "route_004" / "sensor_1.csv"
df = pd.read_csv(file_path)
Anchor_data = pd.read_csv(
    PROJECT_ROOT / "Find_Anchor" / "anchor_cluster" / "anchor_cluster_route_004_refined.csv"
)




def Anchor_into_csv(data, csv_path='Anchor_connection.csv', feature_folder='Anchor_feature_parking'):
    """Process the inputs and return the corresponding result."""
    
    os.makedirs(feature_folder, exist_ok=True)  

    
    if os.path.exists(csv_path):
        existing_data = pd.read_csv(csv_path)
        if 'Cluster_Label' in existing_data.columns:
            max_existing_label = existing_data['Cluster_Label'].max() + 1
        else:
            max_existing_label = 0
    else:
        max_existing_label = 0

    
    
    unique_labels = sorted(data['Sorted_Cluster_Label'].unique())
    label_mapping = {old: new for new, old in enumerate(unique_labels, start=max_existing_label)}
    data['Cluster_Label'] = data['Sorted_Cluster_Label'].map(label_mapping)

    
    feature_files = {}
    for label in sorted(data['Cluster_Label'].unique()):
        subset = data[data['Cluster_Label'] == label]
        feature_file = os.path.join(feature_folder, f"anchor_feature_{label}.csv")
        subset.to_csv(feature_file, index=False)
        feature_files[label] = feature_file

    
    cluster_labels = sorted(data['Cluster_Label'].unique())
    connected_classes = []
    feature_csvs = []

    for i, label in enumerate(cluster_labels):
        connected = []
        
        if i > 0:
            connected.append(cluster_labels[i-1])
        if i < len(cluster_labels)-1:
            connected.append(cluster_labels[i+1])

        connected_classes.append(connected)
        feature_csvs.append(feature_files[label])

    connection_df = pd.DataFrame({
        'Cluster_Label': cluster_labels,
        'Connected_Classes': connected_classes,
        'feature_csv': feature_csvs
    })

    
    if os.path.exists(csv_path):
        connection_df.to_csv(csv_path, mode='a', header=False, index=False)
    else:
        connection_df.to_csv(csv_path, index=False)

    print(f"Anchor_connection updated:{csv_path}")
    print(f"Feature file saved to:{feature_folder}")
    return max_existing_label, label_mapping



def Path_into_csv(paths, csv_path='Paths.csv', index=0):
    
    path_data = []
    
    if os.path.exists(csv_path):
        existing_data = pd.read_csv(csv_path)
        if 'Path_ID' in existing_data.columns:
            max_existing_id = existing_data['Path_ID'].max()  
        else:
            max_existing_id = -1  
    else:
        max_existing_id = -1  

    
    for path in paths:
        start_end = path[0]  
        start_anchor = start_end[0] + index  
        end_anchor = start_end[1] + index

        
        path_length = sum([segment[1] for segment in path[1:]])

        
        path_situation = path[1:]

        
        path_data.append([max_existing_id + 1, start_anchor, end_anchor, path_length, path_situation])
        max_existing_id += 1  

    
    path_data_df = pd.DataFrame(path_data,
                                columns=['Path_ID', 'Start_Anchor', 'End_Anchor', 'Path_Length', 'Path_Situation'])

    
    if os.path.exists(csv_path):
        path_data_df.to_csv(csv_path, mode='a', header=False, index=False)  
    else:
        path_data_df.to_csv(csv_path, index=False)  

    print(f"Path data saved to {csv_path}")



def merge_anchor_features(feature_folder, old_idx, new_idx):
    """Process the inputs and return the corresponding result."""
    old_file = os.path.join(feature_folder, f"anchor_feature_{old_idx}.csv")
    new_file = os.path.join(feature_folder, f"anchor_feature_{new_idx}.csv")

    if not (os.path.exists(old_file) and os.path.exists(new_file)):
        print(f"Anchor file does not exist: {old_file} or {new_file}")
        return

    df_old = pd.read_csv(old_file)
    df_new = pd.read_csv(new_file)
    df_merged = pd.concat([df_old, df_new], ignore_index=True)

    df_merged.to_csv(old_file, index=False)
    os.remove(new_file)  
    print(f"Merged {new_file} into {old_file} and deleted {new_file}")

def update_anchor_labels(anchor_connection, path, feature_folder, new_index, k=3.0, alpha=0.5):
    """Process the inputs and return the corresponding result."""
    
    data = pd.read_csv(anchor_connection)
    if isinstance(data['Connected_Classes'].iloc[0], str):
        data['Connected_Classes'] = data['Connected_Classes'].apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) else x
        )

    
    pairs = Same.same(feature_folder, new_index, k=k, alpha=alpha)
    pairs=[[16,19],[17,21]]
    if not pairs:
        print("No mergeable anchors detected.")
        return

    label_changes = {}

    
    for old_label, new_label in pairs:
        if new_label in label_changes:  
            continue
        label_changes[new_label] = old_label
        print(f"Merge new anchor {new_label} -> existing anchor {old_label}")

        
        old_path = os.path.join(feature_folder, f"anchor_feature_{old_label}.csv")
        new_path = os.path.join(feature_folder, f"anchor_feature_{new_label}.csv")
        if os.path.exists(new_path):
            df_old = pd.read_csv(old_path) if os.path.exists(old_path) else pd.DataFrame()
            df_new = pd.read_csv(new_path)
            pd.concat([df_old, df_new], ignore_index=True).to_csv(old_path, index=False)
            os.remove(new_path)
            print(f"Merged feature file {new_label} -> {old_label}")

    
    data['Cluster_Label'] = data['Cluster_Label'].replace(label_changes)

    
    def update_connections(classes):
        updated = [label_changes.get(cls, cls) for cls in classes]
        return sorted(set(updated))  

    data['Connected_Classes'] = data['Connected_Classes'].apply(update_connections)

    
    merged_rows = []
    for label in sorted(data['Cluster_Label'].unique()):
        same_rows = data[data['Cluster_Label'] == label]
        if same_rows.shape[0] > 1:
            all_connections = set()
            feature_csv = os.path.join(feature_folder, f"anchor_feature_{label}.csv")
            for _, row in same_rows.iterrows():
                all_connections.update(row['Connected_Classes'])
            if label in all_connections:
                all_connections.remove(label)
            merged_rows.append({
                "Cluster_Label": label,
                "Connected_Classes": sorted(all_connections),
                "feature_csv": feature_csv
            })
        else:
            merged_rows.append(same_rows.iloc[0].to_dict())
    data = pd.DataFrame(merged_rows)

    
    path_data = pd.read_csv(path)
    def update_anchor(anchor_id):
        return label_changes.get(anchor_id, anchor_id)
    path_data['Start_Anchor'] = path_data['Start_Anchor'].apply(update_anchor)
    path_data['End_Anchor'] = path_data['End_Anchor'].apply(update_anchor)

    
    data.to_csv(anchor_connection, index=False)
    path_data.to_csv(path, index=False)
    print(f"Updated {anchor_connection} and {path} anchor labels in ")
    print(f"Final merge mapping: {label_changes}")



if __name__ == "__main__":
    median_start_colum,turn_info,paths=PDR_Path.PDR(df, Anchor_data, If_show=True)
    max_existing_label, label_mapping=Anchor_into_csv(
        Anchor_data,
        SCRIPT_DIR / "Anchor_connection.csv",
        SCRIPT_DIR / "Anchor_feature_parking",
    )
    Path_into_csv(paths, csv_path=SCRIPT_DIR / "Paths.csv", index=max_existing_label)

    
    # update_anchor_labels('Anchor_connection.csv','Paths.csv',"Anchor_feature_parking",max_existing_label)
