import random
import os
from sklearn.model_selection import train_test_split

seed = 100
random.seed(seed)

id2disease = [
    "adhd",
    "anxiety",
    "bipolar",
    "depression",
    "mdd",
    "ocd",
    "ppd",
    "ptsd"
]
disease2id = {disease: id for id, disease in enumerate(id2disease)}

def create_data_splits(input_dir, output_dir, train_ratio=0.7, val_ratio=0.10, test_ratio=0.20):
    os.makedirs(output_dir, exist_ok=True)
    
    for folder in os.listdir(input_dir):
        folder_path = os.path.join(input_dir, folder)
        if not os.path.isdir(folder_path):
            continue
        
        profiles = os.listdir(folder_path)
        train_profiles, temp_profiles = train_test_split(profiles, test_size=(1 - train_ratio), random_state=seed)
        val_profiles, test_profiles = train_test_split(temp_profiles, test_size=test_ratio / (val_ratio + test_ratio), random_state=seed)

        for split, profile_list in zip(["train", "val", "test"], [train_profiles, val_profiles, test_profiles]):
            split_folder = os.path.join(output_dir, split, folder)
            os.makedirs(split_folder, exist_ok=True)
            for profile in profile_list:
                src = os.path.join(folder_path, profile)
                dest = os.path.join(split_folder, profile)
                os.symlink(src, dest)  # Creates symbolic links to avoid duplicating data


#create_data_splits("/w/247/baileyng/mental_dataset", "/w/247/baileyng/mental_dataset_split")

create_data_splits("/w/247/abdulbasit", "/w/247/baileyng/mental_dataset_split_compressed")
