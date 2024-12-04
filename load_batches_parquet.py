import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import json
from tqdm import tqdm
from torch.utils.data import Sampler
import math
from transformers import AutoTokenizer

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

col_names = ['Do things easily get painful consequences', 'Worthlessness and guilty',
       'Diminished emotional expression', 'Drastical shift in mood and energy',
       'Avoidance of stimuli', 'Indecisiveness',
       'Decreased energy tiredness fatigue', 'Impulsivity',
       'Loss of interest or motivation', 'Fears of being negatively evaluated',
       'Intrusion symptoms', 'Anger Irritability', 'Flight of ideas',
       'Obsession', 'Inattention', 'Compulsions', 'Poor memory',
       'Catatonic behavior', 'Somatic symptoms others', 'Pessimism',
       'Anxious Mood', 'Fear about social situations', 'Respiratory symptoms',
       'More talkative', 'Panic fear', 'Weight and appetite change',
       'Suicidal ideas', 'Depressed Mood', 'Gastrointestinal symptoms',
       'Hyperactivity agitation', 'Somatic symptoms sensory',
       'Autonomic symptoms', 'Genitourinary symptoms', 'Sleep disturbance',
       'Compensatory behaviors to prevent weight gain',
       'Cardiovascular symptoms', 'Somatic muscle', 'Fear of gaining weight']


def read_scores(path, col_names):
    '''
    returns Nx38 matrix
    '''
    df = pd.read_parquet(path)
    matrix = df[col_names].to_numpy()
    return matrix

class BalanceSampler(Sampler):
    def __init__(self, data_source, control_ratio=0.75) -> None:
        self.data_source = data_source
        self.control_ratio = control_ratio
        self.indexes_control = np.where(data_source.is_control == 1)[0]
        self.indexes_diseases = []
        for idx, disease in enumerate(id2disease):
            disease_indexes = np.where(np.array(data_source.labels)[:, idx] == 1)[0]
            print(f"Disease indexes for {disease}: {disease_indexes}")
            self.indexes_diseases.append(disease_indexes)
        self.len_control = len(self.indexes_control)
        self.len_diseases = [len(disease_idx) for disease_idx in self.indexes_diseases]
        np.random.shuffle(self.indexes_control)
        for i in range(len(self.indexes_diseases)):
            np.random.shuffle(self.indexes_diseases[i])

        self.pointer_control = 0
        self.pointer_disease = [0] * len(id2disease)

    def __iter__(self):
        for i in range(len(self.data_source)):
            rand_num = np.random.rand()
            if rand_num < self.control_ratio:
                id0 = np.random.randint(self.pointer_control, self.len_control)
                sel_id = self.indexes_control[id0]
                self.indexes_control[id0], self.indexes_control[self.pointer_control] = self.indexes_control[self.pointer_control], self.indexes_control[id0]
                self.pointer_control += 1
                if self.pointer_control >= self.len_control:
                    self.pointer_control = 0
                    np.random.shuffle(self.indexes_control)
            else:
                chosen_disease = math.floor((rand_num - self.control_ratio) * 1.0 / ((1 - self.control_ratio) / len(id2disease)))
                id0 = np.random.randint(self.pointer_disease[chosen_disease], self.len_diseases[chosen_disease])
                sel_id = self.indexes_diseases[chosen_disease][id0]
                self.indexes_diseases[chosen_disease][id0] = self.indexes_diseases[chosen_disease][self.pointer_disease[chosen_disease]]
                self.indexes_diseases[chosen_disease][self.pointer_disease[chosen_disease]] = self.indexes_diseases[chosen_disease][id0]
                self.pointer_disease[chosen_disease] += 1
                if self.pointer_disease[chosen_disease] >= self.len_diseases[chosen_disease]:
                    self.pointer_disease[chosen_disease] = 0
                    np.random.shuffle(self.indexes_diseases[chosen_disease])

            yield sel_id

    def __len__(self) -> int:
        return len(self.data_source)

class HierDataset(Dataset):
    def __init__(self, input_dir, tokenizer, max_len, split="train", disease='None', setting='binary', use_symp=True, max_posts=64):
        assert split in {"train", "val", "test"}
        self.input_dir = os.path.join(input_dir, split)  # Use split folder
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.max_posts = max_posts
        self.use_symp = use_symp
        self.data = []
        self.labels = []
        self.is_control = []

        # Load data from directory structure
        for folder in os.listdir(self.input_dir):
            folder_path = os.path.join(self.input_dir, folder)
            if not os.path.isdir(folder_path):
                continue

            label = np.zeros(len(id2disease), dtype=int)
            if folder == "neg":
                self.is_control.append(1)
            else:
                self.is_control.append(0)
                if folder in disease2id:
                    label[disease2id[folder]] = 1
                else:
                    continue

            for profile_folder in os.listdir(folder_path):
                profile_path = os.path.join(folder_path, profile_folder)
                tweets_file = os.path.join(profile_path, "tweets_preprocessed.parquet")
                if not os.path.exists(tweets_file):
                    continue

                # Read preprocessed posts
                df = pd.read_parquet(tweets_file)
                posts = df["text"].tolist()[:max_posts]

                # Compute `symp` scores if `use_symp` is True
                symp = None
                if self.use_symp:
                    symp = read_scores(tweets_file, col_names)
                    if symp.shape[0] > max_posts:  # Truncate to max_posts
                        symp = symp[:max_posts, :]
                    elif symp.shape[0] < max_posts:  # Pad with zeros if less than max_posts
                        padding = np.zeros((max_posts - symp.shape[0], symp.shape[1]))
                        symp = np.vstack((symp, padding))

                # Tokenize posts
                sample = {}
                tokenized = tokenizer(posts, truncation=True, padding='max_length', max_length=max_len)
                for k, v in tokenized.items():
                    sample[k] = v

                if symp is not None:
                    sample["symp"] = symp  # Add symptom features to the sample

                self.data.append(sample)
                self.labels.append(label)

        self.is_control = np.array(self.is_control).astype(int)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int):
        return self.data[index], self.labels[index]

def my_collate_hier(data):
    labels = []
    processed_batch = []
    for item, label in data:
        user_feats = {}
        for k, v in item.items():
            if k != 'symp':
                user_feats[k] = torch.LongTensor(v)
            else:
                user_feats[k] = torch.FloatTensor(v)
        processed_batch.append(user_feats)
        labels.append(label)
    labels = torch.FloatTensor(np.array(labels))
    label_masks = torch.not_equal(labels, -1)
    return processed_batch, labels, label_masks

class HierDataModule(pl.LightningDataModule):
    def __init__(self, bs, input_dir, tokenizer, max_len, disease='None', setting='binary', use_symp=False, bal_sample=False, control_ratio=0.8):
        super().__init__()
        self.bs = bs
        self.input_dir = input_dir
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.disease = disease
        self.control_ratio = control_ratio
        self.bal_sample = bal_sample
        self.use_symp = use_symp
        self.setting = setting

    def setup(self, stage):
        if stage == "fit":
            self.train_set = HierDataset(self.input_dir, self.tokenizer, self.max_len, "train", self.disease, self.setting, self.use_symp)
            self.val_set = HierDataset(self.input_dir, self.tokenizer, self.max_len, "val", self.disease, self.setting, self.use_symp)
            self.test_set = HierDataset(self.input_dir, self.tokenizer, self.max_len, "test", self.disease, self.setting, self.use_symp)
        elif stage == "test":
            self.test_set = HierDataset(self.input_dir, self.tokenizer, self.max_len, "test", self.disease, self.setting, self.use_symp)

    def train_dataloader(self):
        if self.bal_sample:
            sampler = BalanceSampler(self.train_set, self.control_ratio)
            return DataLoader(self.train_set, batch_size=self.bs, collate_fn=my_collate_hier, sampler=sampler, pin_memory=False, num_workers=4)
        else:
            return DataLoader(self.train_set, batch_size=self.bs, collate_fn=my_collate_hier, shuffle=True, pin_memory=False, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.bs, collate_fn=my_collate_hier, pin_memory=False, num_workers=4)
    
    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.bs, collate_fn=my_collate_hier, pin_memory=False, num_workers=4)



# Configuration
input_dir = "/w/247/baileyng/mental_dataset_split_compressed"
max_len = 280
bs = 4
tokenizer = AutoTokenizer.from_pretrained("AIMH/mental-bert-base-cased")

# Initialize dataset and dataloader
dataset = HierDataset(
    input_dir=input_dir,
    tokenizer=tokenizer,
    max_len=max_len,
    split="train",
    use_symp=True,
    max_posts=64
)

dataloader = DataLoader(dataset, batch_size=bs, collate_fn=my_collate_hier)

# Load one batch
for batch_data, batch_labels, batch_masks in dataloader:
    print("Batch Data (Keys):", batch_data[0].keys())
    print("Input IDs Shape:", batch_data[0]["input_ids"].shape)
    print("Symp Tensor Shape:", batch_data[0]["symp"].shape)
    print("Batch Labels Shape:", batch_labels.shape)
    print("Label Masks Shape:", batch_masks.shape)
    break