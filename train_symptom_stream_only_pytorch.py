import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader, Sampler
import json
from tqdm import tqdm
import math
from transformers import AutoTokenizer
import enum
from cv2 import log
from torch import nn, optim
from transformers import AutoModel
from argparse import ArgumentParser
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, accuracy_score
from torch.nn import functional as F

# Constants and Mappings
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

# Helper Functions
def read_scores(path, col_names):
    '''
    returns Nx38 matrix
    '''
    df = pd.read_parquet(path)
    matrix = df[col_names].to_numpy()
    return matrix

def mean_pooling(token_embeddings, attention_mask):
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def get_avg_metrics(all_labels, all_probs, threshold, disease='None', setting='binary', class_names=id2disease):
    labels_by_class = []
    probs_by_class = []
    if disease != 'None':
        dis_id = id2disease.index(disease)
        if setting == 'binary':
            sel_indices = np.where(all_labels[:, dis_id] != -1)
            labels = all_labels[:, dis_id][sel_indices]
            probs = all_probs[:, dis_id][sel_indices]
        else:
            labels = all_labels[:, dis_id]
            probs = all_probs[:, dis_id]
        ret = {}
        preds = (probs > threshold).astype(float)
        ret["macro_acc"]=np.mean(labels == preds)
        ret["macro_p"]=precision_score(labels, preds)
        ret["macro_r"]=recall_score(labels, preds)
        ret["macro_f1"]=f1_score(labels, preds)
        try:
            ret["macro_auc"]=roc_auc_score(labels, probs)
        except:
            ret["macro_auc"]=0.5
    else:
        for i in range(all_labels.shape[1]):
            if setting == 'binary':
                sel_indices = np.where(all_labels[:, i] != -1)
                labels_by_class.append(all_labels[:, i][sel_indices])
                probs_by_class.append(all_probs[:, i][sel_indices])
            else:
                labels_by_class.append(all_labels[:, i])
                probs_by_class.append(all_probs[:, i])
        # macro avg metrics
        ret = {}
        for k in ["macro_acc", "macro_p", "macro_r", "macro_f1", "macro_auc"]:
            ret[k] = []
        for labels, probs in zip(labels_by_class, probs_by_class):
            preds = (probs > threshold).astype(float)
            ret["macro_acc"].append(np.mean(labels == preds))
            ret["macro_p"].append(precision_score(labels, preds))
            ret["macro_r"].append(recall_score(labels, preds))
            ret["macro_f1"].append(f1_score(labels, preds))
            try:
                ret["macro_auc"].append(roc_auc_score(labels, probs))
            except:
                ret["macro_auc"].append(0.5)
        for k in ["macro_acc", "macro_p", "macro_r", "macro_f1", "macro_auc"]:
            # list of diseases
            for class_name, v in zip(class_names, ret[k]):
                ret[class_name+"_"+k[6:]] = v
            ret[k] = np.mean(ret[k])

        if setting != 'binary':
            all_preds = (all_probs > threshold).astype(float)
            ret["micro_p"] = precision_score(all_labels.flatten(), all_preds.flatten())
            ret["micro_r"] = recall_score(all_labels.flatten(), all_preds.flatten())
            ret["micro_f1"] = f1_score(all_labels.flatten(), all_preds.flatten())
            ret["sample_acc"] = accuracy_score(all_labels, all_preds)

    return ret

def masked_logits_loss(logits, labels, masks=None):
    # Align shapes
    if logits.shape != labels.shape:
        logits = logits.view_as(labels)  # Reshape logits to match labels

    # Treat unlabeled samples (-1) as implicit negatives (0)
    labels2 = torch.clamp_min(labels, 0.0)
    losses = F.binary_cross_entropy_with_logits(logits, labels2, reduction='none')

    if masks is not None:
        masked_losses = torch.masked_select(losses, masks)
        return masked_losses.mean()
    else:
        return losses.mean()

# Custom Dataset and Sampler
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

            # Inside HierDataset class
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

                # Tokenize posts only if tokenizer is provided
                sample = {}
                if self.tokenizer:
                    tokenized = self.tokenizer(posts, truncation=True, padding='max_length', max_length=self.max_len)
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

# Model Definition
class SymptomStream(nn.Module):
    def __init__(self, num_heads=8, num_trans_layers=6, max_posts=64):
        super().__init__()
        self.max_posts = max_posts
        self.hidden_dim = 38  # Number of symptom features
        if self.hidden_dim % num_heads != 0:
            raise ValueError(f"`num_heads` must divide `hidden_dim` (38). Use `num_heads` as 1, 2, or 19.")

        # Transformer Encoder for Symptoms
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim, dim_feedforward=self.hidden_dim, nhead=num_heads, activation='gelu'
        )
        self.user_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_trans_layers)

        # Disease-Specific Attention Mechanisms
        self.attn_ff = nn.ModuleList([nn.Linear(self.hidden_dim, 1) for _ in id2disease])

        self.dropout = nn.Dropout(0.1)
        self.clf = nn.ModuleList([nn.Linear(self.hidden_dim, 1) for _ in id2disease])

    def forward(self, batch, **kwargs):
        feats = []
        attn_scores = []
        for user_feats in batch:
            # Symptom features
            x = user_feats["symp"].to(next(self.parameters()).device)  # Shape: [num_posts, hidden_dim]
            x = self.user_encoder(x)  # Apply Transformer

            # Compute disease-specific attention
            disease_attn_scores = []
            disease_feats = []
            for i, disease_attn in enumerate(self.attn_ff):
                attn_score = torch.softmax(disease_attn(x).squeeze(), -1)  # Disease-specific attention over posts
                feat = self.dropout(attn_score @ x)  # Weighted sum of features
                disease_attn_scores.append(attn_score)
                disease_feats.append(feat)

            feats.append(disease_feats)
            attn_scores.append(disease_attn_scores)

        logits = []
        for i in range(len(id2disease)):
            tmp = torch.stack([feats[j][i] for j in range(len(feats))])  # Stack features for each disease
            logit = self.clf[i](tmp)
            logits.append(logit)

        logits = torch.stack(logits, dim=0).transpose(0, 1).squeeze()
        return logits, attn_scores

# Training and Evaluation Code
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, threshold=0.5, disease='None', setting='binary'):
    best_f1 = 0.0

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            x, y, label_masks = batch
            x = [{k: v.to(device) for k, v in user_feats.items()} for user_feats in x]
            y = y.to(device)
            label_masks = label_masks.to(device)

            optimizer.zero_grad()
            y_hat, attn_scores = model(x)
            loss = criterion(y_hat, y, label_masks)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {avg_train_loss:.4f}")

        # Validation
        model.eval()
        total_val_loss = 0.0
        all_labels = []
        all_probs = []
        with torch.no_grad():
            for batch in val_loader:  # or test_loader
                x, y, label_masks = batch
                x = [{k: v.to(device) for k, v in user_feats.items()} for user_feats in x]
                y = y.to(device)
                label_masks = label_masks.to(device)

                y_hat, attn_scores = model(x)
                loss = criterion(y_hat, y, label_masks)
                total_val_loss += loss.item()

                probs = torch.sigmoid(y_hat)
                if probs.dim() == 1:
                    probs = probs.unsqueeze(0)  # Ensure 2D
                all_labels.append(y.cpu().numpy())
                all_probs.append(probs.cpu().numpy())

        all_labels = np.concatenate(all_labels)
        all_probs = np.concatenate(all_probs)
        avg_val_loss = total_val_loss / len(val_loader)
        ret = get_avg_metrics(all_labels, all_probs, threshold, disease, setting)
        print(f"Validation Loss: {avg_val_loss:.4f}")
        print(f"Validation Metrics: {ret}")

        if ret['macro_f1'] > best_f1:
            best_f1 = ret['macro_f1']
            # You can save the model here if needed
            # torch.save(model.state_dict(), 'best_model.pth')
        print(f"Best F1 so far: {best_f1:.4f}")

def test_model(model, test_loader, criterion, device, threshold=0.5, disease='None', setting='binary'):
    model.eval()
    total_test_loss = 0.0
    all_labels = []
    all_probs = []
    with torch.no_grad():
        for batch in val_loader:  # or test_loader
            x, y, label_masks = batch
            x = [{k: v.to(device) for k, v in user_feats.items()} for user_feats in x]
            y = y.to(device)
            label_masks = label_masks.to(device)

            y_hat, attn_scores = model(x)
            loss = criterion(y_hat, y, label_masks)
            total_val_loss += loss.item()

            probs = torch.sigmoid(y_hat)
            if probs.dim() == 1:
                probs = probs.unsqueeze(0)  # Ensure 2D
            all_labels.append(y.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

    all_labels = np.concatenate(all_labels)
    all_probs = np.concatenate(all_probs)

    avg_test_loss = total_test_loss / len(test_loader)
    ret = get_avg_metrics(all_labels, all_probs, threshold, disease, setting)
    print(f"Test Loss: {avg_test_loss:.4f}")
    print(f"Test Metrics: {ret}")

# Main Execution
if __name__ == "__main__":
    # Hyperparameters and configurations
    input_dir = "/w/247/baileyng/mental_dataset_split_compressed" 
    batch_size = 4
    max_len = 280
    max_posts = 64
    epochs = 5
    learning_rate = 1e-4
    control_ratio = 0.5
    disease = 'anxiety'  # Use 'None' for all diseases or specify one like 'anxiety'
    setting = 'binary'
    use_symp = True  # Only use symptom features
    bal_sample = False  # Set to True if you want to use BalanceSampler
    threshold = 0.5  # Threshold for classification

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Instantiate datasets
    train_dataset = HierDataset(
        input_dir=input_dir,
        tokenizer=None,  # No tokenizer needed for symptom-only model
        max_len=max_len,
        split="train",
        disease=disease,
        setting=setting,
        use_symp=use_symp,
        max_posts=max_posts
    )

    val_dataset = HierDataset(
        input_dir=input_dir,
        tokenizer=None,  # No tokenizer needed for symptom-only model
        max_len=max_len,
        split="val",
        disease=disease,
        setting=setting,
        use_symp=use_symp,
        max_posts=max_posts
    )

    test_dataset = HierDataset(
        input_dir=input_dir,
        tokenizer=None,  # No tokenizer needed for symptom-only model
        max_len=max_len,
        split="test",
        disease=disease,
        setting=setting,
        use_symp=use_symp,
        max_posts=max_posts
    )

    # Instantiate data loaders
    if bal_sample:
        sampler = BalanceSampler(train_dataset, control_ratio)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=my_collate_hier, sampler=sampler, num_workers=0)
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=my_collate_hier, shuffle=True, num_workers=0)

    val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=my_collate_hier, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=my_collate_hier, shuffle=False, num_workers=0)

    # Instantiate the model
    model = SymptomStream(
        num_heads=2,  # Use 1, 2, or 19 for symptom stream
        num_trans_layers=6,
        max_posts=max_posts
    ).to(device)

    # Define optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = masked_logits_loss

    # Train the model
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=epochs,
        device=device,
        threshold=threshold,
        disease=disease,
        setting=setting
    )

    # Test the model
    test_model(
        model=model,
        test_loader=test_loader,
        criterion=criterion,
        device=device,
        threshold=threshold,
        disease=disease,
        setting=setting
    )
