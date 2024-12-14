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
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, accuracy_score,classification_report
from torch.nn import functional as F
from tqdm import tqdm
from sklearn.utils.class_weight import compute_class_weight
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, ExponentialLR
from warmup_scheduler import GradualWarmupScheduler

mode = "cosine"   #Choose between "None", "StepLR", "ReduceLROnPlateau", "ExponentialLR", "weight_decay3", "weight_decay4", "AdamW", "RMSprop", "cosine"


def mode_optimizer(mode):
    if mode == "weight_decay3":
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-3)

    elif mode == "weight_decay4":
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    
    elif mode == "AdamW":
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-2)

    elif mode == "RMSprop":
        optimizer = optim.RMSprop(model.parameters(), lr=learning_rate, weight_decay=1e-4, momentum=0.9)
    
    elif mode == "cosine":
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.95))
    
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    return optimizer

def mode_scheduler(mode):
    if mode == "StepLR":
        scheduler = StepLR(optimizer, step_size=10, gamma=0.1)  # Decays the LR by gamma every 10 epochs
    
    elif mode == "ReduceLROnPlateau":
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

    elif mode == "ExponentialLR":
        scheduler = ExponentialLR(optimizer, gamma=0.9)
    
    elif mode == "cosine":
        warmup_epoch = 5  # Number of warm-up epochs
        train_epoch = epochs  # Total training epochs
        lr_min = 1e-6  # Minimum learning rate after cosine annealing

        # Cosine Annealing Scheduler
        scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=(train_epoch - warmup_epoch),
            eta_min=lr_min
        )

        # Gradual Warm-up Scheduler
        scheduler = GradualWarmupScheduler(
            optimizer,
            multiplier=1,  # Final multiplier for learning rate
            total_epoch=warmup_epoch,
            after_scheduler=scheduler_cosine
        )

    else:
        scheduler = None

    return scheduler


    
# Constants and Mappings
id2disease = [
    "adhd",
    "anxiety",
    "bipolar",
    "depression",
    "mdd",
    "ocd",
    "ppd",
    "ptsd",
    "neg"
]
disease2id = {disease: id for id, disease in enumerate(id2disease)}

train_class_weight=[]
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

def read_top_k(path,col_names,k):
    df = pd.read_parquet(path)
    df['sum'] = df[col_names].sum(axis=1)
    # Sort by the new column and get the top k rows
    top_k_rows = df.nlargest(k, 'sum')["text"].tolist()
    return top_k_rows


def get_avg_metrics(all_labels, all_probs, threshold, disease=None, setting='binary', class_names=id2disease):
    predicted=np.argmax(all_probs,axis=1)
    print(f"Accuracy: {accuracy_score(all_labels,predicted)} \n")
    print(classification_report(all_labels,predicted,labels=list(disease2id.values()),target_names=id2disease))


def compute_cross_entropy(predictions, targets, class_weights=None, with_logits=True):
    """
    Computes the cross-entropy loss between predictions and targets.
    
    Args:
        predictions (torch.Tensor): Model outputs (logits or probabilities).
                                    Shape: [batch_size, num_classes] for multi-class,
                                    [batch_size] for binary classification.
        targets (torch.Tensor): Ground truth labels. Shape: [batch_size] (class indices or binary labels).
        class_weights (torch.Tensor, optional): Weights for each class. Shape: [num_classes].
        with_logits (bool, optional): If True, applies `F.cross_entropy` (logits input).
                                      If False, applies `F.nll_loss` (log-softmax input).
    
    Returns:
        torch.Tensor: Cross entropy loss (scalar).
    """
    class_weights=torch.tensor(class_weights,device=predictions.device,dtype=torch.float)
    if with_logits:
        # Use `F.cross_entropy` if predictions are raw logits [bs,9]->[] [bs]->1,2,3,0

        loss = F.cross_entropy(predictions, targets,weight=class_weights)
    else:
        # Use `F.nll_loss` if predictions are log-softmax probabilities
        loss = F.nll_loss(predictions, targets, weight=class_weights)
    
    return loss
    
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
        self.tweets_file_list=[]

        # Load data from directory structure
        for folder in os.listdir(self.input_dir):
            folder_path = os.path.join(self.input_dir, folder)
            if not os.path.isdir(folder_path):
                continue

            if folder in disease2id:
                label=disease2id[folder] 
            else:
                label=disease2id["neg"]

            # Inside HierDataset class
            for profile_folder in os.listdir(folder_path):
                profile_path = os.path.join(folder_path, profile_folder)
                tweets_file = os.path.join(profile_path, "tweets_preprocessed_2.parquet")
                self.tweets_file_list.append(tweets_file)
                self.labels.append(label)
        if(split=="train"):
            global train_class_weight
            train_class_weight=compute_class_weight(class_weight="balanced",classes=np.unique(self.labels),y=self.labels)
        self.is_control = np.array(self.is_control).astype(int)

    def __len__(self) -> int:
        return len(self.tweets_file_list)

    def __getitem__(self, index: int):
        tweets_file=self.tweets_file_list[index]
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
        if self.tokenizer!=None:
            top_k_posts=read_top_k(tweets_file,col_names,25)
            tokenized = self.tokenizer(top_k_posts, truncation=True, padding='max_length', max_length=self.max_len,return_tensors="pt")
            for k, v in tokenized.items():
                sample[k] = v

        if symp is not None:
            sample["symp"] = symp  # Add symptom features to the sample

        return sample, self.labels[index]

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
    labels = torch.tensor(labels,dtype=torch.int64)
    label_masks = torch.not_equal(labels, -1)
    return processed_batch, labels, label_masks

class PsyEx_wo_symp(nn.Module):
    def __init__(self, model_type, num_heads=2, num_trans_layers=1, max_posts=25, freeze=True, pool_type="first") -> None:
        super().__init__()
        self.model_type = model_type
        self.num_heads = num_heads
        self.num_trans_layers = num_trans_layers
        self.pool_type = pool_type
        self.post_encoder = AutoModel.from_pretrained(model_type).to(device)
        if freeze:
            for name, param in self.post_encoder.named_parameters():
                param.requires_grad = False
        self.hidden_dim = self.post_encoder.config.hidden_size
        self.max_posts = max_posts
        self.pos_emb = nn.Parameter(torch.Tensor(max_posts, self.hidden_dim))
        nn.init.xavier_uniform_(self.pos_emb)
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.hidden_dim, dim_feedforward=self.hidden_dim, nhead=num_heads, activation='gelu',batch_first=True)
        self.user_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_trans_layers)
        self.attn_ff = nn.Linear(self.hidden_dim, 1)
        self.dropout = nn.Dropout(self.post_encoder.config.hidden_dropout_prob)
        self.clf = nn.Linear(self.hidden_dim, len(disease2id))
    
    def forward(self, batch, **kwargs):
        # Extract and process tokenized tweets
        input_ids = torch.stack([user_feats["input_ids"] for user_feats in batch], dim=0)  # Shape: [batch_size, num_posts, seq_len]
        attention_mask = torch.stack([user_feats["attention_mask"] for user_feats in batch], dim=0)  # Shape: [batch_size, num_posts, seq_len]
        token_type_ids = torch.stack([user_feats["token_type_ids"] for user_feats in batch], dim=0)  # Shape: [batch_size, num_posts, seq_len]

        batch_size, num_posts, seq_len = input_ids.shape

        # Flatten posts for processing through the encoder
        flat_input_ids = input_ids.view(-1, seq_len)  # Shape: [batch_size * num_posts, seq_len]
        flat_attention_mask = attention_mask.view(-1, seq_len)  # Shape: [batch_size * num_posts, seq_len]
        flat_token_type_ids = token_type_ids.view(-1, seq_len)  # Shape: [batch_size * num_posts, seq_len]

        # Process through post encoder
        self.post_encoder.eval()
        post_outputs = self.post_encoder(flat_input_ids, flat_attention_mask ,flat_token_type_ids)
        last_hidden_state = post_outputs.last_hidden_state  # Shape: [batch_size * num_posts, seq_len, hidden_size]

        # Pooling (first or mean)
        if self.pool_type == "first":
            x = last_hidden_state[:, 0:1, :]  # Shape: [batch_size * num_posts, 1, hidden_size]
        elif self.pool_type == 'mean':
            x = mean_pooling(last_hidden_state, flat_attention_mask).unsqueeze(1)  # Shape: [batch_size * num_posts, 1, hidden_size]

        # Reshape back to [batch_size, num_posts, hidden_size]
        x = x.view(batch_size, num_posts, self.hidden_dim)

        # Add positional embeddings
        pos_emb = self.pos_emb[:num_posts, :].unsqueeze(0)  # Shape: [1, num_posts, hidden_size]
        x = x + pos_emb

        # Process through user encoder
        x = self.user_encoder(x)  # Shape: [batch_size, num_posts, hidden_size]

        # Compute attention scores for all diseases
        attn_scores = torch.softmax(self.attn_ff(x).squeeze(-1), dim=1)   # Shape: [batch_size, num_diseases, num_posts]

        # Compute weighted features for all diseases
        attn_embedding=torch.bmm(attn_scores.unsqueeze(1),x).squeeze(1)
        # Apply dropout

        # Compute logits for all diseases
        logits = self.clf(attn_embedding)

        return logits, attn_scores

def mean_pooling(token_embeddings, attention_mask):
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


# Training and Evaluation Code
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, threshold=0.5, disease='None', setting='binary'):
    best_f1 = 0.0

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        for batch in tqdm(train_loader):
            x, y, label_masks = batch
            x = [{k: v.to(device) for k, v in user_feats.items()} for user_feats in x]
            y = y.to(device)
            label_masks = label_masks.to(device)

            optimizer.zero_grad()
            y_hat, attn_scores = model(x)
            loss = criterion(y_hat, y,train_class_weight)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        
        if mode in ["StepLR", "ExponentialLR", "cosine"]:
            scheduler.step()
            
        avg_train_loss = total_loss / len(train_loader)

        directory = "/scratch/expires-2024-Dec-23/baileyng/models/test/"
        if not os.path.exists(directory):
            os.makedirs(directory)
        torch.save(model.state_dict(), f"{directory}model_{epoch+1}.pth")

        print(f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {avg_train_loss:.4f}")
        print(f"Current Learning Rate: {scheduler.get_last_lr()}")
        
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
                loss = criterion(y_hat, y, train_class_weight)
                total_val_loss += loss.item()

                probs = F.softmax(y_hat,dim=-1)
                if probs.dim() == 1:
                    probs = probs.unsqueeze(0)  # Ensure 2D
                all_labels.append(y.cpu().numpy())
                all_probs.append(probs.cpu().numpy())
                

        all_labels = np.concatenate(all_labels)
        all_probs = np.concatenate(all_probs)
        avg_val_loss = total_val_loss / len(val_loader)
        if mode == "ReduceLROnPlateau":
            scheduler.step(avg_val_loss)
        get_avg_metrics(all_labels, all_probs, threshold, disease, setting)
        print(f"Validation Loss: {avg_val_loss:.4f}")

def test_model(model, test_loader, criterion, device, threshold=0.5, disease='None', setting='binary'):
    model.eval()
    total_test_loss = 0.0  # Initialize the test loss variable
    all_labels = []
    all_probs = []
    with torch.no_grad():
        for batch in test_loader:  # Corrected from `val_loader` to `test_loader`
            x, y, label_masks = batch
            x = [{k: v.to(device) for k, v in user_feats.items()} for user_feats in x]
            y = y.to(device)
            label_masks = label_masks.to(device)

            y_hat, attn_scores = model(x)
            loss = criterion(y_hat, y, [train_class_weight[i] for i in range(len(id2disease))])
            total_test_loss += loss.item()  # Add loss to the total test loss

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
    input_dir = "/w/247/abdulbasit/mental_health_dataset_split" 
    batch_size = 32
    max_len = 280
    max_posts = 500
    epochs = 50
    learning_rate = 1e-4
    control_ratio = 0.5
    disease = None  # Use 'None' for all diseases or specify one like 'anxiety'
    setting = 'binary'
    use_symp = True  # Only use symptom features
    bal_sample = False  # Set to True if you want to use BalanceSampler
    threshold = 0.5  # Threshold for classification
    #model_type = "AIMH/mental-bert-base-cased"
    model_type="mental/mental-bert-base-uncased"
    #model_type="distilbert-base-uncased"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_type)

    # Instantiate datasets
    train_dataset = HierDataset(
        input_dir=input_dir,
        tokenizer=tokenizer,  # No tokenizer needed for symptom-only model
        max_len=max_len,
        split="train",
        disease=disease,
        setting=setting,
        use_symp=use_symp,
        max_posts=max_posts
    )

    val_dataset = HierDataset(
        input_dir=input_dir,
        tokenizer=tokenizer,  # No tokenizer needed for symptom-only model
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
        train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=my_collate_hier, shuffle=True, num_workers=2)

    val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=my_collate_hier, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=my_collate_hier, shuffle=False, num_workers=2)

   
    model=PsyEx_wo_symp(model_type).to(device=device)

    optimizer = mode_optimizer(mode)
    scheduler = mode_scheduler(mode)

    criterion = compute_cross_entropy
    # Print learnable parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Learnable Parameters: {total_params}")
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