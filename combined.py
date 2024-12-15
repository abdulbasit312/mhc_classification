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
from glob import glob
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

def mean_pooling(token_embeddings, attention_mask):
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def get_avg_metrics(all_labels, all_probs):
    predicted=np.argmax(all_probs,axis=1)
    print(f"Accuracy: {accuracy_score(all_labels,predicted)} \n")
    print(classification_report(all_labels,predicted,labels=list(disease2id.values()),target_names=id2disease))

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
    

class HierDataset(Dataset):
    def __init__(self, input_dir, tokenizer, max_len, split="train", use_symp=True, max_posts=64):
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

# Model Definition
class SymptomStream(nn.Module):
    def __init__(self):
        super().__init__()
        
        #self.attn=nn.Linear(38,1)
        #self.activation=nn.Softmax(dim=-1)
        self.projection=nn.Linear(38,512)
        #self.projection2=nn.Linear(4096,2048)
        self.projection3=nn.Linear(512,256)
        self.relu1=nn.ReLU()
        self.relu2=nn.ReLU()
        # Transformer Encoder for Symptoms

        self.dropout = nn.Dropout(0.3)

    def forward(self, batch, **kwargs):
    # Extract symptom tensors from batch and stack into a single tensor
        symp_tensor = torch.stack([user_feats["symp"] for user_feats in batch], dim=0)  # Shape: [batch_size, max_posts, hidden_dim]
        #attn_w=self.activation(self.attn(symp_tensor).squeeze(-1))
        #x=torch.bmm(attn_w.unsqueeze(1),symp_tensor).squeeze(1)
        x=symp_tensor.mean(1)
        projection=self.dropout(self.relu1(self.projection(x)))
        #projection=self.dropout(self.gelu2(self.projection2(projection)))
        projection=self.dropout(self.relu2(self.projection3(projection)))


        return projection
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
        attn_scores = self.dropout(torch.softmax(self.attn_ff(x).squeeze(-1), dim=1))   # Shape: [batch_size, num_diseases, num_posts]

        # Compute weighted features for all diseases
        attn_embedding=torch.bmm(attn_scores.unsqueeze(1),x).squeeze(1)
        # Apply dropout

        # Compute logits for all diseases

        return attn_embedding
class Project_layer(nn.Module):
    def __init__(self,num_projection_layers,input_dim,project_dims,dropout_rate):
        super().__init__()
        self.projection1=nn.Linear(input_dim,project_dims)
        self.num_projection_layer=num_projection_layers
        self.projection_loop=nn.ModuleList()
        self.layer_norms=nn.ModuleList()
        for _ in range(num_projection_layers):
            t=nn.Sequential(nn.GELU(),
                          nn.Linear(project_dims,project_dims),
                          nn.Dropout(dropout_rate))
            self.projection_loop.append(t)
            self.layer_norms.append(nn.LayerNorm(project_dims))
    def forward(self,x):
        embed=self.projection1(x)
        for i in range(self.num_projection_layer):
            x=self.projection_loop[i](embed)
            x=x+embed
            embed=self.layer_norms[i](x)
        return embed
class Combine_results(nn.Module):
    def __init__(self,model_type, num_heads=4, num_trans_layers=1, max_posts=25, freeze=True, pool_type="first",use_projecton=False):
        super().__init__()
        self.left_stream=PsyEx_wo_symp(model_type, num_heads=num_heads, num_trans_layers=num_trans_layers, max_posts=max_posts, freeze=freeze, pool_type=pool_type)
        self.symptom_stream=SymptomStream()
        self.use_projection=use_projecton
        if use_projecton:
            self.projection_dim_left=512
            self.projection_dim_right=128
            self.hidden_size=self.projection_dim_left+self.projection_dim_right
            self.left_projection=Project_layer(1,768,self.projection_dim_left,0.4)
            self.right_projection=Project_layer(1,256,self.projection_dim_right,0.4)
        else:
            self.hidden_size=768+256
        self.clf=nn.Linear(self.hidden_size,len(id2disease))
    def forward(self,batch,**kwargs):
        left_embedding=self.left_stream(batch)
        right_stream=self.symptom_stream(batch)
        if self.use_projection:
            left_embedding=self.left_projection(left_embedding)
            right_stream=self.right_projection(right_stream)
        projection=torch.cat([left_embedding,right_stream],dim=-1)
        logits=self.clf(projection)
        return logits

def mean_pooling(token_embeddings, attention_mask):
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


# Training and Evaluation Code
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, save_path):
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
            y_hat= model(x)
            loss = criterion(y_hat, y,train_class_weight)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        
            
        avg_train_loss = total_loss / len(train_loader)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        torch.save(model.state_dict(),f"{save_path}/model_{epoch+1}.pth")
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

                y_hat = model(x)
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
        get_avg_metrics(all_labels, all_probs)
        print(f"Validation Loss: {avg_val_loss:.4f}")

def test_model(model, test_loader, criterion, device,ckpt):
    weights=sorted(glob(ckpt+"/*.pth"))
    print(weights)
    for weight in weights:
        print(weight)
        total_test_loss = 0.0  # Initialize the test loss variable
        all_labels = []
        all_probs = []
        model.load_state_dict(torch.load(weight,map_location=device,weights_only=True))
        model.eval()        
        with torch.no_grad():
            for batch in tqdm(test_loader):  # Corrected from `val_loader` to `test_loader`
                x, y, label_masks = batch
                x = [{k: v.to(device) for k, v in user_feats.items()} for user_feats in x]
                y = y.to(device)
                label_masks = label_masks.to(device)

                y_hat= model(x)
                loss = criterion(y_hat, y, [train_class_weight[i] for i in range(len(id2disease))])
                total_test_loss += loss.item()  # Add loss to the total test loss

                probs = F.softmax(y_hat,dim=-1)
                if probs.dim() == 1:
                    probs = probs.unsqueeze(0)  # Ensure 2D
                all_labels.append(y.cpu().numpy())
                all_probs.append(probs.cpu().numpy())
                    

            all_labels = np.concatenate(all_labels)
            all_probs = np.concatenate(all_probs)
            avg_val_loss = total_test_loss / len(test_loader)
            get_avg_metrics(all_labels, all_probs)
            print(f"Test Loss: {avg_val_loss:.4f}")

def argument_parser():
    parser=ArgumentParser(prog="Left stream",description="driver code to run and test")
    parser.add_argument('--mode',default="test",help="which mode you wish to run the program in train/test")
    parser.add_argument('--test_ckpt',default="",required=False,help="If in test mode provide the path to folder which contains model ckpt")
    return parser
# Main Execution

if __name__ == "__main__":
    # Hyperparameters and configurations
    args=argument_parser().parse_args()

    input_dir = "/w/247/abdulbasit/mental_health_dataset_split" 
    batch_size = 32
    max_len = 280
    max_posts = 500
    epochs = 50
    learning_rate = 1e-4
    use_symp = True  # Only use symptom features
    model_type="mental/mental-bert-base-uncased"
    save_path="/w/331/abdulbasit/test_run"  # change to run it in your machine
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_type)
    model=Combine_results(model_type=model_type,use_projecton=True).to(device=device)

    # Example of target with class indices
 
    # Example of target with class probabilities
    # Define optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,weight_decay=1e-4)
    criterion = compute_cross_entropy
    # Print learnable parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Learnable Parameters: {total_params}")
    train_dataset = HierDataset(
    input_dir=input_dir,
    tokenizer=tokenizer,  
    max_len=max_len,
    split="train",
    use_symp=use_symp,
    max_posts=max_posts
)
    if(args.mode=="train"):
    # Instantiate datasets


        val_dataset = HierDataset(
            input_dir=input_dir,
            tokenizer=tokenizer,  # No tokenizer needed for symptom-only model
            max_len=max_len,
            split="val",
            use_symp=use_symp,
            max_posts=max_posts
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=my_collate_hier, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=my_collate_hier, shuffle=False, num_workers=2)
            # Train the model
        train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            num_epochs=epochs,
            device=device,
            save_path=save_path
        )
    else:
        test_dataset = HierDataset(
            
            input_dir=input_dir,
            tokenizer=tokenizer,  # No tokenizer needed for symptom-only model
            max_len=max_len,
            split="test",
            use_symp=use_symp,
            max_posts=max_posts
        )
        test_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=my_collate_hier, shuffle=False, num_workers=2)

        # Instantiate data loaders



        # Test the model
        test_model(
            model=model,
            test_loader=test_loader,
            criterion=criterion,
            device=device,
            ckpt=args.test_ckpt if args.test_ckpt!='' else save_path
        )
