import enum
from cv2 import log
import torch
import numpy as np
import pandas as pd
from torch import nn, optim
from transformers import AutoModel
from argparse import ArgumentParser
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, accuracy_score
from torch.nn import functional as F

id2disease=["adhd",
            "anxiety",
            "bipolar",
            "depression",
            "ocd",
            "ppd",
            "ptsd",
            "neg"]  #neg is control
class PsyEx_wo_symp(nn.Module):
    def __init__(self, model_type, num_heads=8, num_trans_layers=6, max_posts=32, freeze=False, pool_type="first") -> None:
        super().__init__()
        self.model_type = model_type
        self.num_heads = num_heads
        self.num_trans_layers = num_trans_layers
        self.pool_type = pool_type
        self.post_encoder = AutoModel.from_pretrained(model_type)
        if freeze:
            for name, param in self.post_encoder.named_parameters():
                param.requires_grad = False
        self.hidden_dim = self.post_encoder.config.hidden_size
        self.max_posts = max_posts
        self.pos_emb = nn.Parameter(torch.Tensor(max_posts, self.hidden_dim))
        nn.init.xavier_uniform_(self.pos_emb)
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.hidden_dim, dim_feedforward=self.hidden_dim, nhead=num_heads, activation='gelu')
        self.user_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_trans_layers)
        self.attn_ff = nn.ModuleList([nn.Linear(self.hidden_dim, 1) for disease in id2disease])
        self.dropout = nn.Dropout(self.post_encoder.config.hidden_dropout_prob)
        self.clf = nn.ModuleList([nn.Linear(self.hidden_dim, 1) for disease in id2disease])
    
    def forward(self, batch, **kwargs):
        feats = []
        attn_scores = []
        for user_feats in batch:
            post_outputs = self.post_encoder(user_feats["tokenized_tweets"]["input_ids"], user_feats["tokenized_tweets"]["attention_mask"], user_feats["tokenized_tweets"]["token_type_ids"])
            # [num_posts, seq_len, hidden_size] -> [num_posts, 1, hidden_size]
            if self.pool_type == "first":
                x = post_outputs.last_hidden_state[:, 0:1, :]
            elif self.pool_type == 'mean':
                x = mean_pooling(post_outputs.last_hidden_state, user_feats["tokenized_tweets"]["attention_mask"]).unsqueeze(1)
            # positional embedding for posts
            x = x + self.pos_emb[:x.shape[0], :].unsqueeze(1)
            x = self.user_encoder(x).squeeze(1) # [num_posts, hidden_size]
            attn_score = [torch.softmax(attn_ff(x).squeeze(), -1) for attn_ff in self.attn_ff]
            # weighted sum [hidden_size, ]
            feat = [self.dropout(score @ x) for score in attn_score]
            feats.append(feat)
            attn_scores.append(attn_score)

        logits = []
        for i in range(len(id2disease)):
            tmp = [feats[j][i] for j in range(len(feats))]
            logit = self.clf[i](torch.stack(tmp))
            logits.append(logit)
        logits = torch.stack(logits, dim=0).transpose(0, 1).squeeze()
        return logits, attn_scores
def mean_pooling(token_embeddings, attention_mask):
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)