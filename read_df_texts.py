import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModel
from huggingface_hub import login
import torch
import time

path = '/w/247/baileyng/tweets_preprocessed.parquet'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
#model_name = "AIMH/mental-bert-large-cased"
model_name = "AIMH/mental-bert-base-cased"
tokenizer = AutoTokenizer.from_pretrained(model_name, token=True)
model = AutoModel.from_pretrained(model_name, token=True).to(device)

def read_texts(path, model, tokenizer, text_column='text'):
    start_time = time.time()
    df = pd.read_parquet(path)

    texts = df[text_column].tolist()
    embeddings = []
    
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze(0).numpy()  # [CLS] token embedding
        embeddings.append(cls_embedding)
    
    matrix = np.vstack(embeddings)
    print(time.time() - start_time)
    return matrix

matrix = read_texts(path, model, tokenizer)
print("Nx768 Matrix Shape:", matrix.shape)
