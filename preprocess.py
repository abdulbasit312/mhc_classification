import os
import numpy as np
import json
import time
import torch
from transformers import AutoTokenizer, AutoModel
import warnings
import pandas as pd
from mentalBertClassifier import mentalBertClassifier
# from huggingface_hub import login

# login()

id2disease = [
    "adhd",
    "anxiety",
    "bipolar",
    "depression",
    "mdd",
    "neg",
    "ocd",
    "ppd",
    "ptsd"
]

symptoms = [
    "Anxious Mood",
    "Autonomic symptoms",
    "Cardiovascular symptoms",
    "Catatonic behavior",
    "Decreased energy tiredness fatigue",
    "Depressed Mood",
    "Gastrointestinal symptoms",
    "Genitourinary symptoms",
    "Hyperactivity agitation",
    "Impulsivity",
    "Inattention",
    "Indecisiveness",
    "Respiratory symptoms",
    "Suicidal ideas",
    "Worthlessness and guilty",
    "Avoidance of stimuli",
    "Compensatory behaviors to prevent weight gain",
    "Compulsions",
    "Diminished emotional expression",
    "Do things easily get painful consequences",
    "Drastical shift in mood and energy",
    "Fear about social situations",
    "Fear of gaining weight",
    "Fears of being negatively evaluated",
    "Flight of ideas",
    "Intrusion symptoms",
    "Loss of interest or motivation",
    "More talkative",
    "Obsession",
    "Panic fear",
    "Pessimism",
    "Poor memory",
    "Sleep disturbance",
    "Somatic muscle",
    "Somatic symptoms others",
    "Somatic symptoms sensory",
    "Weight and appetite change",
    "Anger Irritability"
]


warnings.filterwarnings("ignore", category=UserWarning, module="torch.utils.data.dataloader")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_name = "bert-base-uncased"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
classifier = mentalBertClassifier(model_name, device)
state_dict_path = "mentalBertClassifier.pth"
classifier.load_state_dict(torch.load(state_dict_path, map_location=device))
classifier.eval()

topK = 16
batch_size = 32

base_path = "processed"
for split in ["train", "test", "val"]:
    for disease in id2disease:
        disease_dir = os.path.join(base_path, split, disease)
        for user_dir in os.listdir(disease_dir):
            start_time = time.time()
            user_path = os.path.join(disease_dir, user_dir)
            tweets_file = os.path.join(user_path, 'tweets.json')
            if os.path.isfile(tweets_file):
                with open(tweets_file, 'r') as file:
                    data = json.load(file)

                tweets = []
                for date, tweet_list in data.items():
                    for tweet in tweet_list:
                        tweets.append(tweet)

                df = pd.DataFrame(tweets)
                texts = df['text'].tolist()

                logits = classifier.classify(texts)
                
                max_allocated_memory = torch.cuda.max_memory_allocated(device)
                print(f"Processed {len(texts)} texts in {time.time() - start_time:.2f} seconds, max memory allocated: {max_allocated_memory / 1024 / 1024:.2f} MB")
                
                results = []
                for logit in logits:
                    result = {symptom: logit[i] for i, symptom in enumerate(symptoms)}
                    results.append(result)

                results_df = pd.DataFrame(results)
                df = pd.concat([df, results_df], axis=1)
                results_df['sum'] = results_df.sum(axis=1)
                df['isTopK'] = results_df['sum'].rank(method='first', ascending=False) <= topK

                output_path = os.path.join(user_path, 'tweets_preprocessed.parquet')
                df.to_parquet(output_path)
            
            exit()
                   
            
        


    