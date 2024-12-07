from mentalBertClassifier import mentalBertClassifier
import os
import numpy as np
import json
import time
import torch
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
import warnings
import pandas as pd
# from huggingface_hub import login

warnings.filterwarnings("ignore", message="Length of IterableDataset")

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "bert-base-uncased"
classifier = mentalBertClassifier(model_name, device)
state_dict_path = "mentalBertClassifier2.pth"
classifier.load_state_dict(torch.load(state_dict_path, map_location=device))
classifier.eval()

instructor_name = "valhalla/distilbart-mnli-12-1"
instructor = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
tokenizer_instructor = AutoTokenizer.from_pretrained(model_name)
classifier_pipeline = pipeline("zero-shot-classification", model=instructor, tokenizer=tokenizer_instructor, device=device)


batch_size = 2

base_path = "processed"
for split in ["test"]:
    for disease in id2disease:
        start_time = time.time()
        disease_dir = os.path.join(base_path, split, disease)
        i = 0
        for user_dir in os.listdir(disease_dir):
            user_path = os.path.join(disease_dir, user_dir)
            tweets_file = os.path.join(user_path, 'tweets.json')
            if os.path.isfile(tweets_file):
                with open(tweets_file, 'r') as file:
                    data = json.load(file)

                tweets = []
                for date, tweet_list in data.items():
                    for tweet in tweet_list:
                        if tweet['text'].strip(): # Avoids empty tweets
                            tweets.append(tweet['text'])

                results = []
                try:
                    classifications = classifier_pipeline(tweets[:2], candidate_labels=symptoms, multi_label=True, batch_size=batch_size)
                except ValueError as e:
                    print(f"Error: {e}")
                    print(f"Texts: {tweets[:2]}")
                    print(f"Labels: {symptoms}")
                
                for classification in classifications:
                    scores = {disease: score for disease, score in zip(classification['labels'], classification['scores'])}
                    results.append(scores)
                results = torch.tensor([list(result.values()) for result in results], requires_grad=False).to(device)

                logits = classifier.classify(tweets[:2])
                if(i%20==0):
                    print(logits[0]-results[0])
                    print(torch.nn.functional.mse_loss(logits, results))
                i+=1
                
        print("Max allocated memory:", torch.cuda.max_memory_allocated())
        print(f"Processed {disease} in {time.time() - start_time:.2f} seconds")