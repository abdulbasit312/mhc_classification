import os
import random
import pandas as pd
import requests
import time
import json

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

verbose2ShortDisease = {
    "Attention Deficit Hyperactivity Disorder": "adhd",
    "Anxiety": "anxiety",
    "Bipolar Disorder": "bipolar",
    "Depression": "depression",
    "Major Depressive Disorder": "mdd",
    "Negative (Control group)": "neg",
    "Obsessive Compulsive Disorder": "ocd",
    "Postpartum Depression": "ppd",
    "Post-Traumatic Stress Disorder": "ptsd"
}

base_path = "../../abdulbasit/mental_health_dataset_split/test"

url = "https://api.arliai.com/v1/chat/completions"
ARLIAI_API_KEY = "" # Replace with your own API key from arliai

model = "Llama-3.3+3.1-70B-Nautilus-v0.1"
    

users = []
pred = []
true = []

for i in range(300):
    disease = random.choice(id2disease)
    disease_dir = os.path.join(base_path, disease)

    user_dir = random.choice(os.listdir(disease_dir))
    user_path = os.path.join(disease_dir, user_dir)

    file_path = os.path.join(user_path, "tweets_preprocessed_2.parquet")
    df = pd.read_parquet(file_path)

    df_filtered = df[df['isTop16'] == True]
    text_concatenated = ' '.join(df_filtered['text'].tolist()) # Simple concatenation for now

    system = """You are a psychiatrist specialized in mental health conditions. 
    Your task is to diagnose mental health conditions of an individual based on their tweets.
    Answer using one of the following results, and don't say anything else: 
    Attention Deficit Hyperactivity Disorder, Anxiety, Bipolar Disorder, Depression, Major Depressive Disorder, Negative (Control group), Obsessive Compulsive Disorder, Postpartum Depression, Post-Traumatic Stress Disorder."""

    prompt = f"Here are 16 tweets: {text_concatenated} What mental health condition does this individual have? Do not give an explanation, on answer with one of the following options: Attention Deficit Hyperactivity Disorder, Anxiety, Bipolar Disorder, Depression, Major Depressive Disorder, Negative (Control group), Obsessive Compulsive Disorder, Postpartum Depression, Post-Traumatic Stress Disorder"

    payload = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt}
        ],
        })
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f"Bearer {ARLIAI_API_KEY}"
    }

    response = requests.request("POST", url, headers=headers, data=payload)
    if response.status_code == 200:
        print("User: ", user_dir)
        users.append(user_dir)
        print(f"Correct disease: {disease}")
        true.append(disease)
        data = response.json()
        pred.append(data["choices"][0]["message"]["content"])
        print(data["choices"][0]["message"]["content"])
        print("-----------------------------------")
    else:
        print("Error: ", response.text)

    if(i % 100 == 0 and i != 0): # save every 100 prompts
        df = pd.DataFrame({
            "user": users,
            "true": true,
            "pred": pred
        })

        df.to_csv(f"arliai_predictions{i}.csv", index=False)
        