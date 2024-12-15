import os
import time
import json
from PIL import Image
import requests
from io import BytesIO
import random

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

base_path = "../../abdulbasit"
for iterations in range(1000):
    disease = random.choice(id2disease)
    disease_dir = os.path.join(base_path, disease)

    user_dir = random.choice(os.listdir(disease_dir))
    user_path = os.path.join(disease_dir, user_dir)

    tweets_file = os.path.join(user_path, 'compressed.json')
    images_dir = os.path.join('images', f"{user_dir}_{disease}", )
    
    if os.path.isfile(tweets_file):
        with open(tweets_file, 'r') as file:
            data = json.load(file)
        
        os.makedirs(images_dir, exist_ok=True)
        images_left = 30
        for tweet in data:
            if tweet['media'] and tweet['media'][0]['type'] == 'image' and images_left > 0:
                image_url = tweet['media'][0]['url']
                try:
                    response = requests.get(image_url)
                    image = Image.open(BytesIO(response.content))
                    image = image.convert('RGB')
                    image_filename = os.path.join(images_dir, f"{tweet['tweet_id']}.jpg")
                    image.save(image_filename)
                    images_left -= 1
                except Exception as e:
                    # print(image_url)
                    pass
        if(images_left == 30):
            os.rmdir(images_dir)