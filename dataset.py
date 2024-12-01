import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

disease2id={"adhd":0,
            "anxiety":1,
            "bipolar":2,
            "depression":3,
            "ocd":4,
            "ppd":5,
            "ptsd":6,
            "neg":7}  #neg is control

class DiseaseTweetsDataset(Dataset):
    def __init__(self, root_dir,tokeniser):
        """
        Args:
            root_dir (str): Path to the root directory containing disease folders.
        """
        self.root_dir = root_dir
        self.data=[]
        self.tokeniser=tokeniser
        self._load_data()

    def _load_data(self):
        """Loads the folder structure into a list of file paths."""
        for disease in disease2id:
            disease_path = os.path.join(self.root_dir, disease)
            if os.path.isdir(disease_path):
                for user in os.listdir(disease_path):
                    user_path = os.path.join(disease_path, user)
                    tweet_file = os.path.join(user_path, "tweets_preprocessed.parquet")
                    if os.path.exists(tweet_file):
                        self.data.append(tweet_file)

    def __len__(self):
        """Returns the number of users."""
        return len(self.data)

    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index of the user.

        Returns:
            dict: A dictionary with 'user' and 'tweets', where 'tweets' is a list of tweets for the user.
        """
        #posts = record['selected_posts'][:max_posts]
        #tokenized = tokenizer(posts, truncation=True, padding='max_length', max_length=max_len)
        tweet_file = self.data[idx]
        tweets = pd.read_parquet(tweet_file)["text"].tolist()  # Assuming 'text' column contains the tweets
        tokenized = self.tokeniser(tweets, truncation=True, padding='max_length',return_tensors='pt')

        path=os.path.dirname(tweet_file)
        user=path.split("/")[-1]
        label=disease2id[path.split("/")[-2]]
        return {"user": user, "tokenized_tweets": tokenized, "label":label}


if __name__ == "__main__":
    root_dir = "/w/331/abdulbasit/sampleFolder"
    tokenizer = AutoTokenizer.from_pretrained("mental/mental-bert-base-uncased")
    dataset = DiseaseTweetsDataset(root_dir,tokenizer)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=lambda x: x)

    for batch in dataloader:
        for user_data in batch:
            print(f"User: {user_data['user']}")
            print(f"Tweets: {user_data['tweets']}")