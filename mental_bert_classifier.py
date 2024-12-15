from transformers import AutoTokenizer, AutoModel
import torch

class mentalBertClassifier:
    def __init__(self, model_name, device):
        self.model_name = model_name
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.model.eval()
        self.linear_classifier = torch.nn.Linear(768*100, 38).to(device)
        torch.nn.init.xavier_uniform_(self.linear_classifier.weight)
        self.sigmoid = torch.nn.Sigmoid()
        
    def parameters(self):
        return list(self.model.parameters()) + list(self.linear_classifier.parameters())
    
    def state_dict(self):
        return {
            "model": self.model.state_dict(),
            "linear_classifier": self.linear_classifier.state_dict()
        }
    
    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict["model"])
        self.linear_classifier.load_state_dict(state_dict["linear_classifier"])

    def eval(self):
        self.model.eval()
        self.linear_classifier.eval()

    def classify(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", padding='max_length', truncation=True, max_length=100).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state
            logits = self.linear_classifier(embeddings.view(-1, 768*100))
            probabilities = self.sigmoid(logits)
            return probabilities
        
if __name__ == "__main__":
    print("Testing mentalBertClassifier")
    model_name = "bert-base-uncased"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classifier = mentalBertClassifier(model_name, device)
    example_sentences = ["I am quite sad today", "I am feeling very happy today"]
    logits = classifier.classify(example_sentences)
    print(logits)