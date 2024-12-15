import os
import time
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


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

class SymptomClassifier(nn.Module):
    def __init__(self):
        super(SymptomClassifier, self).__init__()
        self.linear = nn.Linear(38, 9)

    def forward(self, x):
        x = self.linear(x)
        return x

    def parameters(self):
        return list(self.linear.parameters())

    def state_dict(self):
        return {
            "linear.weight": self.linear.weight,
            "linear.bias": self.linear.bias
        }

    def load_state_dict(self, state_dict):
        self.linear.weight = state_dict["linear.weight"]
        self.linear.bias = state_dict["linear.bias"]

    def eval(self):
        self.linear.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
model.eval()
model.float()

classifier = SymptomClassifier().to(device)
criterion = nn.CrossEntropyLoss() 
optimizer = torch.optim.Adam(classifier.parameters(), lr=0.001)

base_path = "images"
losses = []
for epochs in range(10):
    for images_path in os.listdir(base_path):
        disease = images_path.split("_")[1]
        logits = torch.zeros((1, 38), device=device)
        images = []
        for image_path in os.listdir(os.path.join(base_path, images_path)):
            image = Image.open(os.path.join(base_path, images_path, image_path))
            image = image.convert('RGB')
            images.append(image)

        inputs = processor(text=symptoms, images=images, return_tensors="pt", padding=True).to(device)
        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image
        
        logits = torch.mean(logits_per_image, dim=0)
        logits.unsqueeze_(0)
        probs = classifier(logits)

        target = torch.zeros(1, 9, device=device)
        target[0, id2disease.index(disease)] = 1
        
        loss = criterion(probs, target)
        loss.backward()
        losses.append(loss.item())
        optimizer.step()
        optimizer.zero_grad()

torch.save(classifier.state_dict(), 'modelImageStream.pth')

plt.plot(losses)
plt.savefig('loss_image_stream.png')
plt.show()
