import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel
from dataset import DiseaseTweetsDataset
from model import PsyEx_wo_symp
def train(model, dataloader, criterion, optimizer, device, epochs=5):
    model.to(device)
    model.train()
    
    for epoch in range(epochs):
        total_loss = 0
        for i,batch in enumerate(dataloader):
            optimizer.zero_grad()
            
            
            # Forward pass
            predictions = model(batch)
            batch_labels=torch.tensor([batch[j]['label'] for j in range(len(batch))])
            batch_labels = batch_labels.to(device)
            
            # Compute loss
            loss = criterion(predictions[0].unsqueeze(0), batch_labels)
            total_loss += loss.item()
            
            # Backward pass
            loss.backward()
            optimizer.step()
        
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss:.4f}")

if __name__ == "__main__":
    # Dataset and DataLoader
    root_dir = "/w/331/abdulbasit/sampleFolder"
    model_type="mental/mental-bert-base-uncased"
    # Example of target with class indices
 
    # Example of target with class probabilities

    tokenizer = AutoTokenizer.from_pretrained(model_type)
    dataset = DiseaseTweetsDataset(root_dir,tokenizer)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=lambda x: x)
    
    # Model, loss, and optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PsyEx_wo_symp(model_type)
    criterion = nn.CrossEntropyLoss()  
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    # Train the model
    train(model, dataloader, criterion, optimizer, device, epochs=5)