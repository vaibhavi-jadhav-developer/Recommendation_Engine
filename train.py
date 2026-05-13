import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from model import MovieLensDataset, NCF

def train_model():
    print('Loading data')
    train_dataset = MovieLensDataset('data/train_data.csv')
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)

    num_users = 943
    num_items = 1682
    
    model = NCF(num_users=num_users, num_items=num_items)
    print('Model initialized. starting training...\n')

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 5
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for users, items, labels in train_loader:
            optimizer.zero_grad()
            
            predictions = model(users, items)
            
            loss = criterion(predictions, labels)
            
            loss.backward()
            
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs} | Average Loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), 'ncf_model_weights.pth')
    print("\nSuccess! Training complete and model weights saved to 'ncf_model_weights.pth'")

if __name__ == "__main__":
    train_model()
    