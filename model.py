import torch
import torch.nn as nn
from torch.utils.data import Dataset
import pandas as pd

class MovieLensDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)

        self.users = torch.tensor(self.data['user_id'].values, dtype=torch.long)
        self.items = torch.tensor(self.data['item_id'].values, dtype=torch.long)
        self.labels = torch.tensor(self.data['interaction'].values, dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.users[idx], self.items[idx], self.labels[idx]


class NCF(nn.Module):
    def __init__(self, num_users, num_items, embed_dim=8, mlp_layers=[32, 16, 8]):
        super(NCF, self).__init__()
    
        self.user_embed_gmf = nn.Embedding(num_embeddings=num_users, embedding_dim=embed_dim)
        self.item_embed_gmf = nn.Embedding(num_embeddings=num_items, embedding_dim=embed_dim)
        
        self.user_embed_mlp = nn.Embedding(num_embeddings=num_users, embedding_dim=mlp_layers[0]//2)
        self.item_embed_mlp = nn.Embedding(num_embeddings=num_items, embedding_dim=mlp_layers[0]//2)
        
        self.mlp = nn.Sequential(
            nn.Linear(mlp_layers[0], mlp_layers[1]),
            nn.ReLU(),
            nn.Linear(mlp_layers[1], mlp_layers[2]),
            nn.ReLU()
        )
        
        self.output_layer = nn.Linear(embed_dim + mlp_layers[-1], 1)
        self.sigmoid = nn.Sigmoid() 

    def forward(self, user_input, item_input):
        user_gmf = self.user_embed_gmf(user_input)
        item_gmf = self.item_embed_gmf(item_input)
        gmf_vector = torch.mul(user_gmf, item_gmf) 
    
        user_mlp = self.user_embed_mlp(user_input)
        item_mlp = self.item_embed_mlp(item_input)
        mlp_vector = torch.cat([user_mlp, item_mlp], dim=-1) 
        mlp_vector = self.mlp(mlp_vector)
    
        combined_vector = torch.cat([gmf_vector, mlp_vector], dim=-1)
        
        prediction = self.output_layer(combined_vector)
        return self.sigmoid(prediction).squeeze()

if __name__ == "__main__":
    print('PyTorch Dataset and NCF Architecture defined successfully!')