import torch
import pandas as pd
import numpy as np
import math
from model import NCF

def evaluate_model():
    print("Loading data and model for evaluation...")
    train_df = pd.read_csv('data/train_data.csv')
    test_df = pd.read_csv('data/test_data.csv')
    
    positive_train = train_df[train_df['interaction'] == 1]
    user_history = positive_train.groupby('user_id')['item_id'].apply(set).to_dict()
    
    num_users = 943
    num_items = 1682
    
    model = NCF(num_users=num_users, num_items=num_items)
    model.load_state_dict(torch.load('ncf_model_weights.pth', weights_only=True))
    model.eval()
    
    hits = 0
    ndcg = 0
    test_users = test_df['user_id'].values
    test_items = test_df['item_id'].values
    
    print(f"Evaluating Top-10 Recommendations for {len(test_users)} users. Please wait...\n")
    
    with torch.no_grad(): 
        for user, true_item in zip(test_users, test_items):
            interacted_items = user_history.get(user, set())
            neg_items = []
            while len(neg_items) < 99:
                rand_item = np.random.randint(0, num_items)
                if rand_item not in interacted_items and rand_item != true_item:
                    neg_items.append(rand_item)
            
            items_to_test = [true_item] + neg_items
            users_to_test = [user] * 100
            
            user_tensor = torch.tensor(users_to_test, dtype=torch.long)
            item_tensor = torch.tensor(items_to_test, dtype=torch.long)
            
            predictions = model(user_tensor, item_tensor).numpy()
            
            ranked_indices = np.argsort(predictions)[::-1]
            rank = np.where(ranked_indices == 0)[0][0]
            
            if rank < 10:
                hits += 1
                ndcg += math.log(2) / math.log(rank + 2)
                
    hr_10 = hits / len(test_users)
    ndcg_10 = ndcg / len(test_users)
    
    print("--- FINAL BUSINESS METRICS ---")
    print(f"Hit Ratio (HR@10): {hr_10:.4f}")
    print(f"NDCG@10:           {ndcg_10:.4f}")
    print("------------------------------")

if __name__ == "__main__":
    evaluate_model()