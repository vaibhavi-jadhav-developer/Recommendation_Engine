import torch
import pandas as pd
from fastapi import FastAPI
from model import NCF

app = FastAPI()

num_users, num_items = 943, 1682
model = NCF(num_users, num_items)
model.load_state_dict(torch.load('ncf_model_weights.pth', weights_only=True))
model.eval()

item_cols = ['item_id', 'title'] + [f'feat_{i}' for i in range(22)]
movie_titles = pd.read_csv('data/ml-100k/u.item', sep='|', names=item_cols, encoding='latin-1')
movie_map = dict(zip(range(len(movie_titles)), movie_titles['title']))

@app.get("/")
def home():
    return {"message": "Deep Learning Recommendation Engine is Online"}

@app.get("/recommend/{user_id}")
def recommend(user_id: int):
    if user_id < 0 or user_id >= num_users:
        return {"error": "User ID out of range"}

    all_item_ids = torch.arange(num_items)
    user_tensor = torch.full((num_items,), user_id, dtype=torch.long)

    with torch.no_grad():
        predictions = model(user_tensor, all_item_ids).numpy()

    top_indices = predictions.argsort()[-5:][::-1]
    
    recommendations = []
    for idx in top_indices:
        recommendations.append({
            "movie_id": int(idx),
            "title": movie_map.get(idx, "Unknown Movie"),
            "score": float(predictions[idx])
        })

    return {
        "user_id": user_id,
        "recommendations": recommendations
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)