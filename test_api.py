import requests
import json

user_id = 10
url = f"http://127.0.0.1:8000/recommend/{user_id}"

print(f"Asking the AI for recommendations for User {user_id}...\n")

response = requests.get(url)

if response.status_code == 200:
    data = response.json()
    print(json.dumps(data, indent=4))
else:
    print(f"Failed to connect. Status code: {response.status_code}")