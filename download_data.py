import urllib.request
import zipfile
import os
import shutil

os.makedirs('data', exist_ok=True)

url = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
zip_path = "data/ml-100k.zip"

print('Downloading MovieLens 100K dataset')
urllib.request.urlretrieve(url, zip_path)

print('Extracting files')
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall("data/")

extracted_file = "data/ml-100k/u.data"
target_file = "data/u.data"

if os.path.exists(extracted_file):
    shutil.move(extracted_file, target_file)
    print("Success! 'u.data' is now in your data/ folder.")
else:
    print('File already moved or not found.')

os.remove(zip_path)