DEEP LEARNING MOVIE RECOMMENDATION ENGINE
OVERVIEW
This project is an end-to-end Deep Learning Recommendation Engine built using a Neural Collaborative Filtering (NCF) architecture. It predicts user preferences based on implicit feedback (watch history) and serves those predictions through a live FastAPI web server.

BUSINESS VALUE
This project demonstrates the complete lifecycle of an AI product:

Automated Data Ingestion: Scripts to safely fetch and extract benchmark datasets.

Implicit Feedback: Converting raw viewing data into PyTorch-optimized tensors.

Deep Learning: Building an NCF architecture combining Matrix Factorization and a Multi-Layer Perceptron.

Production Deployment: Serving model weights via a REST API for immediate front-end integration.

HOW TO RUN THE PROJECT LOCALLY

Extract the Files
Download the ZIP file and extract it to your local machine. Open your terminal or command prompt and navigate into the extracted folder.

Install Dependencies
Type the following command into your terminal and press Enter:
pip install pandas numpy torch scikit-learn fastapi uvicorn requests

Fetch the Dataset
Run the automated ingestion script to download the MovieLens dataset into the /data folder:
python download_data.py

Start the API Server
The model is pre-trained (weights included). You can immediately start the server:
python app.py

Test the Engine
Open a NEW terminal window and run the test script to see the deep learning recommendations in action:
python test_api.py

RE-TRAINING THE MODEL (OPTIONAL)
If you wish to view the data pipeline and re-train the model from scratch, run the scripts in your terminal in this exact order:

Step A: python data_processing.py
Step B: python split_data.py
Step C: python train.py
Step D: python evaluate.py

(Note: The evaluate.py script will output the Hit Ratio and NDCG ranking metrics to prove model accuracy).

