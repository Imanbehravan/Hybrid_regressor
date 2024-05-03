import pickle
import os

model_filename = 'Hybrid_Model.pkl'
result_path = "/home/karaubuntu/Iman_projects/"
model_path = os.path.join(result_path, model_filename)

with open(model_path, 'rb') as f:
        model = pickle.load(f)

print()