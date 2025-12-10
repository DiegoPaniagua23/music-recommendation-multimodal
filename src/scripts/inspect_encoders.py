import joblib
import os
import sys

path = "checkpoints/prueba_10porciento/encoders.pkl"
if not os.path.exists(path):
    print(f"File not found: {path}")
    sys.exit(1)

encoders = joblib.load(path)
print("Keys:", encoders.keys())

if 'genre_encoder' in encoders:
    cats = encoders['genre_encoder'].categories_[0]
    print(f"Genre Encoder Categories: {len(cats)}")
    # print(cats)

if 'scaler' in encoders:
    print("Scaler mean shape:", encoders['scaler'].mean_.shape)
