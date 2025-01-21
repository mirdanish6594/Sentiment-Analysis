import pickle

# Define the XGBWrapper class (same as in api.py)
class XGBWrapper:
    # Class definition here
    pass

# Load the model
with open("Models/XGBWrapper.pkl", "rb") as f:
    model = pickle.load(f)

print("Model loaded successfully!")