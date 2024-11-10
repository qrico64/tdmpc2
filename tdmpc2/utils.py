import os
import pickle
import sys
from tdmpc2 import TDMPC2

def save_model(model: TDMPC2, filepath: str):
    if os.path.exists(file_path):
        raise FileExistsError(f"The file '{file_path}' already exists.")
    else:
        with open(file_path, 'wb') as file:
            pickle.dump(obj, model, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Model saved to '{file_path}' successfully.")

def load_model(filepath: str) -> TDMPC2:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file '{filepath}' does not exist.")
    else:
        with open(file_path, 'r') as file:
            return pickle.load(file)
