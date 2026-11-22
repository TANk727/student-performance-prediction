
import pandas as pd

def load_data(file_path):
    

    try:
        data = pd.read_csv(file_path, sep=";")
        return data

    except FileNotFoundError:
        print("Error: Dataset file not found.")
        return None
