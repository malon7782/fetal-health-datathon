from types import SimpleNamespace
from src.data import load_data, create_features
import os
import pandas as pd

if __name__ == "__main__":

    DATA_FILE_PATH = 'data/CTG.xls'
    TARGET_COLUMN = 'NSP'
    USEFUL_COLUMNS = ['LB', 'AC', 'FM', 'UC', 'DL', 'DS', 'DP', 'ASTV', 'MSTV', 'ALTV', 'MLTV', 'Width', 'Min', 'Max', 'Nmax', 'Nzeros', 'Mode', 'Mean', 'Median', 'Variance', 'Tendency']
        
    EXPORT_FILE_PATH = 'output/loaded_data.csv'


    raw_df = load_data(DATA_FILE_PATH, sheet_name='Raw Data')

    def run(REQUIRED_COLUMNS):
        from src.model import model_training

        df_cleaned = raw_df[REQUIRED_COLUMNS + [TARGET_COLUMN]]

        os.makedirs(os.path.dirname(EXPORT_FILE_PATH), exist_ok=True)

        df_cleaned.to_csv(EXPORT_FILE_PATH, index=False)

        data_t = SimpleNamespace()
        features_df = df_cleaned.drop(columns=[TARGET_COLUMN], errors='ignore')
        data_t.features = create_features(features_df)
        data_t.targets = pd.DataFrame(df_cleaned[TARGET_COLUMN], columns=[TARGET_COLUMN])
        
        bal_acc, f1, model = model_training(data_t)
        
        return bal_acc, f1, model

    BEST_COMBINAION = ['LB', 'AC', 'FM', 'UC', 'DS', 'DP', 'ASTV', 'MSTV', 'ALTV', 'Nzeros']

    bal_acc, f1, model = run(BEST_COMBINAION)

    print(f"Balanced Accuracy: {bal_acc:.4f}, F1 Score: {f1:.4f}, Model: {type(model).__name__}")
