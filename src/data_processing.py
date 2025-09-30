import pandas as pd
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(file_path: str, sheet_name: str = 'Raw Data') -> pd.DataFrame:
    try:
        data = pd.read_excel(file_path, sheet_name = sheet_name)
        original_rows = len(data)
        data = data.dropna()
        cleaned_rows = len(data)

        print(f"FILE '{file_path}' IS LOADED.")

        if original_rows > cleaned_rows:
            print(f"{original_rows - cleaned_rows} rows are dropped.")
        print(f"Data set includes {cleaned_rows} samples and {len(data.columns)} features.")

        return data

    except FileNotFoundError:
        print(f"{file_path} does not exist.")
        return None

def plot_data(df: pd.DataFrame, target_column: str = 'NSP'):
    
    print(f"Distribution of {target_column}")
    print(df[target_column].value_counts(normalize=True))

    key_features = ['LB', 'AC', 'FM']

    key_features = [f for f in key_features if f in df.columns]

    if key_features:
        plt.figure(figsize=(15, 10))
        for i, feature in enumerate(key_features):
            plt.subplot(2, 2, i+1)
            sns.histplot(data=df, x=feature, hue=target_column, kde=True, multiple="stack")
            plt.title(f"{feature}")
        plt.tight_layout()
        plt.show()

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    
    df_featured = df.copy()

    df_featured['deceleration_risk_score'] = df['DL'] * 1 + \
                                             df['DS'] * 2 + \
                                             df['DP'] * 3
    
    return df_featured

def encode_target(y: pd.Series):
    
    mapping = {
        'Normal': 1,
        'Suspect': 2,
        'Pathologic': 3
    }

    y_encoded = y.map(mapping)

    print(f"mapping: {mapping}")

    inverse_mapping = {v: k for k, v in mapping.items()}

    return y_encoded, inverse_mapping