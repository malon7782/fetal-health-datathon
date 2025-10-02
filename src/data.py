import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

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

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    
    df_featured = df.copy()

    if hasattr(df_featured, 'ASTV') and hasattr(df_featured, 'Mean'):
        pass 
    return df_featured

def plot_feature_importance(model, feature_names):
    importances = model.feature_importances_
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values(by='importance', ascending=False)
    
    print(importance_df)
    
    plt.figure(figsize=(12, 10))
    sns.barplot(x='importance', y='feature', data=importance_df)
    plt.title('RANK', fontsize=16)
    plt.xlabel('score', fontsize=12)
    plt.ylabel('feature', fontsize=12)
    plt.tight_layout()
    
    output_dir = 'output'
    os.makedirs(output_dir, exist_ok=True)
    
    save_path = os.path.join(output_dir, 'feature_importance.png')
    

    plt.savefig(save_path)
    
    plt.close()
