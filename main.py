#   .features .targets: pandas.core.frame.DataFrame
# Returns: balanced_accuracy_score, f1_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, balanced_accuracy_score, f1_score
from types import SimpleNamespace
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

    df_featured['ASTV_x_Mean'] = df['ASTV'] * df['Mean']
    
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
    

def model_training(data):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import balanced_accuracy_score, f1_score
    from lightgbm import LGBMClassifier
    from xgboost import XGBClassifier

    # access data
    # type(X) = <class 'pandas.core.frame.DataFrame'>

    X = data.features
    y = data.targets
    # train model e.g. sklearn.linear_model.LinearRegression().fit(X, y)

    y = y['NSP']
    """
    X_training = X[0:len(X)//5]
    y_training = y[0:len(y)//5]
    X_test = X[len(X)//5:]
    y_test = y[len(y)//5:]
    """
    X_training, X_test, y_training, y_test = train_test_split(
        X, y, 
        test_size=0.5, 
        random_state=42, 
        stratify=y  
    )

    scaler = StandardScaler()
    scaler.fit(X_training)
    X_train_scaled = scaler.transform(X_training)
    X_test_scaled = scaler.transform(X_test)

    model = RandomForestClassifier(random_state = 42, max_depth = 20, max_features = 'sqrt', min_samples_leaf = 1, n_estimators = 300)


    y_pred = model.fit(X_train_scaled, y_training).predict(X_test_scaled)


    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred)
    
    print(cm)





    return balanced_accuracy_score(y_test, y_pred), f1_score(y_test, y_pred, average='macro'), model

    # access metadata
    print(heart_disease.metadata.uci_id)
    print(heart_disease.metadata.num_instances)
    print(heart_disease.metadata.additional_info.summary)

    # access variable info in tabular format
    print(heart_disease.variables)






if __name__ == '__main__':
    
    
    DATA_FILE_PATH = 'data/CTG.xls'
    TARGET_COLUMN = 'NSP'
    REQUIRED_COLUMNS = ['ASTV', 'ALTV', 'MSTV', 'Mean', 'Mode', 'Median', 'Variance', 'AC', 'UC']
    EXPORT_FILE_PATH = 'output/loaded_data.csv'


    raw_df = load_data(DATA_FILE_PATH, sheet_name='Raw Data')

    if raw_df is not None:
        df_cleaned = raw_df[REQUIRED_COLUMNS + [TARGET_COLUMN]]


    #    df_cleaned[TARGET_COLUMN] = df_cleaned[TARGET_COLUMN].apply(lambda x: 1 if x == 1.0 else 2)


        os.makedirs(os.path.dirname(EXPORT_FILE_PATH), exist_ok=True)
    
        df_cleaned.to_csv(EXPORT_FILE_PATH, index=False)





        data_t = SimpleNamespace()
        features_df = df_cleaned.drop(columns=[TARGET_COLUMN], errors='ignore')
        data_t.features = create_features(features_df)
        data_t.targets = pd.DataFrame(df_cleaned[TARGET_COLUMN], columns=[TARGET_COLUMN])
        
        bal_acc, f1 , trained_model= model_training(data_t)
        
        plot_feature_importance(trained_model, data_t.features.columns)

        print(f"Balanced Accuracy: {bal_acc:.4f}")
        print(f"Macro F1 Score: {f1:.4f}")


