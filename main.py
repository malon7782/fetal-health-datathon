# @data
#	.features .targets: pandas.core.frame.DataFrame
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




def model_training(data):
	from sklearn.ensemble import RandomForestClassifier
	from sklearn.metrics import balanced_accuracy_score, f1_score

	data = data_load(data)

	# access data
	# type(X) = <class 'pandas.core.frame.DataFrame'>

	X = data.features
	y = data.targets
	# train model e.g. sklearn.linear_model.LinearRegression().fit(X, y)

	y = y['NSP']

	X_training = X[0:len(X)//2]
	y_training = y[0:len(y)//2]
	X_test = X[len(X)//2:]
	y_test = y[len(y)//2:]

	y_pred = RandomForestClassifier().fit(X_training, y_training).predict(X_test)

	return balanced_accuracy_score(y_test, y_pred), f1_score(y_test, y_pred, average='macro')

	# access metadata
	print(heart_disease.metadata.uci_id)
	print(heart_disease.metadata.num_instances)
	print(heart_disease.metadata.additional_info.summary)

	# access variable info in tabular format
	print(heart_disease.variables)






if __name__ == '__main__':
    
    
    DATA_FILE_PATH = 'data/CTG.xls'
    TARGET_COLUMN = 'NSP'
    USELESS_COLUMNS = ['FileName', 'Date', 'SegFile', 'CLASS', 'b', 'e']

    raw_df = load_data(DATA_FILE_PATH, sheet_name='Raw Data')

    if raw_df is not None:
        df_cleaned = raw_df.drop(columns=USELESS_COLUMNS, errors='ignore')

        data_t = SimpleNamespace()
        data_t.features = df_cleaned.drop(columns=[TARGET_COLUMN], errors='ignore')
        data_t.targets = pd.DataFrame(df_cleaned[TARGET_COLUMN], columns=[TARGET_COLUMN])
        
        bal_acc, f1 = model_training(data_t)
            
        print(f"Balanced Accuracy: {bal_acc:.4f}")
        print(f"Macro F1 Score: {f1:.4f}")
            
