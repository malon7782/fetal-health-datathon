from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder

def model_training(data):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import balanced_accuracy_score, f1_score
    from lightgbm import LGBMClassifier
    from xgboost import XGBClassifier

    X = data.features
    y = data.targets['NSP'].astype(int) - 1

    scaler = StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X)

    from sklearn.model_selection import GridSearchCV

    models = [
    	RandomForestClassifier(random_state = 42, max_depth = 20, max_features = 'sqrt', min_samples_leaf = 1, n_estimators = 300, class_weight='balanced'),
        XGBClassifier( objective="multi:softmax", num_class=3, eval_metric="mlogloss", random_state=42)
    ]

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    results = []

    for model in models:
        bal_acc = cross_val_score(model, X, y, cv=cv, scoring='balanced_accuracy', n_jobs=-1)
        bal_acc = bal_acc.mean()
        f1 = cross_val_score(model, X, y, cv=cv, scoring='f1_macro', n_jobs=-1)
        f1 = f1.mean()
        results.append((bal_acc, f1, model))

    return max(results, key=lambda x: x[0]) if results else (None, 0, 0, None)