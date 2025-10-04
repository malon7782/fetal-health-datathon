import os
import shap
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from joblib import dump

def model_training(data):
    X = data.features
    y = data.targets['NSP'].astype(int) - 1

    models = [
        RandomForestClassifier(
            max_depth=20,
            max_features='sqrt',
            min_samples_leaf=1,
            n_estimators=300,
            class_weight='balanced'
        ),
        XGBClassifier(
            objective="multi:softmax",
            num_class=3,
            eval_metric="mlogloss",
        )
    ]

    cv = StratifiedKFold(n_splits=5, shuffle=True)
    results = []
    oversample = SMOTE()

    for base_model in models:
        pipeline = Pipeline(steps=[
            ('scaler', StandardScaler()),
            ('oversample', oversample),
            ('classifier', base_model)
        ])

        nam = type(base_model).__name__

        bal_acc = cross_val_score(pipeline, X, y, cv=cv, scoring='balanced_accuracy', n_jobs=-1).mean()
        f1 = cross_val_score(pipeline, X, y, cv=cv, scoring='f1_macro', n_jobs=-1).mean()

        results.append((bal_acc, f1, nam))

        pipeline.fit(X, y)
        os.makedirs('models', exist_ok=True)
        dump(pipeline, f'models/{nam}_pipeline.joblib')

    best = max(results, key=lambda x: x[0])
    return best
