def model_training(data):
        from ucimlrepo import fetch_ucirepo, list_available_datasets
        import sklearn

        sklearn.show_versions()

        # check which datasets can be imported
        list_available_datasets()

        # import dataset
        heart_disease = fetch_ucirepo(id=193)
        # alternatively: fetch_ucirepo(name='Heart Disease')

        # access data
        X = heart_disease.data.features
        y = heart_disease.data.targets
        # train model e.g. sklearn.linear_model.LinearRegression().fit(X, y)

        y = y['NSP']

        sklearn.linear_model.LinearRegression().fit(X, y)

        y_pred = sklearn.ensemble.RandomForestClassifier().fit(X, y).predict(X)

        print("Balanced accuracy:", sklearn.metrics.balanced_accuracy_score(y, y_pred))
        print("F1 score", sklearn.metrics.f1_score(y, y_pred, average='macro'))