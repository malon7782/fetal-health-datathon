# @data
#	.features .targets: pandas.core.frame.DataFrame
# Returns: balanced_accuracy_score, f1_score

def data_clean(data):
	pass

def model_training(data):
	from sklearn.ensemble import RandomForestClassifier
	from sklearn.metrics import balanced_accuracy_score, f1_score

	data = data_clean(data)

	# access data
	# type(X) = <class 'pandas.core.frame.DataFrame'>

	X = data.features
	y = data.targets
	# train model e.g. sklearn.linear_model.LinearRegression().fit(X, y)

	y = y['NSP']

	X_training = X[0:len(X)//30]
	y_training = y[0:len(y)//30]
	X_test = X[len(X)//30:]
	y_test = y[len(y)//30:]

	y_pred = RandomForestClassifier().fit(X_training, y_training).predict(X_test)

	return balanced_accuracy_score(y_test, y_pred), f1_score(y_test, y_pred, average='macro')

	# access metadata
	print(heart_disease.metadata.uci_id)
	print(heart_disease.metadata.num_instances)
	print(heart_disease.metadata.additional_info.summary)

	# access variable info in tabular format
	print(heart_disease.variables)

from ucimlrepo import fetch_ucirepo

ctg = fetch_ucirepo(id=193)

print(model_training(ctg.data))