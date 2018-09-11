from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
import pandas as pd
import numpy as np

seed = 20

def gbClf(feature_csv, seed=seed):
	'''
	Takes path to csv containing the training data and prints the accuracy of a Gradient Boosting Classifier using 10-fold cross-validation

	'''
	
	df = pd.read_csv(feature_csv)
	df = df.loc[df['Label-state'] != 'Unlabelled'] 
	df.loc[df['Mode of Transport']=='taxi','Mode of Transport'] = 'car' # group taxis and cars

	for column in df.columns:
		if 'Unnamed' in column:
			df.drop(column, axis=1, inplace=True)

	modes = np.array(df['Mode of Transport'])

	# Encoding modes of transport from here: bit.ly/2LdtVjV (see here also for inverse encoding)
	# integer encode
	label_encoder = LabelEncoder()
	integer_encoded = label_encoder.fit_transform(modes)
	# label_encoder.inverse_transform()

	feature_drop = ['Mode of Transport','Path','Label-state', 'Point Count', 'Duration','Length', 'Turning-angle/Time','Hurst Exponent']

	features = list(df.drop(feature_drop, axis=1).columns)

	# Input/output data
	X = np.array(df.drop(feature_drop, axis=1))
	Y = integer_encoded

	clf = GradientBoostingClassifier(max_depth=2, random_state=0)
	clf.fit(X, Y)

	kfold = KFold(n_splits=10, shuffle=True, random_state=seed)

	results = cross_val_score(clf, X, Y, cv=kfold)
	print("Accuracy: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
	print(pd.DataFrame({'Feature Importance':clf.feature_importances_, 'Feature':features}).loc[:,('Feature','Feature Importance')])

	return