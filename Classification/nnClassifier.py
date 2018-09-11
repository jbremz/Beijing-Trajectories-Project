from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

seed = 20
np.random.seed(seed)

def nnClf(feature_csv, seed=seed):
	'''
	Takes path to csv containing the training data and prints the accuracy of an Neural-network Classifier using 10-fold cross-validation

	'''

	df = pd.read_csv('../Metadata/trajFeatures.csv')
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
	# binary encode
	onehot_encoder = OneHotEncoder(sparse=False)
	integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
	modes = onehot_encoder.fit_transform(integer_encoded)

	feature_drop = ['Mode of Transport','Path','Label-state', 'Point Count', 'Duration','Length', 'Turning-angle/Time','Hurst Exponent']

	features = list(df.drop(feature_drop, axis=1).columns)

	# Input/output data
	X = np.array(df.drop(feature_drop, axis=1))
	Y = modes

	# define baseline model - modified from http://bit.ly/2Lckt0g
	def baseline_model():
		# create model
		model = Sequential()
		model.add(Dense(8, input_dim=len(features), activation='relu'))
		model.add(Dense(8, input_dim=8, activation='relu'))
		model.add(Dense(8, input_dim=8, activation='relu'))
		model.add(Dense(len(np.unique(modes, axis=0)), activation='softmax'))
		# Compile model
		model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
		return model

	estimator = KerasClassifier(build_fn=baseline_model, epochs=200, batch_size=5, verbose=0)

	kfold = KFold(n_splits=5, shuffle=True, random_state=seed)

	results = cross_val_score(estimator, X, Y, cv=kfold)
	print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

	return

