from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
import pandas as pd
import numpy as np

seed = 20

df = pd.read_csv('/Users/JBremner/Desktop/trajFeatures.csv')

for column in df.columns:
	if 'Unnamed' in column:
		df.drop(column, axis=1, inplace=True)

modes = np.array(df['Mode of Transport'])

# Encoding modes of transport from here: bit.ly/2LdtVjV (see here also for inverse encoding)
# integer encode
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(modes)

features = list(df.drop(['Mode of Transport','Path','Label-state'], axis=1).columns)

# Input/output data
X = np.array(df.drop(['Mode of Transport','Path','Label-state'], axis=1))
Y = integer_encoded

clf = GradientBoostingClassifier(max_depth=2, random_state=0)
clf.fit(X, Y)

kfold = KFold(n_splits=10, shuffle=True, random_state=seed)

results = cross_val_score(clf, X, Y, cv=kfold)
print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
print(features)
print(clf.feature_importances_)