import numpy as np
from sklearn.decomposition import PCA
from sklearn import svm
import datetime
from sklearn.externals import joblib
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit

print("Reading all_deep_representations_pca_merged.npy")
all_deep_representations = np.load("./all_deep_representations_pca_merged.npy")

# Generating label matrix
y_pos = np.ones(all_deep_representations.shape[0] / 2)
y_neg = np.zeros(all_deep_representations.shape[0] / 2)
y = np.concatenate((y_pos, y_neg))

print("{} Fitting SVM...".format(datetime.datetime.now()))
clf = svm.SVC()

# First
# clf.fit(all_deep_representations, y)

# print("{} Fitted, calculating score...".format(datetime.datetime.now()))
# score = clf.score(all_deep_representations, y)
# print(score)

# print("{} Saving...".format(datetime.datetime.now()))
# joblib.dump(clf, 'svm.pkl') 
# clf = joblib.load('svm.pkl') 

# Second
# scores = cross_val_score(clf, all_deep_representations, y, cv=10)
# print("Accuracy: %0.4f (+/- %0.4f)" % (scores.mean(), scores.std() * 2))

# Third
cv = ShuffleSplit(n_splits=10, test_size=0.1, random_state=None)
scores = cross_val_score(clf, all_deep_representations, y, cv=cv)
print("Accuracy: %0.4f (+/- %0.4f)" % (scores.mean(), scores.std() * 2))
