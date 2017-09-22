import numpy as np
from sklearn.decomposition import PCA
from sklearn import svm
import datetime
from sklearn.externals import joblib

print("Reading all_deep_representations_pca_merged.npy")
all_deep_representations = np.load("./all_deep_representations_pca_merged.npy")

# Generating label matrix
y_pos = np.ones(all_deep_representations.shape[0] / 2)
y_neg = np.zeros(all_deep_representations.shape[0] / 2)
y = np.concatenate((y_pos, y_neg))

print("{} Fitting SVM...".format(datetime.datetime.now()))
clf = svm.SVC()
clf.fit(all_deep_representations, y)

print("{} Fitted, calculating score...".format(datetime.datetime.now()))
score = clf.score(all_deep_representations, y)
print(score)

print("{} Saving...".format(datetime.datetime.now()))
joblib.dump(clf, 'svm.pkl') 
# clf = joblib.load('svm.pkl') 