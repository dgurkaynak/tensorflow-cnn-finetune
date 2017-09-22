import numpy as np
from sklearn.decomposition import PCA

print("Reading all_deep_representations.npy")
all_deep_representations = np.load("./all_deep_representations.npy")
print(all_deep_representations.shape)

print("Applying PCA...")
pca = PCA(n_components=100, whiten=True)
all_deep_representations_pca = pca.fit_transform(all_deep_representations)
print(all_deep_representations_pca.shape)

print("Saving to all_deep_representations_pca.npy")
np.save("all_deep_representations_pca.npy", all_deep_representations_pca)

