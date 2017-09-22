import numpy as np
from sklearn.decomposition import PCA

print("Reading all_deep_representations_pca.npy")
all_deep_representations = np.load("./all_deep_representations_pca.npy")
print(all_deep_representations.shape)

print("Merging pairs...")
all_deep_representations_merged = np.reshape(all_deep_representations, (all_deep_representations.shape[0] / 2, all_deep_representations.shape[1] * 2))
print(all_deep_representations_merged.shape)

print("Saving to all_deep_representations_pca_merged.npy")
np.save("all_deep_representations_pca_merged.npy", all_deep_representations_merged)

