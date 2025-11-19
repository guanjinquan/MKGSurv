""" File adapted from https://github.com/mahmoodlab/MMP/blob/main/src/utils/proto_utils.py """
import torch
from tqdm import tqdm
import numpy as np

from sklearn.cluster import KMeans
import faiss

def kmeans_clustering(patches: np.ndarray, n_proto: int, n_iter: int, n_init: int):
    """ 
    Find cluster centers of the data (Prototypes) using KMeans (cpu).
    Assumes 'patches' is a np.ndarray.
    """
    print("\nUsing Kmeans for clustering...")
    print(f"\n\tNum of clusters {n_proto}, num of iter {n_iter}, n_init {n_init}")

    # Cluster the data
    kmeans = KMeans(
        n_clusters=n_proto, 
        max_iter=n_iter, 
        n_init=n_init,       
        random_state=42    
    )
    
    kmeans.fit(patches)

    # Get prototypes (shape (n_proto, feature_dim))
    weight = kmeans.cluster_centers_
    return weight

def faiss_clustering(patches: np.ndarray, n_proto: int, n_iter: int, n_init: int, n_proto_patches: int):
    """ 
    Find cluster centers of the data (Prototypes) using Faiss (gpu).
    Assumes 'patches' is a np.ndarray.
    """


    print(f"\tNum of clusters {n_proto}, num of iter {n_iter}")

    # Faiss Kmeans
    kmeans = faiss.Kmeans(patches.shape[1], 
                            n_proto, 
                            niter=n_iter, 
                            nredo=n_init,
                            verbose=True, 
                            max_points_per_centroid=n_proto_patches,
                            gpu=1)
    
    kmeans.train(patches)

    # Get prototypes (shape (n_proto, feature_dim))
    weight = kmeans.centroids
    return weight



def cluster(patches, n_proto, n_iter=50, n_init=3, mode='faiss', n_proto_patches=100000):
    """ 
    Cluster the patch features and save the cluster centers as prototypes. 

    Args:
        patches (np.ndarray | torch.Tensor): 
            Patch features to cluster. 
            np.ndarray is preferred. If torch.Tensor is provided,
            it will be converted (with a warning).
    Returns:
        weight (np.ndarray): Prototypes (cluster centers) with shape (n_proto, feature_dim)
    """
    
    if not isinstance(patches, np.ndarray):
        if torch.is_tensor(patches):
            print("Warning: 'patches' input is a torch.Tensor. Converting to np.ndarray.")
            print("         For optimal performance, please pass a np.ndarray directly.")
            patches = patches.cpu().numpy()
        else:
            # If it's something else (like a list), raise an error
            raise TypeError(f"Input 'patches' must be a np.ndarray (or torch.Tensor), but got {type(patches)}")

    # Faiss (and scikit-learn) often need C-contiguous arrays.
    # .numpy() on a Tensor usually is, but direct array slicing might not be.
    if not patches.flags['C_CONTIGUOUS']:
            print("Warning: 'patches' array is not C-contiguous. Making a copy...")
            # .astype(np.float32) is crucial for Faiss
            patches = np.ascontiguousarray(patches, dtype=np.float32)
    
    # Faiss requires float32
    if mode == 'faiss' and patches.dtype != np.float32:
        print(f"Warning: Faiss requires np.float32, but got {patches.dtype}. Converting...")
        patches = patches.astype(np.float32)
    # --- 检查结束 ---

    n_patches = len(patches)
    print(f"\nTotal of {n_patches} patches picked for clustering.")

    # Find cluster centers according to clustering mode
    if mode == 'kmeans':
        weight = kmeans_clustering(patches, n_proto, n_iter, n_init) 
    elif mode == 'faiss':
        assert torch.cuda.is_available(), f"FAISS requires access to GPU. Please enable use_cuda"
        weight = faiss_clustering(patches, n_proto, n_iter, n_init, n_proto_patches)
    else:
        raise NotImplementedError(f"Clustering not implemented for {mode}!")
    
    # Return prototypes (cluster centers)
    return weight