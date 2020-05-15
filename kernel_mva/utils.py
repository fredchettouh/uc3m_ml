import numpy as np

def get_train_test_idx( n_train=600, n_sample=1200, seed=0):
    np.random.seed(seed)
    orden = np.random.permutation(n_sample)
    idx_train = orden[:n_train]
    idx_test = orden[n_train:]
    return idx_train, idx_test



def get_train_test(X , targets, n_train=600, n_sample=1200, seed=0):
    np.random.seed(0)
    orden = np.random.permutation(n_sample)
    idx_train = orden[:n_train]
    idx_test = orden[n_train:]
    y_train = targets[0][idx_train]
    y_test = targets[0][idx_test]

    X_train = X[idx_train, :]
    X_test = X[idx_test, :]

    return X_train, X_test, y_train, y_test


def combine_kernel(k1, k2, alpha=0.5):
    return alpha*k1 +(1-alpha)*k2

def sorted_spectrum(A):
    complex_eig_val, complex_eig_vec = np.linalg.eig(A)
    eig_val = complex_eig_val.real
    orden = np.argsort(eig_val)[::-1]
    eig_val = eig_val[orden]
    eig_vec = complex_eig_vec.real
    eig_vec = eig_vec[:,orden]
    return eig_val, eig_vec