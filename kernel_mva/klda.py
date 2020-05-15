import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import rbf_kernel, polynomial_kernel,linear_kernel



class centred_kernel(object):
    def __init__(self, kernel, centred=True, gamma=None, degree=3, coef0=1):
        self._kernel_type = kernel
        self._gamma = gamma
        self._degree = degree
        self._coef0 = coef0
        self._centred = centred

    def fit(self, X):
        self._X = X.copy()
        return self

    def fit_transform(self, X):
        self._X = X.copy()
        self._n = X.shape[0]
        if self._kernel_type == 'rbf':
            K = rbf_kernel(X, gamma=self._gamma)
        elif self._kernel_type == 'poly':
            K = polynomial_kernel(X, degree=self._degree, coef0=self._coef0)
        elif self._kernel_type == 'linear':
            K = linear_kernel(X)
        if self._centred == True:

            """
            YOUR CODE

            """
            sumK = np.sum(K, 0)
            K1 = 1. / self._n * np.tile(np.reshape(sumK, (-1, 1)), (1, self._n))
            K2 = 1. / self._n * np.tile(np.reshape(sumK, (1, -1)), (self._n, 1))
            self._K = K.copy()
            Ko = K - K1 - K2 + np.mean(K)
        else:
            Ko = K
        return Ko

    def transform(self, X):
        nt = X.shape[0]
        if self._kernel_type == 'rbf':
            K = rbf_kernel(X, self._X, gamma=self._gamma)
        elif self._kernel_type == 'poly':
            K = polynomial_kernel(X, self._X, degree=self._degree,
                                  coef0=self._coef0)
        elif self._kernel_type == 'linear':
            K = linear_kernel(X, self._X)
        if self._centred == True:

            """
            YOUR CODE
            """
            K1 = (K - 1. / self._n * np.ones((nt, self._n)).dot(self._K))
            K2 = np.eye(self._n) - 1. / self._n * np.ones((self._n, self._n))
            Ko = K1.dot(K2)
        else:
            Ko = K
        return Ko



def sorted_spectrum(A):
    complex_eig_val, complex_eig_vec = np.linalg.eig(A)
    eig_val = complex_eig_val.real
    orden = np.argsort(eig_val)[::-1]
    eig_val = eig_val[orden]
    eig_vec = complex_eig_vec.real
    eig_vec = eig_vec[:,orden]
    return eig_val, eig_vec

def kgda(K, y, tau = 1e-6):
    # K already centred!!
    n = K.shape[0]
    v_classes = np.unique(y)
    M_ = np.mean(K,1)
    P = len(v_classes)
    M_Mp = np.empty((n,P))
    #Sb
    Sb = np.zeros((n,n))
    Sw = np.zeros((n,n))
    for p in range(P):
        idx_class_p = np.where(y==v_classes[p])[0]
        n_p = len(idx_class_p)
        Kp = K[:,idx_class_p]
        Mp = np.mean(Kp,1)
        M_Mp[:,p] = Mp.copy()
        MpM_ = Mp - M_
        Sb += n_p* np.outer(MpM_, MpM_.T) # column * row
        Sw += 1./n_p* Kp.dot(Kp.T) - np.outer(Mp, Mp.T)
    #Sw inv
    if np.linalg.matrix_rank(Sw) < n:
        Sw += tau*np.eye(n)
    iSw = np.linalg.inv(Sw)
    AA = iSw.dot(Sb)
    DD2, UU2 = sorted_spectrum(AA)
    lam = DD2[:P-1]
    A = UU2[:,:P-1]
    VMp = A.T.dot(M_Mp)
    return A, VMp.T

def predict_kgda(K_test, A, Q, v_classes=None):
    # Ktest centred!!
    # Q projection of class means!!
    if v_classes is None:
        P = Q.shape[0]
        v_classes = np.array([int(cc) for cc in range(P)])
    U = K_test.dot(A)
    Distance_sample_mean = pairwise_distances(U, Q)
    closest_mean = np.argmin(Distance_sample_mean,1)
    output = np.array([v_classes[ii] for ii in closest_mean])
    return output, U