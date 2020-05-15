from sklearn.base import BaseEstimator, TransformerMixin
from scipy.linalg import eig
import numpy as np
from utils import sorted_spectrum
from klda import centred_kernel


def normalize(x):
    x_mod = np.linalg.norm(x, 2, axis=0)
    return x / np.tile(x_mod.reshape(1, -1), (x.shape[0], 1))


def _core(eta, K):
    n = K.shape[0]
    thres = eta * np.trace(K)

    nu = np.zeros(n)
    I = np.zeros(n, dtype=int)
    R = np.zeros((n, n))
    d = np.diag(K).copy()

    for jj in range(n):
        if np.sum(d) < thres:
            # print("hola, PGSO limit")
            # print("n= {0:d}, jj={1:d}".format(int(n),jj))
            jj -= 1
            break
        I[jj] = np.argmax(d)
        a = d[I[jj]]
        nu[jj] = np.sqrt(a)
        for ii in range(n):
            R[ii, jj] = (K[ii, I[jj]] - R[ii, :jj].dot(R[I[jj], :jj])) / nu[jj]
            d[ii] = d[ii] - (R[ii, jj] ** 2)
    s_T = jj + 1
    s_R = R[:, :s_T]
    s_nu = nu[:s_T]
    s_I = I[:s_T]
    return s_T, s_R, s_nu, s_I


class PGSO(object):
    def __init__(self, eta=1e-3):
        self._eta = eta

    def fit(self, K):
        self._T, self._R, self._nu, self._I = _core(self._eta, K)
        return self

    def fit_transform(self, K):
        self._T, self._R, self._nu, self._I = _core(self._eta, K)
        return self._R.copy()

    def transform(self, K):
        nt = K.shape[0]
        R = np.zeros((nt, self._T))
        for ii in range(nt):
            for jj in range(self._T):
                R[ii, jj] = (K[ii, self._I[jj]] - R[ii, :].dot(
                    self._R[self._I[jj], :])) / self._nu[jj]
        return R


def reg_cca(Rx, Ry, n_components=None, tau=1e-3, normalized=False):
    n = Rx.shape[0]
    Zxx = Rx.T.dot(Rx) / n
    Zyy = Ry.T.dot(Ry) / n
    Zxy = Rx.T.dot(Ry) / n
    Zyx = Ry.T.dot(Rx) / n

    Mx = Zxx + tau * np.eye(Zxx.shape[0])
    My = Zyy + tau * np.eye(Zyy.shape[0])

    if Mx.shape[0] < My.shape[0]:
        # solve problem in y view
        S = PGSO(eta=0).fit_transform(Mx)
        iS = np.linalg.inv(S)
        iMy = np.linalg.inv(My)
        iMyZ = iMy.dot(Zyx)
        AA = Zxy.dot(iMyZ)
        AA = 0.5 * (AA + AA.T)
        AA = iS.dot(AA).dot(iS.T)
        AA = 0.5 * (AA + AA.T)
        DD1, UU1 = sorted_spectrum(AA)
        lam = np.sqrt(DD1[:n_components])
        w11 = iS.T.dot(UU1[:, :n_components])
        w22 = iMyZ.dot(w11)
        w22 = w22 / np.tile(lam.reshape(1, -1), (w22.shape[0], 1))
    else:
        # solve problem in y view
        S = PGSO(eta=0).fit_transform(My)
        iS = np.linalg.inv(S)
        iMx = np.linalg.inv(Mx)
        iMxZ = iMx.dot(Zxy)
        AA = Zyx.dot(iMxZ)
        AA = 0.5 * (AA + AA.T)
        AA = iS.dot(AA).dot(iS.T)
        AA = 0.5 * (AA + AA.T)
        DD2, UU2 = sorted_spectrum(AA)
        lam = np.sqrt(DD2[:n_components])
        w22 = iS.T.dot(UU2[:, :n_components])

        w11 = iMxZ.dot(w22)
        w11 = w11 / np.tile(lam.reshape(1, -1), (w11.shape[0], 1))
    if normalized:
        return normalize(w11), normalize(w22), lam
    else:
        return w11, w22, lam


# Custom Transformer to compute KCCA
class KCCA(BaseEstimator, TransformerMixin):
    # Class Constructor
    def __init__(self,
                 n_components=None,
                 dim_first_view='half',
                 kernel1='linear',
                 kernel2='linear',
                 center_kernels=True,
                 gamma1=None,
                 gamma2=None,
                 degree1=None,
                 degree2=None,
                 coef0_1=1.0,
                 coef0_2=1.0,
                 tau=1e-6):
        self._n_components = n_components
        self._dim_first_view = dim_first_view
        self._kernel1 = kernel1
        self._kernel2 = kernel2
        self._center_kernels = center_kernels
        self._gamma1 = gamma1
        self._gamma2 = gamma2
        self._degree1 = degree1
        self._degree2 = degree2
        self._coef0_1 = coef0_1
        self._coef0_2 = coef0_2
        self._tau = tau

    # Return self nothing else to do here
    def _compute_K1_K2(self, X):
        n = X.shape[0]
        if self._kernel1 == 'precomputed':
            K1 = X[:, :self._dim_first_view]
        elif self._kernel1 == 'rbf':
            self.centerK1 = centred_kernel(kernel='rbf',
                                           centred=self._center_kernels,
                                           gamma=self._gamma1)
            K1 = self.centerK1.fit_transform(self._X1)
        elif self._kernel1 == 'poly':
            self.centerK1 = centred_kernel(kernel='poly',
                                           centred=self._center_kernels,
                                           degree=self._degree1,
                                           coef0=self._coef0_1)
            K1 = self.centerK1.fit_transform(self._X1)
        elif self._kernel1 == 'linear':
            self.centerK1 = centred_kernel(kernel='linear',
                                           centred=self._center_kernels)
            K1 = self.centerK1.fit_transform(self._X1)
        else:
            print("UNKNOWN KERNEL FOR VIEW 1")
            K1 = None

        if self._kernel2 == 'precomputed':
            K2 = X[:, self._dim_first_view:]
        elif self._kernel2 == 'rbf':
            self.centerK2 = centred_kernel(kernel='rbf',
                                           centred=self._center_kernels,
                                           gamma=self._gamma2)
            K2 = self.centerK2.fit_transform(self._X2)
        elif self._kernel2 == 'poly':
            self.centerK2 = centred_kernel(kernel='poly',
                                           centred=self._center_kernels,
                                           degree=self._degree2,
                                           coef0=self._coef0_2)
            K2 = self.centerK2.fit_transform(self._X2)
        elif self._kernel2 == 'linear':
            self.centerK2 = centred_kernel(kernel='linear',
                                           centred=self._center_kernels)
            K2 = self.centerK2.fit_transform(self._X2)
        else:
            print("UNKNOWN KERNEL FOR VIEW 1")
            K2 = None

        return K1, K2

    def fit(self, X, y=None):
        dim = X.shape[1]
        n = X.shape[0]
        if self._dim_first_view == 'half':
            self._dim_first_view = int(dim / 2)
        if self._kernel1 != 'precomputed':
            self._X1 = X[:, :self._dim_first_view]

        if self._kernel2 != 'precomputed':
            self._X2 = X[:, self._dim_first_view:]
        K1, K2 = self._compute_K1_K2(X)

        self._Chol1 = PGSO()
        self._Chol2 = PGSO()
        R1 = self._Chol1.fit_transform(K1)
        R2 = self._Chol2.fit_transform(K2)

        self._w1, self._w2, self._lambdas = reg_cca(R1,
                                                    R2,
                                                    n_components=self._n_components,
                                                    tau=self._tau)
        return self

    def fit_transform(self, X, y=None):
        dim = X.shape[1]
        if self._dim_first_view == 'half':
            self._dim_first_view = int(dim / 2)
        if self._kernel1 != 'precomputed':
            self._X1 = X[:, :self._dim_first_view]

        if self._kernel2 != 'precomputed':
            self._X2 = X[:, self._dim_first_view:]
        K1, K2 = self._compute_K1_K2(X)
        rank1 = np.linalg.matrix_rank(K1)
        rank2 = np.linalg.matrix_rank(K2)

        self._Chol1 = PGSO()
        self._Chol2 = PGSO()
        R1 = self._Chol1.fit_transform(K1)
        R2 = self._Chol2.fit_transform(K2)
        self._w1, self._w2, self._lambdas = reg_cca(R1,
                                                    R2,
                                                    n_components=self._n_components,
                                                    tau=self._tau)
        output = np.zeros((n, 2 * self._n_components))
        output[:, :self._n_components] = R1.dot(self._w1)
        output[:, self._n_components:] = R2.dot(self._w2)

        return output

    # Method that describes what we need this transformer to do
    def transform(self, X, y=None):
        n = X.shape[0]
        if self._kernel1 != 'precomputed':
            X1_test = X[:, :self._dim_first_view]
        if self._kernel2 != 'precomputed':
            X2_test = X[:, self._dim_first_view:]

        if self._kernel1 == 'precomputed':
            K1_test = X[:, :self._dim_first_view]
        else:
            K1_test = self.centerK1.transform(X1_test)

        if self._kernel2 == 'precomputed':
            K2_test = X[:, self._dim_first_view:]
        else:
            K2_test = self.centerK2.transform(X2_test)

        R1 = self._Chol1.transform(K1_test)
        R2 = self._Chol2.transform(K2_test)
        output = np.zeros((n, 2 * self._n_components))
        output[:, :self._n_components] = R1.dot(self._w1)
        output[:, self._n_components:] = R2.dot(self._w2)

        return output

    def run_experiment(self, K_train_1,
                       K_test_1,
                       K_train_2,
                       K_test_2,
                       kernel,
                       y_test,
                       targets,
                       idx_test,
                       nc_use=5,
                       n_images=10):

        self.fit(np.hstack((K_train_1, K_train_2)))
        K_pred_test = self.transform(np.hstack((K_test_1, K_test_2)))
        K_pred_view_one = K_pred_test[:, :self._n_components]
        K_pred__view_two = K_pred_test[:, self._n_components:]

        inner_product_image_text_5 = kernel(K_pred_view_one[:,:nc_use],
                                            K_pred__view_two[:,:nc_use])
        hits_5 = np.zeros(len(idx_test))
        closest_images_id_all = np.argsort(inner_product_image_text_5, axis=0)

        for ii, correct_label in enumerate(y_test):
            closest_images_id = closest_images_id_all[:, ii][::-1]
            closest_images_id = closest_images_id[:n_images]
            closest_image_in_test_set = [
                idx_test[cc] for cc in closest_images_id
            ]
            hits_5[ii] = np.sum(np.array(
                [1 for jj in range(n_images) if
                 correct_label == targets[0][
                     closest_image_in_test_set[jj]]]))
        outcome = np.mean(hits_5) / n_images * 100.

        return outcome










