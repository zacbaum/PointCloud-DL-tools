# Zachary Baum (zachary.baum.19@ucl.ac.uk),
# Wellcome EPSRC Center for Interventional and Surgical Sciences, University College London, 2020
# This code for research purposes only.
# 
# This is essentially a python re-implementation of the rbfs_kernel.m and rbfs_fit.m
# from @YipengHu's matlab-common-tools.
# See: https://github.com/YipengHu/matlab-common-tools/tree/master/SplineTransform

# Used to compute the Radial Basis Function Kernel between two point 
# sets (of the same size), with a defined sigma.
def compute_RBF(x, y, sigma):

    n, d = x.shape
    m, _ = y.shape
    K = np.zeros([n, m])
    for i in range(d):
        K += np.square((x[:, [i]] * np.ones([1, m]) - np.ones([n, 1]) * y[:, i].T))
    K = np.exp(np.divide(K, (-2 * (sigma ** 2))))

    return K

# Used to compute the deformation on point cloud y given control
# points and weights c and x, as well as sigma.
def compute_RBF_defm(c, x, y, sigma):

    k = compute_RBF(x, y, sigma)

    return np.matmul(k.T, c) + y