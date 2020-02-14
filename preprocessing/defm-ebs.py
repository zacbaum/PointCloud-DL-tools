# Zachary Baum (zachary.baum.19@ucl.ac.uk),
# Wellcome EPSRC Center for Interventional and Surgical Sciences, University College London, 2020
# This code for research purposes only.
# 
# This is essentially a python re-implementation of the ebs_kernel.m, ebs_eval.m,
# and ebs_fit.m from @YipengHu's matlab-common-tools.
# See: https://github.com/YipengHu/matlab-common-tools/tree/master/SplineTransform
#
# Ref: 
# [1] Davis et al (1997), "A Physics-Based Coordinate Transformation for 
#     3-D Image Matching", IEEE TMI, 16(3): 317-328
# [2] Kohlrausch, Rohr & Stiehl (2005), "A New Class of Elastic Body 
#     Splines for Nonrigid Registration of Medical Images", Journal of
#     Mathematical Imaging and Vision 23: 253-280

# Used to compute the Elastic Body Spline Kernel between two point 
# sets (of the same size), with a defined sigma and nu (Useful 
# values found between (0, .49])
def compute_EBS_gauss(p, x, sigma, nu):

    n, d = p.shape
    m, _ = x.shape
    xd = np.zeros((m, n, 3))
    for i in range(d):
        xd[:, :, i] = np.ones((m, 1)) * p[:, i].T - \
                      np.reshape(x[:, i], [-1, 1]) * np.ones((1, n))
    K = np.zeros((m*3, n*3))
    for row in range(m):
        for col in range(n):
            K_r_c = basis_gauss(xd[row, col, :], d, sigma, nu)
            K[3*row:3*row+3, 3*col:3*col+3] = K_r_c
    
    return K

# Computes the gaussian basis for the EBS kernel.
def basis_gauss(y, d, sigma, nu):

    sigma2 = sigma**2
    r2 = np.matmul(y.T, y)
    if r2 == 0: r2 = 1e-8
    r = np.sqrt(r2)
    rhat = r / (np.sqrt(2) * sigma)
    c1 = scipy.special.erf(rhat) / r
    c2 = np.sqrt(2 / np.pi) * sigma * np.exp(-rhat**2) / r2
    g = ((4 * (1 - nu) - 1) * c1 - c2 + sigma2 * c1 / r2) * np.eye(d) + \
        (c1 / r2 + 3 * c2 / r2 - 3 * sigma2 * c1 / (r2 * r2)) * (y * np.reshape(y, [-1, 1]))
    
    return g

# Computes the weights for the EBS Spline Deformation.
def compute_EBS_w(p, q, sigma, nu, lmda=1e-6):
    
    n, d = p.shape
    m, _ = q.shape
    lmda = lmda * np.eye(n * d)
    k = compute_EBS_gauss(p, p, sigma, nu)
    y = np.reshape((q - p).T, [n * d, 1], order='F')
    L = k + lmda
    U, S, V = np.linalg.svd(L)
    w = np.matmul(np.matmul(np.matmul(U, np.diag(np.reciprocal(S))), V), y)
    
    return w

# Computes the deformation of points x given the control points p and 
# q, with a defined sigma and nu (Useful values found between (0, .49])
def compute_EBS_gauss_defm(p, q, x, sigma, nu):

    n, d = p.shape
    m, _ = x.shape
    w = compute_EBS_w(p, q, sigma, nu)
    k = compute_EBS_gauss(p, x, sigma, nu)

    return np.reshape(np.matmul(k, w), [d, m]).T + x