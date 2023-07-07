import random
import numpy as np
import math 

def sobol(gpr, n_sample, dim, bounds):
    seed = None
    rng = np.random.default_rng(seed)
    A = rng.random((n_sample, dim))
    B = rng.random((n_sample, dim))
    #print(A.shape)
    if bounds is not None:
        bounds = np.asarray(bounds)
        min_ = bounds.min(axis=0)
        max_ = bounds.max(axis=0)

        A = (max_ - min_) * A + min_
        B = (max_ - min_) * B + min_
    #print(A)
    f_A = gpr.predict_f(A)
    f_B = gpr.predict_f(B)
    print(f_A)
    print(f_B)
    f_A = f_A.reshape([2000,1])
    f_A = f_A.reshape([2000,1])

    var = np.var(np.vstack([f_A, f_B]), axis=0)
    f_AB = []
    for i in range(dim):
        f_AB.append(gpr.predict(np.column_stack((A[:, 0:i], B[:, i], A[:, i+1:]))))

    f_AB = np.array(f_AB).reshape(dim, n_sample)
    print(f_AB.shape)
    print(f_B.shape)
    #print(f_A.flatten().shape)
    print(f_AB)

    s = 1 / n_sample * np.sum(f_B * (np.subtract(f_AB, f_A.flatten()).T), axis=0) / var

    st = 1 / (2 * n_sample) * np.sum(np.subtract(f_A.flatten(), f_AB).T ** 2, axis=0) / var

    return s, st
