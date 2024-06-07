import torch
import time
from match_solver import myproj2dpam
import config as C

def matchSVT(S, dimGroup, **kwargs):
    alpha = kwargs.get('alpha', 0.1)  # 0.1 0.3 0.5 0.7
    pSelect = kwargs.get('pselect', 1)
    tol = kwargs.get('tol', 5e-4)
    maxIter = kwargs.get('maxIter', 50)
    verbose = kwargs.get('verbose', False)
    eigenvalues = kwargs.get('eigenvalues', False)
    _lambda = kwargs.get('_lambda', 50)  # 25 50 100
    mu = kwargs.get('mu', 64)
    threshold = kwargs.get('threshold', 0.5)  # 0.5 0.3 0.7
    dual_stochastic = kwargs.get('dual_stochastic_SVT', True)
    if verbose:
        print('Running SVT-Matching: alpha = %.2f, pSelect = %.2f _lambda = %.2f \n' % (alpha, pSelect, _lambda))
    info = dict()
    N = S.shape[0]
    S[torch.arange(N), torch.arange(N)] = 0  # diag == 0?
    S = (S + S.t()) / 2
    X = S.clone()
    Y = torch.zeros_like(S)
    W = alpha - S  # W : -S  -Similarity
    t0 = time.time()
    for iter_ in range(maxIter):

        X0 = X

        # update Q with SVT
        U, s, V = torch.svd(1.0 / mu * Y + X)
        diagS = s - _lambda / mu
        diagS[diagS < 0] = 0
        Q = U @ diagS.diag() @ V.t()

        # update X
        X = Q - (W + Y) / mu
        # project X
        for i in range(len(dimGroup) - 1):
            ind1, ind2 = dimGroup[i], dimGroup[i + 1]
            X[ind1:ind2, ind1:ind2] = 0
        if pSelect == 1:
            X[torch.arange(N), torch.arange(N)] = 1 # diag == 1?
        # X[X < 0] = 0
        # X[X > 1] = 1

        if dual_stochastic:
            # Projection for double stochastic constraint
            for i in range(len(dimGroup) - 1):
                row_begin, row_end = int(dimGroup[i]), int(dimGroup[i + 1])
                for j in range(len(dimGroup) - 1):
                    col_begin, col_end = int(dimGroup[j]), int(dimGroup[j + 1])
                    if row_end > row_begin and col_end > col_begin:
                        # Ori
                        X[row_begin:row_end, col_begin:col_end] = myproj2dpam(X[row_begin:row_end, col_begin:col_end],
                                                                             1e-2)
                        # New Sinkhorn
                        # X[row_begin:row_end, col_begin:col_end] = Sinkhorn(X[row_begin:row_end, col_begin:col_end])
        # if dual_stochastic:
            # Projection for double stochastic constraint using Sinkhorn
            # X = Sinkhorn(X)


        X = (X + X.t()) / 2

        # update Y
        Y = Y + mu * (X - Q)
        # test if convergence
        pRes = torch.norm(X - Q) / N
        dRes = mu * torch.norm(X - X0) / N
        if verbose:
            print(f'Iter = {iter_}, Res = ({pRes}, {dRes}), mu = {mu}')

        if pRes < tol and dRes < tol:
            break

        if pRes > 10 * dRes:
            mu = 2 * mu
        elif dRes > 10 * pRes:
            mu = mu / 2

    X = (X + X.t()) / 2
    info['time'] = time.time() - t0
    info['iter'] = iter_

    if eigenvalues:
        info['eigenvalues'] = torch.eig(X)

    X_bin = X > threshold
    if verbose:
        print(f"Alg terminated. Time = {info['time']}, #Iter = {info['iter']}, Res = ({pRes}, {dRes}), mu = {mu} \n")
    # match_mat = transform_closure ( X_bin.numpy() )
    # match_mat = transform_closure(X_bin) + 0
    match_mat = X_bin + 0
    # match_mat = match_mat.numpy()
    return match_mat


def SVT(matrix,each_len):
    # 定义参数，很重要！！！
    # this_threshold=0.7
    this_threshold = C.SVT_threshold
    this_alpha=0
    this_lambda=60
    this_mu=70
    # this_iter=75
    this_iter=C.SVT_iter

    number_of_person=[0]
    for i in range(len(each_len)):
        number_of_person.append((sum(each_len[:(i+1)])))
    result=matchSVT(matrix,number_of_person,alpha=this_alpha,_lambda=this_lambda,threshold=this_threshold,mu=this_mu,maxIter=this_iter,dual_stochastic_SVT=False)
    return result



















