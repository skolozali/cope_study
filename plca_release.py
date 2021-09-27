#  Author Dr Sefki Kolozali - sefki.kolozali@essex.ac.uk
from __future__ import division
from __future__ import print_function
from numpy import linalg as LA, diag, sqrt,shape,reshape, dot, transpose as tr, sum, power, prod, log, zeros, tril, random, tril_indices,diag_indices, identity, array, ones, arange, mean, nanmean
from Kalman_full import *
from TensorDecomposition import *


from numpy import divide




def plca_3d_fast_lds( x, S, A, F,iter, sh, sz, su, sg,w, h,z, u, g,expno, training=None, testing=None,y=None, pid=None, symptomlist=None,templatedirectory=None, alpha=None, beta=None):
    # x input distribution
    # D is the maximum number of days we have monitored our patients for in the given training dataset
    # S is number of symptoms
    # P is number of participants
    # M is number of months/seasons
    # A is number of attack types (i.e. onset, transient, offset)
    # F is number of features
    #  iter number of EM iterations[default = 100]
    # sh sparsity of h
    # sz sparsity of z
    #  su sparsity of u
    # sg sparsity of g
    #  w initial value of w(basis 5 - D tensor)
    #  h initial value of h(monthly/seasonal probability tensor)
    #  z initial value of z(symptom activation matrix)
    #  u initial value of u(participant probability tensor)
    #  g initial value of g(symptom-state/attack-type activation tensor)
    #  expno experiment number

     # Outputs:
     # w spectral bases
     # h monthly/sasonal probability
     # z symptom activation
     # u participant probability
     # xa approximation of input
    y = asarray(y)
    alpha = alpha
    beta = beta

    if testing==True:
        init_m = (1 / S*2) * ones((S*2, 1))
        init_V = eye(S*2)
        ldsQ = alpha * np.eye(S*2)
        ldsR = beta * np.eye(S)


    eps = pow(2.2204,-16)
    x = asarray(x) + eps/2
     # Get sizes , M is features, N is time. sum x is summarised representation of features over time.
    F, T = shape(x)
    sumx = sum(x, axis=0)
    if iter==None:
        iter = 100

     # Initialize
    if np.all(w==None) or (len(w)==0):
        w = random.random((S, A,F))
    else:
        w = w

    # Normalize W
    for s in arange(0,S):
        for a in arange(0,A):
            w[s, a,:] = divide(w[s,a,:], (sum(w[s,a,:])+eps))

    #  P(S|T)
    if z is None or len(z)==0:
        z = random.random((S, T))

    t=arange(0,T)

    z[:,t] = multiply(tile(sumx[t],(S,1)) , (z[:,t] / tile(sum(z[:,t], 0)+eps, (S, 1))+eps))

    #  P(A|S,T)
    if g is None or len(g)==0:
        g = random.random((A, S, T))

    for s in arange(0,S):
        for t in arange(0,T):
            g[:, s, t] = g[:, s, t]/ sum(g[:, s, t] + eps)
    w_reshaped = unfold(w, 2)
    sumx = np.diag(sumx)

    for it in arange(0,iter):
        zg = multiply(g, tile(z, (A,1, 1))).transpose(2,1,0)
        zg_reshaped = unfold(zg,0)
        xa = dot(w_reshaped,np.transpose(zg_reshaped))
        delta = np.divide(x, (xa + eps))
        # M - step(update h, z, u, g)
        WD = dot(transpose(delta),w_reshaped)
        WDUZHGV = fold(zg_reshaped * WD, 0, shape(zg))
        z = pow(np.transpose(sum(WDUZHGV,axis=(2))), sz)
        g = pow(WDUZHGV.transpose(2, 1, 0),sg)

          # Perform LDS smoothing on z
        if (it >= 20):

            if testing == True:
                AA = np.load(templatedirectory + '_A.npy')
                HH = np.load(templatedirectory + '_H.npy')
                if S==1:
                    ldsQQ = np.random.multivariate_normal(np.zeros(S * 2), ldsQ, (S * 2))
                    ldsRR = np.random.normal(0, beta, S)
                if S>1:
                    ldsQQ=np.random.multivariate_normal(np.zeros(S*2),ldsQ,(S*2))
                    ldsRR = np.random.multivariate_normal(np.zeros(S), ldsR,(S))
                [m, V] = Kalman_smoother(z, AA, HH, ldsQQ, ldsRR, init_m, init_V)

                z_lds = m[0:S,:]
                z_lds[z_lds<0]=eps
                z= z_lds


        # Normalize h, z, u, g
        z = dot(divide(z, tile(sum(z, 0), (S,1)) + eps), sumx)
        g_resh = unfold(g, 0)
        g_resh = np.divide(g_resh , np.sum(g_resh, axis=0,dtype="float64") + eps)
        g = fold(g_resh, 0, shape(g))
    return(w,h,z,u,g,xa)



def plca_3d( x, S, A, F,iter, sh, sz, su, sg,w, h,z, u, g,expno, training=None, testing=None,y=None, pid=None, symptomlist=None, templatedirectory=None):
    # x input distribution
    # D is the maximum number of days we have monitored our patients for in the given training dataset
    # S is number of symptoms
    # P is number of participants
    # M is number of months/seasons
    # A is number of attack types (i.e. onset, transient, offset)
    # F is number of features
    #  iter number of EM iterations[default = 100]
    # sh sparsity of h
    # sz sparsity of z
    #  su sparsity of u
    # sg sparsity of g
    #  w initial value of w(basis 5 - D tensor)
    #  h initial value of h(monthly/seasonal probability tensor)
    #  z initial value of z(symptom activation matrix)
    #  u initial value of u(participant probability tensor)
    #  g initial value of g(symptom-state/attack-type activation tensor)
    #  expno experiment number

     # Outputs:
     # w spectral bases
     # h monthly/sasonal probability
     # z symptom activation
     # u participant probability
     # xa approximation of input
    y = asarray(y)


    eps = pow(2.2204,-8)
    x = asarray(x) + eps/2
     # Get sizes , M is features, N is time. sum x is summarised representation of features over time.
    F, T = shape(x)
    sumx = sum(x, axis=0)
    if iter==None:
        iter = 100

     # Initialize
    if np.all(w==None) or (len(w)==0):
        w = random.random((S, A,F))
    else:
        w = w

    for s in arange(0,S):
        for a in arange(0,A):
            w[s, a,:] = divide(w[s,a,:], (sum(w[s,a,:])+eps))

    #  P(S|T)
    if z is None or len(z)==0:
        z = random.random((S, T))

    t=arange(0,T)
    z[:,t] = multiply(tile(sumx[t],(S,1)) , (z[:,t] / tile(sum(z[:,t], 0)+eps, (S, 1))+eps))

    #  P(A|S,T)
    if g is None or len(g)==0:
        g = random.random((A, S, T))

    for s in arange(0,S):
        for t in arange(0,T):
            g[:, s, t] = g[:, s, t]/ sum(g[:, s, t] + eps)

    w_reshaped = unfold(w, 2)
    sumx = np.diag(sumx)


    for it in arange(0,iter):
         # E-step
        #   A, Z, T
        zg = multiply(g, tile(z, (A,1, 1))).transpose(2,1,0)
        zg_reshaped = unfold(zg,0)
        xa = dot(w_reshaped,np.transpose(zg_reshaped))
        delta = np.divide(x, (xa + eps))

        # M - step(update h, z, u, g)
        WD = dot(transpose(delta),w_reshaped)
        WDUZHGV = fold(zg_reshaped * WD, 0, shape(zg))
        z = pow(np.transpose(sum(WDUZHGV,axis=(2))), sz)
        g = pow(WDUZHGV.transpose(2, 1, 0),sg)

        z = dot(divide(z, tile(sum(z, 0), (S,1)) + eps), sumx)
        g_resh = unfold(g, 0)
        g_resh = np.divide(g_resh , (tile(sum(g_resh, axis=0), (A, 1)) + eps))
        g = fold(g_resh, 0, shape(g))

    return(w,h,z,u,g,xa)

def plca_4d( x,M, S, A, F,iter, sh, sz, su, sg,w, h,z, u, g,expno, training=None, testing=None,y=None, pid=None, symptomlist=None, templatedirectory=None):
    # x input distribution
    # D is the maximum number of days we have monitored our patients for in the given training dataset
    # S is number of symptoms
    # P is number of participants
    # M is number of months/seasons
    # A is number of attack types (i.e. onset, transient, offset)
    # F is number of features
    #  iter number of EM iterations[default = 100]
    # sh sparsity of h
    # sz sparsity of z
    #  su sparsity of u
    # sg sparsity of g
    #  w initial value of w(basis 5 - D tensor)
    #  h initial value of h(monthly/seasonal probability tensor)
    #  z initial value of z(symptom activation matrix)
    #  u initial value of u(participant probability tensor)
    #  g initial value of g(symptom-state/attack-type activation tensor)
    #  expno experiment number

     # Outputs:
     # w spectral bases
     # h monthly/sasonal probability
     # z symptom activation
     # u participant probability
     # xa approximation of input


    eps = pow(2.2204,-8)
    x = asarray(x) + eps/2
     # Get sizes , M is features, N is time. sum x is summarised representation of features over time.
    F, T = shape(x)
    sumx = sum(x, axis=0)
    if iter==None:
        iter = 100


     # Initialize
    if np.all(w==None) or (len(w)==0):
        w = random.random((M, S, A,F))
    else:
        w = w
    # Normalize W
    for p in arange(0, M):
            # for m in arange(0, M):
            for s in arange(0,S):
                for a in arange(0,A):
                # print( w[:,p,m,s,a])
                    w[p,s, a,:] = divide(w[p,s,a,:], (sum(w[p,s,a,:])+eps))

    #  P(S|T)
    if z is None or len(z)==0:
        z = random.random((S, T))

    t=arange(0,T)
    z[:,t] = multiply(tile(sumx[t],(S,1)) , (z[:,t] / tile(sum(z[:,t], 0)+eps, (S, 1))+eps))
    # #  P(M|S,T)
    if u is None or len(u)==0:
        u = random.random((M, S, T))

    for s in arange(0,S):
        for t in arange(0,T):
            # u[:,s,t] = divide(u[:,s,t], (sum(u[:,s,t])+eps))
            u[:, s, t] = u[:, s, t]/ (sum(u[:, s, t]) + eps)

    #  P(A|S,T)
    if g is None or len(g)==0:
        g = random.random((A, S, T))

    for s in arange(0,S):
        for t in arange(0,T):
            g[:, s, t] = g[:, s, t]/ sum(g[:, s, t] + eps)

    # Initialize update parameters
    w_reshaped = unfold(w, 3)
    sumx = np.diag(sumx)

     # Iterate
    for it in arange(0,iter):
         # E-step
        #   D, Z, T
        uz = multiply(u, tile(z, (M,1, 1)))
        uz_big = tile(uz, (A, 1, 1, 1)).transpose(3,1,2,0)

        g_big = tile(g, (M, 1, 1, 1)).transpose(3,0,2,1)
        uzhgv = multiply(uz_big, g_big)
        uzhgv_reshaped = unfold(uzhgv,0)
        xa = dot(w_reshaped,np.transpose(uzhgv_reshaped))
        delta = np.divide(x, (xa + eps))

        # M - step(update h, z, u, g)
        WD = dot(transpose(delta),w_reshaped)

        #  calculate global loss p(z,u,g)- WD p(z,u,g)
        WDUZHGV = fold(uzhgv_reshaped * WD, 0, shape(uzhgv))
        z = pow(np.transpose(sum(WDUZHGV,axis=(1,3))), sz)
        u = pow(sum(WDUZHGV, axis=(3)).transpose(1, 2, 0),su)
        g = pow(sum(WDUZHGV,axis=(1)).transpose(2, 1, 0),sg)

        # Normalize h, z, u, g
        z = dot(divide(z, tile(sum(z, 0), (S,1)) + eps), sumx)
        u_resh = unfold(u, 0)
        u_resh = np.divide(u_resh , (tile(sum(u_resh, axis=0), (M, 1)) + eps))
        u = fold(u_resh, 0, shape(u))
        g_resh = unfold(g, 0)
        g_resh = np.divide(g_resh , (tile(sum(g_resh, axis=0), (A, 1)) + eps))
        g = fold(g_resh, 0, shape(g))


    return(w,h,z,u,g,xa)

def plca_4d_fast_lds( x,M, S, A, F,iter, sh, sz, su, sg,w, h,z, u, g,expno, training=None, testing=None,y=None, pid=None, symptomlist=None, templatedirectory=None, alpha=None, beta=None):
    # x input distribution
    # D is the maximum number of days we have monitored our patients for in the given training dataset
    # S is number of symptoms
    # P is number of participants
    # M is number of months/seasons
    # A is number of attack types (i.e. onset, transient, offset)
    # F is number of features
    #  iter number of EM iterations[default = 100]
    # sh sparsity of h
    # sz sparsity of z
    #  su sparsity of u
    # sg sparsity of g
    #  w initial value of w(basis 5 - D tensor)
    #  h initial value of h(monthly/seasonal probability tensor)
    #  z initial value of z(symptom activation matrix)
    #  u initial value of u(participant probability tensor)
    #  g initial value of g(symptom-state/attack-type activation tensor)
    #  expno experiment number

     # Outputs:
     # w spectral bases
     # h monthly/sasonal probability
     # z symptom activation
     # u participant probability
     # xa approximation of input
    y = asarray(y)


    if training==True or testing==True:
        init_m = (1 / S*2) * ones((S*2, 1))
        init_V = eye(S*2)
        ldsQ = alpha * np.eye(S*2)
        ldsR = beta * np.eye(S)


    eps = pow(2.2204,-8)
    x = asarray(x) + eps/2
    F, T = shape(x)
    sumx = sum(x, axis=0)

    if iter==None:
        iter = 100


     # Initialize
    if np.all(w==None) or (len(w)==0):
        w = random.random((M, S, A,F))
    else:
        w = w

    # Normalize W
    for p in arange(0, M):
            for s in arange(0,S):
                for a in arange(0,A):
                    w[p,s, a,:] = divide(w[p,s,a,:], (sum(w[p,s,a,:])+eps))

    #  P(S|T)
    if z is None or len(z)==0:
        z = random.random((S, T))

    # Binary masking of P(S|T)

    t=arange(0,T)
    z[:,t] = multiply(tile(sumx[t],(S,1)) , (z[:,t] / tile(sum(z[:,t], 0)+eps, (S, 1))+eps))

    # #  P(M|S,T)
    if u is None or len(u)==0:
        u = random.random((M, S, T))

    for s in arange(0,S):
        for t in arange(0,T):
            u[:,s,t] = divide(u[:,s,t], (sum(u[:,s,t])+eps))
            # u[:, s, t] = u[:, s, t]/ (sum(u[:, s, t]) + eps)

    #  P(A|S,T)
    if g is None or len(g)==0:
        g = random.random((A, S, T))

    for s in arange(0,S):
        for t in arange(0,T):
            g[:, s, t] = g[:, s, t]/ sum(g[:, s, t] + eps)

    # Initialize update parameters
    w_reshaped = unfold(w, 3)
    sumx = np.diag(sumx)

    for it in arange(0,iter):
         # E-step
        uz = multiply(u, tile(z, (M,1, 1)))
        uz_big = tile(uz, (A, 1, 1, 1)).transpose(3,1,2,0)
        g_big = tile(g, (M, 1, 1, 1)).transpose(3,0,2,1)
        uzhgv = multiply(uz_big, g_big)
        uzhgv_reshaped = unfold(uzhgv,0)
        xa = dot(w_reshaped,np.transpose(uzhgv_reshaped))
        delta = np.divide(x, (xa + eps))
        WD = dot(transpose(delta),w_reshaped)

        WDUZHGV = fold(uzhgv_reshaped * WD, 0, shape(uzhgv))
        z = pow(np.transpose(sum(WDUZHGV,axis=(1,3))), sz)
        u = pow(sum(WDUZHGV, axis=(3)).transpose(1, 2, 0),su)
        g = pow(sum(WDUZHGV,axis=(1)).transpose(2, 1, 0),sg)

          # Perform LDS smoothing on z
        if (it >= 20):

            if testing == True:
                AA = np.load(templatedirectory + '_A.npy')
                HH = np.load(templatedirectory + '_H.npy')

                ldsQQ=np.random.multivariate_normal(np.zeros(S*2),ldsQ,(S*2))
                ldsRR = np.random.multivariate_normal(np.zeros(S), ldsR,(S))

                [m, V] = Kalman_smoother(z, AA, HH, ldsQQ, ldsRR, init_m, init_V)
                z_lds = m[0:S,:]
                z_lds[z_lds <= 0] = eps
                z= z_lds

        # Normalize h, z, u, g
        z = dot(divide(z, tile(sum(z, 0), (S,1)) + eps), sumx)
        u_resh = unfold(u, 0)
        u_resh = np.divide(u_resh , (tile(sum(u_resh, axis=0), (M, 1)) + eps))
        u = fold(u_resh, 0, shape(u))
        g_resh = unfold(g, 0)
        g_resh = np.divide(g_resh , (tile(sum(g_resh, axis=0), (A, 1)) + eps))
        g = fold(g_resh, 0, shape(g))

    return(w,h,z,u,g,xa)