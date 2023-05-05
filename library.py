import numpy as np
import numpy.random as rnd
import copy
from tabulate import tabulate


# Print summary statistics for an expert
def expert_stats(i, GPs, M, S, burnin, RG=False):
  mm = [m for m in range(burnin, M) if i < len(GPs[m])]
  if mm == []:
    return

  ss = S[burnin:]
  step = GPs[mm[-1]][i].step
  step_h = GPs[mm[-1]][i].step_h
  msg = "\nExpert {} ({:.1f}%, step={:.3f}, step_h={:.3f})"
  print(msg.format(i, np.sum(ss==i)/ss.size*100, step, step_h))

  headers = ["", "Mean", "STD", "%Accept"]
  table = []
  if RG == False:
    ## v
    _m = np.array([GPs[m][i].v for m in mm]).mean()
    _s = np.array([GPs[m][i].v for m in mm]).std()
    row = ["v"] + list(np.round((_m, _s), 2)) + ["n/a"]
    table.append(row)
    ## h
    for d in range(len(GPs[0][0].h)):
      row = ["h"+str(d)]
      _t = len(np.unique([GPs[m][i].h[d] for m in mm]))/len(mm)
      _m = np.array([GPs[m][i].h[d] for m in mm]).mean()
      _s = np.array([GPs[m][i].h[d] for m in mm]).std()
      row += list(np.round((_m, _s, _t), 2))
      table.append(row)
  ## Sig2
  _t = len(np.unique([GPs[m][i].sig2 for m in mm]))/len(mm)
  _m = np.array([GPs[m][i].sig2 for m in mm]).mean()
  _s = np.array([GPs[m][i].sig2 for m in mm]).std()
  row = ["sig2"] + list(np.round((_m, _s, _t), 2))
  table.append(row)
  # l
  for d in range(len(GPs[0][0].l)):
    row = ["l"+str(d)]
    _t = len(np.unique([GPs[m][i].l[d] for m in mm]))/len(mm)
    _m = np.array([GPs[m][i].l[d] for m in mm]).mean()
    _s = np.array([GPs[m][i].l[d] for m in mm]).std()
    row += list(np.round((_m, _s, _t), 2))
    table.append(row)
  ## Sig2n
  _t = len(np.unique([GPs[m][i].sig2n for m in mm]))/len(mm)
  _m = np.array([GPs[m][i].sig2n for m in mm]).mean()
  _s = np.array([GPs[m][i].sig2n for m in mm]).std()
  row = ["sig2n"] + list(np.round((_m, _s, _t), 2))
  table.append(row)
  print(tabulate(table, headers, tablefmt="plain"))


def sample_alpha(b, vv, par):
  i_star = len(vv)

  # unnormalised log-posterior
  def lp(a):
    nn = np.arange(a, a+int(b)+1) # from alpha to alpha+beta
    return i_star*np.sum(np.log(nn)) + (a-1)*lpv

  # lpv: log of (1-p_alpha)*prod(vv)
  lpv = np.log(1-par) + np.sum(np.log(vv))
  # serach for a_star by increment it by 1
  a_star = 1
  lphi = i_star*np.log((a_star+b)/a_star) + lpv # log(phi)
  while lphi >= 0:
    a_star += 1
    lphi = i_star*np.log((a_star+b)/a_star) + lpv
  phi = np.exp(lphi)

  # proposal PMF
  p_star = 1/(a_star + phi/(1-phi)) # p_bar/c
    
  # rejection sampling
  while True:
    # a_hat: proposal
    if rnd.rand() < p_star*a_star:
      a_hat = rnd.choice(range(1,a_star+1)) # uniform
    else:
      a_hat = rnd.geometric(1-phi) + a_star

    # lcq: log(c*q(a))
    if a_hat <= a_star:
      lcq = lp(a_star)
    else:
      lcq = lp(a_star) + (a_hat-a_star)*lphi
    if np.log(rnd.rand()) + lcq <= lp(a_hat):
      break

  return a_hat


def sample_beta(a, vv, par):
  i_star = len(vv)

  # unnormalised log-posterior
  def lp(b):
    nn = np.arange(b, int(a)+b+1) # from beta to alpha+beta
    return i_star*np.sum(np.log(nn)) + (b-1)*lpv

  # lpv: log of (1-p_beta)*prod(1-vv)
  lpv = np.log(1-par) + np.sum(np.log(1-vv))
  # serach for b_star by increment it by 1
  b_star = 1
  lphi = i_star*np.log((a+b_star)/b_star) + lpv # log(phi)
  while lphi >= 0:
    b_star += 1
    lphi = i_star*np.log((a+b_star)/b_star) + lpv
  phi = np.exp(lphi)
  
  # proposal PMF
  p_star = 1/(b_star + phi/(1-phi)) # p_bar/c

  # rejection sampling
  while True:
    # b_hat: proposal
    if rnd.rand() < p_star*b_star:
      b_hat = rnd.choice(range(1,b_star+1)) # uniform
    else:
      b_hat = rnd.geometric(1-phi) + b_star

    # lcq: log(c*q(b))
    if b_hat <= b_star:
      lcq = lp(b_star)
    else:
      lcq = lp(b_star) + (b_hat-b_star)*lphi
    if np.log(rnd.rand()) + lcq <= lp(b_hat):
      break

  return b_hat


# For adaptive step size
def update_step(m, acc, lsb, Hb, par_hmc):
  # par
  m0 = par_hmc["m0"]
  target = par_hmc["acc_target"]
  gamma = par_hmc["gamma"]
  kappa = par_hmc["kappa"]
  # compute
  step_mu = np.log(10*par_hmc["step_min"])
  Hb = (1-1/((m+1)+m0))*Hb + (target-acc)/((m+1)+m0)
  ls = step_mu - np.sqrt(m+1)/gamma*Hb
  lsb = (m+1)**(-kappa)*ls + (1-(m+1)**(-kappa))*lsb
  step = min(np.exp(ls), par_hmc["step_max"])

  return step, lsb, Hb


"""
U: potential; K: kinetic; z: auxiliary momentum
U & dU are defined on the original theta space,
while MH and leapfrog proceed in the log space.
Gamma priors assumed for all parameters.
"""
def hmc(expert, GP, par_prior, n_steps, noise=True):
  gp = copy.deepcopy(GP[expert])
  if noise:
    theta0 = np.hstack([gp.sig2, gp.l, gp.sig2n])
  else:
    theta0 = np.hstack([gp.sig2, gp.l])
  
  z0 = rnd.normal(size=len(theta0))
  theta1, z1 = leapfrog(gp, par_prior, theta0,
                        z0, n_steps, noise)

  # Constraints
  l = theta1[1:1+len(gp.l)]
  if (l > 0).all():
    U0 = U(theta0, gp, par_prior, noise)
    K0 = np.sum(z0**2)/2
    U1 = U(theta1, gp, par_prior, noise)
    K1 = np.sum(z1**2)/2
    lacc = min(0, U0+K0-U1-K1) # log of acceptance probability
    if np.log(rnd.rand()) > lacc:
      theta1 = theta0 # rejection
    GP[expert].acc = np.exp(lacc)
  else:
    theta1 = theta0 # rejection
    GP[expert].acc = 0

  # Update
  if (theta1 != theta0).any():
    GP[expert].sig2 = theta1[0]
    GP[expert].l = theta1[1:1+len(gp.l)]
    if noise:
      GP[expert].sig2n = theta1[-1]
    GP[expert].update_C()
    GP[expert].update_K()


def leapfrog(gp, par_prior, theta0, z0, n_steps, noise):
  step = gp.step
  theta = copy.deepcopy(theta0)
  ltheta = np.log(theta)
  z = z0 - step/2*dU(theta, gp, par_prior, noise)
  for _ in range(n_steps):
    ltheta += step*z
    theta = np.exp(ltheta)
    _dU = dU(theta, gp, par_prior, noise)
    z -= step*_dU
  z += step/2*_dU # go half-step back
  return theta, z


# U: potential (negative log posterior density)
def U(theta, gp, par_prior, noise):
  sig2 = theta[0]
  gp.sig2 = sig2
  l = theta[1:1+len(gp.l)]
  gp.l = l
  if noise:
    sig2n = theta[-1]
    gp.sig2n = sig2n
  gp.update_C()
  gp.update_K()
  ll = gp.ll()

  # lp: log priors
  ## (np.log() added due to the change of varianble)
  p = par_prior["sig2"]
  lp_sig2 = (p[0]-1)*np.log(sig2) - sig2/p[1] + np.log(sig2)
  p = par_prior["l"]
  lp_l = [(p[0]-1)*np.log(_l) - _l/p[1] + np.log(_l) for _l in l]
  if noise:
    p = par_prior["sig2n"]
    lp_sig2n = (p[0]-1)*np.log(sig2n) - sig2n/p[1] + np.log(sig2n)
  else:
    lp_sig2n = 0
  lp = lp_sig2 + np.sum(lp_l) + lp_sig2n
  
  return -(ll + lp)


# Gradient of U
def dU(theta, gp, par_prior, noise):
  N = len(gp.members)
  sig2 = theta[0]
  gp.sig2 = sig2
  l = theta[1:1+len(gp.l)]
  gp.l = l
  if noise:
    sig2n = theta[-1]
    gp.sig2n = sig2n
  gp.update_C()
  gp.update_K()
  Y = gp.Y[gp.members].reshape((-1,1))

  """
  dll: grad of log likelihood
  dlp: grad of log priors
  dK: partial derivatives of K
  For dll, first compute dll w.r.t. theta, and then
  multiply it by d(theta)/d(ltheta) due to the chain rule.
  """
  # dll
  ## t: (alpha*alpha - Kinv)
  t = np.linalg.multi_dot([gp.Kinv, Y, Y.T, gp.Kinv]) - gp.Kinv
  ## sig2
  dK = gp.C
  dll_sig2 = np.sum(t*dK)/2 # technically, dK.T but dk.T=dK
  ## l
  dll_l = np.empty(len(gp.l))
  for d, _l in enumerate(l):
    # D2: difference squared precomputed in update_C()
    dK = sig2*2*gp.C*gp.D2[d,:,:]/(_l**3)
    dll_l[d] = np.sum(t*dK.T)/2
  ## sig2n
  if noise:
    dK = np.eye(N)
    dll_sig2n = np.sum(t*dK.T)/2
    dll = np.hstack([dll_sig2, dll_l, dll_sig2n])
  else:
    dll = np.hstack([dll_sig2, dll_l])
  ## d(theta)/d(ltheta) = d(exp(ltheta))/d(ltheta) = theta
  dll *= theta

  # dlp
  ## (n.b. the change of varianble into p(ltheta))
  ## sig2
  p = par_prior["sig2"]
  dlp_sig2 = (p[0]-1) - sig2/p[1] + 1
  ## l
  p = par_prior["l"]
  dlp_l = np.array([(p[0]-1) - _l/p[1] + 1 for _l in l])
  ## sig2n
  if noise:
    p = par_prior["sig2n"]
    dlp_sig2n = (p[0]-1) - sig2n/p[1] + 1
    dlp = np.hstack([dlp_sig2, dlp_l, dlp_sig2n])
  else:
    dlp = np.hstack([dlp_sig2, dlp_l])

  return -(dll + dlp)


# HMC for h
def hmc_h(gp, s, r, n_steps):
  i = gp.id
  B = gp.B
  X = gp.X
  step = gp.step_h
  h0 = gp.h
  z0 = rnd.normal(size=len(h0))
  h1 = copy.deepcopy(h0)
  lh1 = np.log(h1)
  z1 = z0 - step/2*(-dlp_h(h1, i, X, s, B, r))
  for _ in range(n_steps):
    lh1 += step*z1
    # Check the constraint
    if (lh1 > 0).any():
      h1 = np.full_like(h1, np.inf) # for rejection
      _dU = 0. # just to pass z1 += step/2*_dU
      break
    h1 = np.exp(lh1)
    _dU = -dlp_h(h1, i, X, s, B, r)
    z1 -= step*_dU
  z1 += step/2*_dU

  # Constraints
  if (h1 >= 0).all() and (h1 <= 1).all():
    U0 = -lp_h(h0, i, X, s, B, r)
    K0 = np.sum(z0**2)/2
    U1 = -lp_h(h1, i, X, s, B, r)
    K1 = np.sum(z1**2)/2
    lacc = min(0, U0+K0-U1-K1) # log of acceptance probability
    if np.log(rnd.rand()) > lacc:
      h1 = h0
    gp.acc_h = np.exp(lacc)
  else:
    h1 = h0
    gp.acc_h = 0

  return h1


# Posterior log density of h (uniform prior assumed)
def lp_h(h, i, X, s, B, r):
  k = kernel(X[s>=i], h, r)
  Bi = B[s>=i]
  ll = np.sum(Bi*np.log(k) + (1-Bi)*np.log(1-k))
  # log prior is constant and can be dropped but,
  # np.sum(np.log(h)) is added due to the change of variable
  return ll + np.sum(np.log(h))


# Gradient of lp_h (uniform prior assumed)
def dlp_h(h, i, X, s, B, r):
  Xi = X[s>=i] # in N*D
  Bi = B[s>=i].reshape((-1,1)) # in N*1
  k = kernel(Xi, h, h).reshape((-1,1)) # in N*1
  # Xi-h: h subtracted from each row of X
  _t = 2/r**2*(Xi-h)
  dk = _t*k # in N*D
  # n.b. k cancelled in the first term
  dll = np.sum(Bi*_t - (1-Bi)*dk/(1-k), axis=0)
  dll *= h # =dh/dlog(h) by the chain rule
  return dll + 1 # due to the change of variable


# HMC for r
def hmc_r(r0, GP, s, par_prior, step, n_steps):
  r0 = np.atleast_1d(r0)
  z0 = rnd.normal(size=len(r0))
  r1 = copy.deepcopy(r0)
  lr1 = np.log(r1)
  z1 = z0 - step/2*(-dlp_r(r1, GP, s, par_prior))
  for _ in range(n_steps):
    lr1 += step*z1
    r1 = np.exp(lr1)
    if np.isclose(r1, 0).any() or r1 == np.inf:
      r1 = np.full_like(r1, 0.) # for rejection
      _dU = 0. # just to pass z1 += step/2*_dU line below
      break
    _dU = -dlp_r(r1, GP, s, par_prior)
    z1 -= step*_dU
  z1 += step/2*_dU

  if (r1 > 0).all():
    U0 = -lp_r(r0, GP, s, par_prior)
    K0 = np.sum(z0**2)/2
    U1 = -lp_r(r1, GP, s, par_prior)
    K1 = np.sum(z1**2)/2
    lacc = min(0, U0+K0-U1-K1) # log of acceptance probability
    if np.log(rnd.rand()) > lacc:
      r1 = r0
    acc_r = np.exp(lacc)
  else:
    r1 = r0
    acc_r = 0

  return r1.item(), acc_r


# Posterior log density of r (gamma prior assumed)
def lp_r(r, GP, s, par_prior):
  ll = 0
  for i in range(len(GP)):
    if np.sum(s>=i) > 0:
      gp = GP[i]
      Xi = gp.X[s>=i]
      Bi = GP[i].B[s>=i]
      k = kernel(Xi, gp.h, r)
      ll += np.sum(Bi*np.log(k) + (1-Bi)*np.log(1-k))
  p = par_prior["r"]
  return ll + (p[0]-1)*np.log(r) - r/p[1] + np.log(r)


# Gradient of lp_r wrt log(r)
"""
By the change of variable, dll wrt r is multiplied by r.
Since r can be very small, for numerical stability, 
/r**3 in dk is replace by /r**2.
"""
def dlp_r(r, GP, s, par_prior):
  dll = 0
  for i in range(len(GP)):
    if np.sum(s>=i) > 0:
      gp = GP[i]
      Xi = gp.X[s>=i]
      Bi = GP[i].B[s>=i]
      k = kernel(Xi, gp.h, r)
      # dk = 2*k*np.sum((Xi-gp.h)**2, axis=1)/r**3
      dk = 2*k*np.sum((Xi-gp.h)**2, axis=1)/r**2
      dll += np.sum(Bi*dk/k - (1-Bi)*dk/(1-k))
  # dll *= r
  p = par_prior["r"]
  return dll + (p[0]-1) - r/p[1] + 1


# HMC for r in RG
def hmc_rRG(r0, X, beta, s, par_prior, step, n_steps):
  r0 = np.atleast_1d(r0)
  z0 = rnd.normal(size=len(r0))
  r1 = copy.deepcopy(r0)
  lr1 = np.log(r1)
  z1 = z0 - step/2*(-dlp_rRG(r1, X, s, par_prior))
  for _ in range(n_steps):
    lr1 += step*z1
    r1 = np.exp(lr1)
    if np.isclose(r1, 0).any() or r1 == np.inf:
      r1 = np.full_like(r1, 0.) # for rejection
      _dU = 0. # just to pass z1 += step/2*_dU line below
      break
    _dU = -dlp_rRG(r1, X, s, par_prior)
    z1 -= step*_dU
  z1 += step/2*_dU

  if (r1 > 0).all():
    U0 = -lp_rRG(r0, X, beta, s, par_prior)
    K0 = np.sum(z0**2)/2
    U1 = -lp_rRG(r1, X, beta, s, par_prior)
    K1 = np.sum(z1**2)/2
    lacc = min(0, U0+K0-U1-K1) # log of acceptance probability
    if np.log(rnd.rand()) > lacc:
      r1 = r0
    acc_r = np.exp(lacc)
  else:
    r1 = r0
    acc_r = 0

  return r1.item(), acc_r


def lp_rRG(r, X, beta, s, par_prior):
  N = X.shape[0]
  K = kernels(X, r)
  ll = 0
  for n in range(N):
    i = s[n]
    idx = np.delete(range(N), n)
    _s = np.delete(s, n)
    unique_s = np.unique(_s)
    if i in unique_s:
      _t = np.sum(K[n,idx[_s==i]])
      ll += np.log(N-1) + np.log(_t) - np.log(np.sum(K[n,idx]))
    else:
      ll += np.log(beta)
  p = par_prior["r"]
  return ll + (p[0]-1)*np.log(r) - r/p[1] + np.log(r)


# Gradient of lp_rRG wrt log(r)
"""
By the change of variable, dll wrt r is multiplied by r.
Since r can be very small, for numerical stability, 
/r**3 in dk is replace by /r**2.
"""
def dlp_rRG(r, X, s, par_prior):
  N = X.shape[0]
  K = kernels(X, r)
  dll = 0
  for n in range(N):
    i = s[n]
    idx = np.delete(range(N), n)
    _s = np.delete(s, n)
    unique_s = np.unique(_s)
    if i in unique_s:
      k = K[n,idx]
      # dk = 2*k*np.sum((X[n]-X[idx])**2, axis=1)/r**3
      dk = 2*k*np.sum((X[n]-X[idx])**2, axis=1)/r**2
      dll += np.sum(dk[_s==i]) / np.sum(k[_s==i])
      dll -= np.sum(dk) / np.sum(k)
  # dll *= r
  p = par_prior["r"]
  return dll + (p[0]-1) - r/p[1] + 1


# Prior sampler (gamma as a default)
def sample_prior(pars, dist="gamma"):
  if dist == "gamma":
    p1, p2 = pars # [shape, scale]
    return rnd.gamma(p1, p2)
  elif dist == "invgamma":
    p1, p2 = pars # [shape, scale]
    return 1/rnd.gamma(p1, 1/p2)
  elif dist == "lognormal":
    p1, p2 = pars
    _t = np.log((p2/p1)**2+1)
    return rnd.lognormal(np.log(p1) - _t/2, np.sqrt(_t))
  

# Compute a kernel based on distance scaled by r
def kernel(X1, X2, r):
  """"
  With small r, ((X1-X2)/r)**2 can overflow and drives re to 0.
  Also, if X1 and X2 happen to be the same, re will be 1.
  In both cases, sum(kernels) may be 0 and raise warnings when
  used in log or denominator. If desired to avoid exact 0 and 1,
  use a bufferr 1e-9 as follows.  
  """
  re = np.exp(-np.sum(((X1-X2)/r)**2, axis=1))
  re = np.maximum(re, 1e-9)
  re = np.minimum(re, 1-1e-9)
  return re


# Compute the entire kernel matrix
def kernels(X, r):
  N = X.shape[0]
  K = np.zeros((N,N))
  for n in range(1,N):
    K[n,:n] = kernel(X[n], X[:n], r)
  K += K.T + np.eye(N)
  return K


# Compute weights
def weights(v, h, X, r):
  N = X.shape[0]
  s_star = len(v)
  W = np.zeros((N,s_star))
  for n, x in enumerate(X):
    k = kernel(x, h, r)
    VK = np.diag(v*k)
    A = np.eye(s_star) + VK.dot(np.tri(s_star, k=-1))
    b = VK.dot(np.ones((s_star,1)))
    w = np.linalg.solve(A, b).reshape(-1)
    W[n,:] = w
  return W


def cov(X1, X2, sig2, l):
  return sig2*np.exp(-np.sum(((X1-X2)/l)**2, axis=1))


# GP Expert class
## members: a set of indices for datapoints
class Expert():
  def __init__(self,
               id,
               par_prior,
               X,
               Y,
               s,
               noise=True,
               step=0.001):
    self.id = id
    self.X = X
    N, D = X.shape
    self.Y = Y
    self.v = 1.0
    self.h = rnd.random(D)
    self.A = np.full(N, 1)
    self.B = np.full(N, 1)
    self.members = list(np.where(s==id)[0])
    self.sig2 = sample_prior(par_prior["sig2"])
    _l = [sample_prior(par_prior["l"]) for _ in range(D)]
    self.l = np.array(_l)
    if noise:
      self.sig2n = sample_prior(par_prior["sig2n"])
    else:
      self.sig2n = 1e-6
    # D2: Squared distance per dimension of X
    ## Precomute and store to speed up computation of dU
    ## Obviously, also used for computation of C
    self.D2 = np.zeros((D, len(self.members), len(self.members)))
    self.update_C()
    self.update_K()
    # HMC
    self.step = step # step size
    self.lstep_bar = 0 # log of step_bar
    self.H_bar = 0
    self.acc = 0 # acceptance probability
    self.step_h = step
    self.lstep_bar_h = 0
    self.H_bar_h = 0
    self.acc_h = 0


  def update_C(self):
    N = len(self.members)
    X = self.X[self.members]
    self.C = np.zeros(((N,N)))
    self.D2 = np.zeros((X.shape[1], N, N))
    for row in range(1,N):
      for col in range(row):
        D2 = (X[row]-X[col])**2
        self.D2[:,row,col] = D2
        self.D2[:,col,row] = self.D2[:,row,col]
        self.C[row,col] = np.exp(-np.sum(D2/self.l**2))
    self.C = self.C + self.C.T + np.eye(N)


  def update_K(self):
    N = len(self.members)
    self.K = self.sig2*self.C + self.sig2n*np.eye(N)
    self.Kinv = np.linalg.inv(self.K)


  # Remove n and update K & Kinv
  def downdate_K1(self, n):
    if len(self.members) == 1:
      # Cheap if members are empty
      self.members.remove(n)
      self.update_C()
      self.update_K()
    else:
      # Rank-1 downdate
      j = self.members.index(n) # index within the expert
      ii = [x for x in range(len(self.members)) if x != j]
      a = self.Kinv[j,j]
      c = self.Kinv[ii,:][:,[j]]
      d = self.Kinv[ii,:][:,ii]

      self.Kinv = d - np.dot(c,c.T)/a
      self.K = self.K[ii,:][:,ii]
      self.members.remove(n)


  # Add n and update K & Kinv
  def update_K1(self, n):
    X = self.X
    idx = copy.deepcopy(self.members)
    self.members.append(n)
    self.members.sort()
    N = len(self.members)

    if N == 1:
      # Cheap for a single member
      self.update_C()
      self.update_K()
    else:
      # Rank-1 update
      ## K = [A B; C D], Kinv = [a b; c d], Di = Dinv
      j = self.members.index(n) # index within the expert
      ii = [x for x in range(len(self.members)) if x != j]
      K = np.empty((N,N))
      K[j,j] = self.sig2 + self.sig2n
      K[j,ii] = cov(X[n], X[idx], self.sig2, self.l)
      K[ii,j] = K[j,ii]
      K[np.ix_(ii,ii)] = self.K

      Kinv = np.empty((N,N))
      Di = self.Kinv
      A = K[j,j]
      B = K[[j],:][:,ii]
      DiB = Di.dot(B.T) # Di*B
      BDiB = B.dot(DiB) # B*Di*B
      a = 1/(A - BDiB)
      c = -DiB*a
      d = Di - DiB.dot(c.T)
      Kinv[j,j] = a
      Kinv[j,ii] = c.flatten()
      Kinv[ii,j] = c.flatten()
      Kinv[np.ix_(ii,ii)] = d
      
      self.Kinv = Kinv
      self.K = K


  # Log Likelihood (without a normalising constant)
  def ll(self):
    Y = self.Y[self.members].reshape((-1,1))
    sign, t1 = np.linalg.slogdet(self.K)
    t2 = np.linalg.multi_dot([Y.T, self.Kinv, Y]).item()
    return -(t1 + t2)/2


  # n=None for out-of-sample prediction
  def predict(self, x, n=None):
    X = self.X
    Y = self.Y
    sig2 = self.sig2
    l = self.l
    if n is None:
      sig2n = 0
    else:
      sig2n = self.sig2n

    idx = copy.deepcopy(self.members)
    if n in idx:
      is_member = True
      idx = list(set(idx) - set([n])) # this shaffles the order!
      idx.sort() # re-order
    else:
      is_member = False

    if len(idx) == 0: # no datapoint to condition on
      mu = 0
      sig = np.sqrt(sig2 + sig2n)
    else:
      K11 = sig2 + sig2n
      if is_member:
        j = self.members.index(n) # index within the expert
        ii = [x for x in range(len(self.members)) if x != j]
        K12 = self.K[[j],:][:,ii]
        K22 = self.K[ii,:][:,ii]
        # Rank-1 update of Kinv to get np.linalg.inv(K22)
        a = self.Kinv[j,j] # scalar
        c = self.Kinv[ii,:][:,[j]]
        d = self.Kinv[ii,:][:,ii]
        K22inv = d - np.dot(c,c.T)/a
      else:
        K12 = cov(x, X[idx], sig2, l).reshape((1,-1))
        K22 = self.K
        K22inv = self.Kinv
    
      """
      Use of pre-comuputed K22inv is more efficient but unstable
      than use of solve(), and t2 may be negative.
      """
      t1 = np.matmul(K12, K22inv)
      t2 = K11 - np.matmul(t1, K12.T)
      if t2 >= 0:
        mu = np.matmul(t1, Y[idx]).item()
      else:
        mu =  np.matmul(K12, np.linalg.solve(K22, Y[idx])).item()
        t2 = K11 - np.matmul(K12, np.linalg.solve(K22, K12.T))
      sig = np.sqrt(t2).item()

    return mu, sig
