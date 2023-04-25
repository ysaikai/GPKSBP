"""
[Naming]
When variables start with _ (underscore), it means they are used
as temporary placeholders (e.g., _A, _mu, _sig...).
Also, t1, t2,... are also used to indicate temporary variables.

[Key variables]
GP: a list of GP experts
D2: squared distance matrix per dimension
C: correlation matrx
K: covariance matrix
sig2: GP output variances
l: GP lengthscales
sig2n: Noise variances
M: number of MCMC samples
n: index of datapoint
N: number of datapoints
s: expert assignments
v: stick-breaking probabilities
h: locations
r: kernel width
W: individual probabilities (row per datapoint)
u: sliced point; 0<u[n]<W[n,s[n]]
"""

import sys
import csv
import numpy as np
import numpy.random as rnd
import copy
import time
import datetime
import pickle
from tabulate import tabulate
from library import (Expert, expert_stats, kernel, weights,
                     sample_alpha, sample_beta,
                     update_step, hmc, hmc_h, hmc_r)


# Parameters
try:
  seed = int(float(sys.argv[1]))
  dataset = sys.argv[2]
  id = str(seed)+dataset
  experiment = True
except:
  experiment = False

method = "GPKSBP"
if experiment == False:
  M = 20000
  seed = 0
else:
  dir_name = "experiments"

  with open(dir_name+"/par"+id+".csv", "r") as file:
    reader = csv.reader(file)
    for n, line in enumerate(reader):
      if line[0] == "M":
        M = int(line[1])
      elif line[0] == "noise":
        if line[1] == "False":
          noise = False
        else:
          noise = True
np.random.seed(seed)
burnin = max(1, int(0.5*M))
par_prior = {"alpha": .5,
             "beta": .5,
             "sig2": [2., 2.],
             "l": [2., .5],
             "sig2n": [2., .5],
             "r": [2., .5]}
par_hmc = {"n_steps": 5,
           "step_min": 0.001,
           "step_max": 0.05,
           "acc_target": 0.8,
           "gamma": 0.05,
           "kappa": 0.75,
           "m0": 10,
           "m_min": 0}


# Data
"""
When running as part of experiments, train data are
already generated. Otherwise, generate/specify data below.
"""
if experiment:
  # (No header, Y in 1st column, X in the rest)
  data = np.genfromtxt(dir_name+"/train"+id+".csv", delimiter=",")
  Y0 = data[:,0]
  X0 = data[:,1:]
else:
  """GL2008"""
  def gl2008(x):
    x = np.atleast_2d(x)
    return x[:,0]*np.exp(-np.sum(x**2, axis=1))

  X1l = rnd.random((10,2))*np.array([1.,2.]) + np.array([-1.,-1.])
  X1h = rnd.random((10,2))*np.array([1.,2.]) + np.array([0.,-1.])
  X4 = rnd.random((10,2)) + np.array([4.,4.])
  X0 = np.concatenate((X1l,X1h,X4), axis=0)
  Y0 = gl2008(X0)
  dataset = "GL2008"
  noise = False

  time_s = time.time()

N, D = X0.shape
lb = X0.min(axis=0) # lower bounds
ub = X0.max(axis=0) # upper bounds
X = (X0 - lb) / (ub - lb) # normalisation
Y = (Y0 - Y0.mean()) / Y0.std() # standardisation


# Initialisation
s = np.full(N, 0)
S = np.full((M,N), 0)
alpha = 1/par_prior["alpha"] # prior mean
Alpha = np.zeros(M)
beta = 1/par_prior["beta"] # prior mean
Beta = np.zeros(M)
r = par_prior["r"][0]*par_prior["r"][1] # prior mean
R = np.full(M, r)
GP = [Expert(0, par_prior, X, Y, s, noise)]
GPs = []
W = weights([GP[0].v], np.array([GP[0].h]), X, r)
u = rnd.rand(N)*W[range(N),s]
U = np.zeros((M,N))
## for HMC_r
step_r = par_hmc["step_min"]
lstep_bar_r = 0
H_bar_r = 0
acc_r = 0


msg = "{}, {}, N={}, {}, M={}"
print(msg.format(seed, dataset, N, method, M))

# Posterior sampling
for m in range(M):
  # progress output
  if (m+1)%int(M/5) == 0:
    print("{:5d}/{}".format(m+1, M))
    if experiment == False:
      print(np.asarray(np.unique(s, return_counts=True))[:,:10])

  # r
  r, acc_r = hmc_r(r, GP, s, par_prior, step_r, par_hmc["n_steps"])
  if m < burnin and m >= par_hmc["m_min"]:
    _a, _b, _c = update_step(m, acc_r, lstep_bar_r, H_bar_r, par_hmc)
    step_r, lstep_bar_r, H_bar_r = (_a, _b, _c)

  # v, h & u
  """
  Updates of v & h use only those A[n] & B[n] s.t. s[n]>=i, and
  A[n]=B[n]=1 if s[n]=i.
  u is sampled together with v & h. The condition i <= max(s)
  for the loop is to ensure u are sampled for all n.
  Sufficient number of experts satisfies u[n] > 1-w_sum[n]
  for all n.
  """
  vv = [] # used for sampling alpha & beta
  _W = [] # to be converted to W after the block
  w_sum = np.zeros(N)
  i = 0
  while (u < 1 - w_sum).any() or i <= max(s):
    # Create a new expert
    if i > len(GP) - 1:
      GP.append(Expert(i, par_prior, X, Y, s, noise))
    
    # v
    A = GP[i].A
    v = rnd.beta(alpha+np.sum(A[s>=i]), beta+np.sum(1-A[s>=i]))
    vv.append(v)
    GP[i].v = v

    # h
    if np.sum(s>=i) == 0:
      h = rnd.rand(D)
    else:
      h = hmc_h(GP[i], s, r, par_hmc["n_steps"])
      if m < burnin and m >= par_hmc["m_min"]:
        _a, _b, _c = update_step(m, GP[i].acc_h, GP[i].lstep_bar_h,
                                 GP[i].H_bar_h, par_hmc)
        GP[i].step_h, GP[i].lstep_bar_h, GP[i].H_bar_h = (_a, _b, _c)
    GP[i].h = h

    # A & B
    A = np.full(N, 1)
    B = np.full(N, 1)
    for n in np.arange(N)[s>i]:
      k = kernel(X[n].reshape((1,-1)), h, r).item()
      prob = (v*(1-k), (1-v)*k, (1-v)*(1-k))
      j = rnd.choice(range(3), p=prob/np.sum(prob))
      A[n], B[n] = ((1,0),(0,1),(0,0))[j]
    GP[i].A = A
    GP[i].B = B
    
    # A new set of weights for i-th expert
    _w = v*kernel(X, h, r)*(1-w_sum) # N weights
    _W.append(_w)
    w_sum += _w

    # u
    ## u[n] ~ Uniform[0,w[n]]
    ## only for those with s=i as it needs updated w[n]
    idx = np.where(s==i)[0]
    u[idx] = rnd.rand(len(idx))*_w[idx]

    i += 1
  vv = np.array(vv)
  W = np.array(_W).T

  # alpha & beta
  alpha = sample_alpha(beta, vv, par_prior["alpha"])
  beta = sample_beta(alpha, vv, par_prior["beta"])

  # s
  for n in range(N):
    # Gumbel-Max Trick for sampling with log-probabilities
    gmtrick = dict()
    w_sum = 0
    i = 0
    while u[n] < 1 - w_sum:
      if W[n,i] > u[n]:
        mu, sig = GP[i].predict(X[n], n)
        lp = -np.log(sig) - ((Y[n]-mu)/sig)**2/2
        gmtrick[i] = -np.log(-np.log(rnd.rand())) + lp
      w_sum += W[n,i]
      i += 1

    i = max(gmtrick, key=gmtrick.get)
    # If the assignment changes, update GP[s[n]] and GP[i]
    if i != s[n]:
      GP[s[n]].downdate_K1(n) # Remove n
      GP[i].update_K1(n) # Add n
      s[n] = i

  # Delete inactive GPs
  for i in range(len(GP)-1, -1, -1):
    if len(GP[i].members) == 0:
      del GP[i]
  
  # Re-label the experts id to remove gaps
  for i in range(len(GP)):
    if i != GP[i].id:
      s[s==GP[i].id] = i
      GP[i].id = i

  # sig2, l, & sig2n
  ## should be done after deleting and relabeling the experts
  for i in range(len(GP)):
    hmc(i, GP, par_prior, par_hmc["n_steps"], noise)
    if m < burnin and m >= par_hmc["m_min"]:
      _a, _b, _c = update_step(m, GP[i].acc, GP[i].lstep_bar,
                               GP[i].H_bar, par_hmc)
      GP[i].step, GP[i].lstep_bar, GP[i].H_bar = (_a, _b, _c)

  # End of the round
  R[m] = r
  U[m] = u
  Alpha[m] = alpha
  Beta[m] = beta
  S[m] = s
  ## Members changed, and so did C & D2
  for gp in GP:
    gp.update_C()
    gp.update_K() # technically, unnecessary
  GPs.append(copy.deepcopy(GP))


# Post process
# Export results & parameters
## To reduce the file size, delete what can be reconstructed
for GP in GPs:
  for gp in GP:
    del gp.X
    del gp.Y
    del gp.D2
    del gp.C
    del gp.K
    del gp.Kinv

re = {"GPs": GPs,
      "noise": noise,
      "S": S,
      "R": R,
      "Alpha": Alpha,
      "Beta": Beta,
      "U": U,
      "par_prior": par_prior,
      "par_hmc": par_hmc,
      "burnin": burnin,
      "seed": seed,
      "dataset": dataset,
      "method": method}
if experiment == False:
  re["X0"] = X0
  re["Y0"] = Y0

fname = "{:%Y%m%d%H%M%S}.pickle".format(datetime.datetime.now())
if experiment == False:
  print("\nIt took {:.1f} sec.".format(time.time()-time_s))

  # Summary
  headers = ["", "Mean", "STD", "%Accept", "Step"]
  table = []
  mm = [m for m in range(burnin, M)]
  ## r
  _t = len(np.unique([R[m] for m in mm]))/len(mm)
  _m = np.array([R[m] for m in mm]).mean()
  _s = np.array([R[m] for m in mm]).std()
  row = ["r", *np.round((_m, _s, _t), 2), np.round(step_r, 3)]
  table.append(row)
  ## alpha (%Acpt = 1 as sampled till accepted)
  _m = np.array([Alpha[m] for m in mm]).mean()
  _s = np.array([Alpha[m] for m in mm]).std()
  row = ["alpha"] + list(np.round((_m, _s), 2)) + ["n/a"]
  table.append(row)
  ## beta (%Acpt = 1 as sampled till accepted)
  _m = np.array([Beta[m] for m in mm]).mean()
  _s = np.array([Beta[m] for m in mm]).std()
  row = ["beta"] + list(np.round((_m, _s), 2)) + ["n/a"]
  table.append(row)
  print(tabulate(table, headers, tablefmt="plain"))

  ## Expert #0-2
  for expert in range(3):
    expert_stats(expert, GPs, M, S, burnin)

  with open(fname, "wb") as file:
    pickle.dump(re, file)
  print("\n"+fname[:-7])

else:
  with open(dir_name + "/" + fname, "wb") as file:
    pickle.dump(re, file)

