"""
A version of Rasmussen & Ghahramani (2001)
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
from library import (Expert, expert_stats, kernels,
                     update_step, hmc, hmc_rRG)


# Parameters
try:
  seed = int(float(sys.argv[1]))
  dataset = sys.argv[2]
  id = str(seed)+dataset
  experiment = True
except:
  experiment = False

method = "RG"
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
par_prior = {"beta": [2., 1.],
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
n_aux = 1 # number of auxiliaries in Neal's 8 algorithm


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
beta = par_prior["beta"][0]*par_prior["beta"][1] # prior mean
Beta = np.zeros(M)
r = par_prior["r"][0]*par_prior["r"][1] # prior mean
R = np.full(M, r)
GP = [Expert(0, par_prior, X, Y, s, noise)]
GPs = []
K = kernels(X, r) # Kernel matrix
## for HMC_r
step_r = par_hmc["step_min"]
lstep_bar_r = 0
H_bar_r = 0
acc_r = 0


msg = "{}, {}, N={}, {}, M={}"
print(msg.format(seed, dataset, N, method, M))
## Posterior sampling
for m in range(M):
  # progress output
  if (m+1)%int(M/5) == 0:
    print("{:5d}/{}".format(m+1, M))
    if experiment == False:
      print(np.asarray(np.unique(s, return_counts=True))[:,:10])

  # r
  r, acc_r = hmc_rRG(r, X, beta, s, par_prior,
                     step_r, par_hmc["n_steps"])
  if m < burnin and m >= par_hmc["m_min"]:
    _a, _b, _c = update_step(m, acc_r, lstep_bar_r,
                             H_bar_r, par_hmc)
    step_r, lstep_bar_r, H_bar_r = (_a, _b, _c)
  K = kernels(X, r) # Update the kernel matrix

  # beta
  p1, p2 = par_prior["beta"]
  t1 = len(np.unique(s)) # number of active experts
  t2 = rnd.beta(beta+1, N) # auxiliary
  t3 = (p1 + t1 - 1)/(N*(1/p2 - np.log(t2))) # odds
  ## n.b. 1/p2-np.log(t2) is for gamma's rate parameter
  if rnd.rand() < t3/(1+t3):
    beta = rnd.gamma(p1+t1, 1/(1/p2-np.log(t2)))
  else:
    beta = rnd.gamma(p1+t1-1, 1/(1/p2-np.log(t2)))

  # sig2, l, & sig2n
  for i in range(len(GP)):
    hmc(i, GP, par_prior, par_hmc["n_steps"], noise)
    if m < burnin and m >= par_hmc["m_min"]:
      _a, _b, _c = update_step(m, GP[i].acc, GP[i].lstep_bar,
                               GP[i].H_bar, par_hmc)
      GP[i].step, GP[i].lstep_bar, GP[i].H_bar = (_a, _b, _c)

  # s
  for n in range(N):
    # N-1+beta is dropped as the common denominator
    gmtrick = dict()
    idx = np.delete(range(N), n)
    _s = np.delete(s, n)
    unique_s = np.unique(_s)
    n_s = len(unique_s) # number of unique s in _s

    # GPs with some other members
    ## (n may or may not belong to one of these GPs)
    for i in unique_s:
      mu, sig = GP[i].predict(X[n], n)
      lp = -np.log(sig) - ((Y[n]-mu)/sig)**2/2
      _t = np.sum(K[n,idx[_s==i]])
      lp += np.log(N-1) + np.log(_t) - np.log(np.sum(K[n,idx]))
      gmtrick[i] = -np.log(-np.log(rnd.rand())) + lp

    # GP without any other member
    ## (applicable only if n belongs to a GP with no other member)
    if s[n] not in unique_s:
      i = s[n]
      sig = np.sqrt(GP[i].sig2 + GP[i].sig2n)
      lp = -np.log(sig) - ((Y[n]-0)/sig)**2/2
      lp += np.log(beta/n_aux)
      gmtrick[i] = -np.log(-np.log(rnd.rand())) + lp
    
    # New GPs (extra n_aux or n_aux-1 GPs)
    for i in range(len(gmtrick), n_s+n_aux):
      GP.append(Expert(i, par_prior, X, Y, s, noise))
      sig = np.sqrt(GP[i].sig2 + GP[i].sig2n)
      lp = -np.log(sig) - ((Y[n]-0)/sig)**2/2
      lp += np.log(beta/n_aux)
      gmtrick[i] = -np.log(-np.log(rnd.rand())) + lp

    i = max(gmtrick, key=gmtrick.get)
    # If the assignment changes, update GP[s[n]] and GP[i]
    if i != s[n]:
      GP[s[n]].downdate_K1(n) # Remove n
      GP[i].update_K1(n) # Add n
      s[n] = i
    
    # Delete empty GPs
    for i in range(len(GP)-1, -1, -1):
      if len(GP[i].members) == 0:
        del GP[i]
    
    # Re-label the experts id numbers to remove gaps
    for i in range(len(GP)):
      if i != GP[i].id:
        s[s==GP[i].id] = i
        GP[i].id = i

  # End of the round
  R[m] = r
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
      "Beta": Beta,
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
  ## beta (%Acpt = 1 as sampled till accepted)
  _m = np.array([Beta[m] for m in mm]).mean()
  _s = np.array([Beta[m] for m in mm]).std()
  row = ["beta"] + list(np.round((_m, _s), 2)) + ["n/a"]
  table.append(row)
  print(tabulate(table, headers, tablefmt="plain"))

  ## Expert #0-2
  for expert in range(5):
    expert_stats(expert, GPs, M, S, burnin, True)

  with open(fname, "wb") as file:
    pickle.dump(re, file)
  print("\n"+fname[:-7])

else:
  with open(dir_name+"/" + fname, "wb") as file:
    pickle.dump(re, file)
