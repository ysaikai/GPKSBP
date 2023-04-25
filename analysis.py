import sys
import numpy as np
import numpy.random as rnd
from scipy.stats import norm
import time
import datetime
from tabulate import tabulate
import pickle
import glob
import matplotlib.pyplot as plt
import multiprocessing as mp


# Parameter
try:
  dir_name = sys.argv[1]
except:
  dir_name = "experiments"
fpaths = glob.glob(dir_name + "/*.pickle")


def main(fpath):
  from library import Expert, weights, kernel

  # MUSIG: [(mu,sig)] from all the experts
  def compute_crps(MUSIG, w, y):
    def psi(mu, sig2):
      sig = np.sqrt(sig2)
      z = mu/sig
      return 2*sig*norm.pdf(z) + mu*(2*norm.cdf(z) - 1)

    i_star = len(MUSIG)
    crps = 0
    for i in range(i_star):
      mu_i, sig_i = MUSIG[i]
      crps += w[i]*psi(y-mu_i, sig_i**2)
      for j in range(i_star):
        mu_j, sig_j = MUSIG[j]
        crps -= 0.5*w[i]*w[j]*psi(mu_i-mu_j, sig_i**2+sig_j**2)
    return crps

  RMSE = []
  NLPD = []
  CRPS = []

  # Load the data, results & parameters
  with open(fpath, "rb") as file:
    re = pickle.load(file)
  method = re["method"]
  seed = str(re["seed"])
  dataset = re["dataset"]
  par_prior = re["par_prior"]
  burnin = re["burnin"]
  GPs = re["GPs"]
  M = len(GPs)
  if method in ["KSBP", "RG"]:
    S = re["S"]
    R = re["R"]
    Beta = re["Beta"]

  # Trainning data
  data = np.genfromtxt(dir_name+"/train"+seed+dataset+".csv",
                       delimiter=",")
  Y0 = data[:,0]
  X0 = data[:,1:]
  N, D = X0.shape
  lb = X0.min(axis=0) # lower bounds
  ub = X0.max(axis=0) # upper bounds
  X = (X0 - lb) / (ub - lb) # normalisation
  Y = (Y0 - Y0.mean()) / Y0.std() # standardisation

  # Test data
  data = np.genfromtxt(dir_name+"/test"+seed+dataset+".csv",
                       delimiter=",")
  XX = (data[:,1:] - lb) / (ub - lb)
  YY = (data[:,0] - Y0.mean()) / Y0.std()
  NN = len(YY)

  # Thinned and every 100 samples kept
  mm = list(range(burnin, M, max(1,int(M/100))))
  for m in mm:
    # Load & re-construct the parameters
    GP = GPs[m]
    for gp in GP:
      gp.X = X
      gp.Y = Y
      gp.update_C()
      gp.update_K()
    if method == "stationary":
      W = np.ones((NN, 1))
    else:
      r = R[m]
      s = S[m]
      if method == "RG":
        beta = Beta[m]
        W = np.zeros((NN, len(GP)))
        for n, x in enumerate(XX):
          for j in range(len(GP)):
            _t = np.sum(kernel(x.reshape((1,-1)), X[s==j], r))
            num = N*_t/np.sum(kernel(x.reshape((1,-1)), X, r))
            W[n,j] = num/(N+beta)
      elif method == "KSBP":
        v = np.array([gp.v for gp in GP])
        h = np.array([gp.h for gp in GP])
        W = weights(v, h, XX, r)

      # Add a new expert for the remaining weight
      ## In RG, the remaining weight = beta/(N+beta)
      GP.append(Expert(len(GP), par_prior, X, Y, s))
      W = np.hstack((W, 1-np.sum(W, axis=1, keepdims=True)))

    se = 0 # squared error
    nlpd = 0
    crps = 0
    for nn in range(NN):
      y = YY[nn]
      x = XX[nn]
      w = W[nn]
      MUSIG = [gp.predict(x) for gp in GP]
      MU = np.array([musig[0] for musig in MUSIG])
      se += (y - w.dot(MU))**2
      pd = [norm.pdf(y, *musig) for musig in MUSIG]
      nlpd += -np.log(w.dot(pd))
      crps += compute_crps(MUSIG, w, y)

    RMSE.append(np.sqrt(se/NN))
    NLPD.append(nlpd/NN)
    CRPS.append(crps/NN)

  return dataset, method, RMSE, NLPD, CRPS


if __name__ == "__main__":
  results = map(main, fpaths)

  RMSE = {}
  NLPD = {}
  CRPS = {}
  datasets = []
  methods = []
  for re in results:
    if re != None:
      datasets.append(re[0])
      methods.append(re[1])
      if (re[0],re[1]) not in RMSE:
        RMSE[(re[0],re[1])] = []
        NLPD[(re[0],re[1])] = []
        CRPS[(re[0],re[1])] = []
      RMSE[(re[0],re[1])].extend(re[2])
      NLPD[(re[0],re[1])].extend(re[3])
      CRPS[(re[0],re[1])].extend(re[4])
  
  # Summary
  datasets = set(datasets)
  methods = set(methods)
  headers = ["", "RMSE", "NLPD", "CRPS"]
  table = []
  for dataset in datasets:
    table.append(["["+dataset+"]"])
    for method in methods:
      rmse = np.mean(RMSE[(dataset,method)]).round(3)
      nlpd = np.mean(NLPD[(dataset,method)]).round(3)
      crps = np.mean(CRPS[(dataset,method)]).round(3)
      row = [method] + [rmse, nlpd, crps]
      table.append(row)
  print(tabulate(table, headers, tablefmt="plain"))
