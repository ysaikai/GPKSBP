import os
import csv
import numpy as np
import numpy.random as rnd
import pandas as pd
import time
import datetime


if not os.path.isdir("experiments"):
  os.makedirs("experiments")

N = 30 # number of training samples
M = 20000
t = int(datetime.datetime.utcnow().timestamp()*100)
seeds = np.arange(0,30)
datasets = ["Borehole", "detpep108d", "detpep10exp",
            "franke2d", "GL2009"]
methods = ["GPKSBP","RG"]

print("N={}".format(N))
print(seeds)
print(datasets)

time_s = time.time()
for seed in seeds:
  for dataset in datasets:
    np.random.seed(seed)
    id = str(seed)+dataset

    # Data
    if dataset == "Borehole":
      lb_borehole = np.array([0.05,100,63070,63.1,990,700,1120,9855])
      ub_borehole = np.array([0.15,50000,115600,116,1110,820,1680,12045])

      def borehole(rw, r, Tu, Tl, Hu, Hl, L, Kw):
        numerator = 2*np.pi*Tu*(Hu - Hl)
        _t = 2*L*Tu/(np.log(r/rw)*rw**2*Kw)
        denominator = np.log(r/rw)*(1 + _t + Tu/Tl)
        return numerator/denominator

      D = len(lb_borehole)
      # Test set
      X0 = rnd.uniform(lb_borehole, ub_borehole, (300,D))
      Y0 = borehole(X0[:,0],X0[:,1],X0[:,2],X0[:,3],
                    X0[:,4],X0[:,5],X0[:,6],X0[:,7])
      df_test = pd.DataFrame(np.hstack([Y0.reshape(-1,1), X0]))

      # Training set
      X0 = rnd.uniform(lb_borehole, ub_borehole, (N,D))
      Y0 = borehole(X0[:,0],X0[:,1],X0[:,2],X0[:,3],
                    X0[:,4],X0[:,5],X0[:,6],X0[:,7])
      df_train = pd.DataFrame(np.hstack([Y0.reshape(-1,1), X0]))
      noise = False

    elif dataset == "detpep10exp":
      def detpep10exp(X):
        t1 = np.exp(-2/X[:,0]**1.75)
        t2 = np.exp(-2/X[:,1]**1.5)
        t3 = np.exp(-2/X[:,2]**1.25)
        return 100*(t1 + t2 + t3)

      D = 3
      # Test set
      X0 = rnd.random((300,D))
      Y0 = detpep10exp(X0)
      df_test = pd.DataFrame(np.hstack([Y0.reshape(-1,1), X0]))

      # Training set
      X0 = rnd.random((N,D))
      Y0 = detpep10exp(X0)
      df_train = pd.DataFrame(np.hstack([Y0.reshape(-1,1), X0]))
      noise = False

    elif dataset == "detpep108d":
      def detpep108d(x):
        term1 = 4*(x[1] - 2 + 8*x[2] - 8*x[2]**2)**2
        term2 = (3 - 4*x[2])**2
        term3 = 16*np.sqrt(x[3]+1)*(2*x[3]-1)**2
        term4 = 0
        for i in range(4,9):
          term4 += i*np.log(1 + np.sum(x[2:i]))
        return term1 + term2 + term3 + term4

      D = 8
      # Test set
      X0 = rnd.random((300,D))
      Y0 = np.empty(X0.shape[0])
      for i, x in enumerate(X0):
        Y0[i] = detpep108d(x)
      df_test = pd.DataFrame(np.hstack([Y0.reshape(-1,1), X0]))

      # Training set
      X0 = rnd.random((N,D))
      Y0 = np.empty(X0.shape[0])
      for i, x in enumerate(X0):
        Y0[i] = detpep108d(x)
      df_train = pd.DataFrame(np.hstack([Y0.reshape(-1,1), X0]))
      noise = False

    elif dataset == "franke2d":
      def franke2d(x):
        t1 = 0.75*np.exp(-(9*x[0]-2)**2/4 - (9*x[1]-2)**2/4)
        t2 = 0.75*np.exp(-(9*x[0]+1)**2/49 - (9*x[1]+1)/10)
        t3 = 0.5*np.exp(-(9*x[0]-7)**2/4 - (9*x[1]-3)**2/4)
        t4 = 0.2*np.exp(-(9*x[0]-4)**2 - (9*x[1]-7)**2)
        return(t1 + t2 + t3 - t4)

      D = 2
      # Test set
      X0 = rnd.random((300,D))
      Y0 = franke2d(X0.T)
      df_test = pd.DataFrame(np.hstack([Y0.reshape(-1,1), X0]))

      # Training set
      X0 = rnd.random((N,D))
      Y0 = franke2d(X0.T)
      df_train = pd.DataFrame(np.hstack([Y0.reshape(-1,1), X0]))
      noise = False

    elif dataset == "GL2009":
      def gl2009(x):
        re = np.exp(np.sin((0.9*(x[0]+0.48))**10))
        re += x[1]*x[2] + x[3] + rnd.normal(0,0.05)
        return re
      
      D = 6
      # Test set
      X0 = rnd.random((300,D))
      Y0 = gl2009(X0.T)
      df_test = pd.DataFrame(np.hstack([Y0.reshape(-1,1), X0]))

      # Training set
      X0 = rnd.random((N,D))
      Y0 = gl2009(X0.T)
      df_train = pd.DataFrame(np.hstack([Y0.reshape(-1,1), X0]))
      noise = True

    df_test.to_csv("experiments/test"+id+".csv",
                   header=False, index=False)
    df_train.to_csv("experiments/train"+id+".csv",
                    header=False, index=False)

    with open("experiments/par"+id+".csv", "w", newline="") as file:
      writer = csv.writer(file)
      writer.writerow(["M", M])
      writer.writerow(["seed", seed])
      writer.writerow(["dataset", dataset])
      writer.writerow(["noise", noise])
      
    for method in methods:
      # Run
      cmd = "python {}.py {} {}".format(method, seed, dataset)
      os.system(cmd)

print("\nIt took {:.1f} sec.".format(time.time()-time_s))
