from functools import total_ordering
import numpy as np
import itertools
import math
from scipy.stats import qmc
from numpy import *
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
from copy import deepcopy
import matplotlib.pyplot as plt
import tensorflow as tf
from matplotlib.axes import Axes
from matplotlib.cm import coolwarm
import gpflow


def form_sample(lines, dim):
  test_data_size = 0
  x1_test, x2_test, x3_test, x4_test, x5_test, x6_test, x7_test, x8_test, x9_test, x10_test, x11_test, x12_test, x13_test, x14_test, x15_test, x16_test, x17_test, f1_test, f2_test = (np.zeros(test_data_size) for i in range(19))
  test_indices = random.sample(range(720), 0)
  x1, x2, x3, x4, x5,x6, x7, x8, x9, x10,x11, x12, x13, x14, x15, x16, x17, f1,f2 = (np.zeros(len(lines)-test_data_size) for i in range(19))
  k=0
  i=0
  j=0
  for line in lines:
    if(i not in test_indices):
      x1[j] = (line.split()[2])
      x2[j] = (line.split()[3])
      x3[j] = (line.split()[4])
      x4[j] = (line.split()[5])
      x5[j] = (line.split()[6])
      x6[j] = (line.split()[7])
      x7[j] = (line.split()[8])
      x8[j] = (line.split()[9])
      x9[j] = (line.split()[10])
      x10[j] = (line.split()[11])
      x11[j] = (line.split()[12])
      x12[j] = (line.split()[13])
      x13[j] = (line.split()[14])
      x14[j] = (line.split()[15])
      x15[j] = (line.split()[16])
      x16[j] = (line.split()[17])
      x17[j] = (line.split()[18])
      f1[j] = (line.split()[19])
      f2[j] = (line.split()[20])
      j=j+1
    else:
      x1_test[k] = (line.split()[2])
      x2_test[k] = (line.split()[3])
      x3_test[k] = (line.split()[4])
      x4_test[k] = (line.split()[5])
      x5_test[k] = (line.split()[6])
      x6_test[k] = (line.split()[7])
      x7_test[k] = (line.split()[8])
      x8_test[k] = (line.split()[9])
      x9_test[k] = (line.split()[10])
      x10_test[k] = (line.split()[11])
      x11_test[k] = (line.split()[12])
      x12_test[k] = (line.split()[13])
      x13_test[k] = (line.split()[14])
      x14_test[k] = (line.split()[15])
      x15_test[k] = (line.split()[16])
      x16_test[k] = (line.split()[17])
      x17_test[k] = (line.split()[18])
      f1_test[k] = (line.split()[19])
      f2_test[k] = (line.split()[20])
      k=k+1
    i=i+1
  obj1 = np.array(f1)
  obj1 = [float(numeric_string) for numeric_string in obj1]
  obj2 = np.array(f2)
  obj2 = [float(numeric_string) for numeric_string in obj2]
  x1 = np.array(x1)
  x2 = np.array(x2)
  x3 = np.array(x3)
  x4 = np.array(x4)
  x5 = np.array(x5)
  x6 = np.array(x6)
  x7 = np.array(x7)
  x8 = np.array(x8)
  x9 = np.array(x9)
  x10 = np.array(x10)
  x11 = np.array(x11)
  x12 = np.array(x12)
  x13 = np.array(x13)
  x14 = np.array(x14)
  x15 = np.array(x15)
  x16 = np.array(x16)
  x17 = np.array(x17)
  sample  = np.ndarray((len(lines)-test_data_size, dim))
  for i in range(len(lines)-test_data_size):
    sample[i,0]= x1[i]
    sample[i,1]= x2[i]
    sample[i,2]= x3[i]
    sample[i,3]= x4[i]
    sample[i,4]= x5[i]
    sample[i,5]= x6[i]
    sample[i,6]= x7[i]
    sample[i,7]= x8[i]
    sample[i,8]= x9[i]
    sample[i,9]= x10[i]
    sample[i,10]= x11[i]
    sample[i,11]= x12[i]
    sample[i,12]= x13[i]
    sample[i,13]= x14[i]
    sample[i,14]= x15[i]
    sample[i,15]= x16[i]
    sample[i,16]= x17[i]
  sample_test  = np.ndarray((test_data_size, dim))
  for i in range(test_data_size):
    sample_test[i,0]= x1_test[i]
    sample_test[i,1]= x2_test[i]
    sample_test[i,2]= x3_test[i]
    sample_test[i,3]= x4_test[i]
    sample_test[i,4]= x5_test[i]
    sample_test[i,5]= x6_test[i]
    sample_test[i,6]= x7_test[i]
    sample_test[i,7]= x8_test[i]
    sample_test[i,8]= x9_test[i]
    sample_test[i,9]= x10_test[i]
    sample_test[i,10]= x11_test[i]
    sample_test[i,11]= x12_test[i]
    sample_test[i,12]= x13_test[i]
    sample_test[i,13]= x14_test[i]
    sample_test[i,14]= x15_test[i]
    sample_test[i,15]= x16_test[i]
    sample_test[i,16]= x17_test[i]

  return sample, sample_test,obj1, obj2, f1_test, f2_test

def scale(sample, sample_test, dim, l_bounds, u_bounds):
  for i in range(720):
    for j in range(dim):
      t = sample[i][j]
      sample[i][j] = (t-l_bounds[j])/(u_bounds[j]-l_bounds[j])
  for i in range(test_data_size):
    for j in range(dim):
      if(len(sample_test)>0):
          t = sample_test[i][j]
          sample_test[i][j] = (t-l_bounds[j])/(u_bounds[j]-l_bounds[j])
  return sample, sample_test
dim=17
test_data_size = 50
#l_bounds = [10,2.0e3,2.2e3,400000,6000000,0.3414,3.5e-4,0.0,-3.0e-4,0.854,700.0,0.180,1229.0,1513,3233, 2500, 2200, 2000000000]
#u_bounds = [150,3.0e3,3.2e3,600000,10000000,0.3900,4.0e-4,1.7e-5,-2.0e-4,2.854,800.0,0.300,1429.0,1613,3533,2900, 2700, 3500000000]
l_bounds = [24.64,1002.4, 516.5, 400000,6000000 ,0.3414,3.5e-4 ,0.0,0.05912358,618.0,1513,3233,2.0e9,-0.007739495,2800.1,2200,2600]
u_bounds = [36.96,1503.6,574.7,600000,10000000,0.3900,4.0e-4, 1.7e-5,0.06323221,619.0,1613,3533,3.5e9,-0.007223529,3000.1,2600,3000]
with open("720-N-samples.dat", 'r') as pointsFile:
  lines = pointsFile.read().splitlines()
sample, sample_test, obj1, obj2, f1_test, f2_test = form_sample(lines,dim)
sample_s, sample_test_s = scale(sample, sample_test, dim, l_bounds, u_bounds)
sample_s = np.array(sample_s)
obj1 = np.array(obj1)
obj2 = np.array(obj2)

sample_test_s = np.array(sample_test_s)
data = (sample_s.reshape(-1, 17), obj2.reshape(-1,1))
