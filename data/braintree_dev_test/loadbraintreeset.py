#!/usr/bin/env python3

from matplotlib import pyplot as plt
import numpy as np
import scipy as sp
from glob import glob
from collections import namedtuple
from tqdm import tqdm
from datetime import datetime
from scipy import interpolate
import h5py as h5

supply_set = []
return_set = []


with open('braintree/00-0b-57-00-00-2d-e2-51.csv', 'r') as fl_supply:
    supply_set = [i.replace("\"",'').replace('\n','') for i in fl_supply.readlines()][1:]

with open('braintree/00-0b-57-00-00-2d-e2-cc.csv', 'r') as fl_return:
    return_set = [i.replace("\"",'').replace('\n','') for i in fl_return.readlines()][1:]

def parsetime(t):
    return datetime.strptime(t, "%Y-%m-%dT%H:%M:%S.%fZ")

supply_set = [[parsetime(i.split(',')[1]), float(i.split(',')[2])] for i in supply_set if i.split(',')[2] != '']
return_set = [[parsetime(i.split(',')[1]), float(i.split(',')[2])] for i in return_set if i.split(',')[2] != '']


supply_set = sorted(supply_set, key=lambda t: t[0])
return_set = sorted(return_set, key=lambda t: t[0])

t0 = min(supply_set[0][0], return_set[0][0])

for i in range(len(supply_set)):
    supply_set[i][0] = (supply_set[i][0] - t0).total_seconds()
for i in range(len(return_set)):
    return_set[i][0] = (return_set[i][0] - t0).total_seconds()

x_min = 1.25e6

supply_set = [i for i in supply_set if i[0] > x_min]
return_set = [i for i in return_set if i[0] > x_min]

for i in range(len(supply_set)):
    supply_set[i][0] = supply_set[i][0] - x_min
for i in range(len(return_set)):
    return_set[i][0] = return_set[i][0] - x_min

supply_set = np.asarray(supply_set)
return_set = np.asarray(return_set)

dx_s = np.gradient(supply_set[:,0])
dx_r = np.gradient(return_set[:,0])

interp_kind = 'cubic'
supply_interp = sp.interpolate.interp1d(supply_set[:,0], supply_set[:,1], kind=interp_kind)
return_interp = sp.interpolate.interp1d(return_set[:,0], return_set[:,1], kind=interp_kind)

x_end = min(supply_set[-1][0], return_set[-1][0])

sample_ts = 60 # 1 sample/minute - nyquist frequency and all that ;)
sample_num_pts = int(x_end//sample_ts)
sample_pts = np.linspace(sample_ts, sample_ts*sample_num_pts, sample_num_pts)

supply_interpolated = np.vstack([sample_pts, supply_interp(sample_pts)]).T
return_interpolated = np.vstack([sample_pts, return_interp(sample_pts)]).T

#plt.plot(supply_set[:,0], supply_set[:,1])
#plt.plot(return_set[:,0], return_set[:,1])
#plt.legend(["Supply", "Return"])
#plt.xlabel("Time (s)")
#plt.ylabel("Temp (°F)")
#plt.show()

#plt.plot(supply_set[1000:2000,0], supply_set[1000:2000,1])
#plt.plot(return_set[1000:2000,0], return_set[1000:2000,1])
#supply_start = int(supply_set[1000][0]//sample_ts)
#supply_end = int(supply_set[2000][0]//sample_ts)
#return_start = int(return_set[1000][0]//sample_ts)
#return_end = int(return_set[2000][0]//sample_ts)
#plt.plot(supply_interpolated[supply_start:supply_end,0], supply_interpolated[supply_start:supply_end,1])
#plt.plot(return_interpolated[return_start:return_end,0], return_interpolated[return_start:return_end,1])
#plt.legend(["Supply real", "Return real", "Supply interp", "Return interp"])
#plt.xlabel("Time (s)")
#plt.ylabel("Temp (°F)")
#plt.show()

with open("01riser_supply_interp.pkl", 'wb') as f:
    pkl.dump(supply_interpolated, f)
with open("01riser_return_interp.pkl", 'wb') as f:
    pkl.dump(return_interpolated, f)
