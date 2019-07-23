import time
from datetime import datetime
import numpy as np
from matplotlib import pyplot as plt
import scipy as sp
import csv

f = []

with open('testdata.csv', 'r') as c:
    reader = csv.reader(c, delimiter=',')
    for row in reader:
        f.append(row)

f = f[1:]
f = f[180:]
t0 = datetime.strptime(f[0][0], '%Y-%m-%d %H:%M:%S')

risersupply = []
riserreturn = []
for i in f:
    t = (datetime.strptime(i[0], '%Y-%m-%d %H:%M:%S') - t0).total_seconds()
    if len(i) == 2:
        risersupply.append([t, float(i[1])])
    if len(i) == 3:
        riserreturn.append([t, float(i[2])])

risersupply = np.asarray(risersupply)
riserreturn = np.asarray(riserreturn)

plt.plot(risersupply[:,0], risersupply[:,1])
plt.plot(riserreturn[:,0], riserreturn[:,1])
plt.legend(["supply", "return"])
plt.show()

supplysample = risersupply[300:400]
returnsample = riserreturn[300:400]

t0_s = supplysample[0,0]
t0_r = returnsample[0,0]

plt.plot(supplysample[:,0], supplysample[:,1])
plt.plot(returnsample[:,0], returnsample[:,1])
plt.legend(["supply", "return"])
plt.show()
