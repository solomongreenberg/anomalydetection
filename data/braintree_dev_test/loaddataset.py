#!/usr/bin/env python3

import numpy as np
import scipy as sp
from glob import glob
from collections import namedtuple
from tqdm import tqdm
from datetime import datetime

files = glob('s3/*.csv')

text_lines = []

for f in files:
    with open(f, 'r') as fl:
        text_lines.extend([i.strip('\"').strip('\'') for i in fl.readlines() if "deviceid" not in i])

Point = namedtuple('Point', ['timestamp', 'tempf'])
data = {}

for l in tqdm(text_lines):
    deviceid, timestamp, tempf = [l.split(',')[i] for i in [0, 1, 2]]
    if deviceid not in data.keys():
        data[deviceid] = []

    if timestamp.strip("\"") != '' and tempf.strip("\"") != '':
        data[deviceid].append(Point(timestamp.strip("\""), tempf.strip("\"")))


def parsetime(t):
    return datetime.strptime(t, "%Y-%m-%d %H:%M:%S.%f")


for key in tqdm(data.keys()):
    data[key] = sorted(data[key], key=lambda p: parsetime(p.timestamp))
    ts = parsetime(data[key][0].timestamp)
    for entry_idx in range(len(data[key])):
        data[key][entry_idx] = Point(
            timestamp=(parsetime(data[key][entry_idx].timestamp) - ts).total_seconds(),
            tempf=float(data[key][entry_idx].tempf)
        )
