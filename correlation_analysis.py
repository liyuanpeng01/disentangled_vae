# -*- coding: utf-8 -*-
import sys

m = {}
for line in sys.stdin:
  k, v = line.split()
  ki = int(k)
  vf = float(v)
  if ki not in m:
    m[ki] = []
  m[ki].append(vf)

result = []
for k, vlist in m.items():
  vlist.sort()
  size = len(vlist)
  low = vlist[int(round(0.05 * size))]
  open = vlist[int(round(0.25 * size))]
  close = vlist[int(round(0.75 * size))]
  high = vlist[int(round(0.95 * size))]
  avg = sum(vlist) / size
  result.append((k, (low, open, close, high, avg)))
result.sort()
for r in result:
  print r[0], r[1][0], r[1][1], r[1][2], r[1][3], r[1][4]