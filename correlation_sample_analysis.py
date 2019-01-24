# -*- coding: utf-8 -*-
import sys

vt = int(sys.argv[1])
dv = [1, 3, 20, 16, 16]

def get_index(shape, scale, rotate, x, y):
  r = 0
  r += shape * (6 * 40 * 32 * 32)
  r += scale * (40 * 32 * 32)
  r += rotate * (32 * 32)
  r += x * 32
  r += y
  return r

values = []
for line in sys.stdin:
  k, v = line.split()
  ki = int(k)
  vf = float(v)
  values.append(vf)

result = []
if vt == 1:
  for i in range(6):
    result.append(get_index(dv[0], i, dv[2], dv[3], dv[4]))
elif vt == 2:
  for i in range(40):
    result.append(get_index(dv[0], dv[1], i, dv[3], dv[4]))
elif vt == 3:
  for i in range(32):
    result.append(get_index(dv[0], dv[1], dv[2], i, dv[4]))
elif vt == 4:
  for i in range(32):
    result.append(get_index(dv[0], dv[1], dv[2], dv[3], i))
else:
  raise ValueError("Undefined value type " + str(vt))

for r in result:
  print values[r]
