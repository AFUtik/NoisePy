import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import math

M = 5
N = 5
w = 10
h = 10
R = np.round(3 * np.random.rand(N, M))
V = np.array([[0, 1], [0, -1], [1, 0], [-1, 0]])

def f(t):
  return t*t*t*((6*t - 15)*t + 10)
  
def g(a, b, t):
  return a + (b-a) * t
  
def noise(x, y, xCount, yCount):
  xInd = min(math.floor(x / w), xCount - 1)
  yInd = min(math.floor(y / h), yCount - 1)
  if xInd == xCount - 1:
    return 0
  if yInd == yCount - 1:
    return 0
  
  xl = x - xInd * w
  yl = y - yInd * h
  al = f(xl / w)
  be = f(yl / h)

  b10 = np.array([ xl, yl - h ])
  b11 = np.array([ xl - w, yl - h ])
  b00 = np.array([ xl, yl ])
  b01 = np.array([ xl - w, yl ])

  e00 = V[int(R[yInd, xInd])]
  e01 = V[int(R[yInd, xInd+1])]
  e10 = V[int(R[yInd+1, xInd])]
  e11 = V[int(R[yInd+1, xInd+1])]
  
  c00 = np.dot(b00, e00)
  c01 = np.dot(b01, e01)
  c10 = np.dot(b10, e10)
  c11 = np.dot(b11, e11)
  
  cx1 = g(c00, c01, al)
  cx2 = g(c10, c11, al)
  result = g(cx1, cx2, be)
  return result
  
def main():
  MW = (M - 1) * w
  MH = (N - 1) * h
  x = np.linspace(0, 10, MW)
  y = np.linspace(0, 10, MH)
  XX, YY = np.meshgrid(x, y)
  ZZ = np.zeros((MH, MW))
  for inX in range(MW):
    for inY in range(MH):
        ZZ[inY, inX] = noise(inY, inX, M, N)
  fig = plt.figure(figsize=(8, 8))
  ax = fig.add_subplot(projection="3d")
  ax.plot_surface(YY, XX, ZZ, cmap=cm.inferno)
  ax.set_zlim(-80, 80)
  ax.set_xlabel('x')
  ax.set_ylabel('y')
  ax.set_zlabel('z')
  plt.show()

if __name__ == '__main__':
  main()