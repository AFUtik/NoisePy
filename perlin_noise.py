import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import math

M = 5
N = 5
w = 50
h = 50
R = np.round(3 * np.random.rand(M, N))
V = np.array([[0, 1], [0, -1], [1, 0], [-1, 0]])

class PerlinNoise:
  def __init__(self, x, y, xCount, yCount):
    self.x = x
    self.y = y
    self.xCount = xCount 
    self.yCount = yCount

  def f(self, t):
    return t*t*t * ((6*t-15)*t + 10)
  
  def g(self, a, b, t):
    return a + (b-a) * t

  def noise(self):
    xInd = min(math.floor(self.x / w), self.xCount - 1)
    yInd = min(math.floor(self.y / h), self.yCount - 1)
    if xInd == self.xCount - 1:
      return 0
    if yInd == self.yCount - 1:
      return 0

    dx = self.x - xInd * w
    dy = self.y - yInd * h

    al = self.f(dx / w)
    be = self.f(dy / h)
    b10 = np.array([dx, dy - h])
    b00 = np.array([dx, dy])
    b01 = np.array([dx - w, dy])
    b11 = np.array([dx - w, dy - h])

    e00 = V[int(R[yInd, xInd]), :]
    e01 = V[int(R[yInd, xInd+1]), :]
    e10 = V[int(R[yInd+1, xInd]), :]
    e11 = V[int(R[yInd+1, xInd+1]), :]

    c00 = b00 * e00
    c01 = b01 * e01
    c10 = b10 * e10
    c11 = b11 * e11

    cx1 = self.g(c00, c01, al)
    cx2 = self.g(c10, c11, al)
    result = self.g(cx1, cx2, be)
    return result[1]
  
def main():
  MW = (M - 1) * w
  MH = (N - 1) * h
  x = np.linspace(0, M-1, MW)
  y = np.linspace(0, N-1, MH)
  XX, YY = np.meshgrid(x, y)
  ZZ = np.zeros((MH, MW))
  for inX in range(MW):
    for inY in range(MH):
        noise1 = PerlinNoise(inY, inX, M, N)
        noise_value = noise1.noise()
        ZZ[inY, inX] = noise_value
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