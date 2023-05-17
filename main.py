import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import math

class NoiseGenerator:
  def __init__(self, M, N, w, h):
    self.M = M
    self.N = N
    self.w = w
    self.h = h
    self.R = np.round(3 * np.random.rand(self.N, self.M))
    self.V = np.array([[0, 1], [0, -1], [1, 0], [-1, 0]])

  def f(self, t):
    return t * t * t * ((6 * t - 15) * t + 10)

  def g(self, a, b, t):
    return a + (b - a) * t

  def noise(self, x, y, xCount, yCount):
    xInd = min(math.floor(x / self.w), xCount - 1)
    yInd = min(math.floor(y / self.h), yCount - 1)
    if xInd == xCount - 1 or yInd == yCount - 1:
        return 0

    xl = x - xInd * self.w
    yl = y - yInd * self.h
    al = self.f(xl / self.w)
    be = self.f(yl / self.h)
    
    b10 = np.array([xl, yl - self.h])
    b11 = np.array([xl - self.w, yl - self.h])
    b00 = np.array([xl, yl])
    b01 = np.array([xl - self.w, yl])

    e00 = self.V[int(self.R[yInd, xInd])]
    e01 = self.V[int(self.R[yInd, xInd + 1])]
    e10 = self.V[int(self.R[yInd + 1, xInd])]
    e11 = self.V[int(self.R[yInd + 1, xInd + 1])]

    c00 = np.dot(b00, e00)
    c01 = np.dot(b01, e01)
    c10 = np.dot(b10, e10)
    c11 = np.dot(b11, e11)

    cx1 = self.g(c00, c01, al)
    cx2 = self.g(c10, c11, al)
    result = self.g(cx1, cx2, be)
    return result

  def generate_noise_map(self):
    MW = (self.M - 1) * self.w
    MH = (self.N - 1) * self.h
    x = np.linspace(0, self.M, MW)
    y = np.linspace(0, self.N, MH)
    XX, YY = np.meshgrid(x, y)
    ZZ = np.zeros((MH, MW))
    for inX in range(MW):
        for inY in range(MH):
            ZZ[inY, inX] = self.noise(inY, inX, self.M, self.N)
    return XX, YY, ZZ

def main():
  M = 5
  N = 5
  w = 10
  h = 10

  noise1 = NoiseGenerator(w, h, M, N)
  XX, YY, ZZ = noise1.generate_noise_map()
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