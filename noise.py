import matplotlib.pyplot as plt
import numpy as np
import math

np.random.seed(0)

class PerlinNoise:
  def __init__(self, M, N, w, h) -> None:
    self.M = M
    self.N = N
    self.w = w
    self.h = h
    self.R = np.round(7 * np.random.rand(self.N, self.M))
    self.V = np.array([[0, 1], [0, -1], [1, 0], [-1, 0],
                       [1, 1], [-1, -1], [1, -1], [-1, 1]])

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

  def generate(self) -> np.array:
    MW = (self.M - 1) * self.w
    MH = (self.N - 1) * self.h
    x = np.linspace(0, self.M, MW)
    y = np.linspace(0, self.N, MH)
    XX, YY = np.meshgrid(x, y)
    ZZ = np.zeros((MH, MW))
  
    for inX in range(MW):
        for inY in range(MH):
            ZZ[inY, inX] = self.noise(inY, inX, self.M, self.N)
    return ZZ

class RectNoise:
  def __init__(self, genStep : int, zscale : int, mapsizex : int, mapsizey : int, recSizex : int, recSizey : int) -> None:
    self.genStep = genStep
    self.zscale = zscale
    self.mapsizex = mapsizex
    self.mapsizey = mapsizey
    self.recSizex = recSizex
    self.recSizey = recSizey
   
  def generate(self) -> np.array:
    HM = np.zeros((self.mapsizex, self.mapsizey))
    for _ in range(self.genStep):
        x1 = np.random.randint(0, self.mapsizex)
        y1 = np.random.randint(0, self.mapsizey)
        x2 = x1 + self.recSizex // 4 + np.random.randint(0, self.recSizex)
        y2 = y1 + self.recSizey // 4 + np.random.randint(0, self.recSizey)
        
        x2 = min(x2, self.mapsizex)
        y2 = min(y2, self.mapsizey)
        
        for i2 in range(x1, x2):
            for j2 in range(y1, y2):
                HM[i2][j2] += self.zscale / self.genStep + np.random.rand() * 50 / 50.0
    return HM

def merge_noise(arr1: np.array, arr2: np.array) -> np.array:
  return arr1 + arr2
   
def main():
  M = 5
  N = 5
  w = 15
  h = 15

  perlinNoise = PerlinNoise(w, h, M, N)
  noise1 = perlinNoise.generate()
  rectNoise = RectNoise(genStep=1024, zscale=512, mapsizex=M*(w-1), mapsizey=N*(h-1), recSizey=12, recSizex=12)
  noise2 = rectNoise.generate()

  noise3 = merge_noise(noise1, noise2)

  #diamondSquareNoise = DiamondSquare(M*(w-1), N*(h-1), 2)
  #noise1 = diamondSquareNoise.generate()

  plt.imshow(noise1, cmap='gray', origin='upper')
  plt.show()

if __name__ == '__main__':
  main()