import matplotlib.pyplot as plt
import numpy as np
import time as time

from numba import njit

@njit
def fade(t) -> float:
  return t * t * t * ((6 * t - 15) * t + 10)

@njit
def lerp(a, b, t) -> float:
  return a + (b - a) * t

@njit
def grad(hash_val, x, y):
  h = hash_val & 15
  u = x if h < 8 else y
  if h < 4:
    v = y
  elif h == 12 or h == 14:
    v = x
  else:
    v = 0
  res = (u if (h & 1) == 0 else -u) + (v if (h & 2) == 0 else -v)
  return res

@njit(cache=True)
def noise(x, y, p: np.ndarray):
  xi = int(np.floor(x)) & 255
  yi = int(np.floor(y)) & 255
  xf = x - np.floor(x)
  yf = y - np.floor(y)

  u = fade(xf)
  v = fade(yf)

  aa = p[p[xi] + yi]
  ab = p[p[xi] + yi + 1]
  ba = p[p[xi + 1] + yi]
  bb = p[p[xi + 1] + yi + 1]

  x1 = lerp(grad(aa, xf, yf),     grad(ba, xf - 1, yf), u)
  x2 = lerp(grad(ab, xf, yf - 1), grad(bb, xf - 1, yf - 1), u)
  return lerp(x1, x2, v)

class PerlinNoise:
  def __init__(self,
               seed=0,
               scale =(1, 1),
               offset=(0, 0),
               octaves=4,
               persistence=0.5,
               lacunarity=2,
               base_freq=1.0)-> None:
    self.octaves = octaves
    self.persistence = persistence
    self.lacunarity = lacunarity
    self.base_freq = base_freq

    self.offset = offset
    self.scale = scale

    p = np.arange(256, dtype=np.int32)
    rng = np.random.default_rng(seed)
    rng.shuffle(p)
    p = np.concatenate((p, p))
    self.p = p

  def detail(self, x, y) -> float:
    total = 0.0
    max_amplitude = 0
    amplitude = 1.0
    frequency = self.base_freq
    for _ in range(self.octaves):
      total += amplitude * noise((x+self.offset[0]) / self.scale[0] * frequency,
                                 (y+self.offset[1]) / self.scale[1] * frequency, self.p)
      max_amplitude += amplitude
      amplitude *= self.persistence
      frequency *= self.lacunarity
    return total / max_amplitude

  def determine(self, w, h) -> np.array:
    zz = np.zeros((w, h))
    for inX in range(w):
        for inY in range(h):
          zz[inY, inX] = self.detail(inX, inY)
    return zz

# TEST PLOT
def main():
  w = 500
  h = 500

  start = start = time.perf_counter()
  perlinNoise = PerlinNoise(scale=(50, 50))
  noise1 = perlinNoise.determine(w, h)

  end = time.perf_counter()
  print(f"Time of execution: {end - start:.4f} seconds")

  Z = noise1
  X = np.arange(Z.shape[1])
  Y = np.arange(Z.shape[0])
  X, Y = np.meshgrid(X, Y)

  fig = plt.figure(figsize=(10, 6))
  ax = fig.add_subplot(111, projection='3d')

  ax.plot_surface(X, Y, Z, cmap='terrain')
  ax.view_init(elev=30, azim=45)

  ax.set_title("3D Perlin Noise Surface")
  ax.set_xlabel("X")
  ax.set_ylabel("Y")
  ax.set_zlabel("Height")
  ax.set_zlim(0, 4)
  plt.show()

if __name__ == '__main__':
  main()