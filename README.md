# NoisePy
Hello, this project provides 2-Dimensional Perlin Noise with filters for detailed landscape. This noise uses hash to generate values on x and y, so the noise doesn't require providing a width or a height of space,   but you can implement with def `determine(weight, height)` for limited space.


### Example
```py
# TEST PLOT
def main():
  w = 500
  h = 500

  perlinNoise = PerlinNoise(scale=(50, 50))
  noise1 = perlinNoise.determine(w, h)

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
```

Function `main` implies Test plotting of the noise. Parameters `w` and `h` are amounts of points on a discrete plane.    
All filters can be adjusted in initialization of Noise's Object or after initialization of the object.


## Installation Of Dependencies
```pip install -r requirements.txt```


## How many variations does Perlin Noise have?
If you're worried about how unique the noise is, it has `256!` variations (507 decimal digits). It's derived from pseudo-random byte array `p` that's shuffled and has size of 256.
