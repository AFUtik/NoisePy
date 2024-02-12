import numpy as np

def dot(g, x, y):
    return g[0] * x + g[1] * y

def perlin(x, y):
    # Векторы смещения
    vectors = np.array([[1, 1], [-1, 1], [1, -1], [-1, -1],
                        [1, 0], [-1, 0], [0, 1], [0, -1]])
    gradient_idx = np.random.permutation(8)

    # Определение координат узлов сетки
    x0 = int(x)
    y0 = int(y)
    x1 = x0 + 1
    y1 = y0 + 1

    # Вычисление весов для каждого узла
    sx = x - x0
    sy = y - y0

    # Градиенты в узлах сетки
    n00 = dot(vectors[gradient_idx[(x0 + y0) % 8]], sx, sy)
    n10 = dot(vectors[gradient_idx[(x1 + y0) % 8]], sx - 1, sy)
    n01 = dot(vectors[gradient_idx[(x0 + y1) % 8]], sx, sy - 1)
    n11 = dot(vectors[gradient_idx[(x1 + y1) % 8]], sx - 1, sy - 1)

    # Интерполяция по x
    wx = (3 - 2 * sx) * sx * sx
    n0 = n00 + wx * (n10 - n00)
    n1 = n01 + wx * (n11 - n01)

    # Интерполяция по y и возврат окончательного значения
    wy = (3 - 2 * sy) * sy * sy
    return (n0 + wy * (n1 - n0)) / 2

# Пример использования
width = 200
height = 200
scale = 1.5
noise_map = np.zeros((width, height))

for y in range(height):
    for x in range(width):
        noise_map[x][y] = perlin(x * scale, y * scale)

# Визуализация шума
import matplotlib.pyplot as plt

plt.imshow(noise_map, cmap='gray', interpolation='nearest')
plt.colorbar()
plt.show()