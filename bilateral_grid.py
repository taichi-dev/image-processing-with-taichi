import cv2
import taichi as ti
import taichi.math as tm
import numpy as np

ti.init(arch=ti.gpu, debug=True)

grid = ti.Vector.field(2, dtype=ti.f32, shape=(512, 512, 128))
grid_blurred = ti.Vector.field(2, dtype=ti.f32, shape=(512, 512, 128))
weights = ti.field(dtype=ti.f32, shape=(2, 512), offset=(0, -256))


@ti.func
def compute_weights(i, radius, sigma):
    total = 0.0

    # Not much computation here - serialize the for loop to save two more GPU kernel launch costs
    ti.loop_config(serialize=True)
    for j in range(-radius, radius + 1):
        # Drop the normal distribution constant coefficients since we need to normalize later anyway
        val = ti.exp(-0.5 * (j / sigma)**2)
        weights[i, j] = val
        total += val

    ti.loop_config(serialize=True)
    for j in range(-radius, radius + 1):
        weights[i, j] /= total


@ti.func
def sample_grid_spatial(i, j, k):
    g = ti.static(grid_blurred)  # Create an alias
    mix_i_0 = tm.mix(g[int(i), int(j), k], g[int(i) + 1, int(j), k],
                     tm.fract(i))
    mix_i_1 = tm.mix(g[int(i), int(j) + 1, k], g[int(i) + 1,
                                                 int(j) + 1, k], tm.fract(i))
    return tm.mix(mix_i_0, mix_i_1, tm.fract(j))


@ti.func
def sample_grid(i, j, k):
    return tm.mix(sample_grid_spatial(i, j, int(k)),
                  sample_grid_spatial(i, j,
                                      int(k) + 1), tm.fract(k))


@ti.kernel
def bilateral_filter(img: ti.types.ndarray(), s_s: ti.i32, s_r: ti.i32,
                     sigma_s: ti.f32, sigma_r: ti.f32):
    # Reset the grid
    grid.fill(0)
    grid_blurred.fill(0)
    for i, j in ti.ndrange(img.shape[0], img.shape[1]):
        lum = img[i, j]
        grid[ti.round(i / s_s, ti.i32),
             ti.round(j / s_s, ti.i32),
             ti.round(lum / s_r, ti.i32)] += tm.vec2(lum, 1)

    compute_weights(0, ti.ceil(sigma_s * 3, int), sigma_s)
    compute_weights(1, ti.ceil(sigma_r * 3, int), sigma_r)

    # Grid processing (blur)
    grid_n, grid_m = (img.shape[0] + s_s - 1) // s_s, (img.shape[1] + s_s -
                                                       1) // s_s
    grid_l = (255 + s_r - 1) // s_r
    blur_radius = ti.ceil(sigma_s * 3, int)

    # Since grids store affine attributes, no need to normalize in the following three loops (will normalize in slicing anyway)
    for i, j, k in ti.ndrange(grid_n, grid_m, grid_l):
        l_begin, l_end = max(0, i - blur_radius), min(grid_n,
                                                      i + blur_radius + 1)
        total = tm.vec2(0, 0)
        for l in range(l_begin, l_end):
            total += grid[l, j, k] * weights[0, i - l]

        grid_blurred[i, j, k] = total

    for i, j, k in ti.ndrange(grid_n, grid_m, grid_l):
        l_begin, l_end = max(0, j - blur_radius), min(grid_m,
                                                      j + blur_radius + 1)
        total = tm.vec2(0, 0)
        for l in range(l_begin, l_end):
            total += grid_blurred[i, l, k] * weights[0, j - l]

        grid[i, j, k] = total

    blur_radius = ti.ceil(sigma_r * 3, int)
    for i, j, k in ti.ndrange(grid_n, grid_m, grid_l):
        l_begin, l_end = max(0, k - blur_radius), min(grid_l,
                                                      k + blur_radius + 1)
        total = tm.vec2(0, 0)
        for l in range(l_begin, l_end):
            total += grid[i, j, l] * weights[1, k - l]

        grid_blurred[i, j, k] = total

    # Slicing
    for i, j in ti.ndrange(img.shape[0], img.shape[1]):
        lum = img[i, j]
        sample = sample_grid(i / s_s, j / s_s, lum / s_r)
        img[i, j] = ti.u8(sample[0] / sample[1])


src = cv2.imread('images/mountain.jpg')[:, :].copy()

gui_res = 512
gui = ti.GUI('Bilateral Grid', gui_res)
s_s = gui.slider('s_s', 4, 50)
sigma_s = gui.slider('sigma_s', 0.1, 5)
s_r = gui.slider('s_r', 4, 32)
sigma_r = gui.slider('sigma_r', 0.1, 5)

s_s.value = 16
s_r.value = 16

sigma_s.value = 1
sigma_r.value = 1

while gui.running and not gui.get_event(gui.ESCAPE):
    img = src.copy()
    channels = [img[:, :, c].copy() for c in range(3)]
    for c in range(3):
        bilateral_filter(channels[c], int(s_s.value), int(s_r.value),
                         sigma_s.value, sigma_r.value)
        img[:, :, c] = channels[c]
    img = img.swapaxes(0, 1)[:, ::-1, ::-1]
    img_padded = np.zeros(dtype=np.uint8, shape=(gui_res, gui_res, 3))
    img_padded[:img.shape[0], :img.shape[1]] = img
    gui.set_image(img_padded)
    gui.show()
