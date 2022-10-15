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


log_luminance_scale = 16


@ti.func
def log_luminance(c):
    lum = 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]
    return max(min((ti.log(lum) / ti.log(2) * log_luminance_scale) + 256, 256),
               0)


img2d = ti.types.ndarray(element_dim=1)


@ti.kernel
def bilateral_filter(img: img2d, s_s: ti.i32, s_r: ti.i32, sigma_s: ti.f32,
                     sigma_r: ti.f32, blend: ti.f32, alpha: ti.f32,
                     beta: ti.f32):
    # Reset the grid
    grid.fill(0)
    grid_blurred.fill(0)

    # min_log_lum, max_log_lum = 1e10, -1e10

    for i, j in ti.ndrange(img.shape[0], img.shape[1]):
        l = log_luminance(img[i, j])
        grid[ti.round(i / s_s, ti.i32),
             ti.round(j / s_s, ti.i32),
             ti.round(l / s_r, ti.i32)] += tm.vec2(l, 1)
        # ti.atomic_min(min_log_lum, l)
        # ti.atomic_max(max_log_lum, l)

    # print(min_log_lum, max_log_lum)

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
        l = log_luminance(img[i, j])
        sample = sample_grid(i / s_s, j / s_s, l / s_r)
        base = sample[0] / sample[1]
        detail = l - base
        final_log_lum = alpha * base + beta + detail

        linear_scale = ti.pow(2, (final_log_lum - l) / log_luminance_scale)

        img[i, j] = tm.mix(img[i, j], img[i, j] * linear_scale, blend)


src = cv2.imread('images/cambridge.png')[:, :].astype(np.float32) / (2**10)
src = src.swapaxes(0, 1)[:, ::-1, ::-1].copy()

gui_res = (src.shape[0] + 200, src.shape[1])
gui = ti.GUI('Fast Bilateral Filtering', gui_res)
s_s = gui.slider('s_s', 4, 50)
sigma_s = gui.slider('sigma_s', 0.1, 5)
s_r = gui.slider('s_r', 4, 32)
sigma_r = gui.slider('sigma_r', 0.1, 5)

exposure = gui.slider('exposure value', -8, 8)
blend = gui.slider('blend', 0, 1)
gamma = gui.slider('gamma', 0.3, 3)

alpha = gui.slider('alpha', 0.1, 2)
beta = gui.slider('beta', -200, 200)

s_s.value = 16
s_r.value = 16

sigma_s.value = 1
sigma_r.value = 1
exposure.value = 1
blend.value = 1
gamma.value = 1

alpha.value = 0.5
beta.value = 125

while gui.running and not gui.get_event(gui.ESCAPE):
    img = src.copy()
    bilateral_filter(img, int(s_s.value), int(s_r.value), sigma_s.value,
                     sigma_r.value, blend.value, alpha.value, beta.value)
    img_padded = np.zeros(dtype=np.float32, shape=(gui_res[0], gui_res[1], 3))
    img_padded[:img.shape[0], :img.shape[1]] = np.minimum(
        1.0, (img * 2**exposure.value)**(1 / gamma.value))
    gui.set_image(img_padded)
    gui.show()
