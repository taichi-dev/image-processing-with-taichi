import cv2
import taichi as ti
import taichi.math as tm

ti.init(arch=ti.gpu, debug=True)

grid = ti.Vector.field(2, dtype=ti.f32, shape=(512, 512, 128))
# TODO: rename
grid_blurred = ti.Vector.field(2, dtype=ti.f32, shape=(512, 512, 128))
s_s, s_r = 10, 16
sigma_s, sigma_r = 10, 10

weights = ti.field(dtype=ti.f32, shape=(2, 128), offset=(0, -64))


@ti.func
def compute_weight(i, radius, sigma):
    total = 0.0

    # Not much computation here - serialize the for loop to save two more GPU kernel launch costs
    ti.loop_config(serialize=True)
    for j in range(-radius, radius + 1):
        # Drop the normal distribution constant coefficients since we need to normalize later anyway
        val = ti.exp(-0.5 * ((radius - j) / sigma)**2)
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
def bilateral_filter(img: ti.types.ndarray()):
    # Reset the grid
    print('Scattering', img.shape[0], img.shape[1])
    grid.fill(0)
    for i, j in ti.ndrange(img.shape[0], img.shape[1]):
        lum = int(img[i, j])
        grid[ti.round(i / s_s, ti.i32),
             ti.round(j / s_s, ti.i32),
             ti.round(lum / s_r, ti.i32)] += tm.vec2(lum, 1)

    print('Compute blur weights')
    compute_weight(0, sigma_s * 3, sigma_s)
    compute_weight(1, sigma_r * 3, sigma_r)

    print('Blurring')
    # Grid processing (blur)
    grid_n, grid_m = (img.shape[0] + s_s - 1) // s_s, (img.shape[1] + s_s -
                                                       1) // s_s
    grid_l = (255 + s_r - 1) // s_r
    blur_radius = 3 * sigma_s

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

    blur_radius = 3 * sigma_r
    for i, j, k in ti.ndrange(grid_n, grid_m, grid_l):
        l_begin, l_end = max(0, k - blur_radius), min(grid_l,
                                                      k + blur_radius + 1)
        total = tm.vec2(0, 0)
        for l in range(l_begin, l_end):
            total += grid[i, j, k] * weights[1, k - l]

        grid_blurred[i, j, k] = total

    print('Slicing')
    # Slicing
    for i, j in ti.ndrange(img.shape[0], img.shape[1]):
        lum = int(img[i, j])
        sample = sample_grid(i / s_s, j / s_s, lum / s_r)

        img[i, j] = ti.u8(sample[0] / sample[1])


img = cv2.imread('images/lenna_bw.png')[:, :, 0].copy()
cv2.imshow('input', img)
bilateral_filter(img)
'''
for i in range(30):
    g = grid_blurred.to_numpy()[:, :, i]
    cv2.imshow('grid', g[:, :, 0] / 255)
    cv2.waitKey(1)
'''
cv2.imshow('filtered', img)

cv2.waitKey()
