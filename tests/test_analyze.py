import numpy as np

from cifar_dino.analyze import patch_pca_rgb


def test_patch_pca_foreground_is_centre():
    # 8x8 grid: centre 4x4 patches near one direction, the rest near another -> fg must be exactly the centre
    rng = np.random.default_rng(0)
    centre = np.zeros((8, 8), bool)
    centre[2:6, 2:6] = True
    a, b = rng.normal(size=16), rng.normal(size=16)
    x = np.where(centre.reshape(1, 64, 1), a, b) + 0.05 * rng.normal(size=(5, 64, 16))
    rgb_all, rgb_fg, fg = patch_pca_rgb(x)
    assert rgb_all.shape == rgb_fg.shape == (5, 8, 8, 3) and 0 <= rgb_all.min() and rgb_all.max() <= 1
    assert (fg == centre).all()
    assert (rgb_fg[:, ~centre] == 0).all()
