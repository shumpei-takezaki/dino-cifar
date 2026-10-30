"""CIFAR-10 datasets and the ``mrnghia90`` multi-crop augmentation (the only profile implemented)."""

from torch.utils.data import Dataset
from torchvision import datasets, transforms as T

BICUBIC = T.InterpolationMode.BICUBIC


def _crop_transform(a: dict, size: int, scale, blur_prob: float, normalize: T.Normalize) -> T.Compose:
    ops = [
        T.RandomResizedCrop(size, scale=tuple(scale), interpolation=BICUBIC),
        T.RandomHorizontalFlip(a["horizontal_flip_prob"]),
        T.RandomApply([T.ColorJitter(a["brightness"], a["contrast"], a["saturation"], a["hue"])], p=a["color_jitter_prob"]),
        T.RandomGrayscale(a["grayscale_prob"]),
    ]
    if blur_prob > 0:
        ops.append(T.RandomApply([T.GaussianBlur(a["global_blur_kernel_size"], tuple(a["global_blur_sigma"]))], p=blur_prob))
    if a["solarize_prob"] > 0:
        ops.append(T.RandomSolarize(128, p=a["solarize_prob"]))
    return T.Compose(ops + [T.ToTensor(), normalize])


class MultiCropTransform:
    """Returns [global_1, global_2, local_1, ..., local_L] tensors for one PIL image."""

    def __init__(self, augment: dict, mean, std):
        if augment["profile"] != "mrnghia90":
            raise ValueError(f"unknown augmentation profile {augment['profile']!r}; only 'mrnghia90' is implemented")
        if augment["local_blur_prob"] > 0:
            raise NotImplementedError("local blur is not part of the mrnghia90 profile")
        norm = T.Normalize(mean, std)
        self.n_global, self.n_local = augment["num_global_crops"], augment["num_local_crops"]
        self.global_t = _crop_transform(augment, augment["global_size"], augment["global_scale"], augment["global_blur_prob"], norm)
        self.local_t = _crop_transform(augment, augment["local_size"], augment["local_scale"], 0.0, norm)

    def __call__(self, img):
        return [self.global_t(img) for _ in range(self.n_global)] + [self.local_t(img) for _ in range(self.n_local)]


def eval_transform(mean, std) -> T.Compose:
    return T.Compose([T.ToTensor(), T.Normalize(mean, std)])


def build_dataset(ds_cfg: dict, train: bool, transform) -> Dataset:
    """``name: cifar10`` (auto-download) or ``name: fake`` (random images, for tests/smoke runs)."""
    if ds_cfg["name"] == "cifar10":
        return datasets.CIFAR10(ds_cfg["root"], train=train, transform=transform, download=True)
    if ds_cfg["name"] == "fake":
        size = ds_cfg.get("fake_size", 256)
        return datasets.FakeData(size if train else size // 2, (3, 32, 32), 10, transform, random_offset=0 if train else 1)
    raise ValueError(f"unknown dataset {ds_cfg['name']!r}")
