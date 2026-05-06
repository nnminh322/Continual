import os
import numpy as np
from torchvision import datasets, transforms
from utils.toolkit import split_images_labels
from utils.datautils.core50data import CORE50
import ipdb
import yaml
from PIL import Image
from shutil import move, rmtree
import torch
from pathlib import Path
import time

try:
    YAML_LOADER = yaml.CSafeLoader
except AttributeError:
    YAML_LOADER = yaml.SafeLoader

DOMAINNET_ROOT_NAMES = ("DomainNet", "domainnet")
DOMAINNET_DOMAIN_NAMES = ("clipart", "infograph", "painting", "quickdraw", "real", "sketch")


def _domainnet_path_variants(path):
    path = str(path).replace('\\', '/').lstrip('/')
    variants = []

    def add(variant):
        variant = str(variant).replace('\\', '/').lstrip('/')
        if variant and variant not in variants:
            variants.append(variant)

    add(path)

    if path.startswith('data/DomainNet/'):
        tail = path[len('data/DomainNet/'):]
        add(f'DomainNet/{tail}')
        add(f'domainnet/{tail}')
        add(f'data/domainnet/{tail}')
        add(tail)
    elif path.startswith('data/domainnet/'):
        tail = path[len('data/domainnet/'):]
        add(f'domainnet/{tail}')
        add(f'DomainNet/{tail}')
        add(f'data/DomainNet/{tail}')
        add(tail)
    elif path.startswith('DomainNet/'):
        tail = path[len('DomainNet/'):]
        add(f'domainnet/{tail}')
        add(tail)
    elif path.startswith('domainnet/'):
        tail = path[len('domainnet/'):]
        add(f'DomainNet/{tail}')
        add(tail)
    elif path.startswith('data/'):
        add(path[len('data/'):])

    return variants


def _looks_like_domainnet_dir(path):
    path = Path(path).expanduser()
    if not path.exists() or not path.is_dir():
        return False

    present_domains = sum(1 for domain_name in DOMAINNET_DOMAIN_NAMES if (path / domain_name).is_dir())
    return present_domains >= 2


def _iter_domainnet_dir_hints(root):
    root = Path(root).expanduser()
    yield root
    yield root / 'data'
    for root_name in DOMAINNET_ROOT_NAMES:
        yield root / root_name
        yield root / 'data' / root_name

class iData(object):
    train_trsf = []
    test_trsf = []
    common_trsf = []
    class_order = None

def build_transform(is_train, args):
    input_size = 224
    resize_im = input_size > 32
    if is_train:
        scale = (0.05, 1.0)
        ratio = (3. / 4., 4. / 3.)

        transform = [
            transforms.RandomResizedCrop(input_size, scale=scale, ratio=ratio),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
        ]
        return transform

    t = []
    if resize_im:
        size = int((256 / 224) * input_size)
        t.append(
            transforms.Resize(size, interpolation=3),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(input_size))
    t.append(transforms.ToTensor())

    # return transforms.Compose(t)
    return t

class iCUB(iData):
    use_path = True

    train_trsf=[
            transforms.RandomResizedCrop(224, scale=(0.05, 1.0), ratio=(3./4., 4./3.)),
            transforms.RandomHorizontalFlip(p=0.5),
            ]
    test_trsf=[
        transforms.Resize(256, interpolation=3),
        transforms.CenterCrop(224),
        ]
    common_trsf = [transforms.ToTensor()]

    class_order = np.arange(200).tolist()

    def __init__(self, args):
        self.args = args
        class_order = np.arange(200).tolist()
        self.class_order = class_order

    def download_data(self):
        # assert 0, "You should specify the folder of your dataset"
        train_dir = "data/cub/train/"
        test_dir = "data/cub/test/"

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)

class iCIFAR10(iData):
    use_path = False
    train_trsf = [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ]

    test_trsf = [
        transforms.Resize(224),
        transforms.ToTensor()
        ]
    common_trsf = [
        transforms.Normalize(
            mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0)
        ),
    ]

    class_order = np.arange(10).tolist()

    def __init__(self, args):
        self.args = args
        class_order = np.arange(10).tolist()
        self.class_order = class_order

    def download_data(self):
        train_dataset = datasets.cifar.CIFAR10(self.args['data_path'], train=True, download=True)
        test_dataset = datasets.cifar.CIFAR10(self.args['data_path'], train=False, download=True)
        self.train_data, self.train_targets = train_dataset.data, np.array(
            train_dataset.targets
        )
        self.test_data, self.test_targets = test_dataset.data, np.array(
            test_dataset.targets
        )


class iCIFAR100(iData):
    use_path = False
    train_trsf = [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
    ]

    test_trsf = [
        transforms.Resize(224),
        ]

    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0)
        ),
    ]

    # train_trsf = [
    #     transforms.RandomResizedCrop(224, interpolation=3),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.ColorJitter(brightness=63/255)
    # ]
    # test_trsf = [
    #     transforms.Resize(256, interpolation=3),
    #     transforms.CenterCrop(224),
    # ]

    # common_trsf = [
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761)),
    # ]

    class_order = np.arange(100).tolist()

    def __init__(self, args):
        self.args = args
        class_order = np.arange(100).tolist()
        self.class_order = class_order

    def download_data(self):
        train_dataset = datasets.cifar.CIFAR100(self.args['data_path'], train=True, download=True)
        test_dataset = datasets.cifar.CIFAR100(self.args['data_path'], train=False, download=True)
        self.train_data, self.train_targets = train_dataset.data, np.array(
            train_dataset.targets
        )
        self.test_data, self.test_targets = test_dataset.data, np.array(
            test_dataset.targets
        )

class iIMAGENET_R(iData):
    use_path = True
    train_trsf = [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ]

    test_trsf = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()
        ]
    common_trsf = [
        transforms.Normalize(
            mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0)
        ),
    ]

    class_order = np.arange(200).tolist()

    def __init__(self, args):
        self.args = args
        class_order = np.arange(200).tolist()
        self.class_order = class_order

    def download_data(self):
        # load splits from config file
        if not os.path.exists(os.path.join(self.args['data_path'], 'train')) and not os.path.exists(os.path.join(self.args['data_path'], 'train')):
            self.dataset = datasets.ImageFolder(self.args['data_path'], transform=None)

            train_size = int(0.8 * len(self.dataset))
            val_size = len(self.dataset) - train_size

            train, val = torch.utils.data.random_split(self.dataset, [train_size, val_size])
            train_idx, val_idx = train.indices, val.indices

            self.train_file_list = [self.dataset.imgs[i][0] for i in train_idx]
            self.test_file_list = [self.dataset.imgs[i][0] for i in val_idx]

            self.split()

        train_data_config = datasets.ImageFolder(os.path.join(self.args['data_path'], 'train')).samples
        test_data_config = datasets.ImageFolder(os.path.join(self.args['data_path'], 'test')).samples
        self.train_data = np.array([config[0] for config in train_data_config])
        self.train_targets = np.array([config[1] for config in train_data_config])
        self.test_data = np.array([config[0] for config in test_data_config])
        self.test_targets = np.array([config[1] for config in test_data_config])


    def split(self):
        train_folder = os.path.join(self.args['data_path'], 'train')
        test_folder = os.path.join(self.args['data_path'], 'test')

        if os.path.exists(train_folder):
            rmtree(train_folder)
        if os.path.exists(test_folder):
            rmtree(test_folder)
        os.mkdir(train_folder)
        os.mkdir(test_folder)

        for c in self.dataset.classes:
            if not os.path.exists(os.path.join(train_folder, c)):
                os.mkdir(os.path.join(os.path.join(train_folder, c)))
            if not os.path.exists(os.path.join(test_folder, c)):
                os.mkdir(os.path.join(os.path.join(test_folder, c)))

        for path in self.train_file_list:
            if '\\' in path:
                path = path.replace('\\', '/')
            src = path
            dst = os.path.join(train_folder, '/'.join(path.split('/')[-2:]))
            move(src, dst)

        for path in self.test_file_list:
            if '\\' in path:
                path = path.replace('\\', '/')
            src = path
            dst = os.path.join(test_folder, '/'.join(path.split('/')[-2:]))
            move(src, dst)

        for c in self.dataset.classes:
            path = os.path.join(self.args['data_path'], c)
            rmtree(path)


class iIMAGENET_A(iData):
    use_path = True
    train_trsf=[
            transforms.RandomResizedCrop(224, scale=(0.05, 1.0), ratio=(3./4., 4./3.)),
            transforms.RandomHorizontalFlip(p=0.5),
            ]
    test_trsf=[
        transforms.Resize(256, interpolation=3),
        transforms.CenterCrop(224),
        ]
    common_trsf = [transforms.ToTensor()]

    class_order = np.arange(200).tolist()

    def __init__(self, args):
        self.args = args
        class_order = np.arange(200).tolist()
        self.class_order = class_order

    def download_data(self):
        # assert 0, "You should specify the folder of your dataset"
        train_dir = "data/imagenet-a/train/"
        test_dir = "data/imagenet-a/test/"

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)


class iDomainNet(iData):

    use_path = True
    train_trsf = [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
    ]
    test_trsf = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ]
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0]),
    ]

    def __init__(self, args):
        self.args = args
        class_order = np.arange(345).tolist()
        self.class_order = class_order
        self.domain_names = ["clipart", "infograph", "painting", "quickdraw", "real", "sketch", ]
        self._resolved_domainnet_root = None

    def _get_domainnet_root(self):
        if self._resolved_domainnet_root is not None:
            return self._resolved_domainnet_root

        data_root = self.args.get('data_path') if self.args is not None else None
        if not data_root:
            return None

        for candidate_root in _iter_domainnet_dir_hints(data_root):
            if candidate_root.exists() and _looks_like_domainnet_dir(candidate_root):
                self._resolved_domainnet_root = str(candidate_root)
                return self._resolved_domainnet_root

        return None

    def _fast_resolve_from_root(self, path):
        root = self._get_domainnet_root()
        if root is None:
            return None

        norm_path = str(path).replace('\\', '/').lstrip('/')
        for prefix in ('data/DomainNet/', 'data/domainnet/', 'DomainNet/', 'domainnet/'):
            if norm_path.startswith(prefix):
                tail = norm_path[len(prefix):]
                return str(Path(root) / tail)

        return None

    def _resolve_image_path(self, path):
        path = str(path).replace('\\', '/')

        fast_path = self._fast_resolve_from_root(path)
        if fast_path is not None:
            return fast_path

        candidates = _domainnet_path_variants(path)

        for candidate in candidates:
            if os.path.isabs(candidate) and os.path.exists(candidate):
                return candidate
            if os.path.exists(candidate):
                return candidate

        data_root = self.args.get('data_path') if self.args is not None else None
        if data_root:
            for candidate_root in _iter_domainnet_dir_hints(data_root):
                if not candidate_root.exists():
                    continue
                for candidate_tail in candidates:
                    candidate = candidate_root / candidate_tail
                    if candidate.exists():
                        return str(candidate)

        return path

    def _resolve_paths_bulk(self, paths, split_name):
        total = len(paths)
        start = time.time()
        resolved = []

        for idx, path in enumerate(paths, start=1):
            resolved.append(self._resolve_image_path(path))

            if idx == 1 or idx % 200000 == 0 or idx == total:
                elapsed = time.time() - start
                rate = idx / max(elapsed, 1e-9)
                remaining = max(total - idx, 0) / max(rate, 1e-9)
                print(
                    f"[iDomainNet] resolving {split_name}: {idx}/{total} ({(100.0 * idx / max(total, 1)):.1f}%) "
                    f"rate={rate:.0f}/s eta={remaining/60:.1f}m")

        return np.array(resolved)

    def _verify_paths(self, paths, split_name):
        verify_mode = str((self.args or {}).get('domainnet_verify', 'sample')).lower().strip()
        if verify_mode == 'none':
            print(f"[iDomainNet] verify {split_name}: skipped (domainnet_verify=none)")
            return

        n = len(paths)
        if n == 0:
            return

        if verify_mode == 'full':
            idxes = range(n)
            mode_info = f"full ({n} samples)"
        else:
            sample_size = min(2048, n)
            idxes = np.linspace(0, n - 1, num=sample_size, dtype=np.int64)
            mode_info = f"sample ({sample_size}/{n})"

        start = time.time()
        for checked, idx in enumerate(idxes, start=1):
            path = paths[int(idx)]
            if not os.path.exists(path):
                data_root = self.args.get('data_path') if self.args is not None else None
                raise FileNotFoundError(
                    f"DomainNet {split_name} sample not found after path resolution: {path}. "
                    f"Effective data_path={data_root!r}. "
                    f"Set DOMAINNET_ROOT env var or pass --data_path to the directory containing the domain folders "
                    f"(clipart, infograph, painting, quickdraw, real, sketch).")

            if checked == 1 or checked % 500 == 0 or checked == len(idxes):
                elapsed = time.time() - start
                rate = checked / max(elapsed, 1e-9)
                print(
                    f"[iDomainNet] verify {split_name} {mode_info}: {checked}/{len(idxes)} "
                    f"rate={rate:.0f}/s")

    def download_data(self):
        # Auto-detect DomainNet root if the configured data_path doesn't exist as-is.
        # This handles Kaggle mounts where the path in the config is a relative placeholder.
        current_root = self.args.get('data_path') if self.args is not None else None
        current_root_path = Path(current_root).expanduser() if current_root else None
        if current_root_path is None or not current_root_path.exists():
            detected = resolve_domainnet_root(current_root)
            if detected is not None:
                print(f"[iDomainNet] auto-detected DomainNet root: {detected}")
                if self.args is not None:
                    self.args['data_path'] = detected
                    self._resolved_domainnet_root = None
            else:
                print(f"[iDomainNet] WARNING: could not auto-detect DomainNet root from {current_root!r}. "
                      f"Proceeding with path resolution; set DOMAINNET_ROOT or pass --data_path to fix this.")

        effective_root = self._get_domainnet_root()
        print(f"[iDomainNet] effective data_path={self.args.get('data_path') if self.args else None!r}")
        if effective_root is not None:
            print(f"[iDomainNet] using fast root mapping from {effective_root}")

        # load splits from config file
        train_split = os.path.join(os.path.dirname(__file__), '..', 'dataloaders', 'splits', 'domainnet_train.yaml')
        test_split = os.path.join(os.path.dirname(__file__), '..', 'dataloaders', 'splits', 'domainnet_test.yaml')

        t0 = time.time()
        with open(train_split, 'r') as handle:
            train_data_config = yaml.load(handle, Loader=YAML_LOADER)
        print(f"[iDomainNet] loaded train split in {time.time() - t0:.1f}s")

        t1 = time.time()
        with open(test_split, 'r') as handle:
            test_data_config = yaml.load(handle, Loader=YAML_LOADER)
        print(f"[iDomainNet] loaded test split in {time.time() - t1:.1f}s")

        self.train_data = self._resolve_paths_bulk(train_data_config['data'], split_name='train')
        self.train_targets = np.array(train_data_config['targets'])
        self.test_data = self._resolve_paths_bulk(test_data_config['data'], split_name='test')
        self.test_targets = np.array(test_data_config['targets'])

        self._verify_paths(self.train_data, split_name='train')
        self._verify_paths(self.test_data, split_name='test')


def jpg_image_to_array(image_path):
    """
    Loads JPEG image into 3D Numpy array of shape
    (width, height, channels)
    """
    with Image.open(image_path) as image:
        image = image.convert('RGB')
        im_arr = np.fromstring(image.tobytes(), dtype=np.uint8)
        im_arr = im_arr.reshape((image.size[1], image.size[0], 3))
    return im_arr


def resolve_domainnet_root(candidate_root=None):
    def first_existing(paths):
        for path in paths:
            if path is None:
                continue
            resolved = Path(path).expanduser()
            if resolved.exists():
                return str(resolved.resolve())
        return None

    def derive_root_from_match(match_path, probe_variant):
        root = Path(match_path).expanduser()
        for _ in Path(probe_variant).parts:
            root = root.parent
        return root

    def load_probe_paths(limit=8):
        split_path = Path(__file__).resolve().parent / '..' / 'dataloaders' / 'splits' / 'domainnet_train.yaml'
        split_path = split_path.resolve()
        if not split_path.exists():
            return []

        # Parse only the first few entries in the "data:" YAML list without loading the full file.
        probes = []
        in_data_block = False
        with open(split_path, 'r') as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not in_data_block:
                    if line == 'data:' or line.startswith('data:'):
                        in_data_block = True
                    continue

                if line.startswith('targets:'):
                    break

                if not line.startswith('- '):
                    continue

                value = line[2:].strip().strip('"').strip("'")
                if value:
                    probes.append(value.replace('\\', '/'))
                if len(probes) >= limit:
                    break

        return probes

    def iter_candidate_roots():
        direct_roots = [
            candidate_root,
            os.environ.get('DOMAINNET_ROOT'),
            Path.cwd(),
            Path.cwd().parent,
            Path.cwd() / 'data',
            Path.cwd() / 'datasets',
            Path('/kaggle/working'),
            Path('/kaggle/input'),
            Path('/content'),
            Path('/mnt/data'),
        ]

        for root in direct_roots:
            if root is not None:
                yield Path(root).expanduser()

        for base in [Path('/kaggle/input'), Path('/kaggle/working'), Path('/content'), Path('/mnt/data')]:
            if not base.exists():
                continue
            try:
                for child in base.iterdir():
                    if child.is_dir():
                        yield child
            except OSError:
                continue

    probe_paths = load_probe_paths()
    if not probe_paths:
        return first_existing([candidate_root, os.environ.get('DOMAINNET_ROOT')])

    seen = set()
    for root in iter_candidate_roots():
        root_key = str(root)
        if root_key in seen or not root.exists():
            continue
        seen.add(root_key)

        for candidate_dir in _iter_domainnet_dir_hints(root):
            if _looks_like_domainnet_dir(candidate_dir):
                return str(candidate_dir.resolve())

        if any((root / probe_variant).exists() for probe in probe_paths[:8] for probe_variant in _domainnet_path_variants(probe)):
            return str(root.resolve())

    recursive_bases = [
        candidate_root,
        os.environ.get('DOMAINNET_ROOT'),
        Path.cwd(),
        Path.cwd().parent,
        Path('/kaggle/input'),
        Path('/kaggle/working'),
        Path('/content'),
        Path('/mnt/data'),
    ]

    for base in recursive_bases:
        if base is None:
            continue
        base = Path(base).expanduser()
        if not base.exists() or not base.is_dir():
            continue

        for candidate_dir in _iter_domainnet_dir_hints(base):
            if _looks_like_domainnet_dir(candidate_dir):
                return str(candidate_dir.resolve())

        for probe in probe_paths[:4]:
            for probe_variant in _domainnet_path_variants(probe):
                try:
                    match_path = next(base.rglob(probe_variant), None)
                except (OSError, RuntimeError):
                    match_path = None

                if match_path is None or not match_path.exists():
                    continue

                root = derive_root_from_match(match_path, probe_variant)
                for candidate_dir in _iter_domainnet_dir_hints(root):
                    if _looks_like_domainnet_dir(candidate_dir):
                        return str(candidate_dir.resolve())
                if root.exists():
                    return str(root.resolve())

    # Could not find DomainNet — print diagnostics to help the user locate the dataset.
    print("[resolve_domainnet_root] WARNING: could not locate DomainNet. Diagnostics:")
    print(f"  candidate_root={candidate_root!r}")
    print(f"  DOMAINNET_ROOT env={os.environ.get('DOMAINNET_ROOT')!r}")
    print(f"  cwd={str(Path.cwd())!r}")
    for base in [Path('/kaggle/input'), Path('/kaggle/working'), Path('/content'), Path('/mnt/data')]:
        if base.exists():
            try:
                children = [str(c) for c in base.iterdir() if c.is_dir()][:10]
                print(f"  {base}: {children}")
            except OSError:
                print(f"  {base}: (unreadable)")
        else:
            print(f"  {base}: does not exist")
    return None

