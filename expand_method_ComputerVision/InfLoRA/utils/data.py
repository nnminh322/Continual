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

    def _resolve_image_path(self, path):
        path = str(path).replace('\\', '/')

        candidates = []

        def add_candidate(candidate):
            candidate = str(candidate).replace('\\', '/')
            if candidate and candidate not in candidates:
                candidates.append(candidate)

        add_candidate(path)
        if path.startswith('data/'):
            add_candidate(path[len('data/'):])
        if path.startswith('data/DomainNet/'):
            add_candidate(path[len('data/DomainNet/'):])

        for candidate in candidates:
            if os.path.isabs(candidate) and os.path.exists(candidate):
                return candidate
            if os.path.exists(candidate):
                return candidate

        data_root = self.args.get('data_path') if self.args is not None else None
        if data_root:
            for candidate_tail in candidates:
                candidate = os.path.join(data_root, candidate_tail)
                if os.path.exists(candidate):
                    return candidate

        return path

    def download_data(self):
        # load splits from config file
        train_split = os.path.join(os.path.dirname(__file__), '..', 'dataloaders', 'splits', 'domainnet_train.yaml')
        test_split = os.path.join(os.path.dirname(__file__), '..', 'dataloaders', 'splits', 'domainnet_test.yaml')

        train_data_config = yaml.load(open(train_split, 'r'), Loader=yaml.Loader)
        test_data_config = yaml.load(open(test_split, 'r'), Loader=yaml.Loader)

        self.train_data = np.array([self._resolve_image_path(path) for path in train_data_config['data']])
        self.train_targets = np.array(train_data_config['targets'])
        self.test_data = np.array([self._resolve_image_path(path) for path in test_data_config['data']])
        self.test_targets = np.array(test_data_config['targets'])

        missing_train = next((path for path in self.train_data if not os.path.exists(path)), None)
        if missing_train is not None:
            raise FileNotFoundError(
                f"DomainNet train sample not found after path resolution: {missing_train}. "
                f"Check args['data_path']={self.args.get('data_path')!r} and the mounted dataset root.")

        missing_test = next((path for path in self.test_data if not os.path.exists(path)), None)
        if missing_test is not None:
            raise FileNotFoundError(
                f"DomainNet test sample not found after path resolution: {missing_test}. "
                f"Check args['data_path']={self.args.get('data_path')!r} and the mounted dataset root.")


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

    def load_probe_paths(limit=8):
        split_path = Path(__file__).resolve().parent / '..' / 'dataloaders' / 'splits' / 'domainnet_train.yaml'
        split_path = split_path.resolve()
        if not split_path.exists():
            return []

        with open(split_path, 'r') as handle:
            split_config = yaml.load(handle, Loader=yaml.Loader) or {}

        return [str(path).replace('\\', '/') for path in split_config.get('data', [])[:limit]]

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

        if any((root / probe).exists() for probe in probe_paths[:8]):
            return str(root.resolve())

    return None

