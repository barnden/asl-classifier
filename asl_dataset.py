import numpy as np
import torch as th
import lmdb
from PIL import Image
from torchvision import transforms

class ASLDataset(th.utils.data.Dataset):
    def __init__(self, lmdb_path, split="train", transform=None):
        self.labels = [*"ABCDEFGHIKLMNOPQRSTUVWXY"]
        self.transform = transform if transform is not None else transforms.ToTensor()
        self.env = lmdb.open(lmdb_path)
        self.split = split

        with self.env.begin() as txn:
            self.size = int(bytes(txn.get(f'{split}_size'.encode())))

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        with self.env.begin() as txn:
            image = txn.get(f'{self.split}:image:{idx}'.encode())
            image = np.frombuffer(image, dtype=np.uint8).copy().reshape(64, 64, 3)
            image = Image.fromarray(image)
            label = bytes(txn.get(f'{self.split}:label:{idx}'.encode())).decode('utf-8')
            label = self.labels.index(label)

            # image = th.from_numpy(image)
            # image = image.reshape(64, 64, 3).permute(-1, 0, 1)

            image = self.transform(image)

        return image, label
