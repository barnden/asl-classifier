import torch as th
from torchvision import transforms
from torch import nn
import numpy as np
import lmdb
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict
from network import Classifier

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from torch.utils.tensorboard import SummaryWriter

class ASLDataset(th.utils.data.Dataset):
    def __init__(self, lmdb_path, split="train", transform=None):
        self.labels = [*"ABCDEFGHIKLMNOPQRSTUVWXY"]
        self.transform = transform
        self.env = lmdb.open(lmdb_path)
        self.split = split

        with self.env.begin() as txn:
            self.size = int(bytes(txn.get(f'{split}_size'.encode())))

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        with self.env.begin() as txn:
            image = txn.get(f'{self.split}:image:{idx}'.encode())
            image = np.frombuffer(image, dtype=np.float32).copy()
            label = bytes(txn.get(f'{self.split}:label:{idx}'.encode())).decode('utf-8')
            label = self.labels.index(label)

            image = th.from_numpy(image)
            image = image.reshape(64, 64, 3).permute(-1, 0, 1)

            if self.transform is not None:
                image = self.transform(image)

        return image, label

transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10)
])

def initialize(batch_size, lmdb_path, device, resume, save_directory, log_directory, experiment_name, **_):
    data_loader_options = {
        'batch_size': batch_size,
        'shuffle': True,
        'num_workers': 1
    }

    data_set_options = {
        'lmdb_path': lmdb_path,
        'transform': transforms
    }

    train_set = ASLDataset(split="train", **data_set_options)
    test_set = ASLDataset(split="test", **data_set_options)
    validation_set = ASLDataset(split="validate", **data_set_options)

    train_loader = th.utils.data.DataLoader(train_set, **data_loader_options)
    test_loader = th.utils.data.DataLoader(test_set, **data_loader_options)
    validation_loader = th.utils.data.DataLoader(validation_set, **data_loader_options)

    model = Classifier()

    Path(save_directory, experiment_name).mkdir(parents=True, exist_ok=True)
    Path(log_directory, experiment_name).mkdir(parents=True, exist_ok=True)

    if resume is not None:
        model_path = Path(save_directory, experiment_name, f"model-{resume:06}.pt")

        if not model_path.exists():
            print(f"Could not find checkpoint, path: \"{model_path}\"")

        print("Resuming from checkpoint", resume)

        model.load_state_dict(th.load(model_path))

    model = model.to(device)

    return model, train_loader, test_loader, validation_loader

def no_grad_if(condition):
    def decorator(func):
        if not condition:
            return func

        return th.no_grad(func)

    return decorator

def loop(model, loader, loss_fn, epoch, optim=None, callback=None, split='train', device='cuda'):
    logdict = dict()

    prefix = "epoch" if split == 'train' else f"    {split}"
    prefix = f"{prefix}: {epoch}"

    @no_grad_if(split != 'train')
    def __loop():
        nonlocal logdict, prefix

        guesses = 0
        correct = 0
        total_loss = 0

        if split == 'train':
            model.train()
        else:
            model.eval()

        for images, labels in (progress_bar := tqdm(loader)):
            images = images.to(device)
            labels = labels.to(device)

            if split == 'train':
                optim.zero_grad()

            logits = model(images)
            prediction = logits.max(1).indices

            guesses += logits.shape[0]
            correct += th.sum(prediction == labels)

            loss = loss_fn(logits, labels)
            total_loss += loss

            if split == 'train':
                loss.backward()
                optim.step()

            accuracy = 100 * correct / guesses
            progress_bar.set_description(f"{prefix:<25}(accuracy: {accuracy:.3f}% [{correct}/{guesses}], loss: {loss:.4f})")

        if callback is not None:
            callback(epoch)

        logdict['accuracy'] = correct / guesses
        logdict['loss'] = total_loss / len(loader)

    __loop()

    return logdict

def train_loop(model, train, test, validation, epochs, device, resume, save_directory, experiment_name, log_directory, save_every, **_):
    optim = th.optim.AdamW(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    writer = SummaryWriter(log_dir=log_directory)

    resume = resume if resume is not None else 0

    def save(epoch):
        model_path = Path(save_directory, experiment_name)
        th.save(model.state_dict(), model_path.joinpath("latest.pt"))

        if (epoch % save_every) == 0:
            th.save(model.state_dict(), model_path.joinpath(f"model-{epoch:06}"))

    for epoch in range(resume + 1, epochs):
        logobj = defaultdict(dict)

        result = loop(model, train, loss_fn, epoch, optim, save, device=device)
        logobj['accuracy']['train'] = result['accuracy']
        logobj['loss']['train'] = result['loss']

        result = loop(model, validation, loss_fn, epoch, split='validate', device=device)
        logobj['accuracy']['validate'] = result['accuracy']
        logobj['loss']['validate'] = result['loss']

        result = loop(model, test, loss_fn, epoch, split='test', device=device)
        logobj['accuracy']['test'] = result['accuracy']
        logobj['loss']['test'] = result['loss']

        writer.add_scalars(f'{experiment_name}/accuracy', logobj['accuracy'], epoch)
        writer.add_scalars(f'{experiment_name}/loss', logobj['loss'], epoch)
        writer.flush()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('-n', '--experiment_name', type=str, default="classifier")
    parser.add_argument('-e', '--epochs', type=int, default=10_000)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('-s', '--save_directory', type=str, default='./models')
    parser.add_argument('-b', '--batch_size', type=int, default=1792)
    parser.add_argument('-d', '--lmdb_path', type=str, default="asl_64x64.lmdb")
    parser.add_argument('-l', '--log_directory', type=str, default='./logs')
    parser.add_argument('-r', '--resume', type=int, default=None)
    parser.add_argument('--save_every', type=int, default=5)

    args = parser.parse_args()

    if args.device == 'cuda' and not th.cuda.is_available():
        print("cuda unavailable, falling back to cpu")
        args.device = 'cpu'

    model, *loaders = initialize(**vars(args))
    train_loop(model, *loaders, **vars(args))
