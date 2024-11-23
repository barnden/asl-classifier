import torch as th
from torchvision import transforms
from torch import nn
import numpy as np
import lmdb
from tqdm import tqdm
from pathlib import Path

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


class Maxout2D(nn.Module):
    def __init__(self, max_out):
        super().__init__()

        self.max_out = max_out
        self.max_pool = nn.MaxPool1d(max_out)

    def forward(self, x: th.Tensor):
        B, C, H, W = x.shape

        h = x.permute(0, 2, 3, 1).view(B, H * W, C)
        h = self.max_pool(h)
        h = h.permute(0, 2, 1).view(B, C // self.max_out, H, W).contiguous()

        return h

class Block(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.2):
        super().__init__()

        self.layers = nn.Sequential(*[
            nn.Conv2d(in_ch, in_ch, kernel_size=(1, 1)),
            nn.BatchNorm2d(num_features=in_ch),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv2d(in_ch, out_ch, kernel_size=(3, 3)),
            nn.BatchNorm2d(num_features=out_ch),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.MaxPool2d(kernel_size=2),
        ])

    def forward(self, x):
        return self.layers(x)

class Classifier(nn.Module):
    def __init__(self):
        super().__init__()

        self.input_processor = nn.Sequential(*[
            nn.Conv2d(3, 10, kernel_size=(1, 1)),
            nn.BatchNorm2d(num_features=10),
            Maxout2D(2),
            nn.Conv2d(5, 6, kernel_size=(1, 1)),
            nn.BatchNorm2d(num_features=6),
            Maxout2D(2)
        ])

        self.blocks = nn.Sequential(*[
            Block(3, 32, dropout=0.2),
            Block(32, 64, dropout=0.2),
            Block(64, 128, dropout=0.2),
            Block(128, 256, dropout=0.2),
        ])

        self.bottleneck = nn.Sequential(*[
            nn.AvgPool2d(2)
        ])

        self.dense = nn.Sequential(*[
            nn.Linear(256, 1024),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.Dropout(0.3),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(128, 24),
        ])

    def forward(self, x: th.Tensor):
        h = self.input_processor(x)
        h = self.blocks(h)
        h = self.bottleneck(h)
        B, C, *_ = h.shape
        h = self.dense(h.view(B, C))

        return h

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

def loop(model, train, test, validation, epochs, device, resume, save_directory, experiment_name, log_directory, **_):
    optim = th.optim.AdamW(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    writer = SummaryWriter(log_dir=log_directory)

    resume = resume if resume is not None else 0

    for i in range(resume, epochs):
        train_progress = tqdm(train)
        accuracies = dict()
        losses = dict()

        guesses = 0
        correct = 0
        epoch_loss = 0
        for images, labels in train_progress:
            images = images.to(device)
            labels = labels.to(device)

            optim.zero_grad()
            logits = model(images)
            prediction = logits.max(1).indices

            guesses += logits.shape[0]
            correct += th.sum(prediction == labels)

            loss = loss_fn(logits, labels)
            epoch_loss += loss
            train_progress.set_description(f"epoch: {i}        (accuracy: {100 * correct / guesses:.3f}%, loss: {loss:.4f})")
            loss.backward()
            optim.step()

        accuracies['train'] = correct / guesses
        losses['train'] = epoch_loss / len(train_progress)

        model_path = Path(save_directory, experiment_name, f"model-{i:06}.pt")
        th.save(model.state_dict(), model_path)

        validation_progress = tqdm(validation)
        guesses = 0
        correct = 0
        epoch_loss = 0
        with th.no_grad():
            for images, labels in validation_progress:
                images = images.to(device)
                labels = labels.to(device)

                logits = model(images)
                prediction = logits.max(1).indices

                guesses += logits.shape[0]
                correct += th.sum(prediction == labels)

                loss = loss_fn(logits, labels)
                epoch_loss += loss
                validation_progress.set_description(f"    validate: {i} (accuracy: {100 * correct / guesses:.3f}%, loss: {loss:.4f})")

            accuracies['validate'] = correct / guesses
            losses['validate'] = epoch_loss / len(validation_progress)

        test_progress = tqdm(test)
        guesses = 0
        correct = 0
        epoch_loss = 0
        with th.no_grad():
            for images, labels in test_progress:
                images = images.to(device)
                labels = labels.to(device)

                logits = model(images)
                prediction = logits.max(1).indices

                guesses += logits.shape[0]
                correct += th.sum(prediction == labels)

                loss = loss_fn(logits, labels)
                epoch_loss += loss
                test_progress.set_description(f"    test: {i}     (accuracy: {100 * correct / guesses:.3f}%, loss: {loss:.4f})")

            accuracies['test'] = correct / guesses
            losses['test'] = epoch_loss / len(test_progress)

        writer.add_scalars(f'{experiment_name}/accuracy', accuracies, i)
        writer.add_scalars(f'{experiment_name}/loss', losses, i)

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

    args = parser.parse_args()

    if args.device == 'cuda' and not th.cuda.is_available():
        print("cuda is not avaiable, falling back to CPU")
        args.device = 'cpu'

    model, *loaders = initialize(**vars(args))
    loop(model, *loaders, **vars(args))
