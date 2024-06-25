import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, RandomSampler
from torch.utils.tensorboard import SummaryWriter

import os
import time

if __name__ == "__main__":
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))

from system.controller.reachability_estimator.training.utils import load_model
from system.controller.reachability_estimator.training.train_multiframe_dst import DATA_STORAGE_FOLDER, Hyperparameters, TrainDevice
from system.controller.reachability_estimator.networks import AutoAdamOptimizer, NNModuleWithOptimizer
from system.controller.reachability_estimator.autoencoders import ImageAutoencoder
from system.types import Image
from typing import Any

def eval_performance(
    nets: ImageAutoencoder,
    loader: DataLoader[Image],
    loss_function,
) -> dict[str, float]:
    with torch.no_grad():
        log_loss = 0

        for image in loader:
            decoded = nets(image.float())
            log_loss += loss_function(decoded, image)

    metrics : dict[str, Any] = {}
    metrics["Loss/Validation"] = log_loss / len(loader)
    return metrics

def log_performance(
    performance: dict[str, float],
    writer: SummaryWriter,
    epoch: int,
):
    for key, value in performance.items():
        writer.add_scalar(key, value, epoch)


def train(
    nets : ImageAutoencoder, dataset : Dataset[Image],
    resume = False,
    hyperparams : Hyperparameters = Hyperparameters(),
    n_dataset_worker = 0,
    log_interval = 20,
    save_interval = 5,
    model_dir = DATA_STORAGE_FOLDER,
    model_filename = 'autoencoder',
    model_suffix : str = '',
):
    """ Train the model on a multiframe dataset. """

    # For Tensorboard: log the runs
    writer = SummaryWriter(comment=model_suffix)

    model_filename = model_filename + model_suffix
    model_file = os.path.join(model_dir, model_filename)

    epoch = 0
    if resume:
        try:
            state, epoch = load_model(model_file)
            nets.load_state_dict(state)

            torch.manual_seed(231239 + epoch)
            print('loaded saved state. epoch: %d' % epoch)
        except FileNotFoundError:
            epoch = 0
            print('No saved state found. Starting from the beginning. epoch: 0')

    # FIXME: hack to mitigate the bug in torch 1.1.0's schedulers
    last_epoch = epoch - 1 if epoch <= 1 else epoch - 2

    # Scheduler: takes care of learning rate decay
    scheduler = torch.optim.lr_scheduler.StepLR(
        nets.optimizer,
        step_size=hyperparams.lr_decay_epoch,
        gamma=hyperparams.lr_decay_rate,
        last_epoch=last_epoch,
    )

    loss_function = nn.MSELoss()

    # Splitting the Dataset into Train/Validation:
    train_size = int(0.8 * len(dataset))
    valid_size = len(dataset) - train_size
    train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [train_size, valid_size])

    latest_metrics = None

    for epoch in range(epoch + 1, hyperparams.max_epochs + 1):
        print('===== epoch %d =====' % epoch)

        sampler = RandomSampler(train_dataset, True, hyperparams.samples_per_epoch)

        loader = DataLoader(train_dataset,
                            batch_size=hyperparams.batch_size,
                            sampler=sampler,
                            num_workers=n_dataset_worker,
                            pin_memory=True,
                            drop_last=True)

        last_log_time = time.time()

        for idx, item in enumerate(loader):
            nets.optimizer.zero_grad()

            item = item.float()
            decoded = nets(item)
            loss = loss_function(decoded, item)
            loss.backward()
            nets.optimizer.step()

            # Logging the run
            if idx % log_interval == 0:
                print(f'epoch {epoch}; batch time {time.time() - last_log_time}; sec loss: {loss.item()}')
                writer.add_scalar("Loss/train",loss, epoch*hyperparams.samples_per_epoch+idx*hyperparams.batch_size)
                last_log_time = time.time()

        # learning rate decay
        scheduler.step()

        if epoch % save_interval == 0 or epoch == hyperparams.max_epochs:
            print('saving model...')
            writer.flush()
            nets.save(hyperparams, model_file + f'.{epoch}')

        # Validation
        valid_loader = DataLoader(valid_dataset,
                                  batch_size=hyperparams.batch_size,
                                  num_workers=n_dataset_worker)

        # log performance on the validation set
        latest_metrics = eval_performance(nets, valid_loader, loss_function)
        log_performance(latest_metrics, writer, epoch)

    if latest_metrics is None:
        valid_loader = DataLoader(valid_dataset, batch_size=hyperparams.batch_size, num_workers=n_dataset_worker)
        latest_metrics = eval_performance(nets, valid_loader, loss_function)
        log_performance(latest_metrics, writer, hyperparams.max_epochs)

if __name__ == "__main__":
    from system.controller.reachability_estimator.ReachabilityDataset import ReachabilityDataset

    dataset_features = '-3colors'

    dataset = ReachabilityDataset(filename='dataset' + dataset_features + ".hd5")

    class ImageDataset(Dataset[Image]):
        def __init__(self, dataset: ReachabilityDataset):
            self.dataset = dataset
        def __len__(self):
            return len(self.dataset)
        def __getitem__(self, idx) -> Image:
            sample, _ = self.dataset.sample(idx)
            return sample.src.img

    dataset = ImageDataset(dataset)

    net = ImageAutoencoder()

    train(net, dataset, resume=True)
