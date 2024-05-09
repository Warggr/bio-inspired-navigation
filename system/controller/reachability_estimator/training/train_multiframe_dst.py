''' This code has been adapted from:
***************************************************************************************
*    Title: "Scaling Local Control to Large Scale Topological Navigation"
*    Author: "Xiangyun Meng, Nathan Ratliff, Yu Xiang and Dieter Fox"
*    Date: 2020
*    Availability: https://github.com/xymeng/rmp_nav
*
***************************************************************************************
'''

import torch
import time
from torchmetrics import Accuracy, F1Score
from torchmetrics.classification import BinaryPrecision, BinaryRecall, BinaryAccuracy, BinaryF1Score
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.tensorboard import SummaryWriter

import sys
import os
import numpy as np
from typing import Type

if __name__ == "__main__":
    sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))

from system.controller.reachability_estimator.networks import Model
from system.controller.reachability_estimator.ReachabilityDataset import ReachabilityDataset, SampleConfig
from system.controller.reachability_estimator.training.utils import save_model, load_model


def get_path():
    """ returns path to data storage folder """
    dirname = os.path.join(os.path.dirname(__file__), "..")
    return dirname


def _load_weights(model_file, nets : Model, **kwargs):
    state = load_model( os.path.dirname(model_file), os.path.basename(model_file), load_to_cpu=True, **kwargs)
    epoch = int(state['epoch'])

    for name, net in nets.nets.items():
        if name == "img_encoder" and 'conv1.weight' in state['nets'][name].keys(): # Backwards compatibility
            for i in range(4):
                for value_type in ['bias', 'weight']:
                    state['nets'][name][f'layers.{2*i}.{value_type}'] = state['nets'][name][f'conv{i+1}.{value_type}']
                    del state['nets'][name][f'conv{i+1}.{value_type}']
        net.load_state_dict(state['nets'][name])

    for name, opt in nets.optimizers.items():
        opt.load_state_dict(state['optims'][name])
        # Move the parameters stored in the optimizer into gpu
        for opt_state in opt.state.values():
            for k, v in opt_state.items():
                if torch.is_tensor(v):
                    opt_state[k] = v.to(device='cpu')
    return epoch


def _save_model(nets : Model, epoch, global_args, model_file):
    """ save current state of the model """
    state = {
        'epoch': epoch,
        'global_args': global_args,
        'optims': {
            name: opt.state_dict() for name, opt in nets.optimizers.items()
        },
        'nets': {
            name: net.state_dict() for name, net in nets.nets.items()
        }
    }
    save_model(state, epoch, '', model_file)


def run_test_model(dataset):
    """ Test model on dataset. Logs accuracy, precision, recall and f1score. """

    from system.controller.reachability_estimator.reachability_estimation import NetworkReachabilityEstimator
    filename = "trained_model_new.50"
    filepath = os.path.join(os.path.join(os.path.dirname(__file__), "../data/models"), filename)
    reach_estimator = NetworkReachabilityEstimator(weights_file=filepath)

    n_samples = 6400
    sampler = RandomSampler(dataset, True, n_samples)

    loader = DataLoader(dataset,
                        batch_size=reach_estimator.batch_size,
                        sampler=sampler,
                        num_workers=0,
                        pin_memory=True,
                        drop_last=True)

    test_accuracy = 0
    test_precision = 0
    test_recall = 0
    test_f1 = 0

    accuracy = BinaryAccuracy()
    precision = BinaryPrecision()
    recall = BinaryRecall()
    f1 = BinaryF1Score()
    for idx, item in enumerate(loader):
        print(f"Processing batch {idx} out of {n_samples // reach_estimator.batch_size}")
        batch_src_imgs, batch_dst_imgs, batch_reachability, batch_transformation = item
        src_img = batch_src_imgs.to(device="cpu", non_blocking=True)
        dst_imgs = batch_dst_imgs.to(device="cpu", non_blocking=True)
        r = batch_reachability.to(device="cpu", non_blocking=True)

        src_batch = src_img.float()
        dst_batch = dst_imgs.float()

        pred_r = reach_estimator.predict_reachability_batch(src_batch, dst_batch)
        pred_r = torch.from_numpy(pred_r)

        test_accuracy += accuracy(pred_r, r.int())

        test_precision += precision(pred_r, r.int())
        test_recall += recall(pred_r, r.int())

        test_f1 += f1(pred_r, r.int())

    writer = SummaryWriter()
    test_accuracy /= len(loader)
    test_precision /= len(loader)
    test_recall /= len(loader)
    test_f1 /= len(loader)

    writer.add_scalar("Accuracy/Testing", test_accuracy, 1)
    writer.add_scalar("Precision/Testing", test_precision, 1)
    writer.add_scalar("Recall/Testing", test_recall, 1)
    writer.add_scalar("fscore/Testing", test_f1, 1)


def tensor_log(title: str, loader: DataLoader, train_device, writer, epoch, net: Model, position_loss_weight = 0.6, angle_loss_weight = 0.3):
    """ Log accuracy, precision, recall and f1score for dataset in loader."""
    with torch.no_grad():
        log_loss = 0
        log_precision = 0
        log_recall = 0
        log_accuracy = 0
        log_f1 = 0
        accuracy = BinaryAccuracy()
        precision = BinaryPrecision()
        recall = BinaryRecall()
        f1 = BinaryF1Score()
        for idx, item in enumerate(loader):
            batch_src_imgs, batch_dst_imgs, batch_reachability, batch_transformation, *other_data = item
            batch_src_spikings, batch_dst_spikings, batch_src_distances = None, None, None
            other_data = iter(other_data)
            if net.with_grid_cell_spikings:
                batch_src_spikings, batch_dst_spikings = next(other_data), next(other_data)
                batch_src_spikings = batch_src_spikings.to(device=train_device, non_blocking=True).float()
                batch_dst_spikings = batch_dst_spikings.to(device=train_device, non_blocking=True).float()
            if net.with_lidar:
                batch_src_distances = next(other_data)
                batch_src_distances = batch_src_distances.to(device=train_device, non_blocking=True).float()
            # Get predictions
            src_img = batch_src_imgs.to(device=train_device, non_blocking=True)
            dst_imgs = batch_dst_imgs.to(device=train_device, non_blocking=True)
            r = batch_reachability.to(device=train_device, non_blocking=True)
            r = torch.clamp(r, 0.0, 1.0) # to make it into a float (instead of bool)
            transformation = batch_transformation.to(device=train_device, non_blocking=True)
            position = transformation[:, 0:2]
            angle = transformation[:, -1]

            src_batch = src_img.float()
            dst_batch = dst_imgs.float()
            prediction = net.get_prediction(src_batch, dst_batch, batch_transformation, batch_src_spikings, batch_dst_spikings, batch_src_distances)
            reachability_prediction, position_prediction, angle_prediction = prediction

            loss_reachability = torch.nn.functional.binary_cross_entropy(reachability_prediction, r,
                                                                         reduction='none')
            if position_prediction is None:
                new_loss = loss_reachability
            else:
                loss_position = torch.sqrt(torch.sum(
                    torch.nn.functional.mse_loss(position_prediction, position, reduction='none'), dim=1))
                loss_angle = torch.sqrt(torch.nn.functional.mse_loss(angle_prediction, angle, reduction='none'))

                # backwards gradient
                new_loss = loss_reachability + r @ (position_loss_weight * loss_position + angle_loss_weight * loss_angle)

            log_loss += new_loss.sum().item()
            log_precision += precision(reachability_prediction, r.int())
            log_recall += recall(reachability_prediction, r.int())
            accuracy = Accuracy(task="binary")
            log_accuracy += accuracy(reachability_prediction, r.int())
            f1 = F1Score(task="binary")
            log_f1 += f1(reachability_prediction, r.int())

        log_loss /= len(loader)
        log_precision /= len(loader)
        log_recall /= len(loader)
        log_accuracy /= len(loader)
        log_f1 /= len(loader)

        writer.add_scalar("Accuracy/" + title, log_accuracy, epoch)
        writer.add_scalar("Precision/" + title, log_precision, epoch)
        writer.add_scalar("Recall/" + title, log_recall, epoch)
        writer.add_scalar("Loss/" + title, log_loss, epoch)
        writer.add_scalar("Fscore/" + title, log_f1, epoch)


def train_multiframedst(
    nets : Model, dataset : ReachabilityDataset,
    model_file,
    resume : bool,
    batch_size,
    samples_per_epoch,
    max_epochs,
    lr_decay_epoch,
    lr_decay_rate,
    n_dataset_worker,
    train_device,
    log_interval,
    save_interval,
    position_loss_weight,
    angle_loss_weight,
    backbone,
):
    """ Train the model on a multiframe dataset. """

    # For Tensorboard: log the runs
    writer = SummaryWriter()

    epoch = 0

    # Resume: load weights and continue training
    if resume:
        try:
            epoch = _load_weights(model_file, nets)
            torch.manual_seed(231239 + epoch)
            print('loaded saved state. epoch: %d' % epoch)
        except FileNotFoundError:
            epoch = 0
            print('No saved state found. Starting from the beginning. epoch: 0')

    # This is a direct copy of rmp_nav
    # FIXME: hack to mitigate the bug in torch 1.1.0's schedulers
    if epoch <= 1:
        last_epoch = epoch - 1
    else:
        last_epoch = epoch - 2

    # Scheduler: takes care of learning rate decay
    net_scheds = {
        name: torch.optim.lr_scheduler.StepLR(
            opt,
            step_size=lr_decay_epoch,
            gamma=lr_decay_rate,
            last_epoch=last_epoch)
        for name, opt in nets.optimizers.items()
    }

    n_samples = samples_per_epoch

    # Splitting the Dataset into Train/Validation:
    train_size = int(0.8 * len(dataset))
    valid_size = len(dataset) - train_size
    train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [train_size, valid_size])

    while True:
        print('===== epoch %d =====' % epoch)

        sampler = RandomSampler(train_dataset, True, n_samples)

        loader = DataLoader(train_dataset,
                            batch_size=batch_size,
                            sampler=sampler,
                            num_workers=n_dataset_worker,
                            pin_memory=True,
                            drop_last=True)

        last_log_time = time.time()

        for idx, item in enumerate(loader):
            item = [ data.to(device=train_device, non_blocking=True) for data in item ]
            src_imgs, dst_imgs, reachability, transformation, *other_args = item
            src_imgs = src_imgs.to(device=train_device, non_blocking=True).float()
            dst_imgs = dst_imgs.to(device=train_device, non_blocking=True).float()
            try:
                assert not torch.any(src_imgs.isnan())
                assert not torch.any(dst_imgs.isnan())
                assert not torch.any(reachability.isnan())
            except AssertionError:
                print(f'src_imgs with ids [{idx}] contained NaN - skipping')
            reachability = torch.clamp(reachability, 0.0, 1.0) # mainly to make it a tensor of float
            position = transformation[:, 0:2]
            angle = transformation[:, -1]

            # Zeros optimizer gradient
            for _, opt in nets.optimizers.items():
                opt.zero_grad()

            # Get predictions
            prediction = nets.get_prediction(src_imgs, dst_imgs, transformation, *other_args)

            reachability_prediction, position_prediction, angle_prediction = prediction
            assert not np.isnan(sum(reachability_prediction.detach().numpy()))

            # Loss
            loss_reachability = torch.nn.functional.binary_cross_entropy(reachability_prediction, reachability, reduction='none')
            if position_prediction is None:
                loss = loss_reachability
            else:
                loss_position = torch.sum(torch.nn.functional.mse_loss(position_prediction, position, reduction='none'), dim=1)
                loss_angle = torch.nn.functional.mse_loss(angle_prediction, angle, reduction='none')
                loss = loss_reachability + reachability @ (position_loss_weight * loss_position + angle_loss_weight * loss_angle)

            loss = loss.sum()
            assert not loss.isnan()
            # backwards gradient
            loss.backward()

            # optimizer step
            for _, opt in nets.optimizers.items():
                opt.step()

            # Logging the run
            if idx % log_interval == 0:
                print(f'epoch {epoch}; batch time {time.time() - last_log_time}; sec loss: {loss.item()}')
                # print(f"learning rate:\n{tabulate.tabulate([(name, opt.param_groups[0]['lr']) for name, opt in net_opts.items()])}")
                # for name, net in nets.items():
                #     print(f'{name} grad:\n{module_grad_stats(net)}')

                writer.add_scalar("Loss/train",loss, epoch*n_samples+idx*batch_size)
                last_log_time = time.time()

        # learning rate decay
        for _, sched in net_scheds.items():
            sched.step()

        epoch += 1
        if epoch > max_epochs:
            writer.flush()
            break

        if epoch % save_interval == 0:
            print('saving model...')
            writer.flush()
            _save_model(nets, epoch, global_args, model_file)

        # Validation
        valid_loader = DataLoader(valid_dataset,
                                  batch_size=batch_size,
                                  num_workers=n_dataset_worker)

        # log performance on the validation set
        try:
            tensor_log("Validation", valid_loader, train_device, writer, epoch, nets, position_loss_weight, angle_loss_weight)
        except AssertionError:
            print('Could not compute val error due to NaN :(')


def validate_func(model_file, net : Model, dataset, batch_size, train_device, position_loss_weight, angle_loss_weight):
    epoch = _load_weights(model_file, nets)
    print('loaded saved state. epoch: %d' % epoch)
    train_size = int(0.8 * len(dataset))
    valid_size = len(dataset) - train_size
    train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [train_size, valid_size])
    writer = SummaryWriter()

    # Validation
    valid_loader = DataLoader(valid_dataset,
                              batch_size=batch_size,
                              num_workers=0)

    # log performance on the validation set
    tensor_log("Validation", valid_loader, train_device, writer, epoch, net, position_loss_weight, angle_loss_weight)
    writer.flush()


if __name__ == '__main__':

    """ Train or test models for the reachability estimator

        Model Variants:
        -  "pair_conv": model as described in the paper
        -  "the_only_variant": model without added convolutional layer
        -  "with_dist": like "pair_conv", but adds the decoded goal_vector

    """

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['train', 'test', 'validate'])
    parser.add_argument('--spikings', dest='with_grid_cell_spikings', help='Grid cell spikings are included in the dataset', action='store_true')
    parser.add_argument('--lidar', help='LIDAR distances are included in the dataset', choices=['raw_lidar', 'ego_bc', 'allo_bc'])
    parser.add_argument('--pair-conv', dest='with_conv_layer', help='Pair-conv neural network', action='store_true', default=True)
    parser.add_argument('--with_dist', help='<TODO: I\'m not sure what that is>', action='store_true')
    parser.add_argument('--resume', action='store_true')

    args = parser.parse_args()

    config = SampleConfig(
        grid_cell_spikings=args.with_grid_cell_spikings,
        lidar=args.lidar,
    )

    suffix = config.suffix()
    suffix = (suffix
        + ('+conv' if args.with_conv_layer else '')
    )

    global_args = {
        'model_file': os.path.join(os.path.dirname(__file__), "..", "data", "models", "reachability_network" + suffix),
        'resume': args.resume,
        'batch_size': 64,
        'samples_per_epoch': 10000,
        'max_epochs': 50,
        'lr_decay_epoch': 1,
        'lr_decay_rate': 0.7,
        'n_dataset_worker': 0,
        'log_interval': 20,
        'save_interval': 5,
        'train_device': "cpu",
        'position_loss_weight': 0.006,
        'angle_loss_weight': 0.003,
        'backbone': 'convolutional',  # convolutional, res_net
    }
    model_kwargs = { key: getattr(args, key) for key in ['with_conv_layer', 'with_dist'] }

    # Defining the NN and optimizers
    nets = Model.create_from_config(global_args['backbone'], **model_kwargs)

    if args.mode == "validate":
        filepath = os.path.join(get_path(), "data", "reachability", "trajectories.hd5")
        filepath = os.path.realpath(filepath)
        dataset = ReachabilityDataset(filepath)

        validate_func(global_args['model_file'],
                 nets,
                 dataset,
                 global_args['batch_size'],
                 global_args['train_device'],
                 global_args['position_loss_weight'],
                 global_args['angle_loss_weight'])
    elif args.mode == "test":
        # Testing
        filepath = os.path.join(get_path(), "data", "reachability", "trajectories.hd5")
        filepath = os.path.realpath(filepath)
        dataset = ReachabilityDataset(filepath)
        run_test_model(dataset)
    elif args.mode == "train":
        # Training
        filepath = os.path.join(get_path(), "data", "reachability", 'dataset.hd5')
        filepath = os.path.realpath(filepath)

        dataset = ReachabilityDataset(filepath)
        print("Training with dataset of length", len(dataset))

        train_multiframedst(nets, dataset=dataset, **global_args)
