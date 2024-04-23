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
from system.controller.reachability_estimator.training.H5Dataset import H5Dataset
from system.controller.reachability_estimator.training.utils import save_model, load_model


def get_path():
    """ returns path to data storage folder """
    dirname = os.path.join(os.path.dirname(__file__), "..")
    return dirname


def _load_weights(model_file, nets, net_opts):
    state = load_model(os.path.dirname(model_file),
                       os.path.basename(model_file), load_to_cpu=True)
    epoch = int(state['epoch'])

    for name, net in nets.items():
        net.load_state_dict(state['nets'][name])

    for name, opt in net_opts.items():
        opt.load_state_dict(state['optims'][name])
        # Move the parameters stored in the optimizer into gpu
        for opt_state in opt.state.values():
            for k, v in opt_state.items():
                if torch.is_tensor(v):
                    opt_state[k] = v.to(device='cpu')
    return epoch


def _save_model(nets, net_opts, epoch, global_args, model_file):
    """ save current state of the model """
    state = {
        'epoch': epoch,
        'global_args': global_args,
        'optims': {
            name: opt.state_dict() for name, opt in net_opts.items()
        },
        'nets': {
            name: net.state_dict() for name, net in nets.items()
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


def tensor_log(title, loader, train_device, model_variant, writer, epoch, nets, backbone, position_loss_weight = 0.6, angle_loss_weight = 0.3):
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
            if model_variant == 'spikings':
                batch_src_imgs, batch_dst_imgs, batch_reachability, batch_transformation, batch_src_spikings, batch_dst_spikings = item
                batch_src_spikings = batch_src_spikings.to(device=train_device, non_blocking=True).float()
                batch_dst_spikings = batch_dst_spikings.to(device=train_device, non_blocking=True).float()
            else:
                batch_src_imgs, batch_dst_imgs, batch_reachability, batch_transformation = item
                batch_src_spikings = None
                batch_dst_spikings = None
            # Get predictions
            src_img = batch_src_imgs.to(device=train_device, non_blocking=True)
            dst_imgs = batch_dst_imgs.to(device=train_device, non_blocking=True)
            r = batch_reachability.to(device=train_device, non_blocking=True)
            transformation = batch_transformation.to(device=train_device, non_blocking=True)
            position = transformation[:, 0:2]
            angle = transformation[:, -1]

            src_batch = src_img.float()
            dst_batch = dst_imgs.float()
            prediction = get_prediction(nets, backbone, model_variant, src_batch, dst_batch, batch_transformation, batch_src_spikings, batch_dst_spikings)
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


def train_multiframedst(nets, net_opts, dataset, global_args):
    """ Train the model on a multiframe dataset. """
    (
        model_file,
        resume,
        batch_size,
        samples_per_epoch,
        max_epochs,
        lr_decay_epoch,
        lr_decay_rate,
        n_dataset_worker,
        train_device,
        log_interval,
        save_interval,
        model_variant,
        position_loss_weight,
        angle_loss_weight,
        backbone,
        with_grid_cell_spikings
    ) = [global_args[_] for _ in ['model_file',
                                  'resume',
                                  'batch_size',
                                  'samples_per_epoch',
                                  'max_epochs',
                                  'lr_decay_epoch',
                                  'lr_decay_rate',
                                  'n_dataset_worker',
                                  'train_device',
                                  'log_interval',
                                  'save_interval',
                                  'model_variant',
                                  'position_loss_weight',
                                  'angle_loss_weight',
                                  'backbone',
                                  'with_grid_cell_spikings'
    ]]

    # For Tensorboard: log the runs
    writer = SummaryWriter()

    epoch = 0

    # Resume: load weights and continue training
    if resume:
        epoch = _load_weights(model_file, nets, net_opts)
        torch.manual_seed(231239 + epoch)
        print('loaded saved state. epoch: %d' % epoch)

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
        for name, opt in net_opts.items()
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
            if with_grid_cell_spikings:
                batch_src_imgs, batch_dst_imgs, batch_reachability, batch_transformation, batch_src_spikings, batch_dst_spikings = item
                src_spikings_batch = batch_src_spikings.to(device=train_device, non_blocking=True).float()
                dst_spikings_batch = batch_dst_spikings.to(device=train_device, non_blocking=True).float()
            else:
                batch_src_imgs, batch_dst_imgs, batch_reachability, batch_transformation = item
                src_spikings_batch = None
                dst_spikings_batch = None
            # Zeros optimizer gradient
            for _, opt in net_opts.items():
                opt.zero_grad()

            # Get predictions
            src_img = batch_src_imgs.to(device=train_device, non_blocking=True)
            dst_img = batch_dst_imgs.to(device=train_device, non_blocking=True)
            r = batch_reachability.to(device=train_device, non_blocking=True)
            r = torch.clamp(r, 0.0, 1.0)
            transformation = batch_transformation.to(device=train_device, non_blocking=True)
            position = transformation[:, 0:2]
            angle = transformation[:, -1]

            src_batch = src_img.float()
            dst_batch = dst_img.float()

            prediction = get_prediction(nets, backbone, model_variant, src_batch, dst_batch, batch_transformation, src_spikings_batch, dst_spikings_batch)

            reachability_prediction, position_prediction, angle_prediction = prediction

            # Loss
            loss_reachability = torch.nn.functional.binary_cross_entropy(reachability_prediction, r, reduction='none')
            if position_prediction is None:
                loss = loss_reachability
            else:
                loss_position = torch.sqrt(torch.sum(torch.nn.functional.mse_loss(position_prediction, position, reduction='none'), dim=1))
                loss_angle = torch.sqrt(torch.nn.functional.mse_loss(angle_prediction, angle, reduction='none'))
                loss = loss_reachability + r @ (position_loss_weight * loss_position + angle_loss_weight * loss_angle)

            loss = loss.sum()
            # backwards gradient
            loss.backward()

            # optimizer step
            for _, opt in net_opts.items():
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
            _save_model(nets, net_opts, epoch, global_args, model_file)

        # Validation
        valid_loader = DataLoader(valid_dataset,
                                  batch_size=batch_size,
                                  num_workers=n_dataset_worker)

        # log performance on the validation set
        tensor_log("Validation", valid_loader, train_device, model_variant, writer, epoch, nets, backbone, position_loss_weight, angle_loss_weight)


def validate_func(model_file, nets, net_opts, dataset, backbone, batch_size, train_device, model_variant, position_loss_weight, angle_loss_weight):
    epoch = _load_weights(model_file, nets, net_opts)
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
    tensor_log("Validation", valid_loader, train_device, model_variant, writer, epoch, nets, backbone, position_loss_weight,
               angle_loss_weight)
    writer.flush()


if __name__ == '__main__':

    """ Train or test models for the reachability estimator

        Model Variants:
        -  "pair_conv": model as described in the paper
        -  "the_only_variant": model without added convolutional layer
        -  "with_dist": like "pair_conv", but adds the decoded goal_vector

    """

    global_args = {
        'model_file': os.path.join(os.path.dirname(__file__), "../data/models/no_siamese_mse_updated"),
        'resume': False,
        'batch_size': 64,
        'samples_per_epoch': 10000,
        'max_epochs': 50,
        'lr_decay_epoch': 1,
        'lr_decay_rate': 0.7,
        'n_dataset_worker': 0,
        'log_interval': 20,
        'save_interval': 5,
        'model_variant': "spikings",  # "pair_conv",#"with_dist",#"the_only_variant",
        'train_device': "cpu",
        'position_loss_weight': 0.006,
        'angle_loss_weight': 0.003,
        'backbone': 'convolutional',  # convolutional, res_net
        'with_grid_cell_spikings': True,
        'external_link': False
    }

    # Defining the NN and optimizers
    nets = initialize_network(global_args['backbone'], global_args['model_variant'])

    testing = False
    validate = False

    if validate:
        hd5file = "trajectories.hd5"
        directory = get_path()
        directory = os.path.join(directory, "data/reachability")
        filepath = os.path.join(directory, hd5file)
        filepath = os.path.realpath(filepath)
        validate_func(global_args['model_file'],
                 {name: spec['net'] for name, spec in nets.items()},
                 {name: spec['opt'] for name, spec in nets.items()},
                 H5Dataset(filepath),
                 global_args['backbone'],
                 global_args['batch_size'],
                 global_args['train_device'],
                 global_args['model_variant'],
                 global_args['position_loss_weight'],
                 global_args['angle_loss_weight'])
    elif testing:
        # Testing
        hd5file = "trajectories.hd5"
        directory = get_path()
        directory = os.path.join(directory, "data/reachability")
        filepath = os.path.join(directory, hd5file)
        filepath = os.path.realpath(filepath)

        dataset = H5Dataset(filepath)

        run_test_model(dataset)
    else:
        # Training
        hd5file = "dataset_spikings.hd5"

        directory = get_path()
        directory = os.path.join(directory, "data/reachability")
        filepath = os.path.join(directory, hd5file)
        filepath = os.path.realpath(filepath)

        if global_args['with_grid_cell_spikings']:
            dataset = H5DatasetWithSpikings(filepath, global_args['external_link'])
        else:
            dataset = H5Dataset(filepath, global_args['external_link'])

        train_multiframedst(
            nets={
                name: spec['net'] for name, spec in nets.items()
            },
            net_opts={
                name: spec['opt'] for name, spec in nets.items() if spec['opt'] is not None
            },
            dataset=dataset,
            global_args=global_args)
