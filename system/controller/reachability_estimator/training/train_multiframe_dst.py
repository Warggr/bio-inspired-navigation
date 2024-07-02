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
from typing import Type, Literal, Callable, Dict, Any

if __name__ == "__main__":
    sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))

from system.controller.reachability_estimator.networks import Model
from system.controller.reachability_estimator.ReachabilityDataset import ReachabilityDataset, SampleConfig
from system.controller.reachability_estimator.training.utils import load_model, DATA_STORAGE_FOLDER
from system.controller.reachability_estimator.types import Prediction

def _load_weights(model_file, nets : Model, **kwargs) -> int:
    state, epoch = load_model( model_file, load_to_cpu=True, **kwargs)

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


def run_test_model(dataset, filename = "trained_model_new.50"):
    """ Test model on dataset. Logs accuracy, precision, recall and f1score. """

    from system.controller.reachability_estimator.reachability_estimation import NetworkReachabilityEstimator
    filepath = os.path.join(DATA_STORAGE_FOLDER, filename)
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

Batch = Any # TODO
TrainDevice = Literal['cpu', 'gpu'] # TODO (Pierre): not sure about how exactly 'gpu' is called

def process_batch(item : Batch, train_device : TrainDevice):
    model_args, ground_truth = item
    model_args = [ data.to(device=train_device, non_blocking=True) for data in model_args ]
    ground_truth = [ data.to(device=train_device, non_blocking=True) for data in ground_truth ]
    return model_args, ground_truth

LossFunction = Callable[[Prediction, Prediction], torch.Tensor]

def make_loss_function(position_loss_weight = 0.6, angle_loss_weight = 0.3) -> LossFunction:
    def loss_function(prediction : Prediction, truth : Prediction, return_details = False) -> torch.Tensor:
        reachability_prediction, position_prediction, angle_prediction = prediction
        reachability, position, angle = truth

        loss_reachability = torch.nn.functional.binary_cross_entropy(reachability_prediction, reachability, reduction='none')
        if position_prediction is None:
            loss = loss_reachability
        else:
            loss_position = torch.sum(torch.nn.functional.mse_loss(position_prediction, position, reduction='none'), dim=1) / len(position)
            loss_angle = torch.sum((1 - torch.cos(angle_prediction - angle)) ** 2) / len(angle) # see e.g. https://stats.stackexchange.com/a/425270
            loss = loss_reachability + reachability @ (position_loss_weight * loss_position + angle_loss_weight * loss_angle)

        # Loss
        loss = loss.sum()
        assert not loss.isnan()
        if return_details:
            return loss, (loss_reachability, loss_position, loss_angle)
        else:
            return loss
    loss_function.hparams = { 'position_loss_weight': position_loss_weight, 'angle_loss_weight': angle_loss_weight }
    return loss_function

def tensor_log(
    loader : DataLoader,
    train_device,
    writer : SummaryWriter,
    epoch : int,
    net: Model,
    loss_function: LossFunction,
):
    """ Log accuracy, precision, recall and f1score for dataset in loader."""
    with torch.no_grad():
        log_loss = 0

        metrics = {
            'Metrics/Accuracy': BinaryAccuracy(),
            'Metrics/Precision': BinaryPrecision(),
            'Metrics/Recall': BinaryRecall(),
            'Metrics/Fscore': BinaryF1Score(),
        }
        loss_detail_names = ['Loss/Reachability', 'Loss/Position', 'Loss/Angle']
        log_scores = { key: 0 for key in list(metrics.keys()) + loss_detail_names }

        for idx, item in enumerate(loader):
            model_args, ground_truth = process_batch(item, train_device=train_device)

            prediction = nets.get_prediction(*model_args)
            loss, loss_details = loss_function(prediction, ground_truth, return_details=True)

            loss_reachability, loss_position, loss_angle = loss_details
            reachability_prediction, position_prediction, angle_prediction = prediction
            reachability, position, angle = ground_truth

            log_loss += loss.item()
            for key, metric in metrics.items():
                log_scores[key] += metric(reachability_prediction, reachability.int())
            for key, loss_detail in zip(loss_detail_names, loss_details):
                log_scores[key] += loss_detail.sum().item()

    metrics : Dict[str, Any] = {}
    metrics["Loss/Validation"] = log_loss / len(loader)
    for key, value in log_scores.items():
        metrics[key] = value / len(loader)

    for key, value in metrics.items():
        writer.add_scalar(key, value, epoch)
    return metrics

from dataclasses import dataclass

@dataclass
class Hyperparameters:
    batch_size : int = 64
    samples_per_epoch : int = 10000
    max_epochs : int = 25
    lr: float = 3e-4
    lr_decay_epoch : int = 1
    lr_decay_rate : float = 0.7
    eps: float = 1e-5

def train_multiframedst(
    nets : Model, dataset : ReachabilityDataset,
    train_device : TrainDevice,
    resume : bool = False,
    hyperparams : Hyperparameters = Hyperparameters(),
    n_dataset_worker = 0,
    log_interval = 20,
    save_interval = 5,
    model_suffix : str = '',
    model_filename = "reachability_network",
    model_dir = DATA_STORAGE_FOLDER,
    loss_function : LossFunction = make_loss_function(),
):
    """ Train the model on a multiframe dataset. """

    # For Tensorboard: log the runs
    writer = SummaryWriter(comment=model_suffix)

    epoch = 0

    model_filename = model_filename + model_suffix
    model_file = os.path.join(model_dir, model_filename)
    # Resume: load weights and continue training
    if resume:
        try:
            epoch = _load_weights(model_file, nets)
            torch.manual_seed(231239 + epoch)
            print('loaded saved state. epoch: %d' % epoch)
        except FileNotFoundError:
            epoch = 0
            print('No saved state found. Starting from the beginning. epoch: 0')

    if epoch == 0 and nets.image_encoder == 'pretrained': # TODO Pierre: this is ugly
        nets.load_pretrained_model()

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
            step_size=hyperparams.lr_decay_epoch,
            gamma=hyperparams.lr_decay_rate,
            last_epoch=last_epoch)
        for name, opt in nets.optimizers.items()
    }

    n_samples = hyperparams.samples_per_epoch

    # Splitting the Dataset into Train/Validation:
    train_size = int(0.8 * len(dataset))
    valid_size = len(dataset) - train_size
    train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [train_size, valid_size])

    latest_metrics = None

    for epoch in range(epoch + 1, hyperparams.max_epochs + 1):
        print('===== epoch %d =====' % epoch)

        sampler = RandomSampler(train_dataset, True, n_samples)

        loader = DataLoader(train_dataset,
                            batch_size=hyperparams.batch_size,
                            sampler=sampler,
                            num_workers=n_dataset_worker,
                            pin_memory=True,
                            drop_last=True)

        last_log_time = time.time()

        for idx, item in enumerate(loader):
            # Zeros optimizer gradient
            for _, opt in nets.optimizers.items():
                opt.zero_grad()

            model_args, ground_truth = process_batch(item, train_device=train_device)

            prediction = nets.get_prediction(*model_args)
            loss = loss_function(prediction, ground_truth)

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

                writer.add_scalar("Loss/train",loss, epoch*n_samples+idx*hyperparams.batch_size)
                last_log_time = time.time()

        # learning rate decay
        for _, sched in net_scheds.items():
            sched.step()

        if (save_interval is not None and epoch % save_interval == 0) or epoch == hyperparams.max_epochs:
            print('saving model...')
            writer.flush()
            nets.save(epoch, hyperparams, model_file)

        # Validation
        valid_loader = DataLoader(valid_dataset,
                                  batch_size=hyperparams.batch_size,
                                  num_workers=n_dataset_worker)

        # log performance on the validation set
        latest_metrics = tensor_log(valid_loader, train_device, writer, epoch, nets, loss_function)

    if latest_metrics is None:
        valid_loader = DataLoader(valid_dataset, batch_size=hyperparams.batch_size, num_workers=n_dataset_worker)
        latest_metrics = tensor_log(valid_loader, train_device, writer, hyperparams.max_epochs, nets, loss_function)
    nets_hparams = { 'image_encoder': nets.image_encoder, 'with_conv_layer': nets.image_encoder == 'conv' } # with_conv_layer is for backwards compatibility
    hparams = vars(nets.sample_config) | nets_hparams | vars(hyperparams) | getattr(loss_function, 'hparams', {})
    latest_metrics = { "Final/"+key: value for key, value in latest_metrics.items() }
    print("Writing metrics:", latest_metrics)
    writer.add_hparams(hparams, latest_metrics)


def validate_func(net : Model, dataset : ReachabilityDataset, batch_size, train_device,
    loss_function : LossFunction,
    model_suffix : str,
    model_filename = "reachability_network",
    model_dir = DATA_STORAGE_FOLDER,
):
    model_filename = model_filename + model_suffix
    model_file = os.path.join(model_dir, model_filename)
    epoch = _load_weights(model_file, nets)
    print('loaded saved state. epoch: %d' % epoch)
    train_size = int(0.8 * len(dataset))
    valid_size = len(dataset) - train_size
    train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [train_size, valid_size])
    writer = SummaryWriter(comment=model_suffix)

    # Validation
    valid_loader = DataLoader(valid_dataset,
                              batch_size=batch_size,
                              num_workers=0)

    # log performance on the validation set
    tensor_log(valid_loader, train_device, writer, epoch, net, loss_function)
    writer.flush()
    print("Run info written to", writer.log_dir)

def optional(typ):
    def _parser(st):
        if st.lower() in ['none', 'off']:
            return None
        return typ(st)
    return _parser

if __name__ == '__main__':

    """ Train or test models for the reachability estimator

        Model Variants:
        -  "pair_conv": model as described in the paper
        -  "the_only_variant": model without added convolutional layer
        -  "with_dist": like "pair_conv", but adds the decoded goal_vector

    """

    import argparse

    hyperparams_parser = argparse.ArgumentParser(add_help=False)
    for field in Hyperparameters.__dataclass_fields__.values():
        hyperparams_parser.add_argument('--' + field.name.replace('_', '-'), help='hyperparameter ' + field.name, type=field.type, default=field.default)

    model_basename = 'reachability_network'
    parser = argparse.ArgumentParser(parents=[hyperparams_parser], formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('mode', choices=['train', 'test', 'validate'], help='mode')
    parser.add_argument('--dataset-features', nargs='+', default=[])
    parser.add_argument('--dataset-basename', help='The base name of the reachability dataset HD5 file', default='dataset')
    parser.add_argument('--tag', help=f'Network saved in `{model_basename}-{{tag}}`', default='')
    parser.add_argument('--images', help='Images are included in the dataset', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--spikings', help='Grid cell spikings are included in the dataset', action='store_true')
    parser.add_argument('--lidar', help='LIDAR distances are included in the dataset', choices=['raw_lidar', 'ego_bc', 'allo_bc'])
    parser.add_argument('--image-encoder', help='Image encoder', choices=['fc', 'conv', 'pretrained'], default='conv')
    parser.add_argument('--hidden-fc-layers', help='Hidden FC layer dimensions as a comma-separated list', type=lambda s: [int(i) for i in s.split(',')])
    parser.add_argument('--dist', help='Provide the distance and angle to the reachability estimator', action='store_true')
    parser.add_argument('--resume', action='store_true', help='Continue training from last saved model')
    parser.add_argument('--save-interval', type=optional(int))

    args = parser.parse_args()

    config = SampleConfig(
        grid_cell_spikings=args.spikings,
        lidar=args.lidar,
        images=args.images,
        dist=args.dist,
    )

    suffix = ''
    if args.tag:
        suffix += '-' + args.tag
    args.dataset_features = ''.join([ f'-{feature}' for feature in args.dataset_features ])
    suffix += args.dataset_features
    suffix += config.suffix()
    if args.image_encoder:
        suffix += '+' + args.image_encoder
    if args.hidden_fc_layers:
        suffix += '+fc' + ','.join(map(str, args.hidden_fc_layers))

    filename = args.dataset_basename + args.dataset_features + ".hd5"
    dataset = ReachabilityDataset(filename, sample_config=config)

    backbone = 'convolutional' # convolutional, res_net

    # Defining the NN and optimizers
    hyperparameters = Hyperparameters(batch_size=64)

    nets = Model.create_from_config(backbone, config, image_encoder=args.image_encoder, hidden_fc_layers=args.hidden_fc_layers)

    loss_function = make_loss_function(position_loss_weight=0.6, angle_loss_weight=0.3)
    global_args = {
        'train_device': "cpu",
        'loss_function': loss_function,
        'model_filename': model_basename,
        'model_suffix': suffix,
    }

    if args.mode == "validate":
        validate_func(nets, dataset, batch_size=64, **global_args)
    elif args.mode == "test":
        run_test_model(dataset)
    elif args.mode == "train":
        print("Training with dataset of length", len(dataset))
        train_multiframedst(nets, dataset, save_interval=args.save_interval, resume=args.resume, hyperparams=Hyperparameters(batch_size=64), **global_args)
