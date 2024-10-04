''' This code has been adapted from:
***************************************************************************************
*    Title: "Scaling Local Control to Large Scale Topological Navigation"
*    Author: "Xiangyun Meng, Nathan Ratliff, Yu Xiang and Dieter Fox"
*    Date: 2020
*    Availability: https://github.com/xymeng/rmp_nav
*
***************************************************************************************
'''

import sys
import os
import torch
import time
import numpy as np
from torchmetrics.classification import BinaryPrecision, BinaryRecall, BinaryAccuracy, BinaryF1Score
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.tensorboard import SummaryWriter
from system.controller.reachability_estimator.networks import Model
from system.controller.reachability_estimator.ReachabilityDataset import ReachabilityDataset, SampleConfig
from system.controller.reachability_estimator.training.utils import load_model, DATA_STORAGE_FOLDER
from system.controller.reachability_estimator._types import Prediction
from typing import Literal, Callable, Any

TrainDevice = Literal['cpu', 'cuda']
TRAIN_DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using device', TRAIN_DEVICE, file=sys.stderr)

def _load_weights(model_file, nets: Model, **kwargs) -> int:
    state, epoch = load_model(model_file, load_to_cpu=True, **kwargs)

    for name in nets.nets:
        if name == "img_encoder" and 'conv1.weight' in state['nets'][name].keys(): # Backwards compatibility
            for i in range(4):
                for value_type in ['bias', 'weight']:
                    state['nets'][name][f'layers.{2*i}.{value_type}'] = state['nets'][name][f'conv{i+1}.{value_type}']
                    del state['nets'][name][f'conv{i+1}.{value_type}']
        net = nets.nets[name]
        net.load_state_dict(state['nets'][name])
        nets.nets[name] = net.to(TRAIN_DEVICE)

    for name, opt in nets.optimizers.items():
        opt.load_state_dict(state['optims'][name])
        # Move the parameters stored in the optimizer into gpu
        for opt_state in opt.state.values():
            for k, v in opt_state.items():
                if torch.is_tensor(v):
                    opt_state[k] = v.to(device=TRAIN_DEVICE)
    return epoch


def run_test_model(dataset, filename = "trained_model_new.50"):
    """ Test model on dataset. Logs accuracy, precision, recall and f1score. """

    from system.controller.reachability_estimator.reachability_estimation import NetworkReachabilityEstimator
    filepath = os.path.join(DATA_STORAGE_FOLDER, filename)
    reach_estimator = NetworkReachabilityEstimator.from_file(weights_file=filepath)

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

    accuracy = BinaryAccuracy().to(TRAIN_DEVICE)
    precision = BinaryPrecision().to(TRAIN_DEVICE)
    recall = BinaryRecall().to(TRAIN_DEVICE)
    f1 = BinaryF1Score().to(TRAIN_DEVICE)
    for idx, item in enumerate(loader):
        print(f"Processing batch {idx} out of {n_samples // reach_estimator.batch_size}")
        batch_src_imgs, batch_dst_imgs, batch_reachability, batch_transformation = item
        src_img = batch_src_imgs.to(device=TRAIN_DEVICE, non_blocking=True)
        dst_imgs = batch_dst_imgs.to(device=TRAIN_DEVICE, non_blocking=True)
        r = batch_reachability.to(device=TRAIN_DEVICE, non_blocking=True)

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

def process_batch(item: Batch):
    model_args, ground_truth = item
    model_args = [data.to(device=TRAIN_DEVICE, non_blocking=True) for data in model_args ]
    ground_truth = [data.to(device=TRAIN_DEVICE, non_blocking=True) for data in ground_truth ]
    return model_args, ground_truth

LossFunction = Callable[[Prediction, Prediction], torch.Tensor]

weights = [0.9, 1.3]
def BCELoss_class_weighted(input, target):
    input = torch.clamp(input,min=1e-7,max=1-1e-7)
    bce = - weights[1] * target * torch.log(input) - (1 - target) * weights[0] * torch.log(1 - input)
    return torch.mean(bce)

def make_loss_function(position_loss_weight = 0.6, angle_loss_weight = 0.3) -> LossFunction:
    def loss_function(prediction: Prediction, truth: Prediction, return_details = False) -> torch.Tensor:
        reachability_prediction, position_prediction, angle_prediction = prediction
        reachability, position, angle = truth

        loss_reachability = BCELoss_class_weighted(reachability_prediction, reachability)
        if position_prediction is None:
            loss = loss_reachability
        else:
            # d**2 = x**2 + y**2, so MSE(d) = MSE(a) + MSE(b); hence the torch.sum over dim=1
            loss_position = torch.sum(torch.nn.functional.mse_loss(position_prediction, position, reduction='none'), dim=1)
            loss_angle = (1 - torch.cos(angle_prediction - angle)) ** 2 # see e.g. https://stats.stackexchange.com/a/425270
            loss = loss_reachability + np.nan_to_num(reachability @ (position_loss_weight * loss_position + angle_loss_weight * loss_angle) / sum(reachability))

        # Loss
        loss = loss.sum()
        assert not loss.isnan()
        if return_details:
            return loss, (loss_reachability, reachability @ loss_position / sum(reachability), reachability @ loss_angle / sum(reachability))
        else:
            return loss
    loss_function.hparams = { 'position_loss_weight': position_loss_weight, 'angle_loss_weight': angle_loss_weight }
    return loss_function

def tensor_log(
    loader: DataLoader,
    writer: SummaryWriter,
    epoch: int,
    net: Model,
    loss_function: LossFunction,
    print_confusion_matrix=False,
):
    """ Log accuracy, precision, recall and f1score for dataset in loader."""
    with torch.no_grad():
        log_loss = 0

        metrics = {
            'Metrics/Accuracy': BinaryAccuracy().to(TRAIN_DEVICE),
            'Metrics/Precision': BinaryPrecision().to(TRAIN_DEVICE),
            'Metrics/Recall': BinaryRecall().to(TRAIN_DEVICE),
            'Metrics/Fscore': BinaryF1Score().to(TRAIN_DEVICE),
        }
        loss_detail_names = ['Loss/Reachability', 'Loss/Position', 'Loss/Angle']
        log_scores = { key: 0 for key in list(metrics.keys()) + loss_detail_names }

        if print_confusion_matrix:
            confusion_matrix = { key: torch.tensor(0) for key in [(True, True), (True, False), (False, True), (False, False)] }

        for idx, item in enumerate(loader):
            model_args, ground_truth = process_batch(item)

            prediction = net.get_prediction(*model_args)
            loss, loss_details = loss_function(prediction, ground_truth, return_details=True)

            if print_confusion_matrix:
                pred_r = prediction[0] > 0.5
                true_r = ground_truth[0].bool()
                confusion_matrix[(True, True)] += sum(pred_r & true_r)
                confusion_matrix[(True, False)] += sum(pred_r & ~true_r)
                confusion_matrix[(False, True)] += sum(~pred_r & true_r)
                confusion_matrix[(False, False)] += sum(~pred_r & ~true_r)

            loss_reachability, loss_position, loss_angle = loss_details
            reachability_prediction, position_prediction, angle_prediction = prediction
            reachability, position, angle = ground_truth

            log_loss += loss.item()
            for key, metric in metrics.items():
                log_scores[key] += metric(reachability_prediction, reachability.int())
            for key, loss_detail in zip(loss_detail_names, loss_details):
                log_scores[key] += loss_detail.item()

    metrics: dict[str, Any] = {}
    metrics["Loss/Validation"] = log_loss / len(loader)
    for key, value in log_scores.items():
        metrics[key] = value / len(loader)

    if print_confusion_matrix:
        print("Confusion matrix:")
        print("Real: True\tFalse")
        print("P:True", confusion_matrix[(True, True)].item(), confusion_matrix[(True, False)].item(), sep='\t')
        print("P:False", confusion_matrix[(False, True)].item(), confusion_matrix[(False, False)].item(), sep='\t')
        print("Metrics:")
        for key, value in metrics.items():
            print(key, ':', value)

    for key, value in metrics.items():
        writer.add_scalar(key, value, epoch)
    return metrics

from dataclasses import dataclass

@dataclass
class Hyperparameters:
    batch_size: int = 64
    samples_per_epoch: int = 10000
    max_epochs: int = 25
    lr: float = 3e-4
    lr_decay_epoch: int = 1
    lr_decay_rate: float = 0.7
    eps: float = 1e-5

def train_multiframedst(
    nets: Model, dataset: ReachabilityDataset,
    hyperparams: Hyperparameters = Hyperparameters(),
    n_dataset_worker = 0,
    log_interval = 20,
    save_interval = 5,
    model_suffix: str = '',
    loss_function: LossFunction = make_loss_function(),
    start_epoch: int = 0,
):
    """ Train the model on a multiframe dataset. """

    # For Tensorboard: log the runs
    writer = SummaryWriter(comment=model_suffix)

    # This is a direct copy of rmp_nav
    # FIXME: hack to mitigate the bug in torch 1.1.0's schedulers
    if start_epoch <= 1:
        last_epoch = start_epoch - 1
    else:
        last_epoch = start_epoch - 2

    # Scheduler: takes care of learning rate decay
    net_scheds = {
        name: torch.optim.lr_scheduler.StepLR(
            opt,
            step_size=hyperparams.lr_decay_epoch,
            gamma=hyperparams.lr_decay_rate,
            last_epoch=last_epoch)
        for name, opt in nets.optimizers.items()
    }

    nets.to(TRAIN_DEVICE)

    n_samples = hyperparams.samples_per_epoch

    # Splitting the Dataset into Train/Validation:
    train_size = int(0.8 * len(dataset))
    valid_size = len(dataset) - train_size
    train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [train_size, valid_size])

    latest_metrics = None

    for epoch in range(start_epoch + 1, hyperparams.max_epochs + 1):
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

            model_args, ground_truth = process_batch(item)

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
            nets.save(epoch, model_file)

        # Validation
        valid_loader = DataLoader(valid_dataset,
                                  batch_size=hyperparams.batch_size,
                                  num_workers=n_dataset_worker)

        # log performance on the validation set
        latest_metrics = tensor_log(valid_loader, writer, epoch, nets, loss_function)

    if latest_metrics is None:
        valid_loader = DataLoader(valid_dataset, batch_size=hyperparams.batch_size, num_workers=n_dataset_worker)
        latest_metrics = tensor_log(valid_loader, writer, hyperparams.max_epochs, nets, loss_function)
    nets_hparams = { 'image_encoder': nets.image_encoder, 'with_conv_layer': nets.image_encoder == 'conv' } # with_conv_layer is for backwards compatibility
    hparams = vars(nets.sample_config) | nets_hparams | vars(hyperparams) | getattr(loss_function, 'hparams', {})
    latest_metrics = { "Final/"+key: value for key, value in latest_metrics.items() }
    print("Writing metrics:", latest_metrics)
    writer.add_hparams(hparams, latest_metrics)


def validate_func(net: Model, dataset: ReachabilityDataset, batch_size,
    loss_function: LossFunction,
    model_suffix: str,
    start_epoch: int,
):
    train_size = int(0.8 * len(dataset))
    valid_size = len(dataset) - train_size
    train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [train_size, valid_size])
    writer = SummaryWriter(comment=model_suffix)

    # Validation
    valid_loader = DataLoader(valid_dataset,
                              batch_size=batch_size,
                              num_workers=0)

    # log performance on the validation set
    tensor_log(valid_loader, writer, start_epoch, net, loss_function, print_confusion_matrix=True)
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
    parser.add_argument('--train-dataset-tags', nargs='+', default=[])
    # dataset features are an actual configuration option, e.g. a network trained with 3-color walls would not work (or barely work) on single-color walls
    # dataset tags are just a tag added to both the dataset name and the trained network name
    # they also identify only the training config and are ignored during validation.
    # use case: training a network on e.g. `dataset-boolor` and validating it on `dataset-simulation` -> you can add --train-dataset-tags simulation to both training and val runs

    parser.add_argument('--images', help='Images are included in the dataset', nargs='?', default=True, choices=[True, False, 'zeros', 'fixed'], type=lambda s: False if s in ['no', 'off'] else s)
    parser.add_argument('--spikings', help='Grid cell spikings are included in the dataset', action='store_true')
    parser.add_argument('--lidar', help='LIDAR distances are included in the dataset', choices=['raw_lidar', 'ego_bc', 'allo_bc'])
    parser.add_argument('--dist', help='Provide the distance and angle to the reachability estimator', action='store_true')
    parser.add_argument('--image-crop', help='Cover the border (+n) or center (-n) pixels in white', type=int)

    parser.add_argument('--image-encoder', help='Image encoder', choices=['fc', 'conv', 'pretrained'], default='conv')
    parser.add_argument('--hidden-fc-layers', help='Hidden FC layer dimensions as a comma-separated list', type=lambda s: [int(i) for i in s.split(',')])
    parser.add_argument('--dropout', help='Use dropout in the hidden FC layers', action='store_true')

    parser.add_argument('--load', help='Load network from file')
    parser.add_argument('--resume', action='store_true', help='Continue training from last saved model')
    parser.add_argument('--save-interval', type=optional(int))

    args = parser.parse_args()

    if args.images is None:
        args.images = True # --images means --images=True but I don't know how to make argparse do that automatically

    config = SampleConfig(
        grid_cell_spikings=args.spikings,
        lidar=args.lidar,
        images=args.images,
        image_crop=args.image_crop,
        dist=args.dist,
    )

    suffix = ''
    if args.tag:
        suffix += '-' + args.tag
    if args.mode == 'train':
        dataset_tags = ''.join([f'-{tag}' for tag in args.train_dataset_tags])
        suffix += dataset_tags
    else:
        dataset_tags = ''
    args.dataset_features = ''.join([ f'-{feature}' for feature in args.dataset_features ])
    suffix += args.dataset_features
    suffix += config.suffix()
    if args.image_encoder:
        suffix += '+' + args.image_encoder
    if args.hidden_fc_layers:
        suffix += '+fc' + ','.join(map(str, args.hidden_fc_layers))
    if args.dropout:
        suffix += '+dropout'

    filename = args.dataset_basename + dataset_tags + args.dataset_features + ".hd5"
    dataset = ReachabilityDataset(filename, sample_config=config)

    backbone = 'convolutional' # convolutional, res_net

    # Defining the NN and optimizers
    hyperparameters = Hyperparameters(batch_size=64)

    nets = Model.create_from_config(backbone, config, image_encoder=args.image_encoder, **{key: getattr(args, key) for key in ['hidden_fc_layers', 'dropout'] if getattr(args, key) is not None})

    model_filename = model_basename + suffix
    model_file = os.path.join(DATA_STORAGE_FOLDER, model_filename)

    epoch = 0
    if args.load is not None:
        name, epoch = args.load.split('.')
        epoch = int(epoch)
        _load_weights(name, nets, step=epoch)
        assert args.resume == False, "--resume with --load does not make sense"
    elif args.resume or args.mode == "validate":
        try:
            epoch = _load_weights(model_file, nets)
            torch.manual_seed(231239 + epoch)
            print('loaded saved state. epoch: %d' % epoch)
        except FileNotFoundError:
            epoch = 0
            print('No saved state found. Starting from the beginning. epoch: 0')

    if epoch == 0 and nets.image_encoder == 'pretrained': # TODO Pierre: this is ugly
        nets.load_pretrained_model()

    loss_function = make_loss_function(position_loss_weight=0.6, angle_loss_weight=0.3)
    global_args = {
        'loss_function': loss_function,
        'model_suffix': suffix,
        'start_epoch': epoch,
    }

    if args.mode == "validate":
        validate_func(nets, dataset, batch_size=64, **global_args)
    elif args.mode == "test":
        run_test_model(dataset)
    elif args.mode == "train":
        print("Training with dataset of length", len(dataset))
        train_multiframedst(nets, dataset, save_interval=args.save_interval, hyperparams=Hyperparameters(batch_size=64), **global_args)
