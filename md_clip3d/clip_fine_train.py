from __future__ import print_function
import time
import argparse
import importlib

from md_clip3d.utils.base.plot import plot_loss
from md_clip3d.utils.base.logging import setup_logger
from md_clip3d.utils.clip_loss import *
from md_clip3d.utils.clip_training_utils import *


def train(config_file):
    """
    clip network training engine
    :param config_file: the input configuration file
    :return: None
    """

    # load configure file
    cfg = load_clip_cfg_file(config_file)

    # clean the existing folder if not continue training
    setup_workshop(cfg)

    # control randomness during training
    init_torch_and_numpy(cfg)

    # enable logging
    log_file = os.path.join(cfg.general.save_dir, 'train_log.txt')
    logger = setup_logger(log_file, 'classify')

    # define loss function
    if cfg.loss.name == 'Clip':
        loss_func = ClipLoss(**(cfg.loss.params))
    elif cfg.loss.name == 'TopNLabelSmoothingClip':
        loss_func = TopNLabelSmoothingClipLoss(**(cfg.loss.params))
    elif cfg.loss.name == 'SkipTopNClip':
        loss_func = SkipTopNClipLoss(**(cfg.loss.params))
    elif cfg.loss.name == 'ClipHardNegSample':
        loss_func = ClipHardNegSampleLoss(**(cfg.loss.params))
    else:
        raise ValueError('Unknown loss function')

    # define network
    gpu_ids = list(range(cfg.general.num_gpus))

    net_module = importlib.import_module('md_clip3d.network.' + cfg.net.name)
    net = net_module.ClipNet(
        in_channels=cfg.dataset.input_channel,
        pretrained_model_dir=cfg.net.pretrained_model_dir,
        input_size=cfg.dataset.crop_size,
        loss_func=loss_func,
        output_loss=True)
    max_stride = net.max_stride()
    net.parameters_init()
    assert np.all(np.array(cfg.dataset.crop_size) % max_stride == 0), 'crop size not divisible by max stride'

    net = torch.nn.parallel.DataParallel(net, device_ids=gpu_ids)
    net = net.cuda()

    if cfg.net.frozen_text_encoder:
        for name, parameter in net.named_parameters():
            if 'text_encoder' in name:
                parameter.requires_grad = False

    # load checkpoint if resume epoch > 0
    if cfg.general.resume_epoch >= 0:
        last_save_epoch, batch_idx = load_checkpoint(cfg.general.resume_epoch, net, cfg.general.save_dir)
        start_epoch = last_save_epoch
    else:
        start_epoch, last_save_epoch, batch_idx = 0, 0, 0

    # network paramters optimizer
    opt = getattr(torch.optim, cfg.train.optimizer.name)(
        [{'params': filter(lambda p: p.requires_grad, net.parameters()), 'initial_lr': cfg.train.lr}], 
        lr=cfg.train.lr, **(cfg.train.optimizer.params))

    # learning rate scheduler
    if 'last_epoch' not in cfg.train.lr_scheduler.params.keys():
        cfg.train.lr_scheduler.params['last_epoch'] = cfg.general.resume_epoch
    elif cfg.train.lr_scheduler.params['last_epoch'] == -1:
        cfg.train.lr_scheduler.params['last_epoch'] = cfg.general.resume_epoch

    scheduler = getattr(torch.optim.lr_scheduler, cfg.train.lr_scheduler.name+"LR")(optimizer=opt, **(cfg.train.lr_scheduler.params))

    # training data set
    train_data_loader, len_data_set = generate_fine_training_data_set(cfg)

    batch_number = len(train_data_loader)
    data_iter = iter(train_data_loader)

    # loop over batches
    for i in range(batch_number):
        begin_t = time.time()

        epoch_idx = start_epoch + i * cfg.train.batchsize // len_data_set
        batch_idx += 1

        opt.zero_grad()

        if cfg.loss.name in ['Clip', 'TopNLabelSmoothingClip', 'SkipTopNClip']:
            images, texts = next(data_iter)
            images, texts = images.cuda(), texts.cuda()
            train_loss = net(images, texts)
        elif cfg.loss.name == 'ClipHardNegSample':
            images, texts, other_texts = next(data_iter)
            images, texts, other_texts = images.cuda(), texts.cuda(), other_texts.cuda()
            train_loss = net(images, texts, other_texts)

        train_loss = train_loss.mean()
        train_loss.backward()
        opt.step()

        if epoch_idx != scheduler.last_epoch:
            scheduler.step(epoch=epoch_idx)

        # print training information
        sample_duration = (time.time() - begin_t) * 1.0 / cfg.train.batchsize
        msg = 'epoch: {}, batch: {}, train_loss: {:.4f}, time: {:.4f} s/vol, lr:{}'\
            .format(epoch_idx, batch_idx, train_loss.item(), sample_duration, scheduler.optimizer.param_groups[0]['lr'])
        logger.info(msg)

        if (batch_idx + 1) % cfg.train.plot_snapshot == 0:
            train_loss_plot_file = os.path.join(cfg.general.save_dir, 'train_loss.html')
            plot_loss(log_file, train_loss_plot_file, name='train_loss',
                      display='Training Loss ({})'.format(cfg.loss.name))

        if epoch_idx % cfg.train.save_epochs == 0:
            if last_save_epoch != epoch_idx:
                last_save_epoch = epoch_idx

                # save training model
                save_checkpoint(net, epoch_idx, batch_idx, cfg, config_file, max_stride)


def main():

    parser = argparse.ArgumentParser(description="UII Clip3D Train Engine")
    parser.add_argument('-i', '--input', type=str, nargs='?', help='clip3d train config file')
    args = parser.parse_args()

    train(args.input)


if __name__ == '__main__':
    main()
