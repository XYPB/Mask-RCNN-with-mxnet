import argparse
import ast
import pprint
import os

import mxnet as mx
from mxnet.module import Module

from symdata.loader import AnchorGenerator, AnchorSampler, AnchorLoader
from model.logger import logger
from model.model import load_param, infer_data_shape, check_shape, initialize_frcnn, get_fixed_params
from model.metric import *

os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = "0"

def train_net(sym, roidb, args):
    # print config
    logger.info('called with args\n{}'.format(pprint.pformat(vars(args))))
    print(roidb)

    # setup multi-gpu
    ctx = [mx.gpu(int(i)) for i in args.gpus.split(',')]
    batch_size = args.rcnn_batch_size * len(ctx)

    # load training data
    feat_sym_5 = sym.get_internals()['fpn_cls_score_5_output']
    feat_sym_4 = sym.get_internals()['fpn_cls_score_4_output']
    feat_sym_3 = sym.get_internals()['fpn_cls_score_3_output']
    ag_5 = AnchorGenerator(feat_stride=16,
                         anchor_scales=(32,), anchor_ratios=args.rpn_anchor_ratios)
    ag_4 = AnchorGenerator(feat_stride=8,
                         anchor_scales=(16,), anchor_ratios=args.rpn_anchor_ratios)
    ag_3 = AnchorGenerator(feat_stride=4,
                         anchor_scales=(8,), anchor_ratios=args.rpn_anchor_ratios)
    asp = AnchorSampler(allowed_border=args.rpn_allowed_border, batch_rois=args.rpn_batch_rois//3,
                        fg_fraction=args.rpn_fg_fraction, fg_overlap=args.rpn_fg_overlap,
                        bg_overlap=args.rpn_bg_overlap)
    train_data = AnchorLoader(roidb, batch_size, args.img_short_side, args.img_long_side,
                              args.img_pixel_means, args.img_pixel_stds, feat_sym_5, feat_sym_4, feat_sym_3, ag_5, ag_4, ag_3, asp, shuffle=True)
    logger.info("{}/{}".format(len(roidb), batch_size))

    # produce shape max possible
    _, out_shape_5, _ = feat_sym_5.infer_shape(data=(1, 3, args.img_long_side, args.img_long_side))
    _, out_shape_4, _ = feat_sym_4.infer_shape(data=(1, 3, args.img_long_side, args.img_long_side))
    _, out_shape_3, _ = feat_sym_3.infer_shape(data=(1, 3, args.img_long_side, args.img_long_side))
    feat_height_5, feat_width_5 = out_shape_5[0][-2:]
    feat_height_4, feat_width_4 = out_shape_4[0][-2:]
    feat_height_3, feat_width_3 = out_shape_3[0][-2:]
    rpn_num_anchors = len(args.rpn_anchor_ratios)
    data_names = ['data', 'im_info', 'gt_boxes']
    label_names = ['label_5', 'bbox_target_5', 'bbox_weight_5', 'label_4', 'bbox_target_4', 'bbox_weight_4', 'label_3', 'bbox_target_3', 'bbox_weight_3']
    data_shapes = [('data', (batch_size, 3, args.img_long_side, args.img_long_side)),
                   ('im_info', (batch_size, 3)),
                   ('gt_boxes', (batch_size, 100, 5))]
    label_shapes = [
        ('label_5', (batch_size, 1, rpn_num_anchors * feat_height_5, feat_width_5)),
        ('bbox_target_5', (batch_size, 4 * rpn_num_anchors, feat_height_5, feat_width_5)),
        ('bbox_weight_5', (batch_size, 4 * rpn_num_anchors, feat_height_5, feat_width_5)),
        ('label_4', (batch_size, 1, rpn_num_anchors * feat_height_4, feat_width_4)),
        ('bbox_target_4', (batch_size, 4 * rpn_num_anchors, feat_height_4, feat_width_4)),
        ('bbox_weight_4', (batch_size, 4 * rpn_num_anchors, feat_height_4, feat_width_4)),
        ('label_3', (batch_size, 1, rpn_num_anchors * feat_height_3, feat_width_3)),
        ('bbox_target_3', (batch_size, 4 * rpn_num_anchors, feat_height_3, feat_width_3)),
        ('bbox_weight_3', (batch_size, 4 * rpn_num_anchors, feat_height_3, feat_width_3)),
    ]

    # print shapes
    data_shape_dict, out_shape_dict = infer_data_shape(sym, data_shapes + label_shapes)
    logger.info('max input shape\n%s' % pprint.pformat(data_shape_dict))
    logger.info('max output shape\n%s' % pprint.pformat(out_shape_dict))

    # load and initialize params
    if args.resume:
        arg_params, aux_params = load_param(args.resume)
    else:
        arg_params, aux_params = load_param(args.pretrained)
        arg_params, aux_params = initialize_frcnn(sym, data_shapes, arg_params, aux_params)

    # check parameter shapes
    check_shape(sym, data_shapes + label_shapes, arg_params, aux_params)

    # check fixed params
    fixed_param_names = get_fixed_params(sym, args.net_fixed_params)
    logger.info('locking params\n%s' % pprint.pformat(fixed_param_names))

    # metric
    rpn_eval_metric_5 = RPNAccMetric_5()
    rpn_cls_metric_5 = RPNLogLossMetric_5()
    rpn_bbox_metric_5 = RPNL1LossMetric_5()
    rpn_eval_metric_4 = RPNAccMetric_4()
    rpn_cls_metric_4 = RPNLogLossMetric_4()
    rpn_bbox_metric_4 = RPNL1LossMetric_4()
    rpn_eval_metric_3 = RPNAccMetric_3()
    rpn_cls_metric_3 = RPNLogLossMetric_3()
    rpn_bbox_metric_3 = RPNL1LossMetric_3()
    eval_metric = RCNNAccMetric()
    cls_metric = RCNNLogLossMetric()
    bbox_metric = RCNNL1LossMetric()
    eval_metrics = mx.metric.CompositeEvalMetric()
    for child_metric in [
        rpn_eval_metric_5,
        rpn_cls_metric_5,
        rpn_bbox_metric_5,
        rpn_eval_metric_4,
        rpn_cls_metric_4,
        rpn_bbox_metric_4,
        rpn_eval_metric_3,
        rpn_cls_metric_3,
        rpn_bbox_metric_3,
        eval_metric,
        cls_metric,
        bbox_metric,
        ]:
        eval_metrics.add(child_metric)

    # callback
    batch_end_callback = mx.callback.Speedometer(batch_size, frequent=args.log_interval, auto_reset=False)

    epoch_end_callback = mx.callback.do_checkpoint(args.save_prefix)
    # learning schedule
    base_lr = args.lr
    lr_factor = 0.1
    lr_epoch = [int(epoch) for epoch in args.lr_decay_epoch.split(',')]
    lr_epoch_diff = [epoch - args.start_epoch for epoch in lr_epoch if epoch > args.start_epoch]
    lr = base_lr * (lr_factor ** (len(lr_epoch) - len(lr_epoch_diff)))
    lr_iters = [int(epoch * len(roidb) / batch_size) for epoch in lr_epoch_diff]
    logger.info('lr %f lr_epoch_diff %s lr_iters %s' % (lr, lr_epoch_diff, lr_iters))
    lr_scheduler = mx.lr_scheduler.MultiFactorScheduler(lr_iters, lr_factor)
    # optimizer
    optimizer_params = {'momentum': 0.9,
                        'wd': 0.0005,
                        'learning_rate': lr,
                        'lr_scheduler': lr_scheduler,
                        'rescale_grad': (1.0 / batch_size),
                        'clip_gradient': 5}

    # train
    mod = Module(sym, data_names=data_names, label_names=label_names,
                 logger=logger, context=ctx, work_load_list=None,
                 fixed_param_names=fixed_param_names)
    mod.fit(train_data, eval_metric=eval_metrics, epoch_end_callback=epoch_end_callback,
            batch_end_callback=batch_end_callback, kvstore='device',
            optimizer='sgd', optimizer_params=optimizer_params,
            arg_params=arg_params, aux_params=aux_params, begin_epoch=args.start_epoch, num_epoch=args.epochs)


def parse_args():
    parser = argparse.ArgumentParser(description='Train Faster R-CNN network',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--network', type=str, default='resnet50', help='base network')
    parser.add_argument('--pretrained', type=str, default='', help='path to pretrained model')
    parser.add_argument('--dataset', type=str, default='voc', help='training dataset')
    parser.add_argument('--imageset', type=str, default='', help='imageset splits')
    parser.add_argument('--gpus', type=str, default='0', help='gpu devices eg. 0,1')
    parser.add_argument('--epochs', type=int, default=10, help='training epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='base learning rate')
    parser.add_argument('--lr-decay-epoch', type=str, default='7', help='epoch to decay lr')
    parser.add_argument('--resume', type=str, default='', help='path to last saved model')
    parser.add_argument('--start-epoch', type=int, default=0, help='start epoch for resuming')
    parser.add_argument('--log-interval', type=int, default=100, help='logging mini batch interval')
    parser.add_argument('--save-prefix', type=str, default='', help='saving params prefix')
    # faster rcnn params
    parser.add_argument('--img-short-side', type=int, default=600)
    parser.add_argument('--img-long-side', type=int, default=1000)
    parser.add_argument('--img-pixel-means', type=str, default='(0.0, 0.0, 0.0)')
    parser.add_argument('--img-pixel-stds', type=str, default='(1.0, 1.0, 1.0)')
    parser.add_argument('--net-fixed-params', type=str, default='["conv0", "stage1", "gamma", "beta"]')
    parser.add_argument('--rpn-feat-stride', type=int, default=16)
    parser.add_argument('--rpn-anchor-scales', type=str, default='(32,)')
    parser.add_argument('--rpn-anchor-ratios', type=str, default='(0.5, 1, 2)')
    parser.add_argument('--rpn-pre-nms-topk', type=int, default=12000)
    parser.add_argument('--rpn-post-nms-topk', type=int, default=2000)
    parser.add_argument('--rpn-nms-thresh', type=float, default=0.7)
    parser.add_argument('--rpn-min-size', type=int, default=16)
    parser.add_argument('--rpn-batch-rois', type=int, default=256)
    parser.add_argument('--rpn-allowed-border', type=int, default=0)
    parser.add_argument('--rpn-fg-fraction', type=float, default=0.5)
    parser.add_argument('--rpn-fg-overlap', type=float, default=0.7)
    parser.add_argument('--rpn-bg-overlap', type=float, default=0.3)
    parser.add_argument('--rcnn-num-classes', type=int, default=21)
    parser.add_argument('--rcnn-feat-stride', type=int, default=16)
    parser.add_argument('--rcnn-pooled-size', type=str, default='(7, 7)')
    parser.add_argument('--rcnn-batch-size', type=int, default=1)
    parser.add_argument('--rcnn-batch-rois', type=int, default=128)
    parser.add_argument('--rcnn-fg-fraction', type=float, default=0.25)
    parser.add_argument('--rcnn-fg-overlap', type=float, default=0.5)
    parser.add_argument('--rcnn-bbox-stds', type=str, default='(0.1, 0.1, 0.2, 0.2)')
    args = parser.parse_args()
    args.img_pixel_means = ast.literal_eval(args.img_pixel_means)
    args.img_pixel_stds = ast.literal_eval(args.img_pixel_stds)
    args.net_fixed_params = ast.literal_eval(args.net_fixed_params)
    args.rpn_anchor_scales = ast.literal_eval(args.rpn_anchor_scales)
    args.rpn_anchor_ratios = ast.literal_eval(args.rpn_anchor_ratios)
    args.rcnn_pooled_size = ast.literal_eval(args.rcnn_pooled_size)
    args.rcnn_bbox_stds = ast.literal_eval(args.rcnn_bbox_stds)
    return args


def get_voc(args):
    from symimdb.pascal_voc import PascalVOC
    if not args.imageset:
        args.imageset = '2007_trainval'
    args.rcnn_num_classes = len(PascalVOC.classes)

    isets = args.imageset.split('+')
    roidb = []
    for iset in isets:
        imdb = PascalVOC(iset, 'data', 'data/VOCdevkit')
        imdb.filter_roidb()
        imdb.append_flipped_images()
        roidb.extend(imdb.roidb)
    return roidb


def get_coco(args):
    from symimdb.coco import coco
    if not args.imageset:
        args.imageset = 'train2017'
    args.rcnn_num_classes = len(coco.classes)

    isets = args.imageset.split('+')
    roidb = []
    for iset in isets:
        imdb = coco(iset, 'data', 'data/coco')
        imdb.filter_roidb()
        imdb.append_flipped_images()
        roidb.extend(imdb.roidb)
    return roidb


def get_city(args):
    from symimdb.cityscape import Cityscape
    args.rcnn_num_classes = len(Cityscape.classes)
    roidb = []
    imdb = Cityscape('train', 'data', 'data/cityscape')
    imdb.filter_roidb()
    imdb.append_flipped_images()
    roidb.extend(imdb.roidb)
    return roidb



def get_resnet50_train(args):
    from model.symbol_resnet import get_resnet_train
    if not args.pretrained:
        args.pretrained = 'model/resnet-50-0000.params'
    if not args.save_prefix:
        args.save_prefix = 'model/resnet50'
    args.img_pixel_means = (0.0, 0.0, 0.0)
    args.img_pixel_stds = (1.0, 1.0, 1.0)
    args.net_fixed_params = ['conv0', 'stage1', 'gamma', 'beta']
    args.rpn_feat_stride = 16
    args.rcnn_feat_stride = 16
    args.rcnn_pooled_size = (7, 7)
    return get_resnet_train(anchor_scales=args.rpn_anchor_scales, anchor_ratios=args.rpn_anchor_ratios,
                            rpn_feature_stride=args.rpn_feat_stride, rpn_pre_topk=args.rpn_pre_nms_topk,
                            rpn_post_topk=args.rpn_post_nms_topk, rpn_nms_thresh=args.rpn_nms_thresh,
                            rpn_min_size=args.rpn_min_size, rpn_batch_rois=args.rpn_batch_rois,
                            num_classes=args.rcnn_num_classes, rcnn_feature_stride=args.rcnn_feat_stride,
                            rcnn_pooled_size=args.rcnn_pooled_size, rcnn_batch_size=args.rcnn_batch_size,
                            rcnn_batch_rois=args.rcnn_batch_rois, rcnn_fg_fraction=args.rcnn_fg_fraction,
                            rcnn_fg_overlap=args.rcnn_fg_overlap, rcnn_bbox_stds=args.rcnn_bbox_stds,
                            units=(3, 4, 6, 3), filter_list=(256, 512, 1024, 2048))


def get_resnet101_train(args):
    from model.symbol_resnet import get_resnet_train
    if not args.pretrained:
        args.pretrained = 'model/resnet-101-0000.params'
    if not args.save_prefix:
        args.save_prefix = 'model/resnet101'
    args.img_pixel_means = (0.0, 0.0, 0.0)
    args.img_pixel_stds = (1.0, 1.0, 1.0)
    args.net_fixed_params = ['conv0', 'stage1', 'gamma', 'beta']
    args.rpn_feat_stride = 16
    args.rcnn_feat_stride = 16
    args.rcnn_pooled_size = (7, 7)
    return get_resnet_train(anchor_scales=args.rpn_anchor_scales, anchor_ratios=args.rpn_anchor_ratios,
                            rpn_feature_stride=args.rpn_feat_stride, rpn_pre_topk=args.rpn_pre_nms_topk,
                            rpn_post_topk=args.rpn_post_nms_topk, rpn_nms_thresh=args.rpn_nms_thresh,
                            rpn_min_size=args.rpn_min_size, rpn_batch_rois=args.rpn_batch_rois,
                            num_classes=args.rcnn_num_classes, rcnn_feature_stride=args.rcnn_feat_stride,
                            rcnn_pooled_size=args.rcnn_pooled_size, rcnn_batch_size=args.rcnn_batch_size,
                            rcnn_batch_rois=args.rcnn_batch_rois, rcnn_fg_fraction=args.rcnn_fg_fraction,
                            rcnn_fg_overlap=args.rcnn_fg_overlap, rcnn_bbox_stds=args.rcnn_bbox_stds,
                            units=(3, 4, 23, 3), filter_list=(256, 512, 1024, 2048))


def get_dataset(dataset, args):
    datasets = {
        'voc': get_voc,
        'coco': get_coco,
        'city': get_city
    }
    if dataset not in datasets:
        raise ValueError("dataset {} not supported".format(dataset))
    return datasets[dataset](args)


def get_network(network, args):
    networks = {
        'resnet50': get_resnet50_train,
        # 'resnet101': get_resnet101_train
    }
    if network not in networks:
        raise ValueError("network {} not supported".format(network))
    return networks[network](args)


def main():
    args = parse_args()
    roidb = get_dataset(args.dataset, args)
    sym = get_network(args.network, args)
    train_net(sym, roidb, args)


if __name__ == '__main__':
    main()
