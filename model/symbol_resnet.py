import mxnet as mx
from . import proposal_target

eps=2e-5
use_global_stats=True
workspace=1024


def residual_unit(data, num_filter, stride, dim_match, name):
    bn1 = mx.sym.BatchNorm(data=data, fix_gamma=False, eps=eps, use_global_stats=use_global_stats, name=name + '_bn1')
    act1 = mx.sym.Activation(data=bn1, act_type='relu', name=name + '_relu1')
    conv1 = mx.sym.Convolution(data=act1, num_filter=int(num_filter * 0.25), kernel=(1, 1), stride=(1, 1), pad=(0, 0),
                               no_bias=True, workspace=workspace, name=name + '_conv1')
    bn2 = mx.sym.BatchNorm(data=conv1, fix_gamma=False, eps=eps, use_global_stats=use_global_stats, name=name + '_bn2')
    act2 = mx.sym.Activation(data=bn2, act_type='relu', name=name + '_relu2')
    conv2 = mx.sym.Convolution(data=act2, num_filter=int(num_filter * 0.25), kernel=(3, 3), stride=stride, pad=(1, 1),
                               no_bias=True, workspace=workspace, name=name + '_conv2')
    bn3 = mx.sym.BatchNorm(data=conv2, fix_gamma=False, eps=eps, use_global_stats=use_global_stats, name=name + '_bn3')
    act3 = mx.sym.Activation(data=bn3, act_type='relu', name=name + '_relu3')
    conv3 = mx.sym.Convolution(data=act3, num_filter=num_filter, kernel=(1, 1), stride=(1, 1), pad=(0, 0), no_bias=True,
                               workspace=workspace, name=name + '_conv3')
    if dim_match:
        shortcut = data
    else:
        shortcut = mx.sym.Convolution(data=act1, num_filter=num_filter, kernel=(1, 1), stride=stride, no_bias=True,
                                      workspace=workspace, name=name + '_sc')
    sum = mx.sym.ElementWiseSum(*[conv3, shortcut], name=name + '_plus')
    return sum


def get_resnet_feature(data, units, filter_list):
    # res1
    data_bn = mx.sym.BatchNorm(data=data, fix_gamma=True, eps=eps, use_global_stats=use_global_stats, name='bn_data')
    conv0 = mx.sym.Convolution(data=data_bn, num_filter=64, kernel=(7, 7), stride=(2, 2), pad=(3, 3),
                               no_bias=True, name="conv0", workspace=workspace)
    bn0 = mx.sym.BatchNorm(data=conv0, fix_gamma=False, eps=eps, use_global_stats=use_global_stats, name='bn0')
    relu0 = mx.sym.Activation(data=bn0, act_type='relu', name='relu0')
    pool0 = mx.symbol.Pooling(data=relu0, kernel=(3, 3), stride=(2, 2), pad=(1, 1), pool_type='max', name='pool0')

    # res2
    unit_res2 = residual_unit(data=pool0, num_filter=filter_list[0], stride=(1, 1), dim_match=False, name='stage1_unit1')
    for i in range(2, units[0] + 1):
        unit_res2 = residual_unit(data=unit_res2, num_filter=filter_list[0], stride=(1, 1), dim_match=True, name='stage1_unit%s' % i)

    # res3
    unit_res3 = residual_unit(data=unit_res2, num_filter=filter_list[1], stride=(2, 2), dim_match=False, name='stage2_unit1')
    for i in range(2, units[1] + 1):
        unit_res3 = residual_unit(data=unit_res3, num_filter=filter_list[1], stride=(1, 1), dim_match=True, name='stage2_unit%s' % i)

    # res4
    unit_res4 = residual_unit(data=unit_res3, num_filter=filter_list[2], stride=(2, 2), dim_match=False, name='stage3_unit1')
    for i in range(2, units[2] + 1):
        unit_res4 = residual_unit(data=unit_res4, num_filter=filter_list[2], stride=(1, 1), dim_match=True, name='stage3_unit%s' % i)
    return unit_res4, unit_res3, unit_res2, pool0


def get_resnet_top_feature(data, units, filter_list):
    unit = residual_unit(data=data, num_filter=filter_list[3], stride=(2, 2), dim_match=False, name='stage4_unit1')
    for i in range(2, units[3] + 1):
        unit = residual_unit(data=unit, num_filter=filter_list[3], stride=(1, 1), dim_match=True, name='stage4_unit%s' % i)
    bn1 = mx.sym.BatchNorm(data=unit, fix_gamma=False, eps=eps, use_global_stats=use_global_stats, name='bn1')
    relu1 = mx.sym.Activation(data=bn1, act_type='relu', name='relu1')
    pool1 = mx.symbol.Pooling(data=relu1, global_pool=True, kernel=(7, 7), pool_type='avg', name='pool1')
    return pool1


def get_resnet_train(anchor_scales, anchor_ratios, rpn_feature_stride,
                     rpn_pre_topk, rpn_post_topk, rpn_nms_thresh, rpn_min_size, rpn_batch_rois,
                     num_classes, rcnn_feature_stride, rcnn_pooled_size, rcnn_batch_size,
                     rcnn_batch_rois, rcnn_fg_fraction, rcnn_fg_overlap, rcnn_bbox_stds,
                     units, filter_list):
    num_anchors = len(anchor_ratios)

    data = mx.symbol.Variable(name="data")
    im_info = mx.symbol.Variable(name="im_info")
    gt_boxes = mx.symbol.Variable(name="gt_boxes")
    rpn_label_5 = mx.symbol.Variable(name='label_5')
    rpn_label_4 = mx.symbol.Variable(name='label_4')
    rpn_label_3 = mx.symbol.Variable(name='label_3')
    rpn_bbox_target_5 = mx.symbol.Variable(name='bbox_target_5')
    rpn_bbox_weight_5 = mx.symbol.Variable(name='bbox_weight_5')
    rpn_bbox_target_4 = mx.symbol.Variable(name='bbox_target_4')
    rpn_bbox_weight_4 = mx.symbol.Variable(name='bbox_weight_4')
    rpn_bbox_target_3 = mx.symbol.Variable(name='bbox_target_3')
    rpn_bbox_weight_3 = mx.symbol.Variable(name='bbox_weight_3')

    # shared convolutional layers
    conv_feat_5, conv_feat_4, conv_feat_3, conv_feat_2 = get_resnet_feature(data, units=units, filter_list=filter_list)

    
    # C5 to P5, 1x1 dimension reduction to 256
    P5 = mx.symbol.Convolution(data=conv_feat_5, kernel=(1, 1), num_filter=256, name="P5_lateral")

    # P5 2x upsampling + C4 = P4
    P5_up   = mx.symbol.UpSampling(P5, scale=2, sample_type='nearest', workspace=512, name='P5_upsampling', num_args=1)
    P4_la   = mx.symbol.Convolution(data=conv_feat_4, kernel=(1, 1), num_filter=256, name="P4_lateral")
    P5_clip = mx.symbol.Crop(*[P5_up, P4_la], name="P4_clip")
    P4      = mx.sym.ElementWiseSum(*[P5_clip, P4_la], name="P4_sum")
    P4      = mx.symbol.Convolution(data=P4, kernel=(3, 3), pad=(1, 1), num_filter=256, name="P4_aggregate")

    # P4 2x upsampling + C3 = P3
    P4_up   = mx.symbol.UpSampling(P4, scale=2, sample_type='nearest', workspace=512, name='P4_upsampling', num_args=1)
    P3_la   = mx.symbol.Convolution(data=conv_feat_3, kernel=(1, 1), num_filter=256, name="P3_lateral")
    P4_clip = mx.symbol.Crop(*[P4_up, P3_la], name="P3_clip")
    P3      = mx.sym.ElementWiseSum(*[P4_clip, P3_la], name="P3_sum")
    P3      = mx.symbol.Convolution(data=P3, kernel=(3, 3), pad=(1, 1), num_filter=256, name="P3_aggregate")

    # P3 2x upsampling + C2 = P2
    P3_up   = mx.symbol.UpSampling(P3, scale=2, sample_type='nearest', workspace=512, name='P3_upsampling', num_args=1)
    P2_la   = mx.symbol.Convolution(data=conv_feat_2, kernel=(1, 1), num_filter=256, name="P2_lateral")
    P3_clip = mx.symbol.Crop(*[P3_up, P2_la], name="P2_clip")
    P2      = mx.sym.ElementWiseSum(*[P3_clip, P2_la], name="P2_sum")
    P2      = mx.symbol.Convolution(data=P2, kernel=(3, 3), pad=(1, 1), num_filter=256, name="P2_aggregate")

    # P6 2x subsampling P5
    P6 = mx.symbol.Pooling(data=P5, kernel=(3, 3), stride=(2, 2), pad=(1, 1), pool_type='max', name='P6_subsampling')

    fpn_conv_5 = mx.symbol.Convolution(
        data=P5, kernel=(3, 3), pad=(1, 1), num_filter=512, name="fpn_conv_3x3_5")
    fpn_relu_5 = mx.symbol.Activation(data=fpn_conv_5, act_type="relu", name="fpn_relu_5")
    fpn_cls_score_5 = mx.symbol.Convolution(
        data=fpn_relu_5, kernel=(1, 1), pad=(0, 0), num_filter=2 * len(anchor_ratios), name="fpn_cls_score_5")
    fpn_bbox_pred_5 = mx.symbol.Convolution(
        data=fpn_relu_5, kernel=(1, 1), pad=(0, 0), num_filter=4 * len(anchor_ratios), name="fpn_bbox_pred_5")
        
    rpn_cls_score_reshape_5 = mx.symbol.Reshape(
        data=fpn_cls_score_5, shape=(0, 2, -1, 0), name="rpn_cls_score_reshape_5")
    rpn_cls_prob_5 = mx.symbol.SoftmaxOutput(data=rpn_cls_score_reshape_5, label=rpn_label_5, multi_output=True,
                                           normalization='valid', use_ignore=True, ignore_label=-1, name="rpn_cls_prob_5")
    rpn_cls_act_5 = mx.symbol.softmax(
        data=rpn_cls_score_reshape_5, axis=1, name="rpn_cls_act_5")
    rpn_cls_act_reshape_5 = mx.symbol.Reshape(
        data=rpn_cls_act_5, shape=(0, 2 * num_anchors, -1, 0), name='rpn_cls_act_reshape_5')

    # rpn bbox regression
    # rpn_bbox_pred = mx.symbol.Concat(fpn_bbox_pred_3, fpn_bbox_pred_4, fpn_bbox_pred_5, name="rpn_bbox_pred")
    rpn_bbox_loss_5_ = rpn_bbox_weight_5 * mx.symbol.smooth_l1(name='rpn_bbox_loss_5_', scalar=3.0, data=(fpn_bbox_pred_5 - rpn_bbox_target_5))
    rpn_bbox_loss_5 = mx.sym.MakeLoss(name='rpn_bbox_loss_5', data=rpn_bbox_loss_5_, grad_scale=1.0 / rpn_batch_rois)

    # rpn proposal
    rois_5 = mx.symbol.contrib.MultiProposal(
        cls_prob=rpn_cls_act_reshape_5, bbox_pred=fpn_bbox_pred_5, im_info=im_info, name='rois_5',
        feature_stride=rpn_feature_stride, scales=32, ratios=anchor_ratios,
        rpn_pre_nms_top_n=rpn_pre_topk, rpn_post_nms_top_n=rpn_post_topk,
        threshold=rpn_nms_thresh, rpn_min_size=rpn_min_size)

    fpn_conv_4 = mx.symbol.Convolution(
        data=P4, kernel=(3, 3), pad=(1, 1), num_filter=512, name="fpn_conv_3x3_4")
    fpn_relu_4 = mx.symbol.Activation(data=fpn_conv_4, act_type="relu", name="fpn_relu_4")
    fpn_cls_score_4 = mx.symbol.Convolution(
        data=fpn_relu_4, kernel=(1, 1), pad=(0, 0), num_filter=2 * len(anchor_ratios), name="fpn_cls_score_4")
    fpn_bbox_pred_4 = mx.symbol.Convolution(
        data=fpn_relu_4, kernel=(1, 1), pad=(0, 0), num_filter=4 * len(anchor_ratios), name="fpn_bbox_pred_4")
        
    rpn_cls_score_reshape_4 = mx.symbol.Reshape(
        data=fpn_cls_score_4, shape=(0, 2, -1, 0), name="rpn_cls_score_reshape_4")
    rpn_cls_prob_4 = mx.symbol.SoftmaxOutput(data=rpn_cls_score_reshape_4, label=rpn_label_4, multi_output=True,
                                           normalization='valid', use_ignore=True, ignore_label=-1, name="rpn_cls_prob_4")
    rpn_cls_act_4 = mx.symbol.softmax(
        data=rpn_cls_score_reshape_4, axis=1, name="rpn_cls_act_4")
    rpn_cls_act_reshape_4 = mx.symbol.Reshape(
        data=rpn_cls_act_4, shape=(0, 2 * num_anchors, -1, 0), name='rpn_cls_act_reshape_4')

    # rpn bbox regression
    # rpn_bbox_pred = mx.symbol.Concat(fpn_bbox_pred_3, fpn_bbox_pred_4, fpn_bbox_pred_4, name="rpn_bbox_pred")
    rpn_bbox_loss_4_ = rpn_bbox_weight_4 * mx.symbol.smooth_l1(name='rpn_bbox_loss_4_', scalar=3.0, data=(fpn_bbox_pred_4 - rpn_bbox_target_4))
    rpn_bbox_loss_4 = mx.sym.MakeLoss(name='rpn_bbox_loss_4', data=rpn_bbox_loss_4_, grad_scale=1.0 / rpn_batch_rois)

    # rpn proposal
    rois_4 = mx.symbol.contrib.MultiProposal(
        cls_prob=rpn_cls_act_reshape_4, bbox_pred=fpn_bbox_pred_4, im_info=im_info, name='rois_4',
        feature_stride=8, scales=16, ratios=anchor_ratios,
        rpn_pre_nms_top_n=rpn_pre_topk, rpn_post_nms_top_n=rpn_post_topk,
        threshold=rpn_nms_thresh, rpn_min_size=rpn_min_size)

    fpn_conv_3 = mx.symbol.Convolution(
        data=P3, kernel=(3, 3), pad=(1, 1), num_filter=512, name="fpn_conv_3x3_3")
    fpn_relu_3 = mx.symbol.Activation(data=fpn_conv_3, act_type="relu", name="fpn_relu_3")
    fpn_cls_score_3 = mx.symbol.Convolution(
        data=fpn_relu_3, kernel=(1, 1), pad=(0, 0), num_filter=2 * len(anchor_ratios), name="fpn_cls_score_3")
    fpn_bbox_pred_3 = mx.symbol.Convolution(
        data=fpn_relu_3, kernel=(1, 1), pad=(0, 0), num_filter=4 * len(anchor_ratios), name="fpn_bbox_pred_3")
        
    rpn_cls_score_reshape_3 = mx.symbol.Reshape(
        data=fpn_cls_score_3, shape=(0, 2, -1, 0), name="rpn_cls_score_reshape_3")
    rpn_cls_prob_3 = mx.symbol.SoftmaxOutput(data=rpn_cls_score_reshape_3, label=rpn_label_3, multi_output=True,
                                           normalization='valid', use_ignore=True, ignore_label=-1, name="rpn_cls_prob_3")
    rpn_cls_act_3 = mx.symbol.softmax(
        data=rpn_cls_score_reshape_3, axis=1, name="rpn_cls_act_3")
    rpn_cls_act_reshape_3 = mx.symbol.Reshape(
        data=rpn_cls_act_3, shape=(0, 2 * num_anchors, -1, 0), name='rpn_cls_act_reshape_3')

    # rpn bbox regression
    # rpn_bbox_pred = mx.symbol.Concat(fpn_bbox_pred_3, fpn_bbox_pred_3, fpn_bbox_pred_3, name="rpn_bbox_pred")
    rpn_bbox_loss_3_ = rpn_bbox_weight_3 * mx.symbol.smooth_l1(name='rpn_bbox_loss_3_', scalar=3.0, data=(fpn_bbox_pred_3 - rpn_bbox_target_3))
    rpn_bbox_loss_3 = mx.sym.MakeLoss(name='rpn_bbox_loss_3', data=rpn_bbox_loss_3_, grad_scale=1.0 / rpn_batch_rois)

    # rpn proposal
    rois_3 = mx.symbol.contrib.MultiProposal(
        cls_prob=rpn_cls_act_reshape_3, bbox_pred=fpn_bbox_pred_3, im_info=im_info, name='rois_3',
        feature_stride=4, scales=8, ratios=anchor_ratios,
        rpn_pre_nms_top_n=rpn_pre_topk, rpn_post_nms_top_n=rpn_post_topk,
        threshold=rpn_nms_thresh, rpn_min_size=rpn_min_size)

    rois = mx.symbol.Concat(rois_5, rois_4, rois_3, dim=0)

    # rcnn roi proposal target
    group = mx.symbol.Custom(rois=rois, gt_boxes=gt_boxes, op_type='proposal_target',
                             num_classes=num_classes, batch_images=rcnn_batch_size,
                             batch_rois=rcnn_batch_rois, fg_fraction=rcnn_fg_fraction,
                             fg_overlap=rcnn_fg_overlap, box_stds=rcnn_bbox_stds)
    rois = group[0]
    label = group[1]
    bbox_target = group[2]
    bbox_weight = group[3]

    # rcnn roi pool
    roi_pool = mx.symbol.ROIPooling(
        name='roi_pool', data=conv_feat_5, rois=rois, pooled_size=rcnn_pooled_size, spatial_scale=1.0 / rcnn_feature_stride)

    # rcnn top feature
    top_feat = get_resnet_top_feature(roi_pool, units=units, filter_list=filter_list)

    # rcnn classification
    cls_score = mx.symbol.FullyConnected(name='cls_score', data=top_feat, num_hidden=num_classes)
    cls_prob = mx.symbol.SoftmaxOutput(name='cls_prob', data=cls_score, label=label, normalization='batch')

    # rcnn bbox regression
    bbox_pred = mx.symbol.FullyConnected(name='bbox_pred', data=top_feat, num_hidden=num_classes * 4)
    bbox_loss_ = bbox_weight * mx.symbol.smooth_l1(name='bbox_loss_', scalar=1.0, data=(bbox_pred - bbox_target))
    bbox_loss = mx.sym.MakeLoss(name='bbox_loss', data=bbox_loss_, grad_scale=1.0 / rcnn_batch_rois)

    # reshape output
    label = mx.symbol.Reshape(data=label, shape=(rcnn_batch_size, -1), name='label_reshape')
    cls_prob = mx.symbol.Reshape(data=cls_prob, shape=(rcnn_batch_size, -1, num_classes), name='cls_prob_reshape')
    bbox_loss = mx.symbol.Reshape(data=bbox_loss, shape=(rcnn_batch_size, -1, 4 * num_classes), name='bbox_loss_reshape')

    # group output
    group = mx.symbol.Group([rpn_cls_prob_5, rpn_bbox_loss_5, rpn_cls_prob_4, rpn_bbox_loss_4, rpn_cls_prob_3, rpn_bbox_loss_3, cls_prob, bbox_loss, mx.symbol.BlockGrad(label)])
    return group


def get_resnet_test(anchor_scales, anchor_ratios, rpn_feature_stride,
                    rpn_pre_topk, rpn_post_topk, rpn_nms_thresh, rpn_min_size,
                    num_classes, rcnn_feature_stride, rcnn_pooled_size, rcnn_batch_size,
                    units, filter_list):
    
    num_anchors = len(anchor_ratios)

    data = mx.symbol.Variable(name="data")
    im_info = mx.symbol.Variable(name="im_info")

    # shared convolutional layers
    

    # shared convolutional layers
    conv_feat_5, conv_feat_4, conv_feat_3, conv_feat_2 = get_resnet_feature(data, units=units, filter_list=filter_list)

    
    # C5 to P5, 1x1 dimension reduction to 256
    P5 = mx.symbol.Convolution(data=conv_feat_5, kernel=(1, 1), num_filter=256, name="P5_lateral")

    # P5 2x upsampling + C4 = P4
    P5_up   = mx.symbol.UpSampling(P5, scale=2, sample_type='nearest', workspace=512, name='P5_upsampling', num_args=1)
    P4_la   = mx.symbol.Convolution(data=conv_feat_4, kernel=(1, 1), num_filter=256, name="P4_lateral")
    P5_clip = mx.symbol.Crop(*[P5_up, P4_la], name="P4_clip")
    P4      = mx.sym.ElementWiseSum(*[P5_clip, P4_la], name="P4_sum")
    P4      = mx.symbol.Convolution(data=P4, kernel=(3, 3), pad=(1, 1), num_filter=256, name="P4_aggregate")

    # P4 2x upsampling + C3 = P3
    P4_up   = mx.symbol.UpSampling(P4, scale=2, sample_type='nearest', workspace=512, name='P4_upsampling', num_args=1)
    P3_la   = mx.symbol.Convolution(data=conv_feat_3, kernel=(1, 1), num_filter=256, name="P3_lateral")
    P4_clip = mx.symbol.Crop(*[P4_up, P3_la], name="P3_clip")
    P3      = mx.sym.ElementWiseSum(*[P4_clip, P3_la], name="P3_sum")
    P3      = mx.symbol.Convolution(data=P3, kernel=(3, 3), pad=(1, 1), num_filter=256, name="P3_aggregate")

    # P3 2x upsampling + C2 = P2
    P3_up   = mx.symbol.UpSampling(P3, scale=2, sample_type='nearest', workspace=512, name='P3_upsampling', num_args=1)
    P2_la   = mx.symbol.Convolution(data=conv_feat_2, kernel=(1, 1), num_filter=256, name="P2_lateral")
    P3_clip = mx.symbol.Crop(*[P3_up, P2_la], name="P2_clip")
    P2      = mx.sym.ElementWiseSum(*[P3_clip, P2_la], name="P2_sum")
    P2      = mx.symbol.Convolution(data=P2, kernel=(3, 3), pad=(1, 1), num_filter=256, name="P2_aggregate")

    # P6 2x subsampling P5
    P6 = mx.symbol.Pooling(data=P5, kernel=(3, 3), stride=(2, 2), pad=(1, 1), pool_type='max', name='P6_subsampling')

    fpn_conv_5 = mx.symbol.Convolution(
        data=P5, kernel=(3, 3), pad=(1, 1), num_filter=512, name="fpn_conv_3x3_5")
    fpn_relu_5 = mx.symbol.Activation(data=fpn_conv_5, act_type="relu", name="fpn_relu_5")
    fpn_cls_score_5 = mx.symbol.Convolution(
        data=fpn_relu_5, kernel=(1, 1), pad=(0, 0), num_filter=2 * len(anchor_ratios), name="fpn_cls_score_5")
    fpn_bbox_pred_5 = mx.symbol.Convolution(
        data=fpn_relu_5, kernel=(1, 1), pad=(0, 0), num_filter=4 * len(anchor_ratios), name="fpn_bbox_pred_5")
        
    rpn_cls_score_reshape_5 = mx.symbol.Reshape(
        data=fpn_cls_score_5, shape=(0, 2, -1, 0), name="rpn_cls_score_reshape_5")
    rpn_cls_act_5 = mx.symbol.softmax(
        data=rpn_cls_score_reshape_5, axis=1, name="rpn_cls_act_5")
    rpn_cls_act_reshape_5 = mx.symbol.Reshape(
        data=rpn_cls_act_5, shape=(0, 2 * num_anchors, -1, 0), name='rpn_cls_act_reshape_5')

    # rpn proposal
    rois_5 = mx.symbol.contrib.MultiProposal(
        cls_prob=rpn_cls_act_reshape_5, bbox_pred=fpn_bbox_pred_5, im_info=im_info, name='rois_5',
        feature_stride=rpn_feature_stride, scales=32, ratios=anchor_ratios,
        rpn_pre_nms_top_n=rpn_pre_topk, rpn_post_nms_top_n=rpn_post_topk,
        threshold=rpn_nms_thresh, rpn_min_size=rpn_min_size)

    fpn_conv_4 = mx.symbol.Convolution(
        data=P4, kernel=(3, 3), pad=(1, 1), num_filter=512, name="fpn_conv_3x3_4")
    fpn_relu_4 = mx.symbol.Activation(data=fpn_conv_4, act_type="relu", name="fpn_relu_4")
    fpn_cls_score_4 = mx.symbol.Convolution(
        data=fpn_relu_4, kernel=(1, 1), pad=(0, 0), num_filter=2 * len(anchor_ratios), name="fpn_cls_score_4")
    fpn_bbox_pred_4 = mx.symbol.Convolution(
        data=fpn_relu_4, kernel=(1, 1), pad=(0, 0), num_filter=4 * len(anchor_ratios), name="fpn_bbox_pred_4")
        
    rpn_cls_score_reshape_4 = mx.symbol.Reshape(
        data=fpn_cls_score_4, shape=(0, 2, -1, 0), name="rpn_cls_score_reshape_4")
    rpn_cls_act_4 = mx.symbol.softmax(
        data=rpn_cls_score_reshape_4, axis=1, name="rpn_cls_act_4")
    rpn_cls_act_reshape_4 = mx.symbol.Reshape(
        data=rpn_cls_act_4, shape=(0, 2 * num_anchors, -1, 0), name='rpn_cls_act_reshape_4')

    # rpn proposal
    rois_4 = mx.symbol.contrib.MultiProposal(
        cls_prob=rpn_cls_act_reshape_4, bbox_pred=fpn_bbox_pred_4, im_info=im_info, name='rois_4',
        feature_stride=8, scales=16, ratios=anchor_ratios,
        rpn_pre_nms_top_n=rpn_pre_topk, rpn_post_nms_top_n=rpn_post_topk,
        threshold=rpn_nms_thresh, rpn_min_size=rpn_min_size)

    fpn_conv_3 = mx.symbol.Convolution(
        data=P4, kernel=(3, 3), pad=(1, 1), num_filter=512, name="fpn_conv_3x3_3")
    fpn_relu_3 = mx.symbol.Activation(data=fpn_conv_3, act_type="relu", name="fpn_relu_3")
    fpn_cls_score_3 = mx.symbol.Convolution(
        data=fpn_relu_3, kernel=(1, 1), pad=(0, 0), num_filter=2 * len(anchor_ratios), name="fpn_cls_score_3")
    fpn_bbox_pred_3 = mx.symbol.Convolution(
        data=fpn_relu_3, kernel=(1, 1), pad=(0, 0), num_filter=4 * len(anchor_ratios), name="fpn_bbox_pred_3")
        
    rpn_cls_score_reshape_3 = mx.symbol.Reshape(
        data=fpn_cls_score_3, shape=(0, 2, -1, 0), name="rpn_cls_score_reshape_3")
    rpn_cls_act_3 = mx.symbol.softmax(
        data=rpn_cls_score_reshape_3, axis=1, name="rpn_cls_act_3")
    rpn_cls_act_reshape_3 = mx.symbol.Reshape(
        data=rpn_cls_act_3, shape=(0, 2 * num_anchors, -1, 0), name='rpn_cls_act_reshape_3')

    # rpn proposal
    rois_3 = mx.symbol.contrib.MultiProposal(
        cls_prob=rpn_cls_act_reshape_3, bbox_pred=fpn_bbox_pred_3, im_info=im_info, name='rois_3',
        feature_stride=4, scales=8, ratios=anchor_ratios,
        rpn_pre_nms_top_n=rpn_pre_topk, rpn_post_nms_top_n=rpn_post_topk,
        threshold=rpn_nms_thresh, rpn_min_size=rpn_min_size)

    rois = mx.symbol.Concat(rois_5, rois_4, rois_3, dim=0)

    # rcnn roi pool
    roi_pool = mx.symbol.ROIPooling(
        name='roi_pool', data=conv_feat_5, rois=rois, pooled_size=rcnn_pooled_size, spatial_scale=1.0 / rcnn_feature_stride)

    # rcnn top feature
    top_feat = get_resnet_top_feature(roi_pool, units=units, filter_list=filter_list)

    # rcnn classification
    cls_score = mx.symbol.FullyConnected(name='cls_score', data=top_feat, num_hidden=num_classes)
    cls_prob = mx.symbol.softmax(name='cls_prob', data=cls_score)

    # rcnn bbox regression
    bbox_pred = mx.symbol.FullyConnected(name='bbox_pred', data=top_feat, num_hidden=num_classes * 4)

    # reshape output
    cls_prob = mx.symbol.Reshape(data=cls_prob, shape=(rcnn_batch_size, -1, num_classes), name='cls_prob_reshape')
    bbox_pred = mx.symbol.Reshape(data=bbox_pred, shape=(rcnn_batch_size, -1, 4 * num_classes), name='bbox_pred_reshape')

    # group output
    group = mx.symbol.Group([rois, cls_prob, bbox_pred])
    return group
