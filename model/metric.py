import mxnet as mx
import numpy as np


def get_names():
    pred = ['rpn_cls_prob_5', 'rpn_bbox_loss_5', 'rpn_cls_prob_4', 'rpn_bbox_loss_4', 'rpn_cls_prob_3', 'rpn_bbox_loss_3', 'rcnn_cls_prob', 'rcnn_bbox_loss', 'rcnn_label']
    label = ['rpn_label_5', 'rpn_bbox_target_5', 'rpn_bbox_weight_5', 'rpn_label_4', 'rpn_bbox_target_4', 'rpn_bbox_weight_4', 'rpn_label_3', 'rpn_bbox_target_3', 'rpn_bbox_weight_3']
    return pred, label


class RPNAccMetric_5(mx.metric.EvalMetric):
    def __init__(self):
        super(RPNAccMetric_5, self).__init__('RPNAcc5')
        self.pred, self.label = get_names()

    def update(self, labels, preds):
        pred = preds[self.pred.index('rpn_cls_prob_5')]
        label = labels[self.label.index('rpn_label_5')]

        # pred (b, c, p) or (b, c, h, w)
        pred_label = mx.ndarray.argmax_channel(pred).asnumpy().astype('int32')
        pred_label = pred_label.reshape((pred_label.shape[0], -1))
        # label (b, p)
        label = label.asnumpy().astype('int32')

        # filter with keep_inds
        keep_inds = np.where(label != -1)
        pred_label = pred_label[keep_inds]
        label = label[keep_inds]

        self.sum_metric += np.sum(pred_label.flat == label.flat)
        self.num_inst += len(pred_label.flat)


class RPNLogLossMetric_5(mx.metric.EvalMetric):
    def __init__(self):
        super(RPNLogLossMetric_5, self).__init__('RPNLogLoss5')
        self.pred, self.label = get_names()

    def update(self, labels, preds):
        pred = preds[self.pred.index('rpn_cls_prob_5')]
        label = labels[self.label.index('rpn_label_5')]

        # label (b, p)
        label = label.asnumpy().astype('int32').reshape((-1))
        # pred (b, c, p) or (b, c, h, w) --> (b, p, c) --> (b*p, c)
        pred = pred.asnumpy().reshape((pred.shape[0], pred.shape[1], -1)).transpose((0, 2, 1))
        pred = pred.reshape((label.shape[0], -1))

        # filter with keep_inds
        keep_inds = np.where(label != -1)[0]
        label = label[keep_inds]
        cls = pred[keep_inds, label]

        cls += 1e-14
        cls_loss = -1 * np.log(cls)
        cls_loss = np.sum(cls_loss)
        self.sum_metric += cls_loss
        self.num_inst += label.shape[0]


class RPNL1LossMetric_5(mx.metric.EvalMetric):
    def __init__(self):
        super(RPNL1LossMetric_5, self).__init__('RPNL1Loss5')
        self.pred, self.label = get_names()

    def update(self, labels, preds):
        bbox_loss = preds[self.pred.index('rpn_bbox_loss_5')].asnumpy()
        bbox_weight = labels[self.label.index('rpn_bbox_weight_5')].asnumpy()

        # calculate num_inst (average on those fg anchors)
        num_inst = np.sum(bbox_weight > 0) / 4

        self.sum_metric += np.sum(bbox_loss)
        self.num_inst += num_inst


class RPNAccMetric_4(mx.metric.EvalMetric):
    def __init__(self):
        super(RPNAccMetric_4, self).__init__('RPNAcc4')
        self.pred, self.label = get_names()

    def update(self, labels, preds):
        pred = preds[self.pred.index('rpn_cls_prob_4')]
        label = labels[self.label.index('rpn_label_4')]

        # pred (b, c, p) or (b, c, h, w)
        pred_label = mx.ndarray.argmax_channel(pred).asnumpy().astype('int32')
        pred_label = pred_label.reshape((pred_label.shape[0], -1))
        # label (b, p)
        label = label.asnumpy().astype('int32')

        # filter with keep_inds
        keep_inds = np.where(label != -1)
        pred_label = pred_label[keep_inds]
        label = label[keep_inds]

        self.sum_metric += np.sum(pred_label.flat == label.flat)
        self.num_inst += len(pred_label.flat)


class RPNLogLossMetric_4(mx.metric.EvalMetric):
    def __init__(self):
        super(RPNLogLossMetric_4, self).__init__('RPNLogLoss4')
        self.pred, self.label = get_names()

    def update(self, labels, preds):
        pred = preds[self.pred.index('rpn_cls_prob_4')]
        label = labels[self.label.index('rpn_label_4')]

        # label (b, p)
        label = label.asnumpy().astype('int32').reshape((-1))
        # pred (b, c, p) or (b, c, h, w) --> (b, p, c) --> (b*p, c)
        pred = pred.asnumpy().reshape((pred.shape[0], pred.shape[1], -1)).transpose((0, 2, 1))
        pred = pred.reshape((label.shape[0], -1))

        # filter with keep_inds
        keep_inds = np.where(label != -1)[0]
        label = label[keep_inds]
        cls = pred[keep_inds, label]

        cls += 1e-14
        cls_loss = -1 * np.log(cls)
        cls_loss = np.sum(cls_loss)
        self.sum_metric += cls_loss
        self.num_inst += label.shape[0]


class RPNL1LossMetric_4(mx.metric.EvalMetric):
    def __init__(self):
        super(RPNL1LossMetric_4, self).__init__('RPNL1Loss4')
        self.pred, self.label = get_names()

    def update(self, labels, preds):
        bbox_loss = preds[self.pred.index('rpn_bbox_loss_4')].asnumpy()
        bbox_weight = labels[self.label.index('rpn_bbox_weight_4')].asnumpy()

        # calculate num_inst (average on those fg anchors)
        num_inst = np.sum(bbox_weight > 0) / 4

        self.sum_metric += np.sum(bbox_loss)
        self.num_inst += num_inst


class RPNAccMetric_3(mx.metric.EvalMetric):
    def __init__(self):
        super(RPNAccMetric_3, self).__init__('RPNAcc3')
        self.pred, self.label = get_names()

    def update(self, labels, preds):
        pred = preds[self.pred.index('rpn_cls_prob_3')]
        label = labels[self.label.index('rpn_label_3')]

        # pred (b, c, p) or (b, c, h, w)
        pred_label = mx.ndarray.argmax_channel(pred).asnumpy().astype('int32')
        pred_label = pred_label.reshape((pred_label.shape[0], -1))
        # label (b, p)
        label = label.asnumpy().astype('int32')

        # filter with keep_inds
        keep_inds = np.where(label != -1)
        pred_label = pred_label[keep_inds]
        label = label[keep_inds]

        self.sum_metric += np.sum(pred_label.flat == label.flat)
        self.num_inst += len(pred_label.flat)


class RPNLogLossMetric_3(mx.metric.EvalMetric):
    def __init__(self):
        super(RPNLogLossMetric_3, self).__init__('RPNLogLoss3')
        self.pred, self.label = get_names()

    def update(self, labels, preds):
        pred = preds[self.pred.index('rpn_cls_prob_3')]
        label = labels[self.label.index('rpn_label_3')]

        # label (b, p)
        label = label.asnumpy().astype('int32').reshape((-1))
        # pred (b, c, p) or (b, c, h, w) --> (b, p, c) --> (b*p, c)
        pred = pred.asnumpy().reshape((pred.shape[0], pred.shape[1], -1)).transpose((0, 2, 1))
        pred = pred.reshape((label.shape[0], -1))

        # filter with keep_inds
        keep_inds = np.where(label != -1)[0]
        label = label[keep_inds]
        cls = pred[keep_inds, label]

        cls += 1e-14
        cls_loss = -1 * np.log(cls)
        cls_loss = np.sum(cls_loss)
        self.sum_metric += cls_loss
        self.num_inst += label.shape[0]


class RPNL1LossMetric_3(mx.metric.EvalMetric):
    def __init__(self):
        super(RPNL1LossMetric_3, self).__init__('RPNL1Loss3')
        self.pred, self.label = get_names()

    def update(self, labels, preds):
        bbox_loss = preds[self.pred.index('rpn_bbox_loss_3')].asnumpy()
        bbox_weight = labels[self.label.index('rpn_bbox_weight_3')].asnumpy()

        # calculate num_inst (average on those fg anchors)
        num_inst = np.sum(bbox_weight > 0) / 4

        self.sum_metric += np.sum(bbox_loss)
        self.num_inst += num_inst


class RCNNAccMetric(mx.metric.EvalMetric):
    def __init__(self):
        super(RCNNAccMetric, self).__init__('RCNNAcc')
        self.pred, self.label = get_names()

    def update(self, labels, preds):
        pred = preds[self.pred.index('rcnn_cls_prob')]
        label = preds[self.pred.index('rcnn_label')]

        last_dim = pred.shape[-1]
        pred_label = pred.asnumpy().reshape(-1, last_dim).argmax(axis=1).astype('int32')
        label = label.asnumpy().reshape(-1,).astype('int32')

        self.sum_metric += np.sum(pred_label.flat == label.flat)
        self.num_inst += len(pred_label.flat)


class RCNNLogLossMetric(mx.metric.EvalMetric):
    def __init__(self):
        super(RCNNLogLossMetric, self).__init__('RCNNLogLoss')
        self.pred, self.label = get_names()

    def update(self, labels, preds):
        pred = preds[self.pred.index('rcnn_cls_prob')]
        label = preds[self.pred.index('rcnn_label')]

        last_dim = pred.shape[-1]
        pred = pred.asnumpy().reshape(-1, last_dim)
        label = label.asnumpy().reshape(-1,).astype('int32')
        cls = pred[np.arange(label.shape[0]), label]

        cls += 1e-14
        cls_loss = -1 * np.log(cls)
        cls_loss = np.sum(cls_loss)
        self.sum_metric += cls_loss
        self.num_inst += label.shape[0]


class RCNNL1LossMetric(mx.metric.EvalMetric):
    def __init__(self):
        super(RCNNL1LossMetric, self).__init__('RCNNL1Loss')
        self.pred, self.label = get_names()

    def update(self, labels, preds):
        bbox_loss = preds[self.pred.index('rcnn_bbox_loss')].asnumpy()
        label = preds[self.pred.index('rcnn_label')].asnumpy()

        # calculate num_inst
        keep_inds = np.where(label != 0)[0]
        num_inst = len(keep_inds)

        self.sum_metric += np.sum(bbox_loss)
        self.num_inst += num_inst
