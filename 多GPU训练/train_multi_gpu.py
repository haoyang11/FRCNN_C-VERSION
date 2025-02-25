# --------------------------------------------------------
# Written by Bharat Singh
# Modified version of py-R-FCN
# --------------------------------------------------------

"""Train a Fast R-CNN network."""

import caffe
from fast_rcnn.config import cfg
import roi_data_layer.roidb as rdl_roidb
from utils.timer import Timer
import numpy as np
import os

from caffe.proto import caffe_pb2
import google.protobuf as pb2
from multiprocessing import Process
import google.protobuf.text_format

class SolverWrapper(object):
    """A simple wrapper around Caffe's solver.
    This wrapper gives us control over he snapshotting process, which we
    use to unnormalize the learned bounding-box regression weights.
    """

    def __init__(self, solver_prototxt, roidb, output_dir, gpu_id,
                 pretrained_model=None):
        """Initialize the SolverWrapper."""
        self.output_dir = output_dir
        self.gpu_id = gpu_id

        if (cfg.TRAIN.HAS_RPN and cfg.TRAIN.BBOX_REG and
            cfg.TRAIN.BBOX_NORMALIZE_TARGETS):
            # RPN can only use precomputed normalization because there are no
            # fixed statistics to compute a priori
            assert cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED

        if cfg.TRAIN.BBOX_REG:
            print 'Computing bounding-box regression targets...'
            self.bbox_means, self.bbox_stds = \
                    rdl_roidb.add_bbox_regression_targets(roidb)
            print 'done'

        self.solver = caffe.SGDSolver(solver_prototxt)
        if pretrained_model is not None:
            print ('Loading pretrained model '
                   'weights from {:s}').format(pretrained_model)
            self.solver.net.copy_from(pretrained_model)

        self.solver_param = caffe_pb2.SolverParameter()
        with open(solver_prototxt, 'rt') as f:
            pb2.text_format.Merge(f.read(), self.solver_param)

        self.solver.net.layers[0].set_roidb(roidb, gpu_id)

    def snapshot(self):
        """Take a snapshot of the network after unnormalizing the learned
        bounding-box regression weights. This enables easy use at test-time.
        """
        net = self.solver.net

        scale_bbox_params_faster_rcnn = (cfg.TRAIN.BBOX_REG and
                             cfg.TRAIN.BBOX_NORMALIZE_TARGETS and
                             net.params.has_key('bbox_pred'))

        scale_bbox_params_rfcn = (cfg.TRAIN.BBOX_REG and
                             cfg.TRAIN.BBOX_NORMALIZE_TARGETS and
                             net.params.has_key('rfcn_bbox'))

        scale_bbox_params_rpn = (cfg.TRAIN.RPN_NORMALIZE_TARGETS and
                                 net.params.has_key('rpn_bbox_pred'))

        if scale_bbox_params_faster_rcnn:
            # save original values
            orig_0 = net.params['bbox_pred'][0].data.copy()
            orig_1 = net.params['bbox_pred'][1].data.copy()

            # scale and shift with bbox reg unnormalization; then save snapshot
            net.params['bbox_pred'][0].data[...] = \
                    (net.params['bbox_pred'][0].data *
                     self.bbox_stds[:, np.newaxis])
            net.params['bbox_pred'][1].data[...] = \
                    (net.params['bbox_pred'][1].data *
                     self.bbox_stds + self.bbox_means)

        if scale_bbox_params_rpn:
            rpn_orig_0 = net.params['rpn_bbox_pred'][0].data.copy()
            rpn_orig_1 = net.params['rpn_bbox_pred'][1].data.copy()
            num_anchor = rpn_orig_0.shape[0] / 4
            # scale and shift with bbox reg unnormalization; then save snapshot
            self.rpn_means = np.tile(np.asarray(cfg.TRAIN.RPN_NORMALIZE_MEANS),
                                      num_anchor)
            self.rpn_stds = np.tile(np.asarray(cfg.TRAIN.RPN_NORMALIZE_STDS),
                                     num_anchor)
            net.params['rpn_bbox_pred'][0].data[...] = \
                (net.params['rpn_bbox_pred'][0].data *
                 self.rpn_stds[:, np.newaxis, np.newaxis, np.newaxis])
            net.params['rpn_bbox_pred'][1].data[...] = \
                (net.params['rpn_bbox_pred'][1].data *
                 self.rpn_stds + self.rpn_means)

        if scale_bbox_params_rfcn:
            # save original values
            orig_0 = net.params['rfcn_bbox'][0].data.copy()
            orig_1 = net.params['rfcn_bbox'][1].data.copy()
            repeat = orig_1.shape[0] / self.bbox_means.shape[0]

            # scale and shift with bbox reg unnormalization; then save snapshot
            net.params['rfcn_bbox'][0].data[...] = \
                    (net.params['rfcn_bbox'][0].data *
                     np.repeat(self.bbox_stds, repeat).reshape((orig_1.shape[0], 1, 1, 1)))
            net.params['rfcn_bbox'][1].data[...] = \
                    (net.params['rfcn_bbox'][1].data *
                     np.repeat(self.bbox_stds, repeat) + np.repeat(self.bbox_means, repeat))

        infix = ('_' + cfg.TRAIN.SNAPSHOT_INFIX
                 if cfg.TRAIN.SNAPSHOT_INFIX != '' else '')
        filename = (self.solver_param.snapshot_prefix + infix +
                    '_iter_{:d}'.format(self.solver.iter) + '.caffemodel')
        filename = os.path.join(self.output_dir, filename)
        if self.gpu_id == 0:
            net.save(str(filename))
        print 'Wrote snapshot to: {:s}'.format(filename)

        if scale_bbox_params_faster_rcnn:
            # restore net to original state
            net.params['bbox_pred'][0].data[...] = orig_0
            net.params['bbox_pred'][1].data[...] = orig_1
        if scale_bbox_params_rfcn:
            # restore net to original state
            net.params['rfcn_bbox'][0].data[...] = orig_0
            net.params['rfcn_bbox'][1].data[...] = orig_1
        if scale_bbox_params_rpn:
            # restore net to original state
            net.params['rpn_bbox_pred'][0].data[...] = rpn_orig_0
            net.params['rpn_bbox_pred'][1].data[...] = rpn_orig_1

        return filename

    def getSolver(self):
        return self.solver

def solve(proto, roidb, pretrained_model, gpus, uid, rank, output_dir, max_iter):
    caffe.set_mode_gpu()
    caffe.set_device(gpus[rank])
    caffe.set_solver_count(len(gpus))
    caffe.set_solver_rank(rank)
    caffe.set_multiprocess(True)
    cfg.GPU_ID = gpus[rank]

    solverW = SolverWrapper(proto, roidb, output_dir,rank,pretrained_model)
    solver = solverW.getSolver()
    nccl = caffe.NCCL(solver, uid)
    nccl.bcast()
    solver.add_callback(nccl)

    if solver.param.layer_wise_reduce:
        solver.net.after_backward(nccl)
    count = 0
    while count < max_iter:
        solver.step(cfg.TRAIN.SNAPSHOT_ITERS)
        if rank == 0:
            solverW.snapshot()
        count = count + cfg.TRAIN.SNAPSHOT_ITERS

def get_training_roidb(imdb):
    """Returns a roidb (Region of Interest database) for use in training."""
    if cfg.TRAIN.USE_FLIPPED:
        print 'Appending horizontally-flipped training examples...'
        imdb.append_flipped_images()
        print 'done'

    print 'Preparing training data...'
    rdl_roidb.prepare_roidb(imdb)
    print 'done'

    return imdb.roidb

def filter_roidb(roidb):
    """Remove roidb entries that have no usable RoIs."""

    def is_valid(entry):
        # Valid images have:
        #   (1) At least one foreground RoI OR
        #   (2) At least one background RoI
        overlaps = entry['max_overlaps']
        # find boxes with sufficient overlap
        fg_inds = np.where(overlaps >= cfg.TRAIN.FG_THRESH)[0]
        # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
        bg_inds = np.where((overlaps < cfg.TRAIN.BG_THRESH_HI) &
                           (overlaps >= cfg.TRAIN.BG_THRESH_LO))[0]
        # image is only valid if such boxes exist
        valid = len(fg_inds) > 0 or len(bg_inds) > 0
        return valid

    num = len(roidb)
    filtered_roidb = [entry for entry in roidb if is_valid(entry)]
    num_after = len(filtered_roidb)
    print 'Filtered {} roidb entries: {} -> {}'.format(num - num_after,
                                                       num, num_after)
    return filtered_roidb


def train_net_multi_gpu(solver_prototxt, roidb, output_dir, pretrained_model, max_iter, gpus):
    """Train a Fast R-CNN network."""
    uid = caffe.NCCL.new_uid()
    caffe.init_log()
    caffe.log('Using devices %s' % str(gpus))
    procs = []

    for rank in range(len(gpus)):
        p = Process(target=solve,
                    args=(solver_prototxt, roidb, pretrained_model, gpus, uid, rank, output_dir, max_iter))
        p.daemon = False
        p.start()
        procs.append(p)
    for p in procs:
        p.join()
