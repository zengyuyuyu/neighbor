from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import numpy as np

from models.losses import FocalLoss, ClsFocalLoss
from models.losses import RegL1Loss, RegLoss, NormRegL1Loss, RegWeightedL1Loss, RegCtL1Loss
from models.decode import ctdet_decode, _topk
from models.utils import _sigmoid
from utils.debugger import Debugger
from utils.post_process import ctdet_post_process
from utils.oracle_utils import gen_oracle_map
from utils.image import gaussian_radius, draw_umich_gaussian
from .base_trainer import BaseTrainer

class CtdetLoss(torch.nn.Module):
  def __init__(self, opt):
    super(CtdetLoss, self).__init__()
    self.crit = torch.nn.MSELoss() if opt.mse_loss else FocalLoss()
    self.crit_reg = RegL1Loss() if opt.reg_loss == 'l1' else \
              RegLoss() if opt.reg_loss == 'sl1' else None
    self.crit_wh = torch.nn.L1Loss(reduction='sum') if opt.dense_wh else \
              NormRegL1Loss() if opt.norm_wh else \
              RegWeightedL1Loss() if opt.cat_spec_wh else self.crit_reg

    self.crit_ln_cls = ClsFocalLoss()
    self.crit_ln_ct = RegCtL1Loss()

    self.opt = opt

  def forward(self, outputs, batch):
    opt = self.opt

    hm_loss, wh_loss, off_loss, ln_cls_loss, ln_ct_loss = 0, 0, 0, 0, 0

    for s in range(opt.num_stacks):
      output = outputs[s]
      if not opt.mse_loss:
        output['hm'] = _sigmoid(output['hm'])

      if opt.eval_oracle_hm:
        output['hm'] = batch['hm']
      if opt.eval_oracle_wh:
        output['wh'] = torch.from_numpy(gen_oracle_map(
          batch['wh'].detach().cpu().numpy(), 
          batch['ind'].detach().cpu().numpy(), 
          output['wh'].shape[3], output['wh'].shape[2])).to(opt.device)
      if opt.eval_oracle_offset:
        output['reg'] = torch.from_numpy(gen_oracle_map(
          batch['reg'].detach().cpu().numpy(), 
          batch['ind'].detach().cpu().numpy(), 
          output['reg'].shape[3], output['reg'].shape[2])).to(opt.device)

      hm_loss += self.crit(output['hm'], batch['hm']) / opt.num_stacks

      if opt.wh_weight > 0:
        if opt.dense_wh:
          mask_weight = batch['dense_wh_mask'].sum() + 1e-4
          wh_loss += (
            self.crit_wh(output['wh'] * batch['dense_wh_mask'],
            batch['dense_wh'] * batch['dense_wh_mask']) / 
            mask_weight) / opt.num_stacks
        elif opt.cat_spec_wh:
          wh_loss += self.crit_wh(
            output['wh'], batch['cat_spec_mask'],
            batch['ind'], batch['cat_spec_wh']) / opt.num_stacks
        else: # this
          wh_loss += self.crit_reg(
            output['wh'], batch['reg_mask'],
            batch['ind'], batch['wh']) / opt.num_stacks
      
      if opt.reg_offset and opt.off_weight > 0:
        off_loss += self.crit_reg(output['reg'], batch['reg_mask'],
                             batch['ind'], batch['reg']) / opt.num_stacks

      if opt.last_next:
        ln_cls_loss += self.crit_ln_cls(output['ln_cls'], batch['reg_mask'],
                             batch['ind'], batch['ln_cls']) / opt.num_stacks
        ln_ct_loss += self.crit_ln_ct(output['ln_ct'], batch['ind'], batch['ln_ct']) / opt.num_stacks
        
    loss = opt.hm_weight * hm_loss + opt.wh_weight * wh_loss + \
           opt.off_weight * off_loss + 10.0 * ln_cls_loss + 0.01 * ln_ct_loss
    loss_stats = {'loss': loss, 'hm_loss': hm_loss,
                  'wh_loss': wh_loss, 'off_loss': off_loss, 'ln_cls_loss': ln_cls_loss, 'ln_ct_loss': ln_ct_loss}

    return loss, loss_stats

class CtdetTrainer(BaseTrainer):
  def __init__(self, opt, model, optimizer=None):
    super(CtdetTrainer, self).__init__(opt, model, optimizer=optimizer)
  
  def _get_losses(self, opt):
    loss_states = ['loss', 'hm_loss', 'wh_loss', 'off_loss']
    if opt.last_next:
      loss_states = ['loss', 'hm_loss', 'wh_loss', 'off_loss', 'ln_cls_loss', 'ln_ct_loss']
    loss = CtdetLoss(opt)
    return loss_states, loss

  def debug(self, batch, output, iter_id):
    opt = self.opt
    reg = output['reg'] if opt.reg_offset else None
    dets = ctdet_decode(
      output['hm'], output['wh'], reg=reg,
      cat_spec_wh=opt.cat_spec_wh, K=opt.K)
    dets = dets.detach().cpu().numpy().reshape(1, -1, dets.shape[2])
    dets[:, :, :4] *= opt.down_ratio
    dets_gt = batch['meta']['gt_det'].numpy().reshape(1, -1, dets.shape[2])
    dets_gt[:, :, :4] *= opt.down_ratio
    for i in range(1):
      debugger = Debugger(
        dataset=opt.dataset, ipynb=(opt.debug==3), theme=opt.debugger_theme)
      img = batch['input'][i].detach().cpu().numpy().transpose(1, 2, 0)
      img = np.clip(((
        img * opt.std + opt.mean) * 255.), 0, 255).astype(np.uint8)
      pred = debugger.gen_colormap(output['hm'][i].detach().cpu().numpy())
      gt = debugger.gen_colormap(batch['hm'][i].detach().cpu().numpy())
      debugger.add_blend_img(img, pred, 'pred_hm')
      debugger.add_blend_img(img, gt, 'gt_hm')
      debugger.add_img(img, img_id='out_pred')
      for k in range(len(dets[i])):
        if dets[i, k, 4] > opt.center_thresh:
          debugger.add_coco_bbox(dets[i, k, :4], dets[i, k, -1],
                                 dets[i, k, 4], img_id='out_pred')

      debugger.add_img(img, img_id='out_gt')
      for k in range(len(dets_gt[i])):
        if dets_gt[i, k, 4] > opt.center_thresh:
          debugger.add_coco_bbox(dets_gt[i, k, :4], dets_gt[i, k, -1],
                                 dets_gt[i, k, 4], img_id='out_gt')

      if opt.debug == 4:
        debugger.save_all_imgs(opt.debug_dir, prefix='{}'.format(iter_id))
      else:
        debugger.show_all_imgs(pause=True)

  def debug_adj(self, batch, output, iter_id):
    opt = self.opt
    reg = output['reg'] if opt.reg_offset else None

    hm = output['hm']
    bs, cls, height, width = hm.size()
    with torch.no_grad():
      draw_gaussian = draw_umich_gaussian
      topK = 1
      ln_cls = output['ln_cls'].sigmoid()
      ln_ct = output['ln_ct']
      _topk_scores, _topk_inds = torch.topk(hm.view(bs, cls, -1), topK)
      topk_inds = _topk_inds % (height * width)
      topk_scores = _topk_scores.cpu().numpy()
      hm = hm.cpu().numpy()
      draw_hm = np.zeros(hm.shape)
      for b in range(bs):
        for cls_id in range(cls - 1, 0, -1):
          ln_cls_c = _transpose_and_gather_feat(ln_cls, topk_inds[:, cls_id, :]).cpu().numpy()
          ln_ct_c = _transpose_and_gather_feat(ln_ct, topk_inds[:, cls_id, :]).cpu().numpy()
          ct_cur = batch['cls_ct'][b, cls_id].cpu().numpy()
          for k in range(topK):
            if batch['cls_mask'][b, cls_id] == 1 and batch['cls_mask'][b, cls_id-1] == 1:
              draw_gaussian(draw_hm[b, cls_id-1], [ln_ct_c[b, k, 2]+ct_cur[0], ln_ct_c[b, k, 3]+ct_cur[1]], 7)
              hm[b, cls_id-1] = hm[b, cls_id-1] + draw_hm[b, cls_id-1].astype(np.float32)
      draw_hm = torch.from_numpy(draw_hm).to(reg.device)
      hm = torch.from_numpy(hm).to(reg.device)

    dets = ctdet_decode(
      output['hm'], output['wh'], reg=reg,
      cat_spec_wh=opt.cat_spec_wh, K=opt.K)
    dets = dets.detach().cpu().numpy().reshape(1, -1, dets.shape[2])
    dets[:, :, :4] *= opt.down_ratio
    dets_gt = batch['meta']['gt_det'].numpy().reshape(1, -1, dets.shape[2])
    dets_gt[:, :, :4] *= opt.down_ratio
    for i in range(1):
      debugger = Debugger(
        dataset=opt.dataset, ipynb=(opt.debug == 3), theme=opt.debugger_theme)
      img = batch['input'][i].detach().cpu().numpy().transpose(1, 2, 0)
      img = np.clip(((
                             img * opt.std + opt.mean) * 255.), 0, 255).astype(np.uint8)
      pred = debugger.gen_colormap(output['hm'][i].detach().cpu().numpy())
      pred_gaussian = debugger.gen_colormap(draw_hm[i].detach().cpu().numpy())
      pred_add_gaussian = debugger.gen_colormap(hm[i].detach().cpu().numpy())
      gt = debugger.gen_colormap(batch['hm'][i].detach().cpu().numpy())
      debugger.add_blend_img(img, pred, 'pred_hm')
      debugger.add_blend_img(img, pred_gaussian, 'pred_draw_hm')
      debugger.add_blend_img(img, pred_add_gaussian, 'pred_add_hm')
      debugger.add_blend_img(img, gt, 'gt_hm')
      debugger.add_img(img, img_id='out_pred')
      for k in range(len(dets[i])):
        if dets[i, k, 4] > opt.center_thresh:
          debugger.add_coco_bbox(dets[i, k, :4], dets[i, k, -1],
                                 dets[i, k, 4], img_id='out_pred')

      debugger.add_img(img, img_id='out_gt')
      for k in range(len(dets_gt[i])):
        if dets_gt[i, k, 4] > opt.center_thresh:
          debugger.add_coco_bbox(dets_gt[i, k, :4], dets_gt[i, k, -1],
                                 dets_gt[i, k, 4], img_id='out_gt')

      if opt.debug == 4:
        debugger.save_all_imgs(opt.debug_dir, prefix='{}'.format(iter_id))
      else:
        debugger.show_all_imgs(pause=True)

  def save_result(self, output, batch, results, err_adj, err_hm):

    reg = output['reg'] if self.opt.reg_offset else None
    hm = output['hm']
    bs, cls, height, width = hm.size()
    ln_cls = output['ln_cls'] if self.opt.last_next else None
    ln_ct = output['ln_ct'] if self.opt.last_next else None

    if self.opt.last_next:
      with torch.no_grad():
        draw_gaussian = draw_umich_gaussian
        topK = 1
        ln_cls = output['ln_cls'].sigmoid().cpu().numpy()
        ln_ct = output['ln_ct'].cpu().numpy()
        hm = hm.cpu().numpy()
        draw_hm = np.zeros(hm.shape)
        for b in range(bs):
          for cls_id in range(cls-1, 0, -1):
            hm_cls = hm[b, cls_id].flatten()
            sorted_indices = np.argsort(hm_cls)
            topk_inds = sorted_indices[-topK:][::-1] % (height * width)
            topk_scores = hm_cls[topk_inds]
            ln_cls_c = numpy_transpose_and_gather_feat(ln_cls[b], topk_inds)
            ln_ct_c = numpy_transpose_and_gather_feat(ln_ct[b], topk_inds)
            ct_cur = batch['cls_ct'][b, cls_id].cpu().numpy()
            for k in range(topK):
              if ln_cls_c[k, 1] > 0.75 and topk_scores[k] > 0:
                draw_gaussian(draw_hm[b, cls_id-1], [ln_ct_c[k, 2]+ct_cur[0], ln_ct_c[k, 3]+ct_cur[1]], 15, topk_scores[k]/2)  # 15
            hm[b, cls_id-1] = hm[b, cls_id-1] + draw_hm[b, cls_id-1].astype(np.float32)
        hm = torch.from_numpy(hm).to(reg.device)

    dets = ctdet_decode(
      hm, output['wh'], reg=reg, cat_spec_wh=self.opt.cat_spec_wh, K=self.opt.K)

    dets = dets.detach().cpu().numpy().reshape(1, -1, dets.shape[2])
    dets_out = ctdet_post_process(
      dets.copy(), batch['meta']['c'].cpu().numpy(),
      batch['meta']['s'].cpu().numpy(),
      output['hm'].shape[2], output['hm'].shape[3], output['hm'].shape[1])
    results[batch['meta']['img_id'].cpu().numpy()[0]] = dets_out[0]


def _gather_feat(feat, ind, mask=None):
    dim = feat.size(2)
    ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    feat = feat.gather(1, ind)
    if mask is not None:
      mask = mask.unsqueeze(2).expand_as(feat)
      feat = feat[mask]
      feat = feat.view(-1, dim)
    return feat

def _transpose_and_gather_feat(feat, ind):
  feat = feat.permute(0, 2, 3, 1).contiguous()
  feat = feat.view(feat.size(0), -1, feat.size(3))
  feat = _gather_feat(feat, ind)
  return feat


def numpy_gather_feat(feat, ind):
    dim = feat.shape[1]
    ind = np.expand_dims(ind, axis=1)
    ind = np.repeat(ind, dim, axis=1)
    feat = np.take_along_axis(feat, ind, axis=0)
    return feat

def numpy_transpose_and_gather_feat(feat, ind):
    feat = np.transpose(feat, (1, 2, 0))
    feat = feat.reshape(-1, feat.shape[2])
    feat = numpy_gather_feat(feat, ind)
    return feat