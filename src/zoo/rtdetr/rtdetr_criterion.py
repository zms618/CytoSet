"""
reference:
https://github.com/facebookresearch/detr/blob/main/models/detr.py

by lyuwenyu
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from .box_ops import box_cxcywh_to_xyxy, box_iou, generalized_box_iou

from src.misc.dist import get_world_size, is_dist_available_and_initialized
from src.core import register



@register
class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    __share__ = ['num_classes', ]
    __inject__ = ['matcher', ]

    def __init__(self, matcher, weight_dict, losses, alpha=0.2, gamma=2.0, eos_coef=1e-4, num_classes=80,
                 ocaq_enabled=False, ocaq_embed_dim=64, ocaq_momentum=0.9,
                 ocaq_tau=0.1, ocaq_eta=0.5, ocaq_margin=0.20,
                 ocaq_geo_margin=0.20, ocaq_bg_margin=0.30,
                 ocaq_lambda_ord=0.5, ocaq_lambda_cal=0.5, ocaq_lambda_down=0.2,
                 ocaq_lambda_geo=0.5, ocaq_lambda_bg=0.1,
                 class_ranks=None):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses

        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = eos_coef
        self.register_buffer('empty_weight', empty_weight)

        self.alpha = alpha
        self.gamma = gamma
        self.ocaq_enabled = ocaq_enabled
        self.ocaq_tau = ocaq_tau
        self.ocaq_eta = ocaq_eta
        self.ocaq_margin = ocaq_margin
        self.ocaq_geo_margin = ocaq_geo_margin
        self.ocaq_bg_margin = ocaq_bg_margin
        self.ocaq_lambda_ord = ocaq_lambda_ord
        self.ocaq_lambda_cal = ocaq_lambda_cal
        self.ocaq_lambda_down = ocaq_lambda_down
        self.ocaq_lambda_geo = ocaq_lambda_geo
        self.ocaq_lambda_bg = ocaq_lambda_bg
        self.register_buffer('class_centers', torch.zeros(num_classes, ocaq_embed_dim))
        self.register_buffer('center_valid', torch.zeros(num_classes, dtype=torch.bool))
        ranks = class_ranks if class_ranks is not None else list(range(num_classes))
        self.register_buffer('class_ranks', torch.as_tensor(ranks, dtype=torch.long))
        self.ocaq_momentum = ocaq_momentum


    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {'loss_ce': loss_ce}

        if log:
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    def loss_labels_bce(self, outputs, targets, indices, num_boxes, log=True):
        src_logits = outputs['pred_logits']
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        target = F.one_hot(target_classes, num_classes=self.num_classes + 1)[..., :-1]
        loss = F.binary_cross_entropy_with_logits(src_logits, target * 1., reduction='none')
        loss = loss.mean(1).sum() * src_logits.shape[1] / num_boxes
        return {'loss_bce': loss}

    def loss_labels_focal(self, outputs, targets, indices, num_boxes, log=True):
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        target = F.one_hot(target_classes, num_classes=self.num_classes+1)[..., :-1]
        loss = torchvision.ops.sigmoid_focal_loss(src_logits, target, self.alpha, self.gamma, reduction='none')
        loss = loss.mean(1).sum() * src_logits.shape[1] / num_boxes

        return {'loss_focal': loss}

    def loss_labels_vfl(self, outputs, targets, indices, num_boxes, log=True):
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)

        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        ious, _ = box_iou(box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(target_boxes))
        ious = torch.diag(ious).detach()

        src_logits = outputs['pred_logits']
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        target = F.one_hot(target_classes, num_classes=self.num_classes + 1)[..., :-1]

        target_score_o = torch.zeros_like(target_classes, dtype=src_logits.dtype)
        quality = ious.to(target_score_o.dtype)
        qfeatures = outputs.get('query_embeddings', outputs.get('quality_embeddings'))
        if self.ocaq_enabled and qfeatures is not None:
            with torch.no_grad():
                emb = qfeatures[idx].float()
                cls = target_classes_o
                valid = self.center_valid[cls]
                sim = (emb * self.class_centers[cls]).sum(-1)
                consistency = ((1.0 + sim) * 0.5).clamp(0.0, 1.0)
                quality = quality * ((1.0 - self.ocaq_eta) + self.ocaq_eta * consistency * valid + self.ocaq_eta * (~valid))
        target_score_o[idx] = quality
        target_score = target_score_o.unsqueeze(-1) * target

        pred_score = F.sigmoid(src_logits).detach()
        weight = self.alpha * pred_score.pow(self.gamma) * (1 - target) + target_score

        loss = F.binary_cross_entropy_with_logits(src_logits, target_score, weight=weight, reduction='none')
        loss = loss.mean(1).sum() * src_logits.shape[1] / num_boxes
        return {'loss_vfl': loss}

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        losses = {}

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(generalized_box_iou(
                box_cxcywh_to_xyxy(src_boxes),
                box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses

    def loss_masks(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the masks: the focal loss and the dice loss.
           targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_masks = outputs["pred_masks"]
        src_masks = src_masks[src_idx]
        masks = [t["masks"] for t in targets]
        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks.to(src_masks)
        target_masks = target_masks[tgt_idx]

        # upsample predictions to the target size
        src_masks = interpolate(src_masks[:, None], size=target_masks.shape[-2:],
                                mode="bilinear", align_corners=False)
        src_masks = src_masks[:, 0].flatten(1)

        target_masks = target_masks.flatten(1)
        target_masks = target_masks.view(src_masks.shape)
        losses = {
            "loss_mask": sigmoid_focal_loss(src_masks, target_masks, num_boxes),
            "loss_dice": dice_loss(src_masks, target_masks, num_boxes),
        }
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
            'masks': self.loss_masks,

            'bce': self.loss_labels_bce,
            'focal': self.loss_labels_focal,
            'vfl': self.loss_labels_vfl,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if 'aux' not in k}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)
        # All quality targets in this iteration use the same pre-update center bank.
        final_indices = indices

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_available_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            l_dict = self.get_loss(loss, outputs, targets, indices, num_boxes)
            l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
            losses.update(l_dict)

        ocaq_total = outputs['pred_logits'].new_zeros(())
        if self.ocaq_enabled and 'query_embeddings' in outputs:
            ocaq_total = ocaq_total + self._ocaq_losses(outputs, targets, indices)

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                if self.ocaq_enabled and 'query_embeddings' in aux_outputs:
                    ocaq_total = ocaq_total + self._ocaq_losses(aux_outputs, targets, indices)
                for loss in self.losses:
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}

                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
                    l_dict = {k + f'_aux_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        if self.ocaq_enabled:
            losses['loss_ocaq'] = ocaq_total
        if 'loss_dcsr' in outputs:
            losses['loss_dcsr'] = outputs['loss_dcsr']

        # In case of cdn auxiliary losses. For rtdetr
        if 'dn_aux_outputs' in outputs:
            assert 'dn_meta' in outputs, ''
            indices = self.get_cdn_matched_indices(outputs['dn_meta'], targets)
            num_boxes = num_boxes * outputs['dn_meta']['dn_num_group']

            for i, aux_outputs in enumerate(outputs['dn_aux_outputs']):
                for loss in self.losses:
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}

                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
                    l_dict = {k + f'_dn_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        # Update the shared EMA centers only after losses for the final output,
        # decoder layers, encoder proposals, and denoising queries are computed.
        if self.ocaq_enabled and 'query_embeddings' in outputs:
            self._update_centers(outputs['query_embeddings'], targets, final_indices)

        return losses

    @torch.no_grad()
    def _update_centers(self, features, targets, indices):
        src_idx = self._get_src_permutation_idx(indices)
        if src_idx[0].numel() == 0:
            return
        labels = torch.cat([t['labels'][j] for t, (_, j) in zip(targets, indices)])
        emb = features[src_idx].float()
        for cls in labels.unique().tolist():
            mean = F.normalize(emb[labels == cls].mean(0), dim=0)
            if self.center_valid[cls]:
                self.class_centers[cls].mul_(self.ocaq_momentum).add_(mean, alpha=1.0 - self.ocaq_momentum)
                self.class_centers[cls].copy_(F.normalize(self.class_centers[cls], dim=0))
            else:
                self.class_centers[cls].copy_(mean)
                self.center_valid[cls] = True

    def _ocaq_losses(self, outputs, targets, indices):
        features = outputs['query_embeddings']
        idx = self._get_src_permutation_idx(indices)
        labels = torch.cat([t['labels'][j] for t, (_, j) in zip(targets, indices)])
        device = outputs['pred_logits'].device
        zero = outputs['pred_logits'].new_zeros(())
        if labels.numel() == 0:
            return zero
        z = features[idx].float()
        valid_ids = torch.where(self.center_valid)[0]
        if valid_ids.numel() == 0:
            return zero
        centers = self.class_centers[valid_ids].detach()
        ranks = self.class_ranks.to(device)
        sims = z @ centers.t()
        pos_col = torch.searchsorted(valid_ids, labels)
        has_center = self.center_valid[labels]
        keep = has_center
        if not keep.any():
            return zero
        z, labels, sims, pos_col = z[keep], labels[keep], sims[keep], pos_col[keep]
        row = torch.arange(labels.numel(), device=device)
        pos_sim = sims[row, pos_col]
        target_rank = ranks[labels]
        center_rank = ranks[valid_ids]
        rank_gap = (target_rank[:, None] - center_rank[None, :]).abs().float()
        is_other = valid_ids[None, :] != labels[:, None]
        hinge = F.relu(self.ocaq_margin + rank_gap / max(1, int(ranks.max().item())) - pos_sim[:, None] + sims)
        ord_loss = (1.0 - pos_sim).mean() + ((hinge * is_other).sum() / is_other.sum().clamp(min=1))
        posterior = torch.softmax(sims / self.ocaq_tau, dim=-1)
        cal_loss = (posterior * rank_gap).sum(-1).mean()

        high_ids = torch.where((ranks >= 2) & self.center_valid)[0]
        low_ids = torch.where((ranks < 2) & self.center_valid)[0]
        down_mask = (ranks[labels] >= 2)
        down_loss = zero
        if high_ids.numel() and low_ids.numel() and down_mask.any():
            all_centers = self.class_centers.detach()
            all_sims = z @ all_centers.t()
            down_loss = F.relu(all_sims[down_mask][:, low_ids].max(-1).values -
                               all_sims[down_mask][:, high_ids].max(-1).values).mean()

        # Aggregate class representatives across matched queries in the batch.
        class_ids = torch.unique(labels, sorted=True)
        geo_loss = zero
        if class_ids.numel() >= 3:
            class_vecs = torch.stack([
                F.normalize(z[labels == class_id].mean(0), dim=0)
                for class_id in class_ids
            ])
            distances = (1.0 - class_vecs @ class_vecs.t()) * 0.5
            class_ranks = ranks[class_ids]
            gaps = (class_ranks[:, None] - class_ranks[None, :]).abs()
            gap_ab = gaps[:, :, None]
            gap_ac = gaps[:, None, :]
            distinct = ~torch.eye(class_ids.numel(), dtype=torch.bool, device=device)
            valid_triplets = distinct[:, :, None] & distinct[:, None, :] & (gap_ab < gap_ac)
            triplet_loss = F.relu(distances[:, :, None] - distances[:, None, :] +
                                  self.ocaq_geo_margin * (gap_ac - gap_ab).float() /
                                  max(1, int(ranks.max().item())))
            if valid_triplets.any():
                geo_loss = triplet_loss[valid_triplets].mean()

        matched = torch.zeros(features.shape[:2], dtype=torch.bool, device=device)
        matched[idx] = True
        unmatched = features[~matched].float()
        bg_loss = zero
        if unmatched.numel():
            bg_loss = F.relu((unmatched @ centers.t()).max(-1).values - self.ocaq_bg_margin).mean()
        return (self.ocaq_lambda_ord * ord_loss + self.ocaq_lambda_cal * cal_loss +
                self.ocaq_lambda_down * down_loss + self.ocaq_lambda_geo * geo_loss +
                self.ocaq_lambda_bg * bg_loss)

    @staticmethod
    def get_cdn_matched_indices(dn_meta, targets):
        '''get_cdn_matched_indices
        '''
        dn_positive_idx, dn_num_group = dn_meta["dn_positive_idx"], dn_meta["dn_num_group"]
        num_gts = [len(t['labels']) for t in targets]
        device = targets[0]['labels'].device

        dn_match_indices = []
        for i, num_gt in enumerate(num_gts):
            if num_gt > 0:
                gt_idx = torch.arange(num_gt, dtype=torch.int64, device=device)
                gt_idx = gt_idx.tile(dn_num_group)
                assert len(dn_positive_idx[i]) == len(gt_idx)
                dn_match_indices.append((dn_positive_idx[i], gt_idx))
            else:
                dn_match_indices.append((torch.zeros(0, dtype=torch.int64, device=device), \
                    torch.zeros(0, dtype=torch.int64,  device=device)))

        return dn_match_indices





@torch.no_grad()
def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    if target.numel() == 0:
        return [torch.zeros([], device=output.device)]
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res
