import torch
import torch.nn as nn
import torch.nn.functional as F


class DualCodebookSemanticRefinement(nn.Module):
    """Weakly supervised dual-codebook refinement of the projected S5 map."""
    def __init__(self, channels=256, codewords=64, tau=0.1, beta=1.2, delta=0.1,
                 core_fraction=0.30, lambda_com=0.05, lambda_div=0.1,
                 lambda_sep=1.0, lambda_dcsr=0.1):
        super().__init__()
        self.codewords = codewords
        self.tau = tau
        self.beta = beta
        self.delta = delta
        self.core_fraction = core_fraction
        self.lambda_com = lambda_com
        self.lambda_div = lambda_div
        self.lambda_sep = lambda_sep
        self.lambda_dcsr = lambda_dcsr
        self.in_proj = nn.Linear(channels, channels)
        self.out_proj = nn.Linear(channels, channels, bias=False)
        self.gate = nn.Linear(channels, channels)
        self.codebook_pos = nn.Parameter(F.normalize(torch.randn(codewords, channels), dim=-1))
        self.codebook_neg = nn.Parameter(F.normalize(torch.randn(codewords, channels), dim=-1))
        # Initialize the refinement branch as an identity residual mapping.
        nn.init.xavier_uniform_(self.in_proj.weight)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.xavier_normal_(self.gate.weight, gain=0.01)
        nn.init.constant_(self.gate.bias, -3.0)

    def forward(self, feat, targets=None):
        b, c, h, w = feat.shape
        x = feat.flatten(2).transpose(1, 2)
        u = F.normalize(self.in_proj(x), dim=-1)
        e = F.normalize(torch.cat([self.codebook_pos, self.codebook_neg], 0), dim=-1)
        distances = 2.0 - 2.0 * torch.matmul(u, e.t())
        assign = torch.softmax(-distances / self.tau, dim=-1)
        recon = torch.matmul(assign, e)
        out = x + torch.sigmoid(self.gate(x)) * self.out_proj(recon)
        refined = out.transpose(1, 2).reshape(b, c, h, w).contiguous()
        if not self.training or targets is None:
            return refined, feat.new_zeros(())

        # VQ-style detached commitment and within-branch utilization terms.
        recon_cb = torch.matmul(assign.detach(), e)
        loss_cb = (u.detach() - recon_cb).pow(2).sum(-1).mean()
        loss_com = (u - recon.detach()).pow(2).sum(-1).mean()
        div_terms = []
        detached_assign = torch.softmax(-((2.0 - 2.0 * torch.matmul(u.detach(), e.t())) / self.tau), -1)
        for start in (0, self.codewords):
            p = detached_assign[..., start:start + self.codewords].mean((0, 1))
            p = (p + 1e-8) / (p.sum() + self.codewords * 1e-8)
            div_terms.append((p * torch.log(self.codewords * p + 1e-8)).sum())

        pos_dist = 2.0 - 2.0 * torch.matmul(u.detach(), F.normalize(self.codebook_pos, dim=-1).t())
        neg_dist = 2.0 - 2.0 * torch.matmul(u.detach(), F.normalize(self.codebook_neg, dim=-1).t())
        pos_near = pos_dist.min(-1).values
        neg_near = neg_dist.min(-1).values
        pos_pull, neg_pull, pos_push, neg_push = [], [], [], []
        yy, xx = torch.meshgrid(
            (torch.arange(h, device=feat.device, dtype=feat.dtype) + 0.5) / h,
            (torch.arange(w, device=feat.device, dtype=feat.dtype) + 0.5) / w,
            indexing='ij')
        for bi, target in enumerate(targets):
            is_core = torch.zeros((h, w), dtype=torch.bool, device=feat.device)
            inside_box = torch.zeros_like(is_core)
            box_cores, nearest_tokens = [], []
            for box in target.get('boxes', []):
                cx, cy, bw, bh = box
                inside_box |= ((xx >= cx - bw / 2) & (xx <= cx + bw / 2) &
                               (yy >= cy - bh / 2) & (yy <= cy + bh / 2))
                cw, ch = bw * self.core_fraction, bh * self.core_fraction
                box_cores.append((xx >= cx - cw / 2) & (xx <= cx + cw / 2) &
                                 (yy >= cy - ch / 2) & (yy <= cy + ch / 2))
                nearest_tokens.append(((xx - cx).square() + (yy - cy).square()).argmin())

            if box_cores:
                box_cores = torch.stack(box_cores)
                has_core = box_cores.flatten(1).any(dim=1)
                # Tiny cells can be smaller than one feature cell; vectorized
                # fallback guarantees one nearest S5 token per such annotation.
                rows = torch.arange(len(nearest_tokens), device=feat.device)
                nearest_tokens = torch.stack(nearest_tokens)
                flat_cores = box_cores.flatten(1)
                flat_cores[rows, nearest_tokens] |= ~has_core
                is_core = flat_cores.any(dim=0).reshape(h, w)
            # Object-core and outside-all-box tokens are the only region-supervised tokens.
            core = is_core.flatten()
            background = (~inside_box & ~is_core).flatten()
            ppos, pneg = pos_near[bi], neg_near[bi]
            if core.any():
                pos_pull.append(ppos[core])
                pos_push.append(F.relu(self.beta * ppos[core] + self.delta - neg_dist[bi, core].min(-1).values))
            if background.any():
                neg_pull.append(pneg[background])
                neg_push.append(F.relu(self.beta * pneg[background] + self.delta - pos_dist[bi, background].min(-1).values))
        zero = feat.new_zeros(())
        pos_pull_loss = torch.cat(pos_pull).mean() if pos_pull else zero
        neg_pull_loss = torch.cat(neg_pull).mean() if neg_pull else zero
        pos_push_loss = torch.cat(pos_push).mean() if pos_push else zero
        neg_push_loss = torch.cat(neg_push).mean() if neg_push else zero
        pull = pos_pull_loss + neg_pull_loss
        push = pos_push_loss + neg_push_loss
        loss_sep = pull + push
        loss = (loss_cb + self.lambda_com * loss_com + self.lambda_div * sum(div_terms) +
                self.lambda_sep * loss_sep) * self.lambda_dcsr
        return refined, loss


class OrderedCenterHead(nn.Module):
    """Shared normalized query projection used by OCAQ."""
    def __init__(self, input_dim=256, embed_dim=64):
        super().__init__()
        self.projection = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        return F.normalize(self.projection(x), dim=-1)
