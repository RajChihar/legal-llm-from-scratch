import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbones import load_backbone
from .heads import CRF, LinearHead
from .peft import apply_method

class NERModel(nn.Module):
    def __init__(self, backbone, tagset, head="linear", dropout=0.1, head_hidden=0):
        super().__init__()
        self.backbone, self.tagset, self.head_type = backbone, tagset, head
        self.head = LinearHead(backbone.d_model, tagset.n, dropout, head_hidden)
        self.crf = CRF(tagset) if head == "crf" else None
        self.register_buffer("class_w", torch.ones(tagset.n))

    def emissions(self, ids, keep):
        return self.head(self.backbone(ids, keep)).float()

    def loss(self, emis, labels, keep, cfg, teacher=None):
        if self.crf is not None:
            return self.crf.nll(emis, labels, keep)
        K = emis.size(-1)
        ce = F.cross_entropy(emis.view(-1, K), labels.view(-1), weight=self.class_w, ignore_index=-100,
                             label_smoothing=cfg.get("label_smoothing", 0.0))
        if cfg.get("focal_gamma", 0):
            lp = F.log_softmax(emis, -1).view(-1, K)
            y = labels.view(-1); ok = y != -100
            lpt = lp[ok].gather(1, y[ok, None]).squeeze(1)
            ce = (-(1 - lpt.exp()) ** cfg["focal_gamma"] * lpt * self.class_w[y[ok]]).mean()
        if teacher is None:
            return ce
        T, a = cfg.get("kd_T", 2.0), cfg.get("kd_alpha", 0.5)
        ok = (labels != -100)
        s = F.log_softmax(emis[ok] / T, -1)
        t = F.log_softmax(teacher[ok].float() / T, -1)
        kl = F.kl_div(s, t, log_target=True, reduction="batchmean")
        return a * ce + (1 - a) * T * T * kl

def build_model(cfg, tagset):
    bcfg = cfg["backbone"]
    bb, info = load_backbone(bcfg["ckpt"], attention=bcfg.get("attention", "bidirectional"),
                             keep_layers=bcfg.get("keep_layers"), c_segment=bcfg.get("c_segment", 0),
                             grad_ckpt=bcfg.get("grad_ckpt", False))
    apply_method(bb, cfg["method"])
    m = NERModel(bb, tagset, cfg["head"]["type"], cfg["head"].get("dropout", 0.1), cfg["head"].get("hidden", 0))
    return m, info

def trainable_state(model):
    keep = {n for n, p in model.named_parameters() if p.requires_grad}
    return {k: v.detach().cpu() for k, v in model.state_dict().items()
            if k in keep or k.startswith("head.") or k.startswith("crf.") or k == "class_w"}
