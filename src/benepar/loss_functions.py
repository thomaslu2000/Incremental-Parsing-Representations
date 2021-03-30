import torch
import torch.nn as nn
import torch.nn.functional as F


def cel(inp, target):
    target = F.gumbel_softmax(target, tau=1e-8, hard=True)
    inp = F.log_softmax(inp, dim=-1)
    return torch.sum(-target * inp)


def tvd(inp, target):
    # in old version, uses squares instead of abs, log softmax instead of softmax
    inp = F.softmax(inp, dim=-1)
    target = F.softmax(target, dim=-1)
    return torch.sum(torch.abs(inp - target))


def js_gen(**kwargs):
    # log probabilities instead of logits

    kld = nn.KLDivLoss(log_target=False, **kwargs)

    def js(inp, target):
        inp = F.softmax(inp, dim=-1)
        target = F.softmax(target, dim=-1)
        m = (inp + target) / 2
        return kld(inp, m) / 2 + kld(m, target) / 2
    return js


def kl(**kwargs):
    kld = nn.KLDivLoss(log_target=True, **kwargs)

    def k(inp, target):
        inp = F.log_softmax(inp, dim=-1)
        target = F.log_softmax(target, dim=-1)
        return kld(inp, target)
    return k


def emd(inp, target):
    inp = F.softmax(inp, dim=-1)
    target = F.softmax(target, dim=-1)
    emd = torch.cumsum(inp - target, dim=-1)
    return torch.sum(torch.abs(emd))
