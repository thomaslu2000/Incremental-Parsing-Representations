import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from clusopt_core.cluster import Streamkm


def ema_inplace(moving_avg, new, decay):
    moving_avg.data.mul_(decay).add_(new, alpha=(1 - decay))


def laplace_smoothing(x, n_categories, eps=1e-5):
    return (x + eps) / (x.sum() + n_categories * eps)


class VectorQuantize(nn.Module):
    # Based on: https://github.com/lucidrains/vector-quantize-pytorch
    def __init__(
        self,
        dim,
        n_embed,
        decay=0.8,
        commitment=1.0,
        eps=1e-5,
        wait_steps=0,
        observe_steps=1245,
        coreset_size_multiplier=10,
    ):
        super().__init__()

        self.dim = dim
        self.n_embed = n_embed
        self.decay = decay
        self.eps = eps
        self.commitment = commitment

        embed = torch.randn(dim, n_embed)
        self.register_buffer("embed", embed)
        self.register_buffer("cluster_size", torch.zeros(n_embed))
        self.register_buffer("embed_avg", embed.clone())

        self.wait_steps_remaining = wait_steps
        self.observe_steps_remaining = observe_steps
        # self.clustering_model = Streamkm(
        #     coresetsize=n_embed * coreset_size_multiplier,
        #     length=1500000,
        #     seed=42,
        # )
        self.data_chunks = []

    def stream_cluster(self, input, expected_num_tokens=None):
        input_np = input.detach().cpu().numpy()
        assert len(input.shape) == 2
        self.data_chunks.append(input_np)
        if (
            expected_num_tokens is not None
            and sum([chunk.shape[0] for chunk in self.data_chunks])
            < expected_num_tokens
        ):
            return  # This is not the last sub-batch.
        if self.wait_steps_remaining > 0:
            self.wait_steps_remaining -= 1
            self.data_chunks.clear()
            return

        self.observe_steps_remaining -= 1
        input_np = np.concatenate(self.data_chunks, axis=0)
        self.data_chunks.clear()
        self.clustering_model.partial_fit(input_np)
        if self.observe_steps_remaining == 0:
            print("Initializing vq clusters (this may take a while)...")
            clusters, _ = self.clustering_model.get_final_clusters(
                self.n_embed, seed=42
            )
            new_embed = torch.tensor(
                clusters.T, dtype=self.embed.dtype, device=self.embed.device
            )
            self.embed.copy_(new_embed)
            # Don't set initial cluster sizes to zero! If a cluster is rare,
            # embed_avg will be undergoing exponential decay until it's seen for
            # the first time. If cluster_size is zero, this will lead to *embed*
            # also undergoing exponential decay towards the origin before the
            # cluster is ever encountered. Initializing to 1.0 will instead will
            # instead leave embed in place for many iterations, up until
            # cluster_size finally decays to near-zero.
            self.cluster_size.fill_(1.0)
            self.embed_avg.copy_(new_embed)

    def forward(self, input, expected_num_tokens=None):
        if self.observe_steps_remaining > 0:
            if self.training:
                self.stream_cluster(input, expected_num_tokens)
            return (
                input,
                torch.zeros(input.shape[0],
                            dtype=torch.long, device=input.device),
                torch.tensor(0.0, dtype=input.dtype, device=input.device),
                None
            )

        dtype = input.dtype
        flatten = input.reshape(-1, self.dim)
        dist = (
            flatten.pow(2).sum(1, keepdim=True)
            - 2 * flatten @ self.embed
            + self.embed.pow(2).sum(0, keepdim=True)
        )

        _, embed_ind = (-dist).max(1)
        embed_onehot = F.one_hot(embed_ind, self.n_embed).type(dtype)
        embed_ind = embed_ind.view(*input.shape[:-1])
        quantize = F.embedding(embed_ind, self.embed.transpose(0, 1))

        if self.training:
            ema_inplace(self.cluster_size, embed_onehot.sum(0), self.decay)
            embed_sum = flatten.transpose(0, 1) @ embed_onehot
            ema_inplace(self.embed_avg, embed_sum, self.decay)
            cluster_size = (
                laplace_smoothing(self.cluster_size, self.n_embed, self.eps)
                * self.cluster_size.sum()
            )
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(0)
            self.embed.data.copy_(embed_normalized)

        loss = F.mse_loss(quantize.detach(), input) * self.commitment
        quantize = input + (quantize - input).detach()
        return quantize, embed_ind, loss, dist
