#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
import pickle
from torch.multiprocessing import Lock
import time
from typing import List, Optional, Tuple, Union
import math
import einops
from einops import rearrange
from math import sqrt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
import numpy as np
from transformers import AutoConfig, AutoModelForCausalLM, LlamaConfig, LlamaModel, LlamaForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast

from llamavid.model.llamavid_arch import LLaMAVIDMetaModel, LLaMAVIDMetaForCausalLM

def save_ckpt(data, idx):
    file = open('./dec_att_{}.pkl'.format(idx), 'wb')
    pickle.dump(data, file)
    # file.close()

class PredictorLG(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, embed_dim=384):
        super().__init__()
        self.in_conv = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU()
        )

        self.out_conv = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, embed_dim // 4),
            nn.GELU(),
            nn.Linear(embed_dim // 4, 1)
        )

    def forward(self, x):
        x = self.in_conv(x)
        B, N, C = x.size()
        local_x = x[:,:, :C//2]
        global_x = torch.mean(x[:,:, C//2:], dim=1, keepdim=True)
        x = torch.cat([local_x, global_x.expand(B, N, C//2)], dim=-1)
        return self.out_conv(x)

def HardTopK(k, x):
    topk_results = torch.topk(x, k=k, dim=-1, sorted=False)
    indices = topk_results.indices # b, k
    indices = torch.sort(indices, dim=-1).values
    return indices


class PerturbedTopK(nn.Module):
    def __init__(self, k: int, num_samples: int = 1000):
        super(PerturbedTopK, self).__init__()
        self.num_samples = num_samples
        self.k = k

    def __call__(self, x, sigma):
        return PerturbedTopKFunction.apply(x, self.k, self.num_samples, sigma)


class PerturbedTopKFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, k: int, num_samples: int = 1000, sigma: float = 0.05):
        b, d = x.shape
        # for Gaussian: noise and gradient are the same.
        noise = torch.normal(mean=0.0, std=1.0, size=(b, num_samples, d)).to(x.device)
        perturbed_x = x[:, None, :] + noise * sigma # b, nS, d
        topk_results = torch.topk(perturbed_x, k=k, dim=-1, sorted=False)
        indices = topk_results.indices # b, nS, k
        indices = torch.sort(indices, dim=-1).values # b, nS, k

        perturbed_output = torch.nn.functional.one_hot(indices, num_classes=d).float()
        indicators = perturbed_output.mean(dim=1) # b, k, d

        # constants for backward
        ctx.k = k
        ctx.num_samples = num_samples
        ctx.sigma = sigma

        # tensors for backward
        ctx.perturbed_output = perturbed_output
        ctx.noise = noise
        return indicators

    @staticmethod
    def backward(ctx, grad_output):
        if grad_output is None:
            return tuple([None] * 5)

        noise_gradient = ctx.noise
        if ctx.sigma <= 1e-20:
            b, _, k, d = ctx.perturbed_output.size()
            expected_gradient = torch.zeros(b, k, d).to(grad_output.device)
        else:
            expected_gradient = (
                torch.einsum("bnkd,bnd->bkd", ctx.perturbed_output, noise_gradient)
                / ctx.num_samples
                / (ctx.sigma)
            )

        grad_input = torch.einsum("bkd,bkd->bd", grad_output, expected_gradient)

        return (grad_input,) + tuple([None] * 5)


def batched_index_select(input, dim, index):
    for i in range(1, len(input.shape)):
        if i != dim:
            index = index.unsqueeze(i)
    expanse = list(input.shape)
    expanse[0] = -1
    expanse[dim] = -1
    index = index.expand(expanse)
    return torch.gather(input, dim, index)


def extract_patches_from_indices(x, indices):
    batch_size, _, channels = x.shape
    k = indices.shape[-1]
    patches = x
    patches = batched_index_select(patches, 1, indices)
    patches = patches.contiguous().view(batch_size, k, channels)
    return patches


def extract_patches_from_indicators(x, indicators):
    indicators = rearrange(indicators, "b d k -> b k d")
    patches = torch.einsum("b k d, b d c -> b k c", indicators, x)
    return patches

def min_max_norm(x):
    flatten_score_min = x.min(axis=-1, keepdim=True).values
    flatten_score_max = x.max(axis=-1, keepdim=True).values
    norm_flatten_score = (x - flatten_score_min) / (flatten_score_max - flatten_score_min + 1e-5)
    return norm_flatten_score

class PatchNet(nn.Module):
    def __init__(self, ratio, in_channels, stride=None, num_samples=500, kk=22):
        super(PatchNet, self).__init__()
        self.ratio = ratio
        self.stride = stride
        self.in_channels = in_channels
        self.num_samples = num_samples
        self.score_network = PredictorLG(embed_dim=2*in_channels)
        self.kk = kk
        self.merge = Merger()
        
    
    def get_indicator(self, scores, k, sigma):
        indicator = PerturbedTopKFunction.apply(scores, k, self.num_samples, sigma)
        indicator = einops.rearrange(indicator, "b k d -> b d k")
        return indicator
    
    def get_indices(self, scores, k):
        indices = HardTopK(k, scores)
        return indices
    
    def generate_random_indices(self, b, n, k):
        indices = []
        for _ in range(b):
            indice = np.sort(np.random.choice(n, k, replace=False))
            indices.append(indice)
        indices = np.vstack(indices)
        indices = torch.Tensor(indices).long().cuda()
        return indices
    
    def generate_uniform_indices(self, b, n, k):
        indices = torch.linspace(0, n-1, steps=k).long()
        indices = indices.unsqueeze(0).cuda()
        indices = indices.repeat(b, 1)
        return indices

    def forward(self, x, y, N, T, sigma=0.1):
        k = int(np.ceil(T * self.ratio)) 
        B = x.size(0)
        H = W = int(sqrt(N))
        indicator = None
        indices = None
        x = rearrange(x, 'b (t n) m -> b t n m', t=T)
        y = rearrange(y, 'b (t n) m -> b t n m', t=T)
        avg = torch.mean(x, dim=2, keepdim=False)
        max_ = torch.max(x, dim=2).values
        x_ = torch.cat((avg, max_), dim=2)
        scores = self.score_network(x_).squeeze(-1)
        scores = min_max_norm(scores)
        # _, index = torch.topk(scores.detach(), k=self.k, dim=1)

        if self.training:
            x = x.bfloat16()
            y = y.bfloat16()
            indicator = self.get_indicator(scores, k, sigma)
            indicator = indicator.bfloat16()
        else:
            indices = self.get_indices(scores, k)
            # print(scores, "")

        x = rearrange(x, 'b t n m -> b t (n m)')
        y = rearrange(y, 'b t n m -> b t (n m)', n=self.kk)

        if self.training:
            focal_patches = extract_patches_from_indicators(x, indicator)
            focal_patches = rearrange(focal_patches, 'b k (n c) -> b k n c', n = N)
            ample_patches = extract_patches_from_indicators(y, indicator.detach())
            ample_patches = rearrange(ample_patches, 'b k (n c) -> b k n c', n = self.kk)
            ample_patches = ample_patches.squeeze(0) # [T, 256, N]
            focal_patches = focal_patches.squeeze(0) # [T, 16, N]
            #new_patches = self.cross_att(focal_patches, ample_patches)
            new_patches = torch.cat([focal_patches, ample_patches], dim=1)
            indices = indicator.detach().squeeze(0)
            indices = indices.max(dim=0).indices[None]
            return new_patches, indices
        else:
            focal_patches = extract_patches_from_indices(x, indices)
            focal_patches = rearrange(focal_patches, 'b k (n c) -> b k n c', n = N)
            ample_patches = extract_patches_from_indices(y, indices)
            ample_patches = rearrange(ample_patches, 'b k (n c) -> b k n c', n = self.kk)
            ample_patches = ample_patches.squeeze(0) # [T, 256, N]
            # ample_patches = self.merge(ample_patches)
            focal_patches = focal_patches.squeeze(0) # [T, 16, N]
            #new_patches = self.cross_att(focal_patches, ample_patches)
            new_patches = torch.cat([focal_patches, ample_patches], dim=1)
            return new_patches, indices

class SFormer(nn.Module):
    def __init__(self, emb_size):
        super(SFormer, self).__init__()
        self.emb_size = emb_size
        self.slow_proj = nn.Linear(self.emb_size, self.emb_size)
        self.fast_proj = nn.Linear(self.emb_size, self.emb_size)

    def forward(self, fast_feature, slow_feature):
        """
          slow_feature: [T, k1, dim]
          fast_feature: [T, k2, dim]
        """
        fast_key = self.fast_proj(fast_feature)
        slow_key = self.slow_proj(slow_feature)
        sim = fast_key @ slow_key.transpose(-1,-2) # [T, k1, k2]
        sim_norm = sim / (slow_feature.shape[-1] ** 0.5)
        ctx_feature = (sim_norm.softmax(-1) @ slow_feature) # [T, k1, dim], [T, k1, dim]
        fusion_feature = fast_feature + ctx_feature 
        return fusion_feature

# class SFormer(nn.Module):
#     def __init__(self, emb_size):
#         super(SFormer, self).__init__()
#         self.emb_size = emb_size
#         self.slow_proj = nn.Linear(self.emb_size, self.emb_size)
#         self.fast_proj = nn.Linear(self.emb_size, self.emb_size)

#     def forward(self, fast_feature, slow_feature):
#         """
#           slow_feature: [T, k1, dim]
#           fast_feature: [T, k2, dim]
#         """
#         fast_key = self.fast_proj(fast_feature)
#         slow_key = self.slow_proj(slow_feature)
#         sim = fast_key @ slow_key.transpose(-1,-2) # [T, k1, k2]
#         sim_norm = sim / (slow_feature.shape[-1] ** 0.5)
#         index = sim_norm.max(-1).indices
#         N = index.size(0)
#         for idx in range(N):
#             if idx == 0:
#                 matching_one = slow_feature[idx][index[idx]][None]
#             else:
#                 matching_one = torch.cat([matching_one, slow_feature[idx][index[idx]][None]], dim=0)
#         fusion_feature = fast_feature + matching_one 
#         return fusion_feature


# class Merger(nn.Module):

#     def __init__(self):
#         super(Merger, self).__init__()
#         self.hidden_size = 1408
#         self.proj_k = nn.Linear(self.hidden_size, 512)
#         self.proj_q = nn.Linear(self.hidden_size, 512)
#         self.score = nn.Linear(6, 6)
#         self.sgm = nn.Sigmoid()
#         self.cluster_ratio1 = 0.0625
#         self.k_layer1 = 8
#         self.cluster_ratio2 = 0.35
#         self.k_layer2 = 3
#         self.norm = nn.LayerNorm(self.hidden_size)

#     def _l2norm(self, inp, dim):
#         return inp / (1e-6 + inp.norm(dim=dim, keepdim=True))
    
    
#     def group(self, meta_feat1, meta_feat2, idx_cluster):
#         """
#         meta_feat1: [T, 16, 1408]
#         meta_feat2: [T, 6, 1408]
#         idx_cluster: [T, 6] -> [0, 4, 0, 6, 0, 0, 4, 5, 1, 0, 0, 3, 0, 1, 0, 2] # 这其实是树结构
#         """
#         if self.training:
#             meta_feat2 = meta_feat2.bfloat16()
#             meta_feat1 = meta_feat1.bfloat16()
#         else:
#             meta_feat2 = meta_feat2.half()
#             meta_feat1 = meta_feat1.half()
#             # meta_feat2 = meta_feat2.bfloat16()
#             # meta_feat1 = meta_feat1.bfloat16()

#         K1 = meta_feat1.size(1)
#         T, K2, C = meta_feat2.size(0), meta_feat2.size(1), meta_feat2.size(-1)
#         group_feats = torch.zeros(T, K2 + K1 , C).to(meta_feat2.device)
        
#         # Score Network
#         modu = self.sgm(self.score(meta_feat2.mean(dim=2))).unsqueeze(-1).unsqueeze(-1) # [T, 6, 1, 1]

#         for t in range(T):
#             last = 0
#             # group by value
#             for value in range(K2):
#                 meta_feat = meta_feat2[t][value][None]
#                 group_feat = meta_feat1[t, torch.where(idx_cluster[t] == value)[0], :]
#                 cnt = meta_feat.size(0) + group_feat.size(0)
#                 group_feats[t][last:last+cnt] = torch.cat([meta_feat, group_feat], dim=0) * modu[t][value]    
#                 last = last+cnt  

#         # score network
#         if self.training:
#             group_feats = group_feats.bfloat16()
#         else:
#             group_feats = group_feats.half()
#             # group_feats = group_feats.bfloat16()

#         return group_feats


#     def forward(self, vis_embed):
#         """
#            simm: [bs, N], N is the number of tokens
#         """
#         vis_embed = vis_embed.float()
#         T, N, C  = vis_embed.shape
#         # token_weight = self.score(vis_embed) # 通过相似度来获取每个token的权重
#         # token_weight = token_weight.exp()

#         # CTM-1
#         cluster_num1 = max(math.ceil(N * self.cluster_ratio1), 1)
#         idx_cluster1, _ = self.cluster_dpc_knn(vis_embed, cluster_num1, self.k_layer1)
#         meta_emb1 = self.merge_tokens(vis_embed, idx_cluster1, cluster_num1, None) # 16

#         # CTM-2 
#         cluster_num2 = max(math.ceil(cluster_num1 * self.cluster_ratio2), 1)
#         idx_cluster2, _ = self.cluster_dpc_knn(meta_emb1, cluster_num2, self.k_layer2)
#         meta_emb2 = self.merge_tokens(meta_emb1, idx_cluster2, cluster_num2, None)  # 6
        
#         # Group Tokens
#         res = self.group(meta_emb1, meta_emb2, idx_cluster2)

#         if self.training:
#             res = res.bfloat16()
#         else:
#             res = res.half()

#         # meta_emb = torch.cat([meta_emb1, meta_emb2], dim=1) # 22 (不同粒度大小的信息)
#         # k = self.proj_k(meta_emb) # [N, M, C]
#         # q = self.proj_q(vis_embed)
#         # simm = k @ q.transpose(1, 2) / math.sqrt(C) # [T, M, N]
#         # score = simm.detach().sum(dim=2) # 将这个分数detach掉
#         # score = self.score(score)
#         # score = self.sgm(score)
#         # simm = F.softmax(simm, dim=-1) 
#         # res = (simm @ vis_embed) * score.unsqueeze(-1)
#         # res = simm @ vis_embed
#         return res


# def save_func_att(modu):
#     att_data = dict()
#     att_data['att'] = modu.cpu().numpy()
#     with open('/mnt_new/hanyudong.hyd/Slow-Fast-Vid/LVBench/scripts/att.pkl', 'wb') as f:
#         pickle.dump(att_data, f)

# def save_func_group(idx_cluster1, idx_cluster2):
#     group_data = dict()
#     group_data['c1'] = idx_cluster1.cpu()
#     group_data['c2'] = idx_cluster2.cpu()

#     with open('/mnt_new/hanyudong.hyd/Slow-Fast-Vid/LVBench/scripts/group.pkl', 'wb') as f:
#         pickle.dump(group_data, f)


class Merger(nn.Module):

    def __init__(self, hidden_size=1408, cluster_ratio1=0.0625, cluster_ratio2=0.35, k_layer1=8, k_layer2=3):
        super(Merger, self).__init__()
        self.hidden_size = 1408
        self.latent_dim = 512
        self.proj_k = nn.Linear(self.hidden_size, self.latent_dim)
        self.proj_q = nn.Linear(self.hidden_size, self.latent_dim)
        self.score = nn.Linear(6, 6)
        self.sgm = nn.Softmax()
        self.cluster_ratio1 = cluster_ratio1
        self.k_layer1 = k_layer1
        self.cluster_ratio2 = cluster_ratio2
        self.k_layer2 = k_layer2
        self.norm = nn.LayerNorm(self.hidden_size)

    def _l2norm(self, inp, dim):
        return inp / (1e-6 + inp.norm(dim=dim, keepdim=True))
    
    
    def group(self, meta_feat1, meta_feat2, idx_cluster):
        """
        meta_feat1: [T, 16, 1408]
        meta_feat2: [T, 6, 1408]
        idx_cluster: [T, 6] -> [0, 4, 0, 6, 0, 0, 4, 5, 1, 0, 0, 3, 0, 1, 0, 2] # 这其实是树结构
        """
        if self.training:
            meta_feat2 = meta_feat2.bfloat16()
            meta_feat1 = meta_feat1.bfloat16()
        else:
            meta_feat2 = meta_feat2.half()
            meta_feat1 = meta_feat1.half()
            # meta_feat2 = meta_feat2.bfloat16()
            # meta_feat1 = meta_feat1.bfloat16()

        K1 = meta_feat1.size(1)
        T, K2, C = meta_feat2.size(0), meta_feat2.size(1), meta_feat2.size(-1)
        group_feats = torch.zeros(T, K2 + K1 , C).to(meta_feat2.device)
        
        # Score Network
        modu = self.sgm(self.score(meta_feat2.mean(dim=2))).unsqueeze(-1).unsqueeze(-1) # [T, 6, 1, 1]


        for t in range(T):
            last = 0
            # group by value
            for value in range(K2):
                meta_feat = meta_feat2[t][value][None]
                group_feat = meta_feat1[t, torch.where(idx_cluster[t] == value)[0], :]
                cnt = meta_feat.size(0) + group_feat.size(0)
                group_feats[t][last:last+cnt] = torch.cat([meta_feat, group_feat], dim=0) * (modu[t][value])
                last = last+cnt  

        # score network
        if self.training:
            group_feats = group_feats.bfloat16()
        else:
            group_feats = group_feats.half()
            # group_feats = group_feats.bfloat16()

        return group_feats


    def forward(self, vis_embed):
        """
           simm: [bs, N], N is the number of tokens
        """
        vis_embed = vis_embed.float()
        T, N, C  = vis_embed.shape
        # token_weight = self.score(vis_embed) # 通过相似度来获取每个token的权重
        # token_weight = token_weight.exp()

        # CTM-1
        cluster_num1 = max(math.ceil(N * self.cluster_ratio1), 1)  # 8
        idx_cluster1, _ = self.cluster_dpc_knn(vis_embed, cluster_num1, self.k_layer1)
        meta_emb1 = self.merge_tokens(vis_embed, idx_cluster1, cluster_num1, None) # 16  

        # CTM-2 
        cluster_num2 = max(math.ceil(cluster_num1 * self.cluster_ratio2), 1) # 6
        idx_cluster2, _ = self.cluster_dpc_knn(meta_emb1, cluster_num2, self.k_layer2)
        meta_emb2 = self.merge_tokens(meta_emb1, idx_cluster2, cluster_num2, None)  # 6

        # save_func_group(idx_cluster1, idx_cluster2)
    
        # Group Tokens
        res = self.group(meta_emb1, meta_emb2, idx_cluster2)

        if self.training:
            res = res.bfloat16()
        else:
            res = res.half()


        return res

# class Merger(nn.Module):

#     def __init__(self):
#         super(Merger, self).__init__()
#         self.hidden_size = 1408
#         self.proj_k = nn.Linear(self.hidden_size, 512)
#         self.proj_q = nn.Linear(self.hidden_size, 512)
#         self.score = nn.Linear(self.hidden_size, 1)
#         self.cluster_num = 16 
#         self.k = 5
#         self.norm = nn.LayerNorm(self.hidden_size)

#     def _l2norm(self, inp, dim):
#         return inp / (1e-6 + inp.norm(dim=dim, keepdim=True))
     
#     def forward(self, vis_embed):
#         """
#            simm: [bs, N], N is the number of tokens
#         """
#         T, N, C  = vis_embed.shape
#         token_weight = self.score(vis_embed) # 通过相似度来获取每个token的权重
#         token_weight = token_weight.exp()
#         # obtain the number of cluster
#         idx_cluster, cluster_num = self.cluster_dpc_knn(vis_embed, self.cluster_num, self.k)
#         meta_emb = self.merge_tokens(vis_embed, idx_cluster, self.cluster_num, None)
#         k = self.proj_k(meta_emb) # [T, M, C]
#         q = self.proj_q(vis_embed)
#         simm = k @ q.transpose(1, 2) / math.sqrt(C)
#         simm = F.softmax(simm, dim=-1) 
#         res = simm @ vis_embed
#         return res

    def merge_tokens(self, vis_embed, idx_cluster, cluster_num, token_weight=None):

        x = vis_embed

        B, N, C = x.shape
        if token_weight is None:
            token_weight = x.new_ones(B, N, 1)

        idx_batch = torch.arange(B, device=x.device)[:, None]
        idx = idx_cluster + idx_batch * cluster_num

        all_weight = token_weight.new_zeros(B * cluster_num, 1)
        all_weight.index_add_(dim=0, index=idx.reshape(B * N), source=token_weight.reshape(B * N, 1))
        all_weight = all_weight + 1e-6
        norm_weight = token_weight / all_weight[idx]

        # average token features
        x_merged = x.new_zeros(B * cluster_num, C)
        source = x * norm_weight
        x_merged.index_add_(dim=0, index=idx.reshape(B * N), source=source.reshape(B * N, C).type(x.dtype))
        x_merged = x_merged.reshape(B, cluster_num, C)

        # idx_token_new = index_points(idx_cluster[..., None], idx_token).squeeze(-1)
        # weight_t = index_points(norm_weight, idx_token)
        # agg_weight_new = agg_weight * weight_t
        # agg_weight_new / agg_weight_new.max(dim=1, keepdim=True)[0]

        return x_merged

    def index_points(self, points, idx):
        device = points.device
        B = points.shape[0]
        view_shape = list(idx.shape)
        view_shape[1:] = [1] * (len(view_shape) - 1)
        repeat_shape = list(idx.shape)
        repeat_shape[0] = 1
        batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
        new_points = points[batch_indices, idx, :]
        return new_points

    def cluster_dpc_knn(self, vis_embed, cluster_num, k, token_mask=None):
        x = vis_embed 
        # x = self.norm(vis_embed)
        with torch.no_grad():
            B, N, C = x.shape
            dist_matrix = torch.cdist(x, x) / (C ** 0.5)
            if token_mask is not None:
                token_mask = token_mask > 0
                # in order to not affect the local density, the distance between empty tokens
                # and any other tokens should be the maximal distance.
                dist_matrix = dist_matrix * token_mask[:, None, :] + (dist_matrix.max() + 1) * (~token_mask[:, None, :])

            # get local density
            dist_nearest, index_nearest = torch.topk(dist_matrix, k=k, dim=-1, largest=False)

            density = (-(dist_nearest ** 2).mean(dim=-1)).exp()
            # add a little noise to ensure no tokens have the same density.
            density = density + torch.rand(density.shape, device=density.device, dtype=density.dtype) * 1e-6

            if token_mask is not None:
                # the density of empty token should be 0
                density = density * token_mask

            # get distance indicator
            mask = density[:, None, :] > density[:, :, None]
            mask = mask.type(x.dtype)
            dist_max = dist_matrix.flatten(1).max(dim=-1)[0][:, None, None]
            dist, index_parent = (dist_matrix * mask + dist_max * (1 - mask)).min(dim=-1)

            # select clustering center according to score
            score = dist * density
            _, index_down = torch.topk(score, k=cluster_num, dim=-1) # [bs, 8]
            
            # assign tokens to the nearest center
            dist_matrix = self.index_points(dist_matrix, index_down)
            idx_cluster = dist_matrix.argmin(dim=1)

            # make sure cluster center merge to itself
            idx_batch = torch.arange(B, device=x.device)[:, None].expand(B, cluster_num)
            idx_tmp = torch.arange(cluster_num, device=x.device)[None, :].expand(B, cluster_num)
            idx_cluster[idx_batch.reshape(-1), index_down.reshape(-1)] = idx_tmp.reshape(-1)
            # find the meta_emb by index 
            # for row in range(B):
            #     if row == 0:
            #         meta_embed = x[row, index_down[row], ...].unsqueeze(0) # [1, 8, 1408]
            #     else:
            #         meta_embed = torch.cat([meta_embed, x[row, index_down[row], ...].unsqueeze(0)], dim=0)
        return idx_cluster, cluster_num

class LlavaConfig(LlamaConfig):
    model_type = "llava"

class LlavaAttLlamaModel(LLaMAVIDMetaModel, LlamaModel):
    config_class = LlavaConfig

    def __init__(self, config: LlamaConfig):
        super(LlavaAttLlamaModel, self).__init__(config)

# 抽象类：LLaMAVIDMetaForCausalLM
class LlavaLlamaAttForCausalLM(LlamaForCausalLM, LLaMAVIDMetaForCausalLM):
    config_class = LlavaConfig

    def __init__(self, config, hidden_dim):
        super(LlamaForCausalLM, self).__init__(config)
        # model configuration
        self.model = LlavaAttLlamaModel(config) 
        self.selective_net = PatchNet(0.45, hidden_dim, stride=1, num_samples=500)
        self.sf_former = SFormer(hidden_dim)
        self.merge = Merger()
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # Initialize weights and apply final processing
        self.post_init()

    def get_selective_net(self):
        return self.selective_net

    def get_model(self):
        return self.model

    def get_sformer_model(self):
        return self.sf_former

    def get_merge_model(self):
        return self.merge


    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        prompts: Optional[List[str]] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if not self.training:
            if images[0].device != self.device:
                images[0] = images[0].to(device=self.device)
            if input_ids.device != self.device:
                input_ids = input_ids.to(device=self.device)

        input_ids, attention_mask, past_key_values, inputs_embeds, labels = self.prepare_inputs_labels_for_multimodal(input_ids, attention_mask, past_key_values, labels, images, prompts=prompts)
        torch.cuda.empty_cache()

        output_attentions = True
        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        # self.model 为已经封装好的语言模型
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model/pipeline parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels) 
        
        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        # print(CausalLMOutputWithPast(
        #     loss=loss,
        #     logits=logits,
        #     past_key_values=outputs.past_key_values,
        #     hidden_states=outputs.hidden_states,
        #     attentions=outputs.attentions,
        # ))
        # exit()
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "images": kwargs.get("images", None),
            }
        )
        return model_inputs

AutoConfig.register("llava", LlavaConfig)
AutoModelForCausalLM.register(LlavaConfig, LlavaLlamaAttForCausalLM)