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
# ------------------------------------------------------------------------
# Modified from LLaVA (https://github.com/haotian-liu/LLaVA)
# Copyright 2023 Yanwei Li
# ------------------------------------------------------------------------

from abc import ABC, abstractmethod
import os
import json
import copy
import random
import time
import numpy as np
import pdb
import torch
import pickle
import torch.nn as nn
import torch.nn.functional as F
import math
from transformers import BertTokenizer
from transformers.models.bert.modeling_bert import BertLMHeadModel as BertLMHeadModelRaw

from .qformer import BertConfig
from .qformer import BertLMHeadModel as BertLMHeadModelQF

from .multimodal_encoder.builder import build_vision_tower
from .multimodal_projector.builder import build_vision_projector

from llamavid.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
    
class LLaMAVIDMetaModel:

    def __init__(self, config):
        super(LLaMAVIDMetaModel, self).__init__(config)
        
        if hasattr(config, "mm_vision_tower"):
            self.vision_tower = build_vision_tower(config, delay_load=True)
            self.mm_projector = build_vision_projector(config)

    def get_vision_tower(self):
        vision_tower = getattr(self, 'vision_tower', None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower

    def initialize_vision_modules(self, model_args, fsdp=None, max_token=2048):
        vision_tower = model_args.vision_tower
        mm_vision_select_layer = model_args.mm_vision_select_layer
        mm_vision_select_feature = model_args.mm_vision_select_feature
        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter

        self.config.mm_vision_tower = vision_tower
        self.config.image_processor = getattr(model_args, 'image_processor', None)

        vision_tower = build_vision_tower(model_args)

        if fsdp is not None and len(fsdp) > 0:
            self.vision_tower = [vision_tower]
        else:
            self.vision_tower = vision_tower

        self.config.use_mm_proj = True
        self.config.mm_projector_type = getattr(model_args, 'mm_projector_type', 'linear')
        self.config.mm_hidden_size = vision_tower.hidden_size
        self.config.mm_vision_select_layer = mm_vision_select_layer
        self.config.mm_vision_select_feature = mm_vision_select_feature
        self.config.max_token = max_token
        
        if getattr(self, 'mm_projector', None) is None:
            self.mm_projector = build_vision_projector(self.config)
        else:
            # In case it is frozen by LoRA
            for p in self.mm_projector.parameters():
                p.requires_grad = True

        if pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location='cpu')
            def get_w(weights, keyword):
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}

            self.mm_projector.load_state_dict(get_w(mm_projector_weights, 'mm_projector'))

    # 一个初始化的函数
    def initialize_attention_modules(self, model_args, for_eval=False):  
        pretrain_mm_mlp_adapter = getattr(model_args, "pretrain_mm_mlp_adapter", None)
        pretrain_qformer = getattr(model_args, "pretrain_qformer", None)
        self.config.bert_type = getattr(model_args, "bert_type", "qformer")
        self.config.num_query = getattr(model_args, "num_query", 32)
        self.config.compress_type = getattr(model_args, "compress_type", None)

        # bert_type: qformer_pretrain
        if 'pretrain' in self.config.bert_type:
            # for qformer that use evaclip for prtrain
            att_feat_size = 1408
        else:
            att_feat_size = self.config.mm_hidden_size

        # 先进行实例化
        self.vlm_att_tokenlizer, self.vlm_att_encoder, self.vlm_att_query = self.init_bert(att_feat_size, truncation_side="left")
        self.vlm_att_projector = torch.nn.Linear(self.vlm_att_encoder.config.hidden_size, self.config.mm_hidden_size)
        self.vlm_att_em_projector = torch.nn.Linear(self.config.mm_hidden_size, self.config.hidden_size) #(xx, 768)
        self.vlm_att_key_projector  = torch.nn.Linear(self.config.mm_hidden_size, self.config.mm_hidden_size)
        self.vlm_att_val_projector  = torch.nn.Linear(self.config.mm_hidden_size, self.config.hidden_size)
        # self.sf_former = SFormer(1408)
        # self.merge = Merger()

        # 需要加一些额外的导入的参数

        if "raw" in self.config.bert_type:
            self.vlm_att_bert_proj  = torch.nn.Linear(att_feat_size, self.vlm_att_encoder.config.hidden_size)
        elif "pretrain" in self.config.bert_type and self.config.mm_hidden_size!=att_feat_size:
            self.vlm_att_bert_proj = torch.nn.Linear(self.config.mm_hidden_size, att_feat_size)
        else:
            self.vlm_att_bert_proj = None
        
        def get_w(weights, keyword):
            return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}
        
        if 'qformer_pretrain' in self.config.bert_type:
            self.vlm_att_ln = torch.nn.LayerNorm(att_feat_size)
        
        if pretrain_qformer is not None:
            print("Loading pretrained qformer weights...")
            qformer_weight = torch.load(pretrain_qformer, map_location='cpu')['model']
            bert_weight = {_key: qformer_weight[_key] for _key in qformer_weight if 'bert' in _key}
            self.vlm_att_encoder.load_state_dict(get_w(bert_weight, 'Qformer'))
            self.vlm_att_ln.load_state_dict(get_w(qformer_weight, 'ln_vision'))
            self.vlm_att_query.data = qformer_weight['query_tokens']
        
        if 'freeze_all' in self.config.bert_type:
            print("Freezing all qformer weights...")
            self.vlm_att_encoder.requires_grad_(False)
            self.vlm_att_ln.requires_grad_(False)
            self.vlm_att_query.requires_grad_(False)
            self.vlm_att_projector.requires_grad_(False)
            self.vlm_att_key_projector.requires_grad_(False)
            self.vlm_att_val_projector.requires_grad_(False)

        elif 'freeze' in self.config.bert_type:
            print("Freezing pretrained qformer weights...")
            self.vlm_att_encoder.requires_grad_(False)
            self.vlm_att_ln.requires_grad_(False)
            self.vlm_att_query.requires_grad_(False)
        

        if pretrain_mm_mlp_adapter is not None:
            att_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location='cpu')
        else:
            # 在训练阶段执行
            trainable_module = ['vlm_att_encoder', 'vlm_att_projector', 'vlm_att_key_projector', 
                                'vlm_att_val_projector', 'vlm_att_query', 'vlm_att_visual_proj',
                                'vlm_att_ln']

            if hasattr(model_args, 'model_name_or_path'):
                model_save_path = model_args.model_name_or_path
            else:
                model_save_path = model_args.model_path

            model_idx_path = getattr(model_args, 'model_path', model_save_path)
            weight_file = json.load(open(os.path.join(model_idx_path, 'pytorch_model.bin.index.json'), 'r'))['weight_map']
            model_path = set([weight_file[_key] for _key in weight_file if any([_module in _key for _module in trainable_module])])
            att_projector_weights = {}
            for _model in model_path:
                att_projector_weights.update(torch.load(os.path.join(model_idx_path, _model), map_location='cpu'))
            
            if len(att_projector_weights) == 0:
                return

        # 走到这里，att_projector_weights中的参数是第一阶段训练的参数
        bert_dict = get_w(att_projector_weights, 'vlm_att_encoder')
        if "bert.embeddings.position_ids" not in bert_dict and "raw_bert" not in self.config.bert_type:
            bert_dict["bert.embeddings.position_ids"] = self.vlm_att_encoder.bert.embeddings.position_ids

        print('Loading pretrained weights...')
        self.vlm_att_encoder.load_state_dict(bert_dict)
        self.vlm_att_projector.load_state_dict(get_w(att_projector_weights, 'vlm_att_projector'))
        self.vlm_att_key_projector.load_state_dict(get_w(att_projector_weights, 'vlm_att_key_projector'))
        self.vlm_att_val_projector.load_state_dict(get_w(att_projector_weights, 'vlm_att_val_projector'))
        print('Loading slow-fast weights...')
        # self.sf_former.load_state_dict(get_w(att_projector_weights, 'sf_former'))
        # self.merge.load_state_dict(get_w(att_projector_weights, 'merge'))
        

        if "qformer" in self.config.bert_type:
            print('Loading vlm_att_query weights...')
            self.vlm_att_query.data = att_projector_weights['model.vlm_att_query']
            if "pretrain" in self.config.bert_type:
                print('Loading vlm_att_ln weights...')
                self.vlm_att_ln.load_state_dict(get_w(att_projector_weights, 'vlm_att_ln'))

        if self.vlm_att_bert_proj is not None:
            print('Loading vlm_att_bert_proj weights...')
            self.vlm_att_bert_proj.load_state_dict(get_w(att_projector_weights, 'vlm_att_bert_proj'))
        
        if for_eval:
            weight_type = torch.float16
            device_type = self.mm_projector[0].weight.device
            self.vlm_att_encoder = self.vlm_att_encoder.to(device=device_type, dtype=weight_type)
            self.vlm_att_projector = self.vlm_att_projector.to(device=device_type, dtype=weight_type)
            self.vlm_att_key_projector = self.vlm_att_key_projector.to(device=device_type, dtype=weight_type)
            self.vlm_att_val_projector = self.vlm_att_val_projector.to(device=device_type, dtype=weight_type)
            self.vlm_att_em_projector = self.vlm_att_em_projector.to(device=device_type, dtype=weight_type)
            # self.merge = self.merge.to(device=device_type, dtype=weight_type)
            # self.sf_former = self.sf_former.to(device=device_type, dtype=weight_type)
            
            if "qformer" in self.config.bert_type:
                self.vlm_att_query.data = self.vlm_att_query.data.to(device=device_type, dtype=weight_type)
                if "pretrain" in self.config.bert_type:
                    self.vlm_att_ln = self.vlm_att_ln.to(device=device_type, dtype=weight_type)
            
            if self.vlm_att_bert_proj is not None:
                self.vlm_att_bert_proj = self.vlm_att_bert_proj.to(device=device_type, dtype=weight_type)
            

    def init_bert(self, vision_width, cross_attention_freq=2, truncation_side="right"):
        bert_dir = "/root/.cache/huggingface/hub/bert-base-uncased/"
        # initialize BERT tokenizer
        tokenizer = BertTokenizer.from_pretrained(bert_dir, truncation_side=truncation_side)
        tokenizer.add_special_tokens({"bos_token": "[DEC]"})
        # initialize BERT
        encoder_config = BertConfig.from_pretrained(bert_dir)
        encoder_config.encoder_width = vision_width
        # insert cross-attention layer every other block for image tokens
        encoder_config.add_cross_attention = True
        encoder_config.cross_attention_freq = cross_attention_freq
        query_tokens = None
        
        if "qformer" in self.config.bert_type: # qformer_pretrain
            mm_model = BertLMHeadModelQF.from_pretrained(bert_dir, config=encoder_config)
            query_tokens = nn.Parameter(torch.zeros(1, self.config.num_query, encoder_config.hidden_size))
            query_tokens.data.normal_(mean=0.0, std=encoder_config.initializer_range)
        
        mm_model.resize_token_embeddings(len(tokenizer))
        mm_model.cls = None
        
        return tokenizer, mm_model, query_tokens

# def save_func_id(initial_cluster_id, real_cluster_id):
    
#     id_data = dict()
#     id_data['initial_id'] = initial_cluster_id
#     id_data['real_id'] = real_cluster_id

#     with open('/mnt_new/hanyudong.hyd/Slow-Fast-Vid/LVBench/scripts/cluster_id.pkl', 'wb') as f:
#         pickle.dump(id_data, f)

class LLaMAVIDMetaForCausalLM(ABC):

    @abstractmethod
    def get_model(self): # 子类必须重写
        pass

    def get_vision_tower(self):
        return self.get_model().get_vision_tower()

    def encode_images(self, images, prompts=None, image_counts=None, long_video=False):    
        
        # images: [xx, 3, 224, 224]
        image_features = self.get_model().get_vision_tower()(images)
        # image_features: [B*T, 256 + 1, 1408], [4, 256 + 1, 1408]
        image_features = self.vlm_attention(image_features, prompts=prompts, image_counts=image_counts, long_video=long_video)
        return image_features

    def compress_spatial_features(self, image_features, compress_size=8):
        # image_features: [T, P*P, D] -> [435, 257, 1408]
        patch_size = round(math.sqrt(image_features.shape[1])) # 16
        image_features = image_features.view(-1, patch_size, patch_size, image_features.shape[-1]) # [T, P, P, D]
        image_features = image_features.permute(0, 3, 1, 2)  # [T, D, P, P]
        pooled_features = F.avg_pool2d(image_features, (patch_size // compress_size, patch_size // compress_size))
        pooled_features = pooled_features.permute(0, 2, 3, 1)  # [T, P, P, D]
        image_features = pooled_features.view(-1, compress_size * compress_size, pooled_features.shape[-1])
        return image_features

    def weighted_kmeans_feature(self, img_feature, video_max_frames, weights=None):
        
        if weights is None:
            weights = torch.ones(img_feature.size(0), dtype=img_feature.dtype, device=img_feature.device)

        def weighted_kmeans_torch(X, num_clusters, weights=None, distance='euclidean', tol=1e-4, max_iter=10):
            cluster_index = torch.randperm(X.size(0), device=X.device)[:num_clusters]  # 进行随机打乱
            centroids = X[cluster_index]
            for i in range(max_iter):
                if distance == 'euclidean':
                    dists = ((X.unsqueeze(1) - centroids.unsqueeze(0)) ** 2).sum(dim=2).sqrt()
                else:
                    raise NotImplementedError("Only Euclidean distance is supported yet")
                labels = torch.argmin(dists, dim=1)
                weighted_sum = torch.zeros_like(centroids)
                weights_sum = torch.zeros(num_clusters, dtype=X.dtype, device=X.device)
                for j in range(num_clusters): # 遍历所有的聚类中心
                    cluster_mask = (labels == j)
                    weighted_sum[j] = torch.sum(weights[cluster_mask, None] * X[cluster_mask], dim=0)
                    weights_sum[j] = torch.sum(weights[cluster_mask])
                mask = weights_sum > 0
                new_centroids = torch.zeros_like(weighted_sum)
                new_centroids[mask] = weighted_sum[mask] / weights_sum[mask, None]
                if mask.sum() < num_clusters:  # fix nan centroids
                    new_centroids[~mask] = torch.stack([X[random.randint(0, X.size(0) - 1)] for _ in range(num_clusters - mask.sum())])
                diff = torch.norm(centroids - new_centroids, dim=1).sum()
                if diff < tol:
                    break
                centroids = new_centroids

            return centroids, labels, weights_sum, cluster_index
        
        T, P, D = img_feature.shape
        T0 = video_max_frames
        if T <= T0:
            cluster_index = list(range(0, T))
            return img_feature, weights, [[[i] for i in range(T)]], torch.tensor(cluster_index)
        X = img_feature.view(T, -1)  # [T, P, D]  -> [T, P*D] 将后两个维度保留避免丢失大量的信息
        centroids, labels, weights, cluster_index = weighted_kmeans_torch(X, T0, weights)
        reduced_feature = centroids.view(T0, P, D)  # 代表了聚类中心
        step_indices = [[] for _ in range(T0)]
        for i in range(T0):
            step_indices[i] = [j for j in range(T) if labels[j] == i] # 如果第j个数据样本属于第i个类
        return reduced_feature, weights, [step_indices], cluster_index


    def slow_fast_modeling(self, img_feature, q_emb, has_image=False):
        hs_feature = img_feature
        # if has_image > 1:
        # Parameter Setting
        T = hs_feature.size(0)
        # image_feature: [T, 256, C]
        video_long_memory_length = 80 # int(np.ceil(T*0.1))
        compress_long_memory_size = 4 # temporal memory

        long_memory = img_feature  # [L, P*P, D]
        long_memory = self.compress_spatial_features(long_memory, compress_long_memory_size) # [L, P'*P', D]
        N = long_memory.size(1)
        assert T == long_memory.size(0)

        # Key Frame Indicator 
        all_id = list(range(0, T))
        fast_feature, weight, step_long_indices, cluster_index = self.weighted_kmeans_feature(long_memory, video_long_memory_length) # [L_long, P'*P', D], [L_long]
        key_id = cluster_index.cpu().numpy().tolist()
        initial_cluster = copy.deepcopy(key_id)
        # diff_id = list(set(all_id) - set(key_id))
        struc_feature = self.get_merge_model()(hs_feature)
        
        if not self.training:
            # struc_feature = struc_feature.bfloat16()
            # q_emb = q_emb.bfloat16()
            # fast_feature.bfloat16()
            struc_feature = struc_feature.half()
            q_emb = q_emb.half()
            fast_feature.half()
            
        kk = struc_feature.size(1)
        slow_feature = struc_feature[key_id]

        # Select the Salient Frame
        if fast_feature.size(0) != T:
            fast_feature = fast_feature.reshape(1, video_long_memory_length*N, 1408)
            slow_feature = slow_feature.reshape(1, video_long_memory_length*kk, 1408)
            key_frame_feature, pseudo_key_index = self.get_selective_net()(fast_feature, slow_feature, N, video_long_memory_length)
        else:
            fast_feature = fast_feature.reshape(1, T*N, 1408)
            slow_feature = slow_feature.reshape(1, T*kk, 1408)
            key_frame_feature, pseudo_key_index = self.get_selective_net()(fast_feature, slow_feature, N, T)
        
        pseudo_key_index = pseudo_key_index[0].cpu().numpy().tolist()
        cluster_index = cluster_index[pseudo_key_index] # stem from the learnable indicator
        key_id = cluster_index.cpu().numpy().tolist()
        diff_id = list(set(all_id) - set(key_id))

        # preserve the visualization result
        # save_func_id(initial_cluster, key_id) # segment the video clip according to the initial_cluster

        # struc_feature = self.get_merge_model()(hs_feature)
        # slow_feature = struc_feature[key_id]
        # key_frame_feature = torch.cat([fast_feature, slow_feature], dim=1)  # 直接级联
        # key_frame_feature = self.get_sformer_model()(fast_feature, slow_feature) # fast_feature和slow_feature已经保证对齐
        # key_frame_feature = key_frame_feature.float()
        key_frame_feature = self.get_model().mm_projector(key_frame_feature)
        o_sorted_indices = torch.argsort(cluster_index)
        key_frame_feature = key_frame_feature[o_sorted_indices] # 按照帧出现的大小排序(可能是因为这个顺序导致的) 
        # For Non-Key Frame
        assert q_emb.size(0) == struc_feature.size(0)
        if len(diff_id) != 0:
            non_key_id = sorted(diff_id) # 必须进行排序
            non_key_frame_feature = self.one_token_generation(struc_feature[non_key_id], q_emb[non_key_id]) # Here, non_key_set和non_key_set是相互对应的, [T2, 1, 1408]
            # Two-Ponter Sequential Interpolation
            key_pointer = 0  # xxxxx
            non_key_pointer = 0 # xxxxxxxxx
            for cnt, frame_idx in enumerate(range(0, T)):
                if cnt == 0:
                    if frame_idx in key_id:
                        summarized_feature = key_frame_feature[key_pointer][None]
                        key_pointer = key_pointer + 1
                    elif frame_idx in non_key_id:
                        summarized_feature = non_key_frame_feature[non_key_pointer][None]
                        non_key_pointer = non_key_pointer + 1
                else:
                    if frame_idx in key_id:
                        cur_feature = key_frame_feature[key_pointer][None]
                        summarized_feature = torch.cat([summarized_feature, cur_feature], dim=1)
                        key_pointer = key_pointer + 1
                    elif frame_idx in non_key_id:
                        cur_feature = non_key_frame_feature[non_key_pointer][None]
                        summarized_feature = torch.cat([summarized_feature, cur_feature], dim=1)
                        non_key_pointer = non_key_pointer + 1
        else:

            key_id = sorted(key_id) # 必须进行排序
            non_key_frame_feature = self.one_token_generation(struc_feature[key_id], q_emb[key_id]) # [T, 2, 1408] [T, 16, 1408]
            summarized_feature = torch.cat([non_key_frame_feature, key_frame_feature], dim=1)
        
        # else:
        #     summarized_feature = hs_feature.mean(dim=1, keepdim=True)
        #     summarized_feature = self.get_model().mm_projector(summarized_feature)
            # struc_feature = self.get_merge_model()(hs_feature) # [1, 6, 1408]
            # coarse_emb = self.one_token_generation(struc_feature, q_emb) # [1, 2, 4096]
            # fine_emb = self.get_model().mm_projector(struc_feature) # [1, 6, 4096]
            # summarized_feature = torch.cat([coarse_emb, fine_emb], dim=1) # [1, 6+2, 4096]

        summarized_feature = summarized_feature.flatten(0, 1)
        return summarized_feature
        
    def reshape_2x2_image_features(self, image_features):
        B, P, D = image_features.shape
        patch_size = round(math.sqrt(P))
        assert patch_size % 2 == 0, "Patch size must be divisible by 2."
        image_features = image_features.reshape(B, patch_size, patch_size, D)
        image_features_2x2 = image_features.reshape(B, patch_size // 2, 2, patch_size // 2, 2, D)
        image_features_2x2 = image_features_2x2.permute(0, 1, 3, 2, 4, 5)  
        image_features_2x2 = image_features_2x2.reshape(B, patch_size // 2, patch_size // 2, 4 * D)  # concat 2x2 neighbor patches
        image_features = image_features_2x2.reshape(B, (patch_size // 2) ** 2, 4 * D)
        return image_features

    def vlm_attention(self, image_features, prompts=None, image_counts=None, long_video=False):  
        img_feat_lst = []
        # assign prompt for each frame feature
        if image_counts is None:
            assert len(image_features) == len(prompts), f"Size mismatch! image_features: {len(image_features)}, prompts: {len(prompts)}"
        else:
            assert len(prompts) == len(image_counts), f"Size mismatch! prompts: {len(prompts)}, image_counts: {len(image_counts)}"
        image_atts = torch.ones(image_features.size()[:-1], dtype=torch.long).to(image_features.device)    
        total_count = 0
        # calculate each image feat according to the prompt
        for _idx in range(len(prompts)): # [[has many prompt?], [], []]
            assert isinstance(prompts[_idx], list), f"Prompt should be a list, but got {type(prompts[_idx])}"
            input_token = self.get_model().vlm_att_tokenlizer(
                prompts[_idx], 
                padding='longest', 
                truncation=True,
                max_length=256,
                return_tensors="pt"
                ).to(image_features.device)

            input_ids = input_token.input_ids # (1, 15)
            attention_masks = input_token.attention_mask # [1,1,1,1,0]
            # if image_counts is None:
            #     img_feat_prompt = image_features[_idx, None].expand(len(prompts[_idx]), -1, -1)
            #     img_att_prompt = image_atts[_idx, None].expand(len(prompts[_idx]), -1)
            # else:
                # shape: [prompt_num*frame_num, image_shape, feat_dim]
            img_feat_prompt = image_features[total_count:total_count+image_counts[_idx]]
            img_feat_prompt = img_feat_prompt[None].expand(len(prompts[_idx]), -1, -1, -1).flatten(0,1)
            img_att_prompt = image_atts[total_count:total_count+image_counts[_idx]]
            img_att_prompt = img_att_prompt[None].expand(len(prompts[_idx]), -1, -1).flatten(0,1)
            input_ids = input_ids[:,None].expand(-1, image_counts[_idx], -1).flatten(0,1) # 将一个问题扩展和视频帧数一样的长度
            attention_masks = attention_masks[:,None].expand(-1, image_counts[_idx], -1).flatten(0,1)
            total_count += image_counts[_idx]
            
            if "pretrain" in self.config.bert_type and self.get_model().vlm_att_bert_proj is not None:
                bert_feat = self.get_model().vlm_att_bert_proj(img_feat_prompt)
            else:
                bert_feat = img_feat_prompt.clone()

            # remove cls embedding
            if self.config.mm_vision_select_feature == 'patch':
                if img_feat_prompt.shape[1] % 2 == 1: # 判断patch是否冗余
                    img_feat_prompt = img_feat_prompt[:, 1:] # 257

            if "qformer" in self.config.bert_type:
                # self.get_model().vlm_att_query是qformer初始化的query token
                query_tokens = self.get_model().vlm_att_query.expand(bert_feat.shape[0], -1, -1) # 将query扩展到和T一个维度
                query_atts = torch.cat([torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(bert_feat.device), attention_masks],dim=1)
                
                if 'pretrain' in self.config.bert_type:
                    mm_img_in = self.get_model().vlm_att_ln(bert_feat)
                else:
                    mm_img_in = bert_feat
                
                # Qformer
                mm_output = self.get_model().vlm_att_encoder.bert(
                    input_ids,
                    query_embeds=query_tokens,
                    attention_mask=query_atts,
                    encoder_hidden_states=mm_img_in,
                    encoder_attention_mask=img_att_prompt,
                    return_dict=True,
                )
                mm_output = mm_output.last_hidden_state[:,:query_tokens.shape[1]] # 得到Qformer的query(只取前32 token)
                
            else:

                raise ValueError(f'Unexpected bert type: {self.config.bert_type}')

            q_emb = self.get_model().vlm_att_projector(mm_output)  # 这里采用Qformer的形式
            q_emb = q_emb.reshape(len(prompts[_idx]), image_counts[_idx], 32, -1)
            img_feat_prompt = img_feat_prompt.reshape(len(prompts[_idx]), image_counts[_idx], 256, -1)

            for sub_idx in range(len(prompts[_idx])):
                if sub_idx == 0:
                    final_token = self.slow_fast_modeling(img_feat_prompt[sub_idx], q_emb[sub_idx], image_counts[_idx])
                    final_token = final_token.unsqueeze(0)                
                else:
                    token = self.slow_fast_modeling(img_feat_prompt[sub_idx], q_emb[sub_idx], image_counts[_idx])
                    token = token.unsqueeze(0)  
                    final_token = torch.cat([final_token, token], dim=0)

            img_feat_lst.append(final_token)

        return img_feat_lst

    def one_token_generation(self, vis_embed, q_emb, long_video=False):
        """
        vis_embed: [16, 256, 1408]
        ctx_embed: [31, 256, 1408]
        text_q: [31, 256, 1408]
        """
        ctx_key = self.get_model().vlm_att_key_projector(vis_embed)
        # Key part 1: calculate context-related embedding
        sim = q_emb @ ctx_key.transpose(-1,-2) 
        sim_norm = sim / (vis_embed.shape[-1] ** 0.5)
        ctx_embed = (sim_norm.softmax(-1) @ vis_embed).mean(1)
        ctx_embed = self.get_model().vlm_att_val_projector(ctx_embed[:,None]) # 4096
        vis_embed = vis_embed.mean(dim=1, keepdim=True)
        vis_embed = self.get_model().mm_projector(vis_embed)
        final_token = torch.cat([vis_embed, ctx_embed], dim=1) # [T, 2, 1408]
        return final_token

    def update_prompt(self, prompts=None):
        self.prompts = prompts

    def prepare_inputs_labels_for_multimodal(self, input_ids, attention_mask, past_key_values, labels, images, prompts=None):   # 准备各种数据 
        if prompts is None and hasattr(self, 'prompts'):
            prompts = self.prompts
        
        vision_tower = self.get_vision_tower()

        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            print("step there when inference...")
            if past_key_values is not None and vision_tower is not None and images is not None and input_ids.shape[1] == 1:
                attention_mask = torch.ones((attention_mask.shape[0], past_key_values[-1][-1].shape[-2] + 1), dtype=attention_mask.dtype, device=attention_mask.device)
            return input_ids, attention_mask, past_key_values, None, labels

        images = [image if len(image.shape) == 4 else image.unsqueeze(0) for image in images]
        image_counts = [image.shape[0] for image in images]
        concat_images = torch.cat(images, dim=0)
        image_features = self.encode_images(concat_images, prompts, image_counts) # 822 + 57 = 879 tokens

        new_input_embeds = []
        new_labels = [] if labels is not None else None
        cur_image_idx = 0

        for batch_idx, cur_input_ids in enumerate(input_ids):
            if (cur_input_ids == IMAGE_TOKEN_INDEX).sum() == 0:
                # multimodal LLM, but the current sample is not multimodal
                # FIXME: this is a hacky fix, for deepspeed zero3 to work
                half_len = cur_input_ids.shape[0] // 2
                if isinstance(image_features, list):
                    cur_image_features = image_features[cur_image_idx][0]
                else:
                    cur_image_features = image_features[cur_image_idx]
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids[:half_len])
                cur_input_embeds_2 = self.get_model().embed_tokens(cur_input_ids[half_len:])
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0], cur_input_embeds_2], dim=0)
                new_input_embeds.append(cur_input_embeds)
                if labels is not None:
                    new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                continue

            image_token_indices = torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0]
            # print(image_token_indices)
            cur_new_input_embeds = []
            if labels is not None:
                cur_labels = labels[batch_idx]
                cur_new_labels = []
                assert cur_labels.shape == cur_input_ids.shape
            
            token_idx = 0
            # print(image_token_indices.numel(), image_token_indices, "*********")
            # only one image token
            while image_token_indices.numel() > 0:
                if isinstance(image_features, list):
                    cur_image_features = image_features[cur_image_idx][token_idx]
                else:
                    cur_image_features = image_features[cur_image_idx]

                image_token_start = image_token_indices[0]
                
                if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
                    cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids[:image_token_start-1]).detach())
                    cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids[image_token_start-1:image_token_start]))
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids[image_token_start+1:image_token_start+2]))
                    if labels is not None:
                        cur_new_labels.append(cur_labels[:image_token_start])
                        cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=labels.device, dtype=labels.dtype))
                        cur_new_labels.append(cur_labels[image_token_start:image_token_start+1])
                        cur_labels = cur_labels[image_token_start+2:]
                else:
                    cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids[:image_token_start]))
                    cur_new_input_embeds.append(cur_image_features)

                    if labels is not None:
                        cur_new_labels.append(cur_labels[:image_token_start])
                        cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=labels.device, dtype=labels.dtype))
                        cur_labels = cur_labels[image_token_start+1:]
                
                if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
                    cur_input_ids = cur_input_ids[image_token_start+2:]
                else:
                    cur_input_ids = cur_input_ids[image_token_start+1:]
                image_token_indices = torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0]
                token_idx += 1
            
            # changle image idx after processing one sample
            cur_image_idx += 1

            if cur_input_ids.numel() > 0:
                if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
                    cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids).detach())
                else:
                    cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids))
                if labels is not None:
                    cur_new_labels.append(cur_labels)

            cur_new_input_embeds = [x.to(device=self.device) for x in cur_new_input_embeds]
            cur_new_input_embeds = torch.cat(cur_new_input_embeds, dim=0)
            new_input_embeds.append(cur_new_input_embeds)  # 对不同batch进行封装
            if labels is not None:
                cur_new_labels = torch.cat(cur_new_labels, dim=0)
                new_labels.append(cur_new_labels)
        
        # 分batch对数据进行组装
        if any(x.shape != new_input_embeds[0].shape for x in new_input_embeds):  # 如果维度不一致，就要进行padding
            max_len = max(x.shape[0] for x in new_input_embeds)
            new_input_embeds_align = []
            for cur_new_embed in new_input_embeds:
                cur_new_embed = torch.cat((cur_new_embed, torch.zeros((max_len - cur_new_embed.shape[0], cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)), dim=0)
                new_input_embeds_align.append(cur_new_embed)
            new_input_embeds = torch.stack(new_input_embeds_align, dim=0)

            if labels is not None:
                new_labels_align = []
                _new_labels = new_labels
                for cur_new_label in new_labels:
                    cur_new_label = torch.cat((cur_new_label, torch.full((max_len - cur_new_label.shape[0],), IGNORE_INDEX, dtype=cur_new_label.dtype, device=cur_new_label.device)), dim=0)
                    new_labels_align.append(cur_new_label)
                new_labels = torch.stack(new_labels_align, dim=0)

            # only used for right padding in tokenlizer
            if attention_mask is not None:
                new_attention_mask = []
                for cur_attention_mask, cur_new_labels, cur_new_labels_align in zip(attention_mask, _new_labels, new_labels):
                    new_attn_mask_pad_left = torch.full((cur_new_labels.shape[0] - labels.shape[1],), True, dtype=attention_mask.dtype, device=attention_mask.device)
                    new_attn_mask_pad_right = torch.full((cur_new_labels_align.shape[0] - cur_new_labels.shape[0],), False, dtype=attention_mask.dtype, device=attention_mask.device)
                    cur_new_attention_mask = torch.cat((new_attn_mask_pad_left, cur_attention_mask, new_attn_mask_pad_right), dim=0)
                    new_attention_mask.append(cur_new_attention_mask)
                attention_mask = torch.stack(new_attention_mask, dim=0)
                assert attention_mask.shape == new_labels.shape
        else:

            new_input_embeds = torch.stack(new_input_embeds, dim=0)  # 如果维度一致，就能直接进行级联了

            if labels is not None:
                new_labels  = torch.stack(new_labels, dim=0)
            
            # only used for right padding in tokenlizer
            if attention_mask is not None:
                new_attn_mask_pad_left = torch.full((attention_mask.shape[0], new_input_embeds.shape[1] - input_ids.shape[1]), True, dtype=attention_mask.dtype, device=attention_mask.device)
                attention_mask = torch.cat((new_attn_mask_pad_left, attention_mask), dim=1)
                assert attention_mask.shape == new_input_embeds.shape[:2]

        return None, attention_mask, past_key_values, new_input_embeds, new_labels


    def initialize_vision_tokenizer(self, model_args, tokenizer):

        # 往之前的token里加入新的patch
        if model_args.mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

        if model_args.mm_use_im_start_end:
            num_new_tokens = tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

            if model_args.pretrain_mm_mlp_adapter:
                mm_projector_weights = torch.load(model_args.pretrain_mm_mlp_adapter, map_location='cpu')
                embed_tokens_weight = mm_projector_weights['model.embed_tokens.weight']
                assert num_new_tokens == 2
                if input_embeddings.shape == embed_tokens_weight.shape:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight[-num_new_tokens:]
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    raise ValueError(f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}.")
        elif model_args.mm_use_im_patch_token:
            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = False
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False
