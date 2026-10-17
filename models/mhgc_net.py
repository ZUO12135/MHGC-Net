"""MHGC with paper-aligned module names and legacy-checkpoint support."""

from collections import OrderedDict
from typing import Mapping, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter
import math
from torch.nn.utils.rnn import pad_sequence
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.nn.parameter import Parameter
from torch.nn import init
from torch.nn import functional as F
import fairseq
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'pretrained_models', 'Speech-Emotion-Recognition-main'))
from .ACRNN import acrnn
from .AASIST import *


class BCAM(nn.Module):
    """Bidirectional Cross-Attention Module (BCAM) for graph feature alignment."""

    def __init__(self, dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by num_heads ({num_heads})")

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        # projections for X -> Y attention
        self.q_x = nn.Linear(dim, dim)
        self.k_y = nn.Linear(dim, dim)
        self.v_y = nn.Linear(dim, dim)
        self.out_x = nn.Linear(dim, dim)

        # projections for Y -> X attention
        self.q_y = nn.Linear(dim, dim)
        self.k_x = nn.Linear(dim, dim)
        self.v_x = nn.Linear(dim, dim)
        self.out_y = nn.Linear(dim, dim)

        self.norm_x = nn.LayerNorm(dim)
        self.norm_y = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)

    def _reshape(self, tensor: torch.Tensor) -> torch.Tensor:
        b, n, _ = tensor.size()
        tensor = tensor.view(b, n, self.num_heads, self.head_dim)
        return tensor.permute(0, 2, 1, 3)  # (B, heads, N, head_dim)

    def _merge(self, tensor: torch.Tensor) -> torch.Tensor:
        b, heads, n, head_dim = tensor.size()
        tensor = tensor.permute(0, 2, 1, 3).contiguous()
        return tensor.view(b, n, heads * head_dim)

    def _attend(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.drop(attn)
        return torch.matmul(attn, v)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if x.dim() != 3 or y.dim() != 3:
            raise RuntimeError(f"Expected 3D tensors, got {x.shape} and {y.shape}")

        x_norm = self.norm_x(x)
        y_norm = self.norm_y(y)

        # X attends to Y
        q_x = self._reshape(self.q_x(x_norm))
        k_y = self._reshape(self.k_y(y_norm))
        v_y = self._reshape(self.v_y(y_norm))
        ctx_x = self._attend(q_x, k_y, v_y)
        ctx_x = self._merge(ctx_x)
        x = x + self.drop(self.out_x(ctx_x))

        # Y attends to X
        q_y = self._reshape(self.q_y(y_norm))
        k_x = self._reshape(self.k_x(x_norm))
        v_x = self._reshape(self.v_x(x_norm))
        ctx_y = self._attend(q_y, k_x, v_x)
        ctx_y = self._merge(ctx_y)
        y = y + self.drop(self.out_y(ctx_y))

        return x, y


# =========================================================================
#  ASR Model (High Performance Version - No MaxPool, Uses LayerNorm)
# =========================================================================
class SemanticViewEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        cp_path = os.path.realpath(os.path.expanduser(os.environ.get(
            'MHGC_XLSR_CHECKPOINT',
            os.path.join(PROJECT_ROOT, 'pretrained_models', 'xlsr2_300m.pt'),
        )))
        if not os.path.isfile(cp_path):
            raise FileNotFoundError(
                'Missing XLS-R checkpoint: {}. Set MHGC_XLSR_CHECKPOINT.'.format(cp_path)
            )
        model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([cp_path])
        self.model = model[0]
        self.model.feature_grad_mult = 0.0  # Freeze Frontend to save memory
        
        # 使用 LayerNorm 替代 BN，且不做暴力池化，保留细节
        self.proj = nn.Sequential(
            nn.Linear(1024, 128),
            nn.LayerNorm(128),
            nn.SELU(inplace=True),
            nn.Linear(128, 64),     # Project to 64 for GAT
            nn.LayerNorm(64),
            nn.SELU(inplace=True)
        )

    def forward(self, x):
        # Extract features (B, T, 1024)
        emb = self.model(x, mask=False, features_only=True)['x']
        # Project keeping time dimension (B, T, 64)
        emb = self.proj(emb)
        return emb


# =========================================================================
#  SER Model (High Performance Version - No MaxPool)
# =========================================================================
class ProsodicViewEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        cp_path = os.path.realpath(os.path.expanduser(os.environ.get(
            'MHGC_SER_CHECKPOINT',
            os.path.join(PROJECT_ROOT, 'pretrained_models', 'ser_acrnn.pth'),
        )))
        if not os.path.isfile(cp_path):
            raise FileNotFoundError(
                'Missing ACRNN checkpoint: {}. Set MHGC_SER_CHECKPOINT.'.format(cp_path)
            )
        model = acrnn()
        model.load_state_dict(torch.load(cp_path, map_location='cpu'))
        self.model = model
        
        # Projection head to align with GAT dim (64)
        self.proj = nn.Sequential(
            nn.Linear(256, 128),  # Assuming ACRNN outputs 256 dim channels
            nn.LayerNorm(128),
            nn.SELU(inplace=True),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.SELU(inplace=True)
        )

    def forward(self, x):
        emb = self.model(x) # Expected (B, 256, T) or (B, T, 256)
        
        # Ensure shape is (Batch, Time, Channels) before Linear
        if emb.dim() == 4:
            emb = emb.squeeze(-1)
        if emb.size(1) == 256: # If Channel first (B, C, T)
            emb = emb.permute(0, 2, 1) # -> (B, T, C)
            
        emb = self.proj(emb) # -> (B, T, 64)
        return emb


class AcousticViewEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        filts = [[1, 32], [32, 32], [32, 64], [64, 64]]
        self.conv_time = CONV(out_channels=70,
                              kernel_size=128,
                              in_channels=1)
        self.first_bn = nn.BatchNorm2d(num_features=1)
        self.selu = nn.SELU(inplace=True)
        self.encoder = nn.Sequential(
            nn.Sequential(Residual_block(nb_filts=filts[0], first=True)),
            nn.Sequential(Residual_block(nb_filts=filts[1])),
            nn.Sequential(Residual_block(nb_filts=filts[2])),
            nn.Sequential(Residual_block(nb_filts=filts[3])),
            nn.Sequential(Residual_block(nb_filts=filts[3])),
            nn.Sequential(Residual_block(nb_filts=filts[3])))

    def forward(self, x, Freq_aug):
        x = x.unsqueeze(1)
        x = self.conv_time(x, mask=Freq_aug)
        x = x.unsqueeze(dim=1)
        x = F.max_pool2d(torch.abs(x), (3, 3))
        x = self.first_bn(x)
        x = self.selu(x)
        emb = self.encoder(x)
        out, _ = torch.max(torch.abs(emb), dim=2)
        out = out.transpose(1, 2)
        return out


class MRGF(nn.Module):
    def __init__(self):
        super().__init__()
        self.HtrgGAT_layer1 = HtrgGraphAttentionLayer(64, 64, temperature=100)
        self.HtrgGAT_layer2 = HtrgGraphAttentionLayer(64, 64, temperature=100)
        self.drop_way = nn.Dropout(0.2, inplace=True)
        self.stack_node = nn.Parameter(torch.randn(1, 1, 64))

    def forward(self, x1, x2):
        # 在HtrgGAT层调用之前先对齐节点数量
        if x1.size(1) != x2.size(1):
            min_nodes = min(x1.size(1), x2.size(1))
            # 截取到相同的节点数量
            x1 = x1[:, :min_nodes, :]
            x2 = x2[:, :min_nodes, :]
        
        stack_node = self.stack_node.expand(x1.size(0), -1, -1)
        x1, x2, stack_node, _ = self.HtrgGAT_layer1(x1, x2, master=stack_node)
        x1_aug, x2_aug, stack_node2, attmap = self.HtrgGAT_layer2(x1, x2, master=stack_node)
        
        if x1.dim() == 2:
            x1 = x1.unsqueeze(0)
        if x2.dim() == 2:
            x2 = x2.unsqueeze(0)
            
        x1 = x1 + x1_aug 
        x2 = x2 + x2_aug
        stack_node = stack_node + stack_node2 
        x1 = self.drop_way(x1)
        x2 = self.drop_way(x2)
        stack_node = self.drop_way(stack_node)
        
        return x1+x2, stack_node, attmap


class HGRM(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool1 = GraphPool(0.5, 64, 0.3)
        self.pool2 = GraphPool(0.5, 64, 0.3)
        
    def forward(self, x_list):
        pool_list=[]
        for i in x_list:
            pool_list.append(self.pool2(self.pool1(i)))
        pool_cat=torch.cat(pool_list, dim=1)
        pool_max, _=torch.max(torch.abs(pool_cat),dim=1)
        pool_avg=torch.mean(pool_cat,dim=1)
        return torch.cat([pool_max,pool_avg], dim=1)


class SVGE(nn.Module):
    """Single-View Graph Encoder (SVGE)."""

    def __init__(self):
        super().__init__()
        self.gat = GraphAttentionLayer(64, 64, temperature=2.0)
        self.pool = GraphPool(0.5, 64, 0.3)

    def forward(self, view):
        return self.pool(self.gat(view))


class LatentStyleBlending(nn.Module):
    """Label-conditioned Latent Style Blending (LSB)."""

    def forward(self, views, labels):
        indices = same_class_permutation(labels)
        blended = tuple(
            blend_latent_statistics(view, view[indices], dim=-1) for view in views
        )
        return blended, indices


class StochasticFeatureReorganization(nn.Module):
    """Stochastic Feature Reorganization (SFR) of the acoustic view."""

    def forward(self, acoustic_view):
        indices = torch.randperm(acoustic_view.size(0), device=acoustic_view.device)
        return acoustic_view[indices], indices


class MHGCNet(nn.Module):
    def __init__(self, use_lsb=True, use_sfr=True, 
                 lsb_prob=0.5, sfr_prob=0.3,
                 use_mrgf=True, use_bcam=True, use_svge=True, use_hgrm=True,
                 active_views=None):
        super().__init__()
        self.use_svge = use_svge
        self.use_hgrm = use_hgrm
        # active_views: 激活的模态列表，如 ['audio']、['audio','text']、['audio','text','emo']
        # None 表示全部三个模态都激活（默认full model）
        if active_views is None:
            active_views = ['acoustic', 'semantic', 'prosodic']
        self.active_views = set(active_views)
        """multi-view feature extractor"""
        self.semantic_view_encoder = SemanticViewEncoder()
        self.prosodic_view_encoder = ProsodicViewEncoder()
        self.acoustic_view_encoder = AcousticViewEncoder()
        """SVGE"""
        self.svge_semantic = SVGE()
        self.svge_prosodic = SVGE()
        self.svge_acoustic = SVGE()
        """MRGF"""
        self.use_mrgf = use_mrgf
        self.use_bcam = use_bcam
        if self.use_mrgf:
            self.mrgf_ae = MRGF()
            self.mrgf_at = MRGF()
            self.mrgf_et = MRGF()
            self.mrgf_aet = MRGF()
        else:
            self.mrgf_ae = None
            self.mrgf_at = None
            self.mrgf_et = None
            self.mrgf_aet = None
        if self.use_bcam:
            self.bcam_ae_at = BCAM(64)
            self.bcam_et_aet = BCAM(64)
        else:
            self.bcam_ae_at = None
            self.bcam_et_aet = None
        """HGRM"""
        self.hgrm_view=HGRM()
        self.hgrm_relation=HGRM()
        self.hgrm_master=HGRM()
        self.drop = nn.Dropout(0.5, inplace=True)
        self.fusion_projection = nn.Linear(384, 64)
        self.classifier = nn.Linear(64, 2)
        
        # ========== 特征增强策略配置 ==========
        self.use_lsb = use_lsb
        self.use_sfr = use_sfr
        self.lsb_prob = lsb_prob
        self.sfr_prob = sfr_prob
        self.lsb = LatentStyleBlending()
        self.sfr = StochasticFeatureReorganization()

    def _graph_fusion_forward(self, audio_view, emo_view, text_view, return_feat: bool = False):
        """Run SVGE + MRGF + BCAM + HGRM for a given set of views.

        If return_feat is True, returns a tuple (logits, feat) where feat is the
        64-D representation right before the final 2-class classifier.
        """
        if self.use_svge:
            emo_gat = self.svge_prosodic(emo_view)
            audio_gat = self.svge_acoustic(audio_view)
            text_gat = self.svge_semantic(text_view)
        else:
            # Fallback: bypass SVGE but keep a graph-like (B, N, C) shape.
            # Downstream MRGF / HGRM expects a node dimension.
            emo_gat = emo_view
            audio_gat = audio_view
            text_gat = text_view

        (AE_HG, AT_HG, ET_HG, AET_HG,
         AE_SN, AT_SN, ET_SN, AET_SN) = self._hetero_forward(audio_gat, emo_gat, text_gat)

        GAT_Group=[audio_gat,emo_gat,text_gat]
        HGAT_Group=[AE_HG,AT_HG,ET_HG,AET_HG]
        SN_Group=[AE_SN,AT_SN,ET_SN,AET_SN]
        
        if self.use_hgrm:
            out1=self.hgrm_view(GAT_Group)
            out2=self.hgrm_relation(HGAT_Group)
            out3=self.hgrm_master(SN_Group)
            last_hidden = torch.cat([out1,out2,out3], dim=1)
            last_hidden = self.drop(last_hidden)
            feat = self.fusion_projection(last_hidden)
        else:
            # Fallback: no-parameter readout that is shape-safe even when node counts differ.
            # We mimic a simple global (max, mean) pooling over concatenated nodes per group,
            # producing the same 128-D group embeddings as HGRM, but without GraphPool.
            def _simple_readout(x_list):
                nodes = []
                for t in x_list:
                    if t.dim() == 2:
                        t = t.unsqueeze(1)
                    elif t.dim() != 3:
                        raise RuntimeError(f"Expected 2D/3D tensor in readout, got shape={tuple(t.shape)}")
                    nodes.append(t)
                cat = torch.cat(nodes, dim=1)  # (B, sum_N, 64)
                pool_max, _ = torch.max(torch.abs(cat), dim=1)  # (B, 64)
                pool_avg = torch.mean(cat, dim=1)               # (B, 64)
                return torch.cat([pool_max, pool_avg], dim=1)   # (B, 128)

            out1 = _simple_readout(GAT_Group)
            out2 = _simple_readout(HGAT_Group)
            out3 = _simple_readout(SN_Group)
            last_hidden = torch.cat([out1, out2, out3], dim=1)   # (B, 384)
            last_hidden = self.drop(last_hidden)
            feat = self.fusion_projection(last_hidden)
            
        logits = self.classifier(feat)

        if return_feat:
            return logits, feat
        return logits

    def forward(self, inputs, inputs2, Freq_aug, stage="test", batch=None, return_feat: bool = False):
        x=inputs
        x2=inputs2
        batch_size = x.size(0)
        
        """multi-view features"""
        audio_view=self.acoustic_view_encoder(x, Freq_aug=Freq_aug) 
        emo_view=self.prosodic_view_encoder(x2)
        text_view=self.semantic_view_encoder(x)

        # ---- 消融实验：对未激活的模态置零（不影响模型参数，仅屏蔽输入信号）----
        if 'acoustic' not in self.active_views:
            audio_view = torch.zeros_like(audio_view)
        if 'prosodic' not in self.active_views:
            emo_view = torch.zeros_like(emo_view)
        if 'semantic' not in self.active_views:
            text_view = torch.zeros_like(text_view)

        shuffle_info = {}  # 用于存储洗牌信息，便于计算额外损失
        
        if stage == "train" and batch is not None and batch.get('label') is not None:
            labels = batch['label']
            # 1. Style Shuffle - 在同类样本间交换统计特性
            if self.use_lsb and np.random.random() < self.lsb_prob:
                (audio_view, emo_view, text_view), lsb_indices = self.lsb(
                    (audio_view, emo_view, text_view), labels
                )
                shuffle_info["lsb_applied"] = True
                shuffle_info["lsb_indices"] = lsb_indices

        # --- 主分支前向 ---
        output = self._graph_fusion_forward(audio_view, emo_view, text_view, return_feat=return_feat)

        # --- 特征洗牌并行分支（不污染主分支标签） ---
        if stage == "train" and self.use_sfr and np.random.random() < self.sfr_prob:
            acoustic_reorganized, sfr_indices = self.sfr(audio_view)
            sfr_logits = self._graph_fusion_forward(
                acoustic_reorganized, emo_view, text_view, return_feat=False
            )
            shuffle_info["sfr_applied"] = True
            shuffle_info["sfr_logits"] = sfr_logits
            shuffle_info["sfr_indices"] = sfr_indices

        # 返回结果
        if stage == "train" and shuffle_info:
            return output, shuffle_info
        else:
            return output

    def _hetero_forward(self, audio_gat, emo_gat, text_gat):
        """Dispatch heterogeneous graph blocks with optional co-attention."""
        if self.use_mrgf:
            AE_HG, AE_SN, _ = self.mrgf_ae(audio_gat, emo_gat)
            AT_HG, AT_SN, _ = self.mrgf_at(audio_gat, text_gat)
            ET_HG, ET_SN, _ = self.mrgf_et(emo_gat, text_gat)
            AET_HG, AET_SN, _ = self.mrgf_aet(AE_HG, ET_HG)
        else:
            AE_HG = audio_gat
            AT_HG = audio_gat
            ET_HG = emo_gat
            AET_HG = text_gat
            AE_SN = self._zero_stack(audio_gat)
            AT_SN = self._zero_stack(audio_gat)
            ET_SN = self._zero_stack(emo_gat)
            AET_SN = self._zero_stack(text_gat)

        if self.use_bcam and self.bcam_ae_at is not None:
            AE_HG, AT_HG = self.bcam_ae_at(AE_HG, AT_HG)
            ET_HG, AET_HG = self.bcam_et_aet(ET_HG, AET_HG)

        return AE_HG, AT_HG, ET_HG, AET_HG, AE_SN, AT_SN, ET_SN, AET_SN

    @staticmethod
    def _zero_stack(ref):
        batch, _, dim = ref.size()
        return ref.new_zeros(batch, 1, dim)


# Paper-aligned LSB/SFR helpers
def same_class_permutation(label):
    """按标签生成置换索引，确保同类样本间进行混淆"""
    x = label.cpu().numpy()
    index0 = np.where(x == 0)[0]  # 真实样本索引
    index1 = np.where(x == 1)[0]  # 虚假样本索引

    shuffle_index0 = np.random.permutation(index0)
    shuffle_index1 = np.random.permutation(index1)

    new_index = np.ones_like(x, dtype=int)
    for i in range(len(index0)):
        new_index[index0[i]] = shuffle_index0[i]
    for i in range(len(index1)):
        new_index[index1[i]] = shuffle_index1[i]
    
    return torch.from_numpy(new_index).to(label.device)


def blend_latent_statistics(x, y, dim=None, alpha_range=(0.5, 1.0)):
    """
    交换两个特征张量的均值和标准差
    """
    mu_x = torch.mean(x, dim=dim, keepdim=True)
    mu_y = torch.mean(y, dim=dim, keepdim=True)
    std_x = torch.std(x, dim=dim, keepdim=True)
    std_y = torch.std(y, dim=dim, keepdim=True)

    # 随机混合系数
    alpha = np.random.uniform(alpha_range[0], alpha_range[1])
    target_mu = alpha * mu_x + (1 - alpha) * mu_y
    target_std = alpha * std_x + (1 - alpha) * std_y

    # 应用新的统计特性
    z = target_std * ((x - mu_x) / (std_x + 1e-9)) + target_mu
    
    # 添加轻微噪声增强鲁棒性
    noise_level = 0.01
    if np.random.random() < 0.3:  # 30%概率添加噪声
        noise = noise_level * torch.randn_like(z) * torch.std(z, dim=dim, keepdim=True)
        z = z + noise
    
    return z


def reorganize_feature_list(features_list, shuffle_prob=0.5):
    if np.random.random() > shuffle_prob:
        return features_list, None
    
    batch_size = features_list[0].size(0)
    shuffle_indices = torch.randperm(batch_size).to(features_list[0].device)
    
    shuffled_features = []
    for feat in features_list:
        shuffled_features.append(feat[shuffle_indices])
    
    return shuffled_features, shuffle_indices

LEGACY_PREFIX_MAP = (
    ("text_view_extract.", "semantic_view_encoder."),
    ("emo_view_extract.", "prosodic_view_encoder."),
    ("audio_view_extract.", "acoustic_view_encoder."),
    ("GAT_text.", "svge_semantic.gat."),
    ("pool_text.", "svge_semantic.pool."),
    ("GAT_emo.", "svge_prosodic.gat."),
    ("pool_emo.", "svge_prosodic.pool."),
    ("GAT_audio.", "svge_acoustic.gat."),
    ("pool_audio.", "svge_acoustic.pool."),
    ("Core_AE.", "mrgf_ae."),
    ("Core_AT.", "mrgf_at."),
    ("Core_ET.", "mrgf_et."),
    ("Core_AET.", "mrgf_aet."),
    ("coattn_ae_at.", "bcam_ae_at."),
    ("coattn_et_aet.", "bcam_et_aet."),
    ("GRS_group1.", "hgrm_view."),
    ("GRS_group2.", "hgrm_relation."),
    ("GRS_group3.", "hgrm_master."),
    ("out_layer2.", "classifier."),
    ("out_layer.", "fusion_projection."),
)


def translate_legacy_state_dict(state_dict: Mapping[str, Tensor]) -> OrderedDict:
    """Translate historical parameter keys to the MHGC namespace."""
    translated = OrderedDict()
    for raw_key, value in state_dict.items():
        key = raw_key[7:] if raw_key.startswith("module.") else raw_key
        for old_prefix, new_prefix in LEGACY_PREFIX_MAP:
            if key.startswith(old_prefix):
                key = new_prefix + key[len(old_prefix):]
                break
        if key in translated:
            raise KeyError(f"Duplicate key after legacy translation: {key}")
        translated[key] = value
    return translated


def load_mhgc_checkpoint(model: nn.Module, checkpoint, strict: bool = True):
    """Load a native MHGC or a historical legacy checkpoint."""
    if isinstance(checkpoint, (str, os.PathLike)):
        checkpoint = torch.load(checkpoint, map_location="cpu")
    state = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
    return model.load_state_dict(translate_legacy_state_dict(state), strict=strict)


__all__ = [
    "MHGCNet", "SVGE", "MRGF", "BCAM", "HGRM",
    "LatentStyleBlending", "StochasticFeatureReorganization",
    "load_mhgc_checkpoint", "translate_legacy_state_dict",
]
