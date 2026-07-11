import os
from time import sleep, time as tt
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"]="expandable_segments:True"
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import ToPILImage
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import numpy as np
import clip
import torch.fft
import math
import random
from typing import Optional, Tuple, List, Union
from diffusers import StableDiffusionPipeline
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img import (
    retrieve_latents, 
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
Tensor = torch.Tensor

def create_vae():
    pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5",
                                                             torch_dtype=torch.float16, safety_checker=None)
    vae = pipe.vae
    del pipe
    torch.cuda.empty_cache()

    return vae

class LayerScale(nn.Module):
    def __init__(self, dim, init_value=1e-4):
        super().__init__()
        self.gamma = nn.Parameter(init_value * torch.ones(dim))
        
    def forward(self, x):
        return self.gamma * x

def dct_2d(x):
    return dct(dct(x, norm='ortho', dim=-1), norm='ortho', dim=-2)

def dct(x: Tensor, norm: Optional[str] = None, dim: int = -1) -> Tensor:
    N = x.shape[dim]
    v = torch.cat([x, x.flip([dim])], dim=dim)
    Vc = torch.fft.fft(v, dim=dim)
    k = torch.arange(N, device=x.device)
    factor = torch.exp(-1j * math.pi * k / (2 * N))
    V = Vc.index_select(dim, k) * factor

    if norm == 'ortho':
        V[..., 0] /= math.sqrt(N) * 2
        V[..., 1:] /= math.sqrt(N / 2) * 2
    else:
        V /= 2
    return V.real

class DCTExtractor(nn.Module):
    def __init__(self, cutoff_freq=8):
        super().__init__()
        self.cutoff_freq = cutoff_freq

    def forward(self, x):
        dct_val = dct_2d(x)

        B, C, H, W = x.shape
        mask = torch.ones_like(dct_val)
        mask[:, :, :self.cutoff_freq, :self.cutoff_freq] = 0

        high_freq_dct = dct_val * mask
        return high_freq_dct

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class WindowAttention(nn.Module):
    def __init__(self, dim, num_heads, window_size, qkv_bias=True):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        
    def forward(self, x):
        B, N, C = x.shape
        
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        q, k, v = qkv.unbind(2) 
        
        q = q.transpose(1, 2) 
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x

class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(
            dim=dim,
            num_heads=num_heads,
            window_size=7 
        )
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = FeedForward(dim, mlp_dim, dropout)
        self.layer_scale = LayerScale(dim)
        
    def forward(self, x):
        x = x + self.layer_scale(self.attn(self.norm1(x)))
        x = x + self.layer_scale(self.mlp(self.norm2(x)))
        return x

class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)

        self.out_proj = nn.Linear(dim, dim)

    def forward(self, query, context):
        if context.dim() == 2:
            context = context.unsqueeze(1) 
        if query.dim() == 2:
            query = query.unsqueeze(1) 

        B, Nq, D = query.size()
        Nc = context.size(1)

        q = self.query(query).view(B, Nq, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key(context).view(B, Nc, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value(context).view(B, Nc, self.num_heads, self.head_dim).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)

        out = (attn @ v).transpose(1, 2).contiguous().view(B, Nq, D)
        out = self.out_proj(out)

        return out.squeeze(1)

class TensorAsymmetricResizeAug(nn.Module):
    def __init__(self, target_size=224, bottleneck_range=(112, 200), apply_prob=0.2):
        super().__init__()
        self.target_size = (target_size, target_size)
        self.bottleneck_range = bottleneck_range
        self.apply_prob = apply_prob
        self.interpolation = 'bicubic'

    def forward(self, x):
        if self.training and random.random() < self.apply_prob:
            bottleneck_size = random.randint(self.bottleneck_range[0], self.bottleneck_range[1])
            x = F.interpolate(x, size=(bottleneck_size, bottleneck_size), mode=self.interpolation, align_corners=False)
            x = F.interpolate(x, size=self.target_size, mode=self.interpolation, align_corners=False)
        return x

class MVA_VIT(nn.Module):
    def __init__(self, clip_model_name="ViT-L/14", num_classes=1, 
                 patch_sizes=[16, 32], dim=768, depth=8, heads=12):
        super().__init__()
        print("Initializing MVA_VIT model...")
        self.dct_extract = DCTExtractor()
        
        clip_model, _ = clip.load(clip_model_name, device="cuda")
        self.visual = clip_model.visual
        self.text_encoder = clip_model.encode_text

        self.vae = create_vae()
        self.vae.eval()
        self.decode_dtype = next(iter(self.vae.post_quant_conv.parameters())).dtype
        for param in self.vae.parameters():
            param.requires_grad = False
        self.vae = self.vae.to(device)
   
        for param in self.visual.parameters():
            param.requires_grad = False

        for block in self.visual.transformer.resblocks[-4:]:
            block.float() 
            for param in block.parameters():
                param.requires_grad = True
            
        self.npr_embeddings = nn.ModuleList([
            nn.Sequential(
                Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', 
                         p1=p_size, p2=p_size),
                nn.Linear(3 * p_size * p_size, dim),
                nn.LayerNorm(dim)
            ) for p_size in patch_sizes
        ])

        self.error_map_embeddings = nn.ModuleList([
            nn.Sequential(
                Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', 
                         p1=p_size, p2=p_size),
                nn.Linear(3 * p_size * p_size, dim),
                nn.LayerNorm(dim)
            ) for p_size in patch_sizes
        ])
        
        self.num_patches = [(256 // p_size) ** 2 for p_size in patch_sizes]
        total_patches = sum(self.num_patches)
        
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, total_patches + 1, dim))
        self.dropout = nn.Dropout(0.1)
        
        clip_dim = self.visual.transformer.width 
        self.clip_proj = nn.Linear(clip_dim, dim)
        self.clip_text_proj = nn.Linear(clip_dim, dim)
        
        self.fusion_blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, num_heads=heads, dim_head=dim//heads, 
                                mlp_dim=dim*4, dropout=0.1)
            for _ in range(depth//2)
        ])
        
        self.cross_attention = CrossAttention(dim=dim, num_heads=heads)

        self.adaptive_fusion = nn.Sequential(
            nn.Linear(dim*2, dim),
            nn.Sigmoid()
        )
        
        self.norm = nn.LayerNorm(dim)
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim//2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(dim//2, num_classes)
        )

        self.inject_mlp = nn.Sequential(
            nn.Linear(5, dim)
        )
        
        self.projection_head = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim)
        )
        
        self.layer_attention = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, 1),
            nn.Softmax(dim=1)
        )

        self.dct_aug = TensorAsymmetricResizeAug(
            apply_prob=0.1 
        )
        
        self.clip_aug = TensorAsymmetricResizeAug(
            apply_prob=0.8 
        )

        print("MVA_VIT model initialized.")

    def extract_clip_features(self, x):
        if x.shape[-1] != 224:
            x_clip = F.interpolate(x, size=(224, 224), mode='bicubic', align_corners=False)
        else:
            x_clip = x
        
        target_dtype = self.visual.conv1.weight.dtype
        x = self.visual.conv1(x_clip.to(target_dtype)) 
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)
        cls_token = self.visual.class_embedding.to(x.dtype).reshape(1, 1, -1).expand(x.shape[0], -1, -1)
        x = torch.cat([cls_token, x], dim=1)
        x = x + self.visual.positional_embedding.to(x.dtype)
        x = self.visual.ln_pre(x)
        
        x = x.permute(1, 0, 2) 
        projected_features = []
        projection_dtype = self.clip_proj.weight.dtype 

        for i, block in enumerate(self.visual.transformer.resblocks):
            x = block(x.to(projection_dtype))
            
            if i >= len(self.visual.transformer.resblocks) - 4: 
                feat_nld = x.permute(1, 0, 2)
                feat_to_proj = feat_nld.to(projection_dtype)
                projected_features.append(self.clip_proj(feat_to_proj))

        return projected_features

    def extract_text_features(self, prompt_text: Union[str, List[str]]) -> Tensor:
        text_inputs = clip.tokenize(prompt_text, truncate=True).to("cuda")
        with torch.no_grad():
            text_features = self.text_encoder(text_inputs)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features

    def _get_patch_embeddings(self, x: Tensor, embedding_layer: nn.Module) -> Tensor:
        embeddings = []
        for embed in embedding_layer:
            patch_embed = embed(x)
            embeddings.append(patch_embed)
        return torch.cat(embeddings, dim=1)
    
    def process_dct_features(self, x: Tensor, error_map: Tensor) -> Tensor:
        dct_feat = self.dct_extract(x)
        dct_tokens = self._get_patch_embeddings(dct_feat, self.npr_embeddings)
        error_tokens = self._get_patch_embeddings(error_map, self.error_map_embeddings)
        x = dct_tokens + error_tokens
        
        b, n, _ = x.shape
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)
        
        return x
    
    def adaptive_feature_fusion(self, npr_feat: Tensor, clip_features: List[Tensor]) -> Tensor:
        selected_features = clip_features[1:] if len(clip_features) > 1 else clip_features
        
        fused_features = []
        for i, clip_feat in enumerate(selected_features):
            if clip_feat.size(1) != npr_feat.size(1):
                b, clip_seq_len, c = clip_feat.shape
                b, npr_seq_len, c = npr_feat.shape
                
                if clip_seq_len < npr_seq_len:
                    clip_feat_reshaped = clip_feat.transpose(1, 2)
                    clip_feat_resized = F.interpolate(
                        clip_feat_reshaped, 
                        size=npr_seq_len,
                        mode='linear',
                        align_corners=False
                    )
                    clip_feat = clip_feat_resized.transpose(1, 2)
                else:
                    clip_feat = clip_feat[:, :npr_seq_len, :]
            
            concat_feat = torch.cat([npr_feat, clip_feat], dim=-1)
            base_weight = self.adaptive_fusion(concat_feat)
            
            if i == 0: 
                fusion_weight = torch.clamp(base_weight + 0.3, 0.0, 1.0)
            elif i == len(selected_features) - 1: 
                fusion_weight = torch.clamp(base_weight - 0.2, 0.0, 1.0)
            else:
                fusion_weight = base_weight
            
            fused = npr_feat * fusion_weight  + clip_feat *  (1 - fusion_weight)
            fused_features.append(fused)
        
        if len(fused_features) > 1:
            final_fused = torch.mean(torch.stack(fused_features, dim=0), dim=0)
        else:
            final_fused = fused_features[0]
        
        return final_fused
    
    def shuffle_patches(self, x: Tensor, patch_size):
        if not self.training:
            return x
        B, C, H, W = x.size()
        patches = F.unfold(x, kernel_size=patch_size, stride=patch_size, dilation=1)
        shuffled_patches = patches[:, :, torch.randperm(patches.size(-1))]
        shuffled_images = F.fold(shuffled_patches, output_size=(H, W), kernel_size=patch_size, stride=patch_size)
        return shuffled_images

    def forward(self, x: Tensor, prompt_text: Optional[Union[str, List[str]]] = None) -> Union[Tuple[Tensor, Tensor, Tensor], Tensor]:
        x_dct =  self.dct_aug(x)
        x_clip = self.clip_aug(x)
        x_shuf = self.shuffle_patches(x, patch_size=32)

        with torch.no_grad():
            latents_x = retrieve_latents(self.vae.encode(x.to(self.decode_dtype)))
            latents_x = latents_x * self.vae.config.scaling_factor
            reconstructions_x = self.vae.decode(latents_x.to(self.decode_dtype),return_dict=False)[0]
            reconstructions_x = (reconstructions_x / 2.0 + 0.5).clamp(0, 1)
        error_map = torch.abs(reconstructions_x - x.to(self.decode_dtype))
        npr_features = self.process_dct_features(x_dct, error_map)
        clip_features = self.extract_clip_features(x_clip)
        x1_features = self.extract_clip_features(x_shuf)
        clip_features = [self.cross_attention(clip, x1) for clip,x1 in zip(clip_features, x1_features)]

        fused = npr_features        
        for i, block in enumerate(self.fusion_blocks):
            fused = block(fused)

            if i == len(self.fusion_blocks) // 2:
                fused = self.adaptive_feature_fusion(fused, clip_features)
        
        fused = self.norm(fused)

        cls_feature = fused[:, 0]

        if not self.training:
            return self.mlp_head(cls_feature)

        text_features = self.extract_text_features(prompt_text)

        return self.mlp_head(cls_feature), \
                self.projection_head(cls_feature), \
                text_features


def create_model(num_classes=1, **kwargs):
    return MVA_VIT(
        clip_model_name="ViT-L/14",
        num_classes=num_classes,
        patch_sizes=[16, 32],
        dim=768,
        depth=8,
        heads=12,
        **kwargs
    )
