import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import DistilBertModel, DistilBertConfig, ASTModel, ASTConfig, SwinModel, SwinConfig
import timm

class BottleneckAdapter(nn.Module):
    """Lightweight adapter module for transformer layers."""
    def __init__(self, dim, bottleneck_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, bottleneck_dim),
            nn.ReLU(inplace=True),
            nn.Linear(bottleneck_dim, dim)
        )
        # Initialize as identity
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        return x + self.net(x)

class QstTransformerEncoder(nn.Module):
    def __init__(self, model_name='distilbert-base-uncased', embed_size=512, use_adapters=False):
        super().__init__()
        self.bert = DistilBertModel.from_pretrained(model_name)
        self.use_adapters = use_adapters
        self.embed_size = embed_size
        
        # Hidden dimension for DistilBert is 768
        self.projection = nn.Linear(768, embed_size)
        
        if use_adapters:
            # Add adapters to each transformer block
            for layer in self.bert.transformer.layer:
                layer.adapter = BottleneckAdapter(768)
                layer.forward = self._make_adapter_forward(layer, layer.forward)

    def _make_adapter_forward(self, block, orig_forward):
        def adapter_forward(*args, **kwargs):
            output = orig_forward(*args, **kwargs)
            # output is (hidden_states,)
            if isinstance(output, tuple):
                return (block.adapter(output[0]),) + output[1:]
            return block.adapter(output)
        return adapter_forward

    def forward(self, input_ids, attention_mask=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # pooled_output = outputs.last_hidden_state[:, 0, :] # [CLS] token
        # Or mean pooling
        pooled_output = torch.mean(outputs.last_hidden_state, dim=1)
        return self.projection(pooled_output)

class VisualTransformerEncoder(nn.Module):
    def __init__(self, model_name='swin_tiny_patch4_window7_224', embed_size=512, use_adapters=False):
        super().__init__()
        # Use timm for Swin
        self.swin = timm.create_model(model_name, pretrained=True, num_classes=0)
        self.use_adapters = use_adapters
        self.embed_size = embed_size
        
        # Swin-Tiny feature dim is 768
        self.projection = nn.Linear(768, embed_size)
        
        if use_adapters:
            # Inject adapters into Swin blocks
            for layer in self.swin.layers:
                for block in layer.blocks:
                    block.adapter = BottleneckAdapter(block.norm1.normalized_shape[0])
                    block.forward = self._make_adapter_forward(block, block.forward)

    def _make_adapter_forward(self, block, orig_forward):
        def adapter_forward(*args, **kwargs):
            return block.adapter(orig_forward(*args, **kwargs))
        return adapter_forward

    def forward(self, x):
        # x: (B, T, 3, 224, 224)? Actually Swin usually takes (B, 3, 224, 224)
        # If we have multiple frames, we might need to process them and then pool or keep sequence
        if x.ndim == 5: # (B, T, C, H, W)
            B, T, C, H, W = x.shape
            x = x.view(B * T, C, H, W)
            feats = self.swin.forward_features(x) # (B*T, L, C_feat)
            feats = feats.view(B, T, -1, feats.size(-1)) # (B, T, L, C_feat)
            # Global pool spatial
            pooled_feats = torch.mean(feats, dim=2) # (B, T, C_feat)
        else:
            pooled_feats = self.swin.forward_features(x)
        
        return self.projection(pooled_feats)

class AudioTransformerEncoder(nn.Module):
    def __init__(self, model_name='MIT/ast-finetuned-audioset-10-10-0.4593', embed_size=512, use_adapters=False):
        super().__init__()
        self.ast = ASTModel.from_pretrained(model_name)
        self.use_adapters = use_adapters
        self.embed_size = embed_size
        
        # AST hidden dim is 768
        self.projection = nn.Linear(768, embed_size)
        
        if use_adapters:
            for layer in self.ast.encoder.layer:
                layer.adapter = BottleneckAdapter(768)
                layer.forward = self._make_adapter_forward(layer, layer.forward)

    def _make_adapter_forward(self, block, orig_forward):
        def adapter_forward(*args, **kwargs):
            output = orig_forward(*args, **kwargs)
            if isinstance(output, tuple):
                return (block.adapter(output[0]),) + output[1:]
            return block.adapter(output)
        return adapter_forward

    def forward(self, input_values):
        # input_values: (B, T_s, F_s)
        outputs = self.ast(input_values)
        # pooled_output = outputs.last_hidden_state[:, 0, :] # [CLS] token
        pooled_output = torch.mean(outputs.last_hidden_state, dim=1)
        return self.projection(pooled_output)

class NoveltyAudioVisualNet(nn.Module):
    def __init__(self, args, step_out_ans_num):
        super().__init__()
        self.args = args
        self.embed_size = 512
        
        self.use_adapters = getattr(args, 'use_adapters', False)
        
        self.visual_encoder = VisualTransformerEncoder(use_adapters=self.use_adapters)
        self.audio_encoder = AudioTransformerEncoder(use_adapters=self.use_adapters)
        self.question_encoder = QstTransformerEncoder(use_adapters=self.use_adapters)
        
        # Fusion-related (similar to original but with transformer sequence)
        self.attn_a = nn.MultiheadAttention(512, 4, dropout=0.1)
        self.attn_v = nn.MultiheadAttention(512, 4, dropout=0.1)
        
        self.fc_fusion = nn.Linear(1024, 512)
        self.classifier = nn.Linear(512, step_out_ans_num)
        self.tanh = nn.Tanh()
        
        # Contrastive projection
        self.contrast_proj_a = nn.Linear(512, 128)
        self.contrast_proj_v = nn.Linear(512, 128)
        
        if self.use_adapters:
            self.freeze_backbones()

    def freeze_backbones(self):
        """Freeze everything except adapters and new heads."""
        for name, param in self.named_parameters():
            if 'adapter' not in name and 'projection' not in name and 'fc_fusion' not in name and 'classifier' not in name and 'attn' not in name and 'contrast_proj' not in name:
                param.requires_grad = False

    def forward(self, audio, visual, question, attention_mask=None, out_logits=True, out_features=False, out_contrastive=False):
        # audio: (B, 1024, 128) - Mel spectrogram for AST
        # visual: (B, T, 3, 224, 224)
        # question: (B, L) - input_ids
        
        qst_feat = self.question_encoder(question, attention_mask) # (B, 512)
        xq = qst_feat.unsqueeze(0) # (1, B, 512)
        
        vis_seq = self.visual_encoder(visual) # (B, T, 512)
        aud_seq = self.audio_encoder(audio) # (B, T_a?, 512) - AST might need adjustment for sequence
        
        # For simplicity, if AST returns pooled, we might need its sequence
        # Update AudioTransformerEncoder to return sequence if needed
        # Let's assume aud_seq is pooled for now (B, 512)
        
        # Fusion (following baseline logic)
        vis_att = self.attn_v(xq, vis_seq.transpose(0, 1), vis_seq.transpose(0, 1))[0].squeeze(0)
        # If aud_seq is pooled, we might just use it or treat it as a sequence of 1
        aud_seq_expanded = aud_seq.unsqueeze(1) if aud_seq.ndim == 2 else aud_seq
        aud_att = self.attn_a(xq, aud_seq_expanded.transpose(0, 1), aud_seq_expanded.transpose(0, 1))[0].squeeze(0)
        
        feat = torch.cat((aud_att, vis_att), dim=-1)
        feat = self.tanh(self.fc_fusion(feat))
        
        combined_feature = torch.mul(feat, qst_feat)
        combined_feature = self.tanh(combined_feature)
        
        logits = self.classifier(combined_feature)
        
        outputs = ()
        if out_logits:
            outputs += (logits,)
        if out_features:
            outputs += (combined_feature,)
        if out_contrastive:
            # Projected embeddings for InfoNCE
            proj_a = self.contrast_proj_a(aud_att)
            proj_v = self.contrast_proj_v(vis_att)
            outputs += (proj_a, proj_v,)
            
        if len(outputs) == 1:
            return outputs[0]
        return outputs

    def incremental_classifier(self, num_ans):
        # Same logic as original
        weight = self.classifier.weight.data
        bias = self.classifier.bias.data
        in_features = self.classifier.in_features
        out_features = self.classifier.out_features
        self.classifier = nn.Linear(in_features, num_ans, bias=True)
        self.classifier.weight.data[:out_features] = weight
        self.classifier.bias.data[:out_features] = bias
        nn.init.kaiming_uniform_(self.classifier.weight[out_features:])
        self.classifier.bias.data[out_features:].zero_()

def info_nce_loss(proj_a, proj_v, temperature=0.07):
    """Bimodal InfoNCE loss."""
    # Normalize features
    proj_a = F.normalize(proj_a, dim=-1)
    proj_v = F.normalize(proj_v, dim=-1)
    
    # Cosine similarity matrix
    logits = torch.matmul(proj_a, proj_v.T) / temperature
    
    # Symmetric loss
    labels = torch.arange(proj_a.size(0)).to(proj_a.device)
    loss_a2v = F.cross_entropy(logits, labels)
    loss_v2a = F.cross_entropy(logits.T, labels)
    
    return (loss_a2v + loss_v2a) / 2
