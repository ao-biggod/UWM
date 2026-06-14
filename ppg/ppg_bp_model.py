import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ==========================================
# 1. LTAM: 可学习任务感知掩码 (空间净化器)
# ==========================================
class LTAM(nn.Module):
    def __init__(self, in_channels=3):
        super(LTAM, self).__init__()
        self.mask_generator = nn.Sequential(
            nn.Conv1d(in_channels, 16, kernel_size=7, padding=3),
            nn.BatchNorm1d(16),
            nn.GELU(),
            nn.Conv1d(16, 1, kernel_size=3, padding=1),
            nn.Sigmoid() # 输出 0-1 的软掩码
        )

    def forward(self, x):
        M = self.mask_generator(x)
        view_a = x * M          # 核心生理特征
        view_b = x * (1 - M)    # 背景噪音杂质
        return view_a, view_b, M


# ==========================================
# 2. Inception1D: 多尺度局部特征提取
# ==========================================
class ConvBlock1D(nn.Module):
    """封装标准的 Conv1d + BatchNorm1d + GELU，防止梯度消失"""
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding),
            nn.BatchNorm1d(out_channels),
            nn.GELU()
        )
    def forward(self, x):
        return self.block(x)

class Inception1D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        dim = out_channels // 4 
        
        self.branch1 = ConvBlock1D(in_channels, dim, kernel_size=3, padding=1)
        self.branch2 = ConvBlock1D(in_channels, dim, kernel_size=5, padding=2)
        self.branch3 = ConvBlock1D(in_channels, dim, kernel_size=7, padding=3)
        self.branch4 = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            ConvBlock1D(in_channels, dim, kernel_size=1, padding=0)
        )

    def forward(self, x):
        return torch.cat([self.branch1(x), self.branch2(x), self.branch3(x), self.branch4(x)], dim=1) 


# ==========================================
# 3. Transformer: 全局时序建模与共享主干
# ==========================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=2000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]

class SharedBackbone(nn.Module):
    def __init__(self, in_channels=3, d_model=256, n_heads=8, num_layers=4):
        super(SharedBackbone, self).__init__()
        self.inception = Inception1D(in_channels, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model*2, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.pool_proj = nn.Linear(d_model, d_model) # 投影层，学习最优压缩

    def forward(self, x, src_key_padding_mask=None):
        # x: (B, C, L)
        x = self.inception(x)
        x = x.permute(0, 2, 1) # -> (B, L, C) 适应 Transformer
        x = self.pos_encoder(x)
        
        # 序列级特征 (供 MAE 重构)
        seq_out = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask) 
        
        # 全局池化特征 (供 血压回归 / 对比学习)
        pool_in = seq_out.permute(0, 2, 1) # -> (B, C, L)
        pooled_out = self.global_pool(pool_in).squeeze(-1) # -> (B, C)
        pooled_out = self.pool_proj(pooled_out)
        
        return seq_out, pooled_out


# ==========================================
# 4. 终极架构组装: PPG-BP Network
# ==========================================
class PPGBP_Network(nn.Module):
    def __init__(self, seq_len=875, in_channels=3, d_model=256):
        super(PPGBP_Network, self).__init__()
        self.seq_len = seq_len
        self.in_channels = in_channels
        
        self.ltam = LTAM(in_channels)
        self.backbone = SharedBackbone(in_channels, d_model)
        
        # --- 左脑：MAE Decoder ---
        # 核心：定义可学习的 [MASK] Token
        self.mask_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.mask_token, std=0.02)
        
        self.mae_decoder = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, in_channels)
        )
        
        # --- 右脑：决策头 ---
        self.regression_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Linear(64, 2) # 输出 SBP 和 DBP
        )
        
        self.projection_head = nn.Sequential(
            nn.LayerNorm(d_model), # 稳定连续态 SupCon
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 128) 
        )

    def forward(self, x, phase='finetune', src_key_padding_mask=None):
        view_a, view_b, mask_m = self.ltam(x)
        
        if phase == 'pretrain':
            # Phase 1: 获取两条通路的特征
            seq_a, pooled_a = self.backbone(view_a, src_key_padding_mask)
            _, pooled_b = self.backbone(view_b, src_key_padding_mask) 
            
            # MAE 完形填空：用 mask_token 填补被遮挡的特征
            m_prob = mask_m.permute(0, 2, 1) # 维度 (B, L, 1)
            decoder_input = seq_a * m_prob + self.mask_token * (1 - m_prob)
            
            reconstructed = self.mae_decoder(decoder_input) # (B, L, 3)
            reconstructed = reconstructed.permute(0, 2, 1)   # (B, 3, L) 匹配原始输入
            
            return reconstructed, pooled_a, pooled_b, mask_m
            
        elif phase == 'finetune':
            # Phase 2-4: 卸磨杀驴，完全丢弃 MAE，全速回归
            _, pooled_a = self.backbone(view_a, src_key_padding_mask)
            
            bp_preds = self.regression_head(pooled_a)
            p_features = self.projection_head(pooled_a)
            
            # 使用 F.normalize 确保在超球面上，供后续连续态 SupCon 使用
            p_features = F.normalize(p_features, dim=1)
            
            return bp_preds, p_features


# ==========================================
# 5. 训练实战演示 (直接运行查看结果)
# ==========================================
if __name__ == "__main__":
    # 配置
    BATCH_SIZE = 16
    CHANNELS = 3    # PPG, VPG, APG
    SEQ_LEN = 875   # 波形长度
    
    # 初始化模型与假数据
    print("🚀 正在初始化 PPG-BP 万总终极版架构...\n")
    model = PPGBP_Network(seq_len=SEQ_LEN, in_channels=CHANNELS)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    # 模拟真实数据 (包含血压标签)
    raw_ppg = torch.randn(BATCH_SIZE, CHANNELS, SEQ_LEN)
    true_bp = torch.tensor([[120.0, 80.0]] * BATCH_SIZE) # 模拟真实血压
    
    # ---------------------------------------
    # 【实战一：第一级火箭 (预训练 Phase 1)】
    # ---------------------------------------
    print(">>> 启动阶段 1：无监督物理预训练 (MAE + LTAM净化)")
    model.train()
    optimizer.zero_grad()
    
    recon_wave, lat_a, lat_b, mask = model(raw_ppg, phase='pretrain')
    
    # 1. 重构损失
    loss_recon = F.mse_loss(recon_wave, raw_ppg)
    # 2. LTAM 空间对比损失 (推开 View A 和 View B)
    target_push = torch.full((BATCH_SIZE,), -1).to(raw_ppg.device)
    loss_cont = F.cosine_embedding_loss(lat_a, lat_b, target_push, margin=0.5)
    # 3. 掩码稀疏正则化 (逼迫掩码极化)
    loss_l1 = torch.mean(torch.abs(mask))
    
    total_pretrain_loss = loss_recon + 0.1 * loss_cont + 1e-4 * loss_l1
    total_pretrain_loss.backward()
    optimizer.step()
    
    print(f"✅ Phase 1 跑通！")
    print(f"   [输出维度] 重构波形: {recon_wave.shape}, 潜特征: {lat_a.shape}")
    print(f"   [Loss追踪] 重构: {loss_recon.item():.4f} | 对比: {loss_cont.item():.4f} | 掩码稀疏度: {loss_l1.item():.4f}\n")


    # ---------------------------------------
    # 【实战二：第三级火箭 (微调 Phase 3)】
    # ---------------------------------------
    print(">>> 启动阶段 3：全数据有监督微调 (卸磨杀驴)")
    model.train()
    optimizer.zero_grad()
    
    preds, supcon_feats = model(raw_ppg, phase='finetune')
    
    # 1. 血压精准回归损失
    loss_huber = F.huber_loss(preds, true_bp, delta=1.0)
    # (注：连续态 SupCon 和 Delta BP Loss 需要自定义复杂逻辑，这里暂以 Huber 代表)
    
    loss_huber.backward()
    optimizer.step()
    
    print(f"✅ Phase 3 跑通！(MAE Decoder 已被彻底屏蔽)")
    print(f"   [输出维度] 血压预测: {preds.shape}, 聚类特征: {supcon_feats.shape}")
    print(f"   [Loss追踪] Huber 回归误差: {loss_huber.item():.4f}\n")
    
    print("🎉 万总，全链路打通！模型可以上 GPU 正式开始烧显卡了！")