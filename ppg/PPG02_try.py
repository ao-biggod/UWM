import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
import glob
import argparse
import random
import pandas as pd
from tfrecord.torch.dataset import TFRecordDataset
from torch.utils.data import ChainDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# ==========================================
# 0. 高级自定义损失函数模块
# ==========================================
def ppg_sparsity_penalty(mask_m):
    """ 🌟 ADIOS 正弦惩罚项：防止掩码摆烂全 0 或全 1 """
    mean_activation = mask_m.mean(dim=[-2, -1]) 
    sin_val = torch.sin(torch.pi * mean_activation)
    return (1.0 / (sin_val + 1e-6)).mean()

class ContinuousSupConLoss(nn.Module):
    """ 🌟 连续态监督对比学习 Loss：重塑血压特征流形 """
    def __init__(self, temperature=0.1, sigma=10.0):
        super().__init__()
        self.temperature = temperature
        self.sigma = sigma 

    def forward(self, features, labels):
        device = features.device
        batch_size = features.shape[0]
        similarity_matrix = torch.matmul(features, features.T) / self.temperature
        sbp = labels[:, 0]
        sbp_diff = torch.abs(sbp.unsqueeze(1) - sbp.unsqueeze(0))
        
        continuous_weights = torch.exp(-(sbp_diff ** 2) / (2 * self.sigma ** 2))
        mask = torch.eye(batch_size, dtype=torch.bool).to(device)
        continuous_weights = continuous_weights.masked_fill(mask, 0.0)
        
        sim_max, _ = torch.max(similarity_matrix, dim=1, keepdim=True)
        logits = similarity_matrix - sim_max.detach()
        exp_logits = torch.exp(logits).masked_fill(mask, 0.0)
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-8)
        
        mean_log_prob_pos = (continuous_weights * log_prob).sum(1) / (continuous_weights.sum(1) + 1e-8)
        return -mean_log_prob_pos.mean()

# ==========================================
# 1. 核心模型架构 (Phase 1 与 Phase 3 闭环路由)
# ==========================================
class LTAM(nn.Module):
    def __init__(self, in_channels=3):
        super(LTAM, self).__init__()
        self.mask_generator = nn.Sequential(
            nn.Conv1d(in_channels, 16, kernel_size=7, padding=3),
            nn.BatchNorm1d(16),
            nn.GELU(),
            nn.Conv1d(16, 1, kernel_size=3, padding=1),
            nn.Sigmoid() 
        )
    def forward(self, x):
        M = self.mask_generator(x)
        return x * M, x * (1 - M), M

class ConvBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding),
            nn.BatchNorm1d(out_channels),
            nn.GELU()
        )
    def forward(self, x): return self.block(x)

class Inception1D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        dim = out_channels // 4 
        self.branch1 = ConvBlock1D(in_channels, dim, kernel_size=5, padding=2)
        self.branch2 = ConvBlock1D(in_channels, dim, kernel_size=15, padding=7)
        self.branch3 = ConvBlock1D(in_channels, dim, kernel_size=31, padding=15)
        self.branch4 = nn.Sequential(nn.MaxPool1d(3, 1, 1), ConvBlock1D(in_channels, dim, 1, 0))
    def forward(self, x):
        return torch.cat([self.branch1(x), self.branch2(x), self.branch3(x), self.branch4(x)], dim=1) 

class PPGBP_Network(nn.Module):
    def __init__(self, seq_len=875, in_channels=3, d_model=256):
        super().__init__()
        self.ltam = LTAM(in_channels)
        self.inception = Inception1D(in_channels, d_model)
        self.transformer = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model, 8, d_model*2, batch_first=True), 4)
        
        self.mask_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.mask_token, std=0.02)
        self.mae_decoder = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.GELU(), nn.Linear(d_model // 2, in_channels)
        )
        
        self.regression_head = nn.Sequential(nn.Linear(d_model, 64), nn.GELU(), nn.Linear(64, 2))
        self.projection_head = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, 128))

    def forward(self, x, phase='finetune'):
        if phase == 'pretrain':
            view_a, view_b, mask_m = self.ltam(x)
            feat_a = self.transformer(self.inception(view_a).permute(0,2,1)) 
            feat_b = self.transformer(self.inception(view_b).permute(0,2,1))
            
            m_prob = mask_m.permute(0, 2, 1) 
            decoder_input = feat_a * m_prob + self.mask_token * (1 - m_prob)
            reconstructed = self.mae_decoder(decoder_input).permute(0, 2, 1) 
            
            return reconstructed, feat_a.mean(1), feat_b.mean(1), mask_m
        
        else: # Phase 3: 闭环不对称路由
            view_a, view_b, mask_m = self.ltam(x)
            
            # View A (生理信号) 被送入回归头预测血压
            feat_a = self.transformer(self.inception(view_a).permute(0,2,1)).mean(1)
            preds = self.regression_head(feat_a)
            proj_feats = F.normalize(self.projection_head(feat_a))
            
            # View B (反面教材) 仅用于隐空间推开，绝对不参与预测
            feat_b = self.transformer(self.inception(view_b).permute(0,2,1)).mean(1)
            
            return preds, proj_feats, feat_a, feat_b, mask_m

# ==========================================
# 2. SQA 过滤引擎与数据加载 (防断流装甲版)
# ==========================================
BAD_SUBJECTS_SET = set()

def load_sqa_reject_list(csv_path, threshold=50):
    global BAD_SUBJECTS_SET
    if not os.path.exists(csv_path): return
    df = pd.read_csv(csv_path)
    bad_counts = df['subject_id'].value_counts()
    for subj_id, count in bad_counts.items():
        if count >= threshold: BAD_SUBJECTS_SET.add(int(float(subj_id)))
    print(f"✅ SQA 就绪，拉黑 {len(BAD_SUBJECTS_SET)} 名劣质受试者。")

def decode_and_process(element):
    global BAD_SUBJECTS_SET
    subj_id = int(float(element['subject_idx'][0]))
    
    if subj_id in BAD_SUBJECTS_SET: return None
    
    ppg = torch.tensor(element['ppg'], dtype=torch.float32).view(-1)
    
    diff_vpg = torch.diff(ppg, dim=-1)
    vpg = torch.cat([diff_vpg[0:1], diff_vpg], dim=-1)
    diff_apg = torch.diff(vpg, dim=-1)
    apg = torch.cat([diff_apg[0:1], diff_apg], dim=-1)
    
    return torch.stack([ppg, vpg, apg], dim=0), torch.tensor(element['label'], dtype=torch.float32)

class SafeTFRecordDataset(TFRecordDataset):
    def __iter__(self):
        try:
            for record in super().__iter__(): yield record
        except RuntimeError: pass

def filter_collate_fn(batch):
    batch = [item for item in batch if item is not None]
    return torch.utils.data.dataloader.default_collate(batch) if len(batch) > 0 else None

def get_dataloader(data_dir, batch_size, limit_files=0):
    files = sorted(glob.glob(os.path.join(data_dir, "*.tfrecord")))
    random.shuffle(files)
    if limit_files > 0:
        files = files[:limit_files]
        print(f"🔬 调试模式：限制读取前 {limit_files} 个 TFRecord 文件分片。")
    
    ds_list = [SafeTFRecordDataset(f, None, {"ppg": "float", "label": "float", "subject_idx": "float"}, transform=decode_and_process) for f in files]
    return DataLoader(ChainDataset(ds_list), batch_size=batch_size, collate_fn=filter_collate_fn, num_workers=4)

# ==========================================
# 3. 炼丹主程序 
# ==========================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', type=int, choices=[1, 3], required=True)
    parser.add_argument('--mode', type=str, choices=['train', 'val', 'test'], default='train')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--start_epoch', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--resume', type=str, default='')
    parser.add_argument('--reject_thresh', type=int, default=50)
    parser.add_argument('--limit_files', type=int, default=0)
    # 🌟 新增：Phase 3 闭环博弈的三方超参数，可通过命令行直接调整
    parser.add_argument('--lambda_push', type=float, default=0.1, help="闭环微调时，推开噪声的惩罚权重")
    parser.add_argument('--lambda_adios', type=float, default=0.1, help="闭环微调时，保持掩码稀疏的惩罚权重")
    args = parser.parse_args()
    
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🔥 启动万总 MAE+对比学习+闭环微调 黄金炼丹炉！设备: {DEVICE} | Phase: {args.phase}")
    
    load_sqa_reject_list("/root/autodl-tmp/SQA/结果报告及分析/three_layer_problem_windows_new.csv", threshold=args.reject_thresh)
    
    model = PPGBP_Network().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion_supcon = ContinuousSupConLoss().to(DEVICE)
    
    if args.resume and os.path.exists(args.resume):
        model.load_state_dict(torch.load(args.resume, map_location=DEVICE))
        print(f"✅ 成功加载盆地断点权重: {args.resume}")

    os.makedirs("/root/autodl-tmp/checkpoints", exist_ok=True)
    writer = SummaryWriter(log_dir="/root/autodl-tmp/logs")
    loader = get_dataloader(f"/root/autodl-tmp/data/{args.mode}", args.batch_size, limit_files=args.limit_files)
    
    for epoch in range(args.start_epoch, args.start_epoch + args.epochs):
        model.train()
        total_loss, step_count = 0, 0
        total_sbp_mae, total_dbp_mae = 0.0, 0.0
        pbar = tqdm(loader, desc=f"Phase {args.phase} Epoch {epoch}")
        
        for batch in pbar:
            if batch is None: continue  
            raw_ppg, true_bp = batch
            raw_ppg, true_bp = raw_ppg.to(DEVICE), true_bp.to(DEVICE)
            
            optimizer.zero_grad()
            
            if args.phase == 1:
                # ================= Phase 1: MAE 重构打底座 =================
                recon_wave, lat_a, lat_b, mask_m = model(raw_ppg, phase='pretrain')
                
                loss_recon = F.mse_loss(recon_wave, raw_ppg) 
                # 纯血对抗分离：target=-1, margin=0.0
                loss_push = F.cosine_embedding_loss(lat_a, lat_b, torch.full((raw_ppg.size(0),), -1).to(DEVICE), margin=0.0)
                loss_adios = ppg_sparsity_penalty(mask_m)
                
                loss = loss_recon + 0.1 * loss_push + 0.1 * loss_adios
                loss.backward()
                
                # 物理防线：阻断尖峰梯度
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                total_loss += loss.item(); step_count += 1
                pbar.set_postfix({'Loss': f"{loss.item():.4f}", 'MSE': f"{loss_recon.item():.4f}"})
                
            else:
                # ================= Phase 3: 误差倒逼掩码闭环进化 =================
                preds, feats, lat_a, lat_b, mask_m = model(raw_ppg, phase='finetune')
                
                # 主线任务：预测误差 (只针对 View A)
                loss_bp = F.huber_loss(preds, true_bp)
                loss_supcon = criterion_supcon(feats, true_bp)
                
                # 护航任务：维持掩码的物理意义，防止全透摆烂
                loss_push = F.cosine_embedding_loss(lat_a, lat_b, torch.full((raw_ppg.size(0),), -1).to(DEVICE), margin=0.0)
                loss_adios = ppg_sparsity_penalty(mask_m)
                
                # 动态博弈：由超参数控制主辅关系
                loss = loss_bp + 0.5 * loss_supcon + args.lambda_push * loss_push + args.lambda_adios * loss_adios
                loss.backward()
                
                # 物理防线：保护已收敛的 Phase 1 权重盆地
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                total_loss += loss.item(); step_count += 1
                
                # 实时透视血压预测绝对误差 (MAE)
                with torch.no_grad():
                    mae_batch = torch.abs(preds - true_bp).mean(dim=0)
                    total_sbp_mae += mae_batch[0].item()
                    total_dbp_mae += mae_batch[1].item()
                
                pbar.set_postfix({
                    'Loss': f"{loss.item():.2f}",
                    'S_MAE': f"{total_sbp_mae/step_count:.1f}",
                    'D_MAE': f"{total_dbp_mae/step_count:.1f}"
                })
            
        avg_loss = total_loss / max(1, step_count)
        writer.add_scalar(f'Phase{args.phase}/Total_Loss', avg_loss, epoch)
        if args.phase == 3:
            writer.add_scalar('Phase3/SBP_MAE', total_sbp_mae/max(1, step_count), epoch)
            writer.add_scalar('Phase3/DBP_MAE', total_dbp_mae/max(1, step_count), epoch)
            
        torch.save(model.state_dict(), f"/root/autodl-tmp/checkpoints/phase{args.phase}_epoch_{epoch}.pth")

if __name__ == "__main__":
    main()