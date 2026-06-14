#!/usr/bin/env python3
"""E2-6A: Unified corrected v3 diagnostics for all 4 model variants.

Metrics computed with corrected v3 convention:
  gt16 = act_test[:, :16]
  exec_start = 1, exec_end = 9
  fixed seed for static predictions
  same diffusion seed for obs sensitivity
"""
import json, os, sys, numpy as np, torch, torch.nn.functional as F
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "unified-world-model-main"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "diffusion_policy-main"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from stepB_retrain_lowdim import LowdimStatePolicyV2, MinMaxNormalizer
from models.dp.transformer import TransformerNoisePredictionNet
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler as DDPMScheduler

DIAG_SEED = 202506
OBS_HORIZON = 2
PRED_HORIZON = 16
EXEC_START = 1
EXEC_END = 9
SEQ_LEN = 19
OBS_DIM = 20

# ─── data ───
def load_zarr_data_keypoint(zarr_path):
    import zarr
    z = zarr.open(zarr_path, "r")
    kp = z["data/keypoint"][:]; st = z["data/state"][:]; ac = z["data/action"][:]
    ap = st[:, :2]
    obs20 = np.concatenate([kp.reshape(kp.shape[0], -1), ap], axis=-1)
    eps = []; s = 0
    for e in z["meta/episode_ends"][:]:
        eps.append((obs20[s:e], ac[s:e])); s = e
    return eps

def build_dataset(eps, n, sl, oh=2):
    ao, aa = [], []
    for o, a in eps[:n]:
        T = len(o)
        for i in range(T - sl):
            ao.append(o[i:i+oh]); aa.append(a[i:i+sl])
    return torch.utils.data.TensorDataset(
        torch.tensor(np.stack(ao), dtype=torch.float32),
        torch.tensor(np.stack(aa), dtype=torch.float32))

def get_gt16(act_seq):
    return act_seq[:, :PRED_HORIZON]

def normalize_obs(obs, norm):
    return (obs - torch.tensor(norm.offset, device=obs.device).float()) / torch.tensor(norm.scale, device=obs.device).float()

def unnormalize_action(act, norm):
    return act * torch.tensor(norm.scale, device=act.device).float() + torch.tensor(norm.offset, device=act.device).float()

# ─── model loading ───
def load_b3(device):
    m = LowdimStatePolicyV2(obs_dim=20, action_len=16, action_dim=2).to(device)
    ck = torch.load("artifacts_keep/B3_keypoint20_local_dit_20k/latest.pt", map_location=device)
    m.load_state_dict(ck["model"]); m.eval()
    ns = MinMaxNormalizer(); ns.offset = np.array(ck["state_normalizer"]["offset"]); ns.scale = np.array(ck["state_normalizer"]["scale"])
    na = MinMaxNormalizer(); na.offset = np.array(ck["action_normalizer"]["offset"]); na.scale = np.array(ck["action_normalizer"]["scale"])
    return m, ns, na

def _build_uwm_policy(obs_dim, action_dim, action_len, device, embed_dim=768, depth=12, num_heads=12):
    class LOE(torch.nn.Module):
        def __init__(self, od, nf, ed):
            super().__init__()
            self.net = torch.nn.Sequential(torch.nn.Linear(od*nf, ed), torch.nn.Mish(), torch.nn.Linear(ed, ed))
        def forward(self, o): return self.net(o.reshape(o.shape[0], -1))
    class LDP(torch.nn.Module):
        def __init__(self, od, al, ad, ed=768, d=12, nh=12):
            super().__init__()
            self.action_len=al; self.action_dim=ad
            self.obs_encoder=LOE(od, OBS_HORIZON, ed)
            self.noise_pred_net=TransformerNoisePredictionNet(input_len=al, input_dim=ad, global_cond_dim=ed, timestep_embed_dim=256, embed_dim=ed, depth=d, num_heads=nh, mlp_ratio=4, qkv_bias=True)
            self.noise_scheduler=DDPMScheduler(num_train_timesteps=100, beta_schedule="squaredcos_cap_v2", clip_sample=True, prediction_type="epsilon")
            self.num_inference_steps=10
        def forward(self, obs, action):
            oe=self.obs_encoder(obs); n=torch.randn_like(action)
            t=torch.randint(0,self.noise_scheduler.config.num_train_timesteps,(action.shape[0],),device=action.device).long()
            na=self.noise_scheduler.add_noise(action,n,t)
            return F.mse_loss(self.noise_pred_net(na,t,global_cond=oe), n)
        @torch.no_grad()
        def sample(self, obs, seed=None):
            B,d=obs.shape[0],obs.device; oe=self.obs_encoder(obs)
            if seed is not None:
                g=torch.Generator(device=d).manual_seed(seed)
                a=torch.randn(B,self.action_len,self.action_dim,device=d,generator=g)
            else: a=torch.randn(B,self.action_len,self.action_dim,device=d)
            self.noise_scheduler.set_timesteps(self.num_inference_steps)
            for ts in self.noise_scheduler.timesteps:
                t=torch.full((B,),ts,device=d,dtype=torch.long)
                a=self.noise_scheduler.step(self.noise_pred_net(a,t,global_cond=oe),ts,a).prev_sample
            return a
    return LDP(od=obs_dim, al=action_len, ad=action_dim, ed=embed_dim, d=depth, nh=num_heads).to(device)

def load_uwm_base(device):
    m = _build_uwm_policy(20,2,16,device,768,12,12)
    ck = torch.load("artifacts_keep/uwm_dp_only_keypoint20_20k/latest.pt", map_location=device)
    m.load_state_dict(ck["model"]); m.eval()
    ns = MinMaxNormalizer(); ns.offset=np.array(ck["state_normalizer"]["offset"]); ns.scale=np.array(ck["state_normalizer"]["scale"])
    na = MinMaxNormalizer(); na.offset=np.array(ck["action_normalizer"]["offset"]); na.scale=np.array(ck["action_normalizer"]["scale"])
    return m, ns, na

def load_uwm_hybrid(device):
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "hybrid_mod", str(Path(__file__).resolve().parent / "uwm_dp_only_keypoint20_obstoken_hybrid.py"))
    hybrid_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(hybrid_mod)
    m = hybrid_mod.LowdimDiTPolicyHybrid(obs_dim=20, action_len=16, action_dim=2).to(device)
    ck = torch.load("outputs/e2_uwm_kp_obstoken_hybrid/latest.pt", map_location=device)
    m.load_state_dict(ck["model"]); m.eval()
    ns = MinMaxNormalizer(); ns.offset=np.array(ck["state_normalizer"]["offset"]); ns.scale=np.array(ck["state_normalizer"]["scale"])
    na = MinMaxNormalizer(); na.offset=np.array(ck["action_normalizer"]["offset"]); na.scale=np.array(ck["action_normalizer"]["scale"])
    return m, ns, na

def load_uwm_small(device):
    m = _build_uwm_policy(20,2,16,device,256,6,8)
    ck = torch.load("outputs/e2_uwm_kp_small/latest.pt", map_location=device)
    m.load_state_dict(ck["model"]); m.eval()
    ns = MinMaxNormalizer(); ns.offset=np.array(ck["state_normalizer"]["offset"]); ns.scale=np.array(ck["state_normalizer"]["scale"])
    na = MinMaxNormalizer(); na.offset=np.array(ck["action_normalizer"]["offset"]); na.scale=np.array(ck["action_normalizer"]["scale"])
    return m, ns, na

# ─── sample seeded ───
@torch.no_grad()
def sample_seeded(model, obs_norm, seed, model_name):
    B, device = obs_norm.shape[0], obs_norm.device
    if model_name == "uwm_base" or model_name == "uwm_small":
        return model.sample(obs_norm, seed=seed)
    elif model_name == "uwm_hybrid":
        torch.manual_seed(seed)
        out = model.sample(obs_norm)
        torch.manual_seed(torch.randint(0, 2**31, (1,)).item())  # re-randomize
        return out
    elif model_name == "b3":
        obs_feat = model.obs_proj(obs_norm.reshape(B, -1)).unsqueeze(1)
        generator = torch.Generator(device=device).manual_seed(seed)
        action = torch.randn(B, model.action_len, model.action_dim, device=device, generator=generator)
        model.noise_scheduler.set_timesteps(model.num_inference_steps)
        for t_step in model.noise_scheduler.timesteps:
            t = torch.full((B,), t_step, device=device, dtype=torch.long)
            temb = model._time_emb(t, B, device)
            act_emb = model.action_embed(action) + model.pos_embed
            x = torch.cat([obs_feat, act_emb], dim=1) + temb
            x = model.transformer(x)
            noise_pred = model.action_decoder(x[:, 1:])
            action = model.noise_scheduler.step(noise_pred, t_step, action).prev_sample
        return action
    raise ValueError(model_name)

# ─── diagnostics ───
def run_diagnostics(name, model, n_state, n_action, device, dataset):
    N = 128; K = 32
    obs_test, act_test = dataset[:N]
    obs_test = obs_test.to(device); act_test = act_test.to(device)
    gt16 = get_gt16(act_test)
    obs_norm = normalize_obs(obs_test, n_state).float()

    # E2-0: offline MSE
    pred_raw = unnormalize_action(sample_seeded(model, obs_norm, DIAG_SEED, name), n_action)
    all16 = F.mse_loss(pred_raw[:,:16], gt16[:,:16]).item()
    exec8 = F.mse_loss(pred_raw[:,EXEC_START:EXEC_END], gt16[:,EXEC_START:EXEC_END]).item()
    first_l2 = torch.norm(pred_raw[:,EXEC_START] - gt16[:,EXEC_START], dim=-1).mean().item()

    # E2-1: exec8 L2
    exec8_l2 = torch.norm((pred_raw[:,EXEC_START:EXEC_END] - gt16[:,EXEC_START:EXEC_END]).reshape(N,-1), dim=-1).mean().item()

    # E2-2: resampling variance
    all_actions = []
    for k in range(K):
        all_actions.append(unnormalize_action(sample_seeded(model, obs_norm, 10000+k, name), n_action).cpu().numpy())
    all_actions = np.stack(all_actions)
    first_std_l2 = np.mean([np.sqrt(all_actions[:,i,EXEC_START,0].var()+all_actions[:,i,EXEC_START,1].var()) for i in range(N)])
    exec8_std = np.mean([all_actions[:,i,EXEC_START:EXEC_END].std(axis=0).mean() for i in range(N)])

    # E2-2.5: obs sensitivity (same seed)
    act_normal = unnormalize_action(sample_seeded(model, obs_norm, DIAG_SEED, name), n_action)
    shuf_idx = torch.randperm(N)
    act_shuffled = unnormalize_action(sample_seeded(model, obs_norm[shuf_idx], DIAG_SEED, name), n_action)
    act_zero = unnormalize_action(sample_seeded(model, torch.zeros_like(obs_norm), DIAG_SEED, name), n_action)
    shuff_l2 = float(torch.norm((act_normal[:,EXEC_START:EXEC_END]-act_shuffled[:,EXEC_START:EXEC_END]).reshape(N,-1), dim=-1).mean())

    eps = 0.01; Bfd = 32
    torch.manual_seed(42)
    nv = torch.randn_like(obs_norm[:Bfd])
    nv = nv / nv.reshape(Bfd,-1).norm(dim=-1).view(Bfd,1,1)
    obs_pert = obs_norm[:Bfd] + eps * nv
    act_base = unnormalize_action(sample_seeded(model, obs_norm[:Bfd], DIAG_SEED, name), n_action)
    act_pert = unnormalize_action(sample_seeded(model, obs_pert, DIAG_SEED, name), n_action)
    fd_sens = torch.norm(act_pert - act_base, dim=-1).mean().item() / eps

    return {"all16_mse": float(all16), "exec8_mse": float(exec8), "first_l2": float(first_l2),
            "exec8_l2": float(exec8_l2), "first_std_l2": float(first_std_l2), "exec8_std": float(exec8_std),
            "shuff_exec8_l2": float(shuff_l2), "fd_sensitivity": float(fd_sens)}

# ─── main ───
def main():
    device = torch.device("cuda:0")
    os.makedirs("outputs/e2_all_variants", exist_ok=True)

    eps_data = load_zarr_data_keypoint("diffusion_policy-main/data/pusht/pusht_cchi_v7_replay.zarr")
    dataset = build_dataset(eps_data, 90, SEQ_LEN, OBS_HORIZON)

    variants = [
        ("b3", load_b3),
        ("uwm_base", load_uwm_base),
        ("uwm_hybrid", load_uwm_hybrid),
        ("uwm_small", load_uwm_small),
    ]

    all_results = {}
    for name, loader in variants:
        print(f"\n{'='*60}\n  {name}\n{'='*60}")
        model, ns, na = loader(device)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  params: {n_params/1e6:.1f}M")
        r = run_diagnostics(name, model, ns, na, device, dataset)
        r["params"] = n_params
        all_results[name] = r
        print(f"  all16_mse={r['all16_mse']:.1f}  exec8_mse={r['exec8_mse']:.1f}  first_l2={r['first_l2']:.1f}")
        print(f"  exec8_l2={r['exec8_l2']:.1f}  first_std_l2={r['first_std_l2']:.2f}  exec8_std={r['exec8_std']:.2f}")
        print(f"  shuff_exec8={r['shuff_exec8_l2']:.1f}  fd_sens={r['fd_sensitivity']:.1f}")

    # FD ratio vs B3
    b3_fd = all_results["b3"]["fd_sensitivity"]
    for name in all_results:
        all_results[name]["fd_ratio_vs_b3"] = all_results[name]["fd_sensitivity"] / b3_fd

    with open("outputs/e2_all_variants_diagnostics.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # Markdown table
    rows = [
        ("B3 local DiT", "6L/256E/8H", "5.0M", "obs-as-token TransformerEncoder"),
        ("UWM-DP-KP baseline", "12L/768E/12H", "149.8M", "AdaLN"),
        ("UWM-DP-KP obstoken hybrid", "12L/768E/12H", "149.8M", "AdaLN + obs-token"),
        ("UWM-DP-KP small", "6L/256E/8H", "10.9M", "AdaLN"),
    ]
    keys = ["b3", "uwm_base", "uwm_hybrid", "uwm_small"]

    md = "# E2-6A: Unified Diagnostics (corrected v3 convention)\n\n"
    md += f"GT convention: gt16 = act[:, :16]. exec_start={EXEC_START}, exec_end={EXEC_END}. Seed={DIAG_SEED}.\n\n"
    md += "| Model | Backbone | Params | Conditioning | all16 MSE | exec8 MSE | first L2 | exec8 L2 | std L2 | std exec8 | shuff L2 | FD sens | FD/B3 |\n"
    md += "|-------|----------|--------|-------------|----------:|----------:|---------:|---------:|-------:|---------:|---------:|--------|------:|\n"

    for i, key in enumerate(keys):
        r = all_results[key]
        md += f"| {rows[i][0]} | {rows[i][1]} | {rows[i][2]} | {rows[i][3]} | "
        md += f"{r['all16_mse']:.1f} | {r['exec8_mse']:.1f} | {r['first_l2']:.1f} | {r['exec8_l2']:.1f} | "
        md += f"{r['first_std_l2']:.2f} | {r['exec8_std']:.2f} | {r['shuff_exec8_l2']:.1f} | {r['fd_sensitivity']:.1f} | {r['fd_ratio_vs_b3']:.2f} |\n"

    with open("outputs/e2_all_variants_diagnostics.md", "w") as f:
        f.write(md)

    print("\n" + "="*60)
    print("E2-6A complete. Saved to outputs/e2_all_variants_diagnostics.*")
    print(md)


if __name__ == "__main__":
    main()
