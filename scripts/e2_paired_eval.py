#!/usr/bin/env python3
"""E2-6B: Paired eval on same 50 seeds for all 4 model variants."""
import json, os, sys, time, numpy as np, torch
from pathlib import Path
from collections import deque

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "unified-world-model-main"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "diffusion_policy-main"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from stepB_retrain_lowdim import LowdimStatePolicyV2, MinMaxNormalizer

OBS_HORIZON = 2

def load_b3(device):
    m = LowdimStatePolicyV2(obs_dim=20, action_len=16, action_dim=2).to(device)
    ck = torch.load("artifacts_keep/B3_keypoint20_local_dit_20k/latest.pt", map_location=device)
    m.load_state_dict(ck["model"]); m.eval()
    ns = MinMaxNormalizer(); ns.offset=np.array(ck["state_normalizer"]["offset"]); ns.scale=np.array(ck["state_normalizer"]["scale"])
    na = MinMaxNormalizer(); na.offset=np.array(ck["action_normalizer"]["offset"]); na.scale=np.array(ck["action_normalizer"]["scale"])
    return m, ns, na

def _build_uwm(device, ed=768, d=12, nh=12):
    from models.dp.transformer import TransformerNoisePredictionNet
    from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
    class LOE(torch.nn.Module):
        def __init__(self, od, nf, ed):
            super().__init__(); self.net = torch.nn.Sequential(torch.nn.Linear(od*nf, ed), torch.nn.Mish(), torch.nn.Linear(ed, ed))
        def forward(self, o): return self.net(o.reshape(o.shape[0], -1))
    class LDP(torch.nn.Module):
        def __init__(self, od, al, ad, ed=768, d=12, nh=12):
            super().__init__(); self.action_len=al; self.action_dim=ad
            self.obs_encoder=LOE(od, OBS_HORIZON, ed)
            self.noise_pred_net=TransformerNoisePredictionNet(input_len=al, input_dim=ad, global_cond_dim=ed, timestep_embed_dim=256, embed_dim=ed, depth=d, num_heads=nh, mlp_ratio=4, qkv_bias=True)
            self.noise_scheduler=DDPMScheduler(num_train_timesteps=100, beta_schedule="squaredcos_cap_v2", clip_sample=True, prediction_type="epsilon")
            self.num_inference_steps=10
        def sample(self, obs):
            B,d=obs.shape[0],obs.device; oe=self.obs_encoder(obs)
            a=torch.randn(B,self.action_len,self.action_dim,device=d)
            self.noise_scheduler.set_timesteps(self.num_inference_steps)
            for ts in self.noise_scheduler.timesteps:
                t=torch.full((B,),ts,device=d,dtype=torch.long)
                a=self.noise_scheduler.step(self.noise_pred_net(a,t,global_cond=oe),ts,a).prev_sample
            return a
    return LDP(od=20, al=16, ad=2, ed=ed, d=d, nh=nh).to(device)

def load_ckpt_custom(model_class, ckpt_path, device):
    ck = torch.load(ckpt_path, map_location=device)
    model = model_class.to(device)
    model.load_state_dict(ck["model"]); model.eval()
    ns = MinMaxNormalizer(); ns.offset=np.array(ck["state_normalizer"]["offset"]); ns.scale=np.array(ck["state_normalizer"]["scale"])
    na = MinMaxNormalizer(); na.offset=np.array(ck["action_normalizer"]["offset"]); na.scale=np.array(ck["action_normalizer"]["scale"])
    return model, ns, na

def eval_one_episode(model, norm_state, norm_action, device, seed):
    from diffusion_policy.env.pusht.pusht_keypoints_env import PushTKeypointsEnv
    from diffusion_policy.gym_util.multistep_wrapper import MultiStepWrapper

    model.eval()
    env = MultiStepWrapper(PushTKeypointsEnv(legacy=True, keypoint_visible_rate=1.0), n_obs_steps=2, n_action_steps=8)
    env.seed(seed)
    raw_obs = env.reset()
    obs_buffer = deque(maxlen=2)
    Do = raw_obs.shape[-1] // 2
    obs_buffer.append(raw_obs[0, :Do]); obs_buffer.append(raw_obs[1, :Do])
    rewards = []; done = False; step = 0

    while not done and step < 300:
        state_np = np.stack(list(obs_buffer))
        state_t = torch.from_numpy(state_np).float().to(device).unsqueeze(0)
        state_norm = (state_t - torch.tensor(norm_state.offset, device=device).float()) / torch.tensor(norm_state.scale, device=device).float()
        with torch.no_grad():
            action_norm = model.sample(state_norm)[0]
        action_raw = action_norm * torch.tensor(norm_action.scale, device=device).float() + torch.tensor(norm_action.offset, device=device).float()
        action_raw = np.clip(action_raw.cpu().numpy(), 0.0, 512.0)
        exec_actions = action_raw[:8]
        raw_obs, reward, done, info = env.step(exec_actions)
        rewards.append(float(reward))
        done = bool(np.all(done))
        Do = raw_obs.shape[-1] // 2
        obs_buffer.append(raw_obs[1, :Do])
        step += 1

    return float(max(rewards)) if rewards else 0.0

def main():
    device = torch.device("cuda:0")
    os.makedirs("outputs/e2_all_variants", exist_ok=True)
    N_EPS = 50
    SEEDS = [100000 + ep for ep in range(N_EPS)]

    # Load all models
    print("Loading models...")
    import importlib.util
    spec = importlib.util.spec_from_file_location("hm", str(Path(__file__).resolve().parent / "uwm_dp_only_keypoint20_obstoken_hybrid.py"))
    hm = importlib.util.module_from_spec(spec); spec.loader.exec_module(hm)

    models = {
        "b3": load_b3(device),
        "uwm_base": load_ckpt_custom(_build_uwm(device, 768, 12, 12), "artifacts_keep/uwm_dp_only_keypoint20_20k/latest.pt", device),
        "uwm_hybrid": load_ckpt_custom(hm.LowdimDiTPolicyHybrid(obs_dim=20, action_len=16, action_dim=2), "outputs/e2_uwm_kp_obstoken_hybrid/latest.pt", device),
        "uwm_small": load_ckpt_custom(_build_uwm(device, 256, 6, 8), "outputs/e2_uwm_kp_small/latest.pt", device),
    }

    all_scores = {n: [] for n in models}
    t0 = time.time()

    for ep, seed in enumerate(SEEDS):
        print(f"  ep {ep:3d} (seed={seed})", end="", flush=True)
        for name, (model, ns, na) in models.items():
            score = eval_one_episode(model, ns, na, device, seed)
            all_scores[name].append(score)
            print(f"  {name}={score:.3f}", end="", flush=True)
        print()

    elapsed = time.time() - t0
    print(f"\nPaired eval done in {elapsed:.1f}s ({N_EPS} eps × {len(models)} models)")

    # Statistics
    results = {}
    print(f"\n{'Model':<20} {'Mean':>8} {'Median':>8} {'Std':>8} {'ep>0.5':>8}")
    print("-" * 56)
    for name in models:
        s = np.array(all_scores[name])
        print(f"  {name:<18} {s.mean():8.4f} {np.median(s):8.4f} {s.std():8.4f} {(s>0.5).sum():>6}/{N_EPS}")
        results[name] = {"mean": float(s.mean()), "median": float(np.median(s)), "std": float(s.std()),
                         "ep_gt_05": int((s>0.5).sum()), "scores": [float(x) for x in s]}

    # Paired deltas
    b3_arr = np.array(all_scores["b3"])
    base_arr = np.array(all_scores["uwm_base"])
    small_arr = np.array(all_scores["uwm_small"])
    hybrid_arr = np.array(all_scores["uwm_hybrid"])

    for label, arr in [("small", small_arr), ("hybrid", hybrid_arr)]:
        delta = arr - base_arr
        print(f"\n  paired delta: UWM-{label} - UWM-base:")
        print(f"    mean_delta={delta.mean():.4f}  median_delta={np.median(delta):.4f}")
        print(f"    delta>0: {(delta>0).sum()}/{N_EPS}  delta>0.1: {(delta>0.1).sum()}/{N_EPS}")
        print(f"    delta<0: {(delta<0).sum()}/{N_EPS}  delta<-0.1: {(delta<-0.1).sum()}/{N_EPS}")
        results[f"paired_delta_{label}"] = {
            "mean": float(delta.mean()), "median": float(np.median(delta)),
            "gt0": int((delta>0).sum()), "gt_01": int((delta>0.1).sum()),
            "lt0": int((delta<0).sum()), "lt_neg_01": int((delta<-0.1).sum()),
        }

    # Save paired table
    with open("outputs/e2_paired_eval_scores.json", "w") as f:
        json.dump(results, f, indent=2)

    # Paired table (first 20 rows)
    with open("outputs/e2_paired_eval_table.csv", "w") as f:
        f.write("ep,seed,b3,uwm_base,uwm_hybrid,uwm_small\n")
        for ep, seed in enumerate(SEEDS):
            f.write(f"{ep},{seed},{all_scores['b3'][ep]:.4f},{all_scores['uwm_base'][ep]:.4f},{all_scores['uwm_hybrid'][ep]:.4f},{all_scores['uwm_small'][ep]:.4f}\n")

    print("\nPaired eval complete. outputs/e2_paired_eval_scores.json, outputs/e2_paired_eval_table.csv")


if __name__ == "__main__":
    main()
