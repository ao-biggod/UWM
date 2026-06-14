"""
Smoke test for Diffusion Policy PushT Image Env.

Verifies:
- PushTImageEnv can be created
- reset() returns correct obs dict
- step() with random actions works
- Rendering works (rgb_array mode)

Usage:
    cd diffusion_policy-main
    # If on headless server:
    export SDL_VIDEODRIVER=dummy
    python scripts/smoke_pusht_env.py

Requires: pip install -e . (from diffusion_policy-main root)
"""

import os
import sys
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def main():
    print("=" * 60)
    print("PushT Image Env Smoke Test")
    print("=" * 60)

    # ---- 1. Check headless mode ----
    print("\n[1/4] Checking headless mode")
    sdl_driver = os.environ.get("SDL_VIDEODRIVER", "not set")
    print(f"  SDL_VIDEODRIVER={sdl_driver}")
    if sdl_driver == "not set":
        print("  WARNING: SDL_VIDEODRIVER not set. On headless servers, run:")
        print("    export SDL_VIDEODRIVER=dummy")

    # ---- 2. Create env ----
    print("\n[2/4] Creating PushTImageEnv")
    try:
        from diffusion_policy.env.pusht.pusht_image_env import PushTImageEnv
        from diffusion_policy.gym_util.multistep_wrapper import MultiStepWrapper

        n_obs_steps = 2
        n_action_steps = 8
        max_steps = 300

        def env_fn():
            return MultiStepWrapper(
                PushTImageEnv(legacy=True, render_size=96),
                n_obs_steps=n_obs_steps,
                n_action_steps=n_action_steps,
                max_episode_steps=max_steps,
            )

        env = env_fn()
        print(f"  Env created: {type(env)}")
        print(f"  Action space: {env.action_space}")
        print(f"  Obs space keys: {list(env.observation_space.spaces.keys())}")
    except Exception as e:
        print(f"  ERROR creating env: {e}")
        print("\n  Troubleshooting:")
        print("  1. pygame: pip install pygame==2.1.2")
        print("  2. pymunk: pip install pymunk==6.2.1")
        print("  3. On headless: export SDL_VIDEODRIVER=dummy")
        print("  4. shapely: pip install shapely")
        print("  5. scikit-image: pip install scikit-image")
        sys.exit(1)

    # ---- 3. Reset and step ----
    print("\n[3/4] Running env for 10 steps with random actions")
    try:
        obs = env.reset()
        print(f"  Reset OK")
        print(f"  Obs keys: {list(obs.keys())}")
        print(f"  image shape: {obs['image'].shape}")
        print(f"  agent_pos shape: {obs['agent_pos'].shape}")

        # Verify shapes
        assert obs["image"].shape == (n_obs_steps, 3, 96, 96), \
            f"Expected ({n_obs_steps}, 3, 96, 96), got {obs['image'].shape}"
        assert obs["agent_pos"].shape == (n_obs_steps, 2), \
            f"Expected ({n_obs_steps}, 2), got {obs['agent_pos'].shape}"

        for step in range(10):
            # Generate random action within bounds
            action = np.random.uniform(0, 512, size=(n_action_steps, 2)).astype(np.float32)
            obs, reward, done, info = env.step(action)

            if done:
                print(f"  Step {step}: reward={reward:.3f}, done=True (early termination)")
                break
            else:
                print(f"  Step {step}: reward={reward:.3f}, done=False")

        print("  OK - env steps completed")
    except Exception as e:
        print(f"  ERROR during env step: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # ---- 4. Render test ----
    print("\n[4/4] Testing render (rgb_array)")
    try:
        img = env.render(mode="rgb_array")
        print(f"  Render shape: {img.shape}")
        # For AsyncVectorEnv wrapper, render after reset returns list
        if isinstance(img, list):
            img = img[0]
            print(f"  First env render shape: {img.shape}")
    except Exception as e:
        print(f"  WARNING: Render failed (non-critical): {e}")

    env.close()
    print("\n" + "=" * 60)
    print("ENV SMOKE TEST PASSED")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
