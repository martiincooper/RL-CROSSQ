from mario_wrapper import make_mario_env
from sbx import SAC
from sbx.sac.utils import ReLU
import numpy as np
import jax
import flax.linen as nn

print("Testing env creation...")
try:
    env = make_mario_env()
    print("Env created successfully.")
    print("Observation Space:", env.observation_space)
    print("Action Space:", env.action_space)
    
    obs, info = env.reset()
    print("Reset successful. Obs shape:", obs.shape)

    print("Testing Model Initialization (CrossQ/SAC with CNN)...")
    
    # Must pass activation_fn class (from sbx utils)
    policy_kwargs = dict(
        activation_fn=ReLU,
        net_arch=[64]
    )
    
    model = SAC("CnnPolicy", env, policy_kwargs=policy_kwargs, verbose=1)
    print("Model initialized.")

    print("Testing Prediction...")
    action, _ = model.predict(obs, deterministic=True)
    print("Action:", action)
    
    print("Testing Training Step...")
    model.learn(total_timesteps=10)
    print("Training step successful.")

    print("VERIFICATION PASSED")

except Exception as e:
    print("VERIFICATION FAILED")
    import traceback
    print(traceback.format_exc())
