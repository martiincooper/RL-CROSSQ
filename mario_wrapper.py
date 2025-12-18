
import gymnasium as gym
from gymnasium.wrappers import ResizeObservation, GrayScaleObservation, FrameStack, TransformObservation, NormalizeObservation
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT, RIGHT_ONLY
import numpy as np

# Robust wrapper to force Gymnasium API (5-tuple step)
class MarioGymnasiumWrapper(gym.Wrapper):
    def __init__(self, env):
        # We manually wrap the gym environment
        self.env = env
        # Do not copy attributes manually, let Wrapper delegate or Properties handle it.
        # But we might need to expose spaces if they are hidden.
        # gymnasium wrapper might need properties to be set on init or override
        
        
        # We try to get them from the unwrapped env if possible
        # And convert to gymnasium spaces
        gym_obs = env.observation_space
        self._observation_space = gym.spaces.Box(
            low=gym_obs.low, high=gym_obs.high, shape=gym_obs.shape, dtype=gym_obs.dtype
        )
        
        gym_act = env.action_space
        if hasattr(gym_act, 'n'):
            self._action_space = gym.spaces.Discrete(gym_act.n)
        else:
             self._action_space = gym.spaces.Box(
                low=gym_act.low, high=gym_act.high, shape=gym_act.shape, dtype=gym_act.dtype
             )

        self._metadata = getattr(env, 'metadata', {})
        self._reward_range = getattr(env, 'reward_range', (-float('inf'), float('inf')))

    @property
    def action_space(self):
        return self._action_space

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def metadata(self):
        return self._metadata

    @property
    def reward_range(self):
        return self._reward_range

    def reset(self, **kwargs):
        # gym-super-mario-bros reset returns (obs) or (obs, info) depending on version/wrapper
        # Handle seed manually for old gym compatibility
        seed = kwargs.get('seed', None)
        options = kwargs.get('options', None)
        
        # Pop them to avoid passing to old reset which might not accept them
        if 'seed' in kwargs: kwargs.pop('seed')
        if 'options' in kwargs: kwargs.pop('options')
        
        if seed is not None:
             try:
                 self.env.seed(seed)
             except AttributeError:
                 pass # Env might not support seed
        
        # usage: obs = env.reset()
        ret = self.env.reset(**kwargs)
        if isinstance(ret, tuple):
            if len(ret) == 2:
                return ret
            return ret[0], {}
        return ret, {}

    def step(self, action):
        ret = self.env.step(action)
        if len(ret) == 4:
            obs, reward, done, info = ret
            truncated = False
            terminated = done
            return obs, reward, terminated, truncated, info
        elif len(ret) == 5:
            return ret
        else:
            raise ValueError(f"Unexpected step return length: {len(ret)}")
    
    def render(self):
        return self.env.render()
    
    def close(self):
        return self.env.close()

class ContinuousMarioWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        n_actions = env.action_space.n
        # CrossQ/SAC expect a Box action space
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(n_actions,), dtype=np.float32)
        
    def action(self, action):
        # Convert continuous vector to discrete index
        return int(np.argmax(action))

# Transpose to (H, W, C) = (84, 84, 4)
class TransposeWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        old_shape = env.observation_space.shape # (4, 84, 84)
        self.observation_space = gym.spaces.Box(
            low=0, high=255, 
            shape=(old_shape[1], old_shape[2], old_shape[0]), 
            dtype=env.observation_space.dtype
        )
    
    def observation(self, obs):
        # obs is (4, 84, 84) -> (84, 84, 4)
        return np.moveaxis(obs, 0, -1)

def make_mario_env(env_id='SuperMarioBros-v0', action_space=SIMPLE_MOVEMENT, stack_frames=4, render_mode='rgb_array'):
    # Create the original environment
    # nes-py returns 4-tuple, gym 0.26 expects 5-tuple. compatibility=True might help.
    env = gym_super_mario_bros.make(env_id, apply_api_compatibility=True) 
    env = JoypadSpace(env, action_space)
    
    # Force load (nes_py lazy initialization might be an issue)
    env.reset()
    
    # Wrap it to gymnasium manually
    env = MarioGymnasiumWrapper(env)

    # Convert to continuous for CrossQ/SAC
    env = ContinuousMarioWrapper(env)
    
    # Apply wrappers
    env = GrayScaleObservation(env, keep_dim=False)
    env = ResizeObservation(env, (84, 84))
    env = FrameStack(env, stack_frames)
    
    env = TransposeWrapper(env)
    
    return env
