import random, datetime
from pathlib import Path

import gym
import gym_super_mario_bros
from gym.wrappers import FrameStack, GrayScaleObservation, TransformObservation
from nes_py.wrappers import JoypadSpace

from metrics import MetricLogger
from agent_ddqn import MarioDDQN
from agent_pg import MarioPG
from wrappers import ResizeObservation, SkipFrame

env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0')

env = JoypadSpace(
    env,
    [['right'],
    ['right', 'A']]
)

env = SkipFrame(env, skip=4)
env = GrayScaleObservation(env, keep_dim=False)
env = ResizeObservation(env, shape=84)
env = TransformObservation(env, f=lambda x: x / 255.)
env = FrameStack(env, num_stack=4)

env.reset()

save_dir = Path('checkpoints') / datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%S')
save_dir.mkdir(parents=True)
file = 'trained_mario_pg.pth'
checkpoint = Path(file)
print(checkpoint,type(checkpoint))
mario = MarioPG(state_dim=(4, 84, 84), action_dim=env.action_space.n, save_dir=save_dir, checkpoint=checkpoint)

if mario.__class__.__name__ == "MarioDDQN":
    mario.exploration_rate = mario.exploration_rate_min

logger = MetricLogger(save_dir)

episodes = 100

for e in range(episodes):

    state = env.reset()

    while True:

        env.render()

        action = mario.act(state)

        next_state, reward, done, info = env.step(action)
        #Episode Roll out depend on which agent is used
        if mario.__class__.__name__ == "MarioDDQN":
            mario.cache(state, next_state, action, reward, done)
            logger.log_step(reward, None, None)
            state = next_state
            if done or info['flag_get']:
                break
        elif mario.__class__.__name__ == "MarioPG":
            mario.cache(state, next_state, action, reward, done)
            logger.log_step(reward, None, None)
            state = next_state
            if done or info['flag_get']:
                break
        elif mario.__class__.__name__ == "MarioPPO":
            pass #TO DO
        else:
            raise ValueError(f"{mario.__class__.__name__} does not exist")


    logger.log_episode()

    if e % 20 == 0:
        logger.record(
            episode=e,
            epsilon=mario.exploration_rate,
            step=mario.curr_step
        )
