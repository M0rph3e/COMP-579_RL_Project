import random, datetime
from pathlib import Path
import matplotlib.pyplot as plt
import torch
import gym
import gym_super_mario_bros
from gym.wrappers import FrameStack, GrayScaleObservation, TransformObservation
from nes_py.wrappers import JoypadSpace

from metrics import MetricLogger
from agent_ddqn import MarioDDQN
from agent_pg import MarioPG
from wrappers import ResizeObservation, SkipFrame

def compare_rewards(logger_a,logger_b,name_a,name_b,k,title,path,xlabel='Episodes', ylabel='Rewards'):
    """
    Compare and plot rewards alongs episodes for tested agents
    Inputs:
    logger_a (MetricLogger)
    logger_b (MetricLogger)
    """
    rewards_1 = logger_a.moving_avg_ep_rewards
    rewards_2 = logger_b.moving_avg_ep_rewards
    assert(len(rewards_1)==len(rewards_2))

    x = [i*k for i in range(len(rewards_1))]
    
    plt.plot(x, rewards_1, label=name_a)
    plt.plot(x, rewards_2, label=name_b)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title('Avg Rewards per '+str(k)+' episodes with ' + title + ' env')
    plt.legend()
    plt.savefig(path)

def compare_length(logger_a,logger_b,name_a,name_b,k,title,path,xlabel='Episodes', ylabel='Rewards'):
    """
    Compare and plot number of steps alongs episodes for tested agents
    Inputs:
    logger_a (MetricLogger)
    logger_b (MetricLogger)
    """
    rewards_1 = logger_a.moving_avg_ep_lengths
    rewards_2 = logger_b.moving_avg_ep_lengths
    assert(len(rewards_1)==len(rewards_2))

    x = [i*k for i in range(len(rewards_1))]
    
    plt.plot(x, rewards_1, label=name_a)
    plt.plot(x, rewards_2, label=name_b)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title('Avg Number of steps per '+str(k)+' episodes with ' + title + ' env')
    plt.legend()
    plt.savefig(path)


env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0') #to change if try different
modif = 'standard'
render = False
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
date = datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%S')
save_dir_ddqn = Path('checkpoints') / date / "ddqn"
save_dir_pg = Path('checkpoints') / date / "pg"
save_dir_ddqn.mkdir(parents=True)
save_dir_pg.mkdir(parents=True)

file_ddqn = 'trained_mario_ddqn.chkpt'
file_pg = 'trained_mario_pg.pth'
save_final_plot_r = Path(str(save_dir_ddqn.parent.resolve()) + '/reward_comparison_'+modif+'.jpg')
save_final_plot_l = Path(str(save_dir_ddqn.parent.resolve()) + '/length_comparison_'+modif+'.jpg')
checkpoint_ddqn = Path(file_ddqn)
checkpoint_pg = Path(file_pg)

mario_ddqn = MarioDDQN(state_dim=(4, 84, 84), action_dim=env.action_space.n, save_dir=save_dir_ddqn, checkpoint=checkpoint_ddqn)
mario_pg = MarioPG(state_dim=(4, 84, 84), action_dim=env.action_space.n, save_dir=save_dir_pg, checkpoint=checkpoint_pg)

marios = [mario_ddqn,mario_pg]


logger_ddqn = MetricLogger(save_dir_ddqn)
logger_pg = MetricLogger(save_dir_pg)

loggers = [logger_ddqn,logger_pg]

episodes = 1000
k=10 # NUMBER OF EP WE LOG REWARD
for (mario,logger) in zip(marios,loggers):
    if mario.__class__.__name__ == "MarioDDQN":
        mario.exploration_rate = mario.exploration_rate_min
    for e in range(episodes):

        state = env.reset()

        while True:
            if render:
                env.render()

            action = mario.act(state)

            next_state, reward, done, info = env.step(action)
            #Episode Roll out depend on which agent is used
            if mario.__class__.__name__ == "MarioDDQN":
                #mario.cache(state, next_state, action, reward, done)
                logger.log_step(reward, None, None)
                state = next_state
                if done or info['flag_get']:
                    break
            elif mario.__class__.__name__ == "MarioPG":
                #mario.cache(state, next_state, action, reward, done)
                logger.log_step(reward, None, None)
                state = next_state
                if done or info['flag_get']:
                    break
            elif mario.__class__.__name__ == "MarioPPO":
                pass #TO DO SOMEDAY
            else:
                raise ValueError(f"{mario.__class__.__name__} does not exist")


        logger.log_episode()

        if e % k == 0:
            logger.record(
                episode=e,
                epsilon=mario.exploration_rate,
                step=mario.curr_step
            )
        torch.cuda.empty_cache() # because memory overflow


compare_rewards(logger_ddqn,logger_pg,
                mario_ddqn.__class__.__name__,
                mario_pg.__class__.__name__,
                k,modif,save_final_plot_r)
compare_length(logger_ddqn,logger_pg,
                mario_ddqn.__class__.__name__,
                mario_pg.__class__.__name__,
                k,modif,save_final_plot_l)





