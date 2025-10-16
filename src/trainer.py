
import matplotlib.pyplot as plt
import time
import numpy as np
from collections import deque
import torch
import gymnasium as gym
from pathlib import Path
from datetime import datetime
import os
import itertools
import torch
from tqdm import tqdm
from src.utility.logger import logger
from src.agent import RLSeq2Seq
from torch.utils.tensorboard import SummaryWriter


os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'



class Seq2SeqTrainer:
    def __init__(self, agent:RLSeq2Seq, env:gym, env_name:str, config):
        
        self.env = env
        self.env_name = env_name
        self.agent = agent
        self.best_score = 0.0
        self.score_history = []
        self.config = config
        self.episode_rewards = []  # Stores total reward per episode
        self.step_rewards = []     # Stores every single reward at each timestep

        self.log_dir = "model_logs"
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir, exist_ok=True)

        self.log_dir = self.log_dir + '/' + env_name + '/'
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir, exist_ok=True)

        run_num = 0
        current_num_files = next(os.walk(self.log_dir))[2]
        run_num = len(current_num_files)

        #### create new log file for each run
        self.log_f_name = self.log_dir + '/Seq2SeqRL_' + env_name + "_log_" + str(run_num) + ".csv"

        logger.info("current logging run number for " + env_name + " : ", run_num)
        logger.info("logging at : " + self.log_f_name)

        self.directory = "Models"
        if not os.path.exists(self.directory):
            os.makedirs(self.directory, exist_ok=True)

        self.directory = self.directory + '/' + env_name + '/'
        if not os.path.exists(self.directory):
            os.makedirs(self.directory, exist_ok=True)

        self.reward_folder = 'rewards'
        if not os.path.exists(self.reward_folder):
            os.makedirs(self.reward_folder, exist_ok=True)

        self.reward_folder = self.reward_folder + '/' + env_name + '/'
        if not os.path.exists(self.reward_folder):
            os.makedirs(self.reward_folder, exist_ok=True)
        
    

    def train(self):
        start_time = datetime.now().replace(microsecond=0)
        logger.info("Started training at (GMT) : ", start_time)

        print("============================================================================================")

        # logging file
        log_f = open(self.log_f_name,"w+")
        log_f.write('episode,timestep,reward\n')

        # printing and logging variables
        print_running_reward = 0
        print_running_episodes = 0

        log_running_reward = 0
        log_running_episodes = 0

        time_step = 0
        i_episode = 0
        
        while time_step <= self.config['max_training_timesteps']:

            state, _ = self.env.reset()
            current_ep_reward = 0
            prev_obs = None
            prev_action = None


            for t in range(1, self.config['max_ep_len']+1):
                if prev_obs is not None and prev_action is not None:
                    prev_context = np.concatenate([prev_obs, np.eye(self.config['action_dim'])[prev_action]])
                else:
                    prev_context = None

                # select action with policy

                action, new_context = self.agent.select_action(state, prev_context=prev_context)

                next_state, reward, done, _, _ = self.env.step(action)
                self.step_rewards.append(reward)
                self.agent.buffer.next_state.append(next_state)
                

                decoder_input = None
                if prev_obs is not None and prev_action is not None:
                    
                    one_hot_action = np.eye(self.config['action_dim'])[prev_action]
                    decoder_input = np.concatenate([prev_obs, one_hot_action])
                else:
                    decoder_input = np.zeros(self.config['input_dim'] + self.config['action_dim'])

                


                # saving reward and is_terminals
                self.agent.buffer.rewards.append(reward)
                self.agent.buffer.is_terminals.append(done)
                if prev_obs is not None and prev_action is not None:
                    prev_context = np.concatenate([prev_obs, np.eye(self.config['action_dim'])[prev_action]])
                    self.agent.buffer.prev_contexts.append(torch.FloatTensor(prev_context))
                else:
                    self.agent.buffer.prev_contexts.append(torch.zeros(self.config['input_dim'] + self.config['action_dim']))



                prev_obs = state
                prev_action = action
                state = next_state

                time_step +=1
                current_ep_reward += reward

                # update PPO agent
                if time_step % self.config['update_timestep'] == 0:
                    self.agent.update()

                # if continuous action space; then decay action std of ouput action distribution
                if self.config['has_continuous_action_space'] and time_step % self.config['action_std_decay_freq'] == 0:
                    self.agent.decay_action_std(self.config['action_std_decay_rate'], self.config['min_action_std'])

                # log in logging file
                if time_step % self.config['log_freq'] == 0:

                    # log average reward till last episode
                    log_avg_reward = log_running_reward / log_running_episodes
                    log_avg_reward = round(log_avg_reward, 4)

                    log_f.write('{},{},{}\n'.format(i_episode, time_step, log_avg_reward))
                    log_f.flush()

                    log_running_reward = 0
                    log_running_episodes = 0

                # printing average reward
                if time_step % self.config['print_freq'] == 0:

                    # print average reward till last episode
                    print_avg_reward = print_running_reward / print_running_episodes
                    print_avg_reward = round(print_avg_reward, 2)

                    logger.info("Episode : {} \t\t Timestep : {} \t\t Average Reward : {}".format(i_episode, time_step, print_avg_reward))

                    print_running_reward = 0
                    print_running_episodes = 0

                # save model weights
                if time_step % self.config['save_model_freq'] == 0:
                    logger.info("--------------------------------------------------------------------------------------------")
                    logger.info("saving model at : " + self.directory)
                    self.agent.save_models(self.directory)
                    logger.info("model saved")
                    logger.info("Elapsed Time  : ", datetime.now().replace(microsecond=0) - start_time)
                    logger.info("--------------------------------------------------------------------------------------------")

                # break; if the episode is over
                if done:
                    break
                
            self.episode_rewards.append(current_ep_reward)
            print_running_reward += current_ep_reward
            print_running_episodes += 1

            log_running_reward += current_ep_reward
            log_running_episodes += 1

            i_episode += 1

        log_f.close()
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")   # e.g., 20251016-184223
        out_dir = Path(self.log_dir) / f"render_logs_{ts}"
        out_dir.mkdir(parents=True, exist_ok=True)
        self.env.save_for_render(dir = str(out_dir))
        self.env.close()

        # print total training time
        logger.info("============================================================================================")
        end_time = datetime.now().replace(microsecond=0)
        logger.info("Started training at (GMT) : ", start_time)
        logger.info("Finished training at (GMT) : ", end_time)
        logger.info("Total training time  : ", end_time - start_time)
        logger.info("============================================================================================")

        np.save(os.path.join(self.reward_folder, f"seq2seqBasic_{self.env_name}_{self.config['horizon']}_step_rewards.npy"), np.array(self.step_rewards))
        np.save(os.path.join(self.reward_folder, f"seq2seqBasic_{self.env_name}_{self.config['horizon']}_episode_rewards.npy"), np.array(self.episode_rewards))
        logger.info(f"Saved step_rewards and episode_rewards to {self.log_dir}")


