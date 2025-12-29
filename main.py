import gymnasium as gym
from src.agent import RLSeq2Seq, Encoder, Decoder, PPO
from src.trainer import Seq2SeqTrainer, Trainer
from src.utility.logger import logger
from src.utility.data_prep import build_trading_features
from src.utility.config import config
import gym_trading_env

url = "https://raw.githubusercontent.com/ClementPerroud/Gym-Trading-Env/main/examples/data/BTC_USD-Hourly.csv"
df = build_trading_features(source=url)

env = gym.make("TradingEnv",
        name= "BTCUSD",
        df = df, # Your dataset with your custom features
        positions = [ -1, 0, 1], # -1 (=SHORT), 0(=OUT), +1 (=LONG)
        trading_fees = 0.01/100, # 0.01% per stock buy / sell (Binance fees)
        borrow_interest_rate= 0.0003/100, # 0.0003% per timestep (one timestep = 1h here)
    )

def seq2seq_main():
    logger.info(f"Environment created.")
    config['input_dim'] = env.observation_space.shape[0]
    config['action_dim'] = env.action_space.n if hasattr(env.action_space, 'n') else env.action_space.shape[0]


    encoder = Encoder(config)
    decoder = Decoder(config)
    trader = RLSeq2Seq(config=config, decoder=decoder, encoder=encoder)
    num_params = sum(p.numel() for p in trader.parameters() if p.requires_grad)
    logger.info(f"Agent initialized with {num_params/1e6:.3f}M trainable parameters")


    trainer = Seq2SeqTrainer(trader, env=env, env_name="stock", config=config)
    logger.info("trainer initiliazed")
    trainer.train()



def ppo_train():
    config['input_dim'] = env.observation_space.shape[0]
    config['action_dim'] = env.action_space.n if hasattr(env.action_space, 'n') else env.action_space.shape[0]

    trader = PPO(config=config)
    num_params = sum(p.numel() for p in trader.policy.parameters() if p.requires_grad)
    logger.info(f"Agent initialized with {num_params/1e6:.3f}M trainable parameters")

    trainer = Trainer(trader, env=env, env_name="stock", config=config)
    logger.info("trainer initiliazed")
    trainer.train()



def ppo_eval():
    from src.agent import PPO
    from src.utility.config import config
    from datetime import datetime
    from pathlib import Path
    import numpy as np

    config['input_dim'] = env.observation_space.shape[0]
    config['action_dim'] = env.action_space.n if hasattr(env.action_space, 'n') else env.action_space.shape[0]
    agent = PPO(config)
    agent.load(checkpoint_path="models\\stock")

    

    log_dir = "model_logs/stock"

    terminated, truncated = False, False
    obs, info = env.reset()
    ep_rewards = []
    cum_reward = 0.0

    while not (terminated or truncated):
        action = agent.select_action(state=obs)  # your policy here
        obs, reward, terminated, truncated, info = env.step(action)
        cum_reward += reward
        ep_rewards.append(reward)


    print(f"Episode cumulative reward: {cum_reward:.6f}")

    # If you want the running cumulative reward at each step:
    cum_rewards_per_step = np.cumsum(ep_rewards)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")   # e.g., 20251016-184223
    out_dir = Path(log_dir) / f"ppo_render_logs_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    env.save_for_render(dir = str(out_dir))
    env.close()


if __name__ == "__main__":
    ppo_train()