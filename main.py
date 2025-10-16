import gymnasium as gym
from src.agent import RLSeq2Seq, Encoder, Decoder
from src.trainer import Seq2SeqTrainer
from src.utility.logger import logger
from src.utility.data_prep import build_trading_features
from src.utility.config import config


url = "https://raw.githubusercontent.com/ClementPerroud/Gym-Trading-Env/main/examples/data/BTC_USD-Hourly.csv"
df = build_trading_features(source=url)

env = gym.make("TradingEnv",
        name= "BTCUSD",
        df = df, # Your dataset with your custom features
        positions = [ -1, 0, 1], # -1 (=SHORT), 0(=OUT), +1 (=LONG)
        trading_fees = 0.01/100, # 0.01% per stock buy / sell (Binance fees)
        borrow_interest_rate= 0.0003/100, # 0.0003% per timestep (one timestep = 1h here)
    )

logger.info(f"Environment created.")
config['state_dim'] = env.observation_space.shape[0]
config['action_dim'] = env.action_space.n if hasattr(env.action_space, 'n') else env.action_space.shape[0]


encoder = Encoder(config)
decoder = Decoder(config)
trader = RLSeq2Seq(config=config, decoder=decoder, encoder=encoder)
logger.info(f"Agent initialized with {sum(p.numel() for p in trader.parameters() if p.requires_grad)} trainable parameters")


trainer = Seq2SeqTrainer(trader, env=env, env_name="stock", config=config)
logger.info("trainer initiliazed")
trainer.train()