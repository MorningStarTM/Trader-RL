# Trader-RL
![Candlestick Chart](https://img.freepik.com/free-vector/dynamic-candlestick-chart-financial-analysis_1308-182195.jpg?semt=ais_hybrid&w=740&q=80)



This project involves training a Proximal Policy Optimization (PPO) agent for stock market trading within a Gym-based environment.


# 🧠 Problem & Approach

Financial markets are sequential, partially observable, and non-stationary, making traditional supervised learning approaches insufficient for directly optimizing trading decisions. The objective of this project is to learn a trading policy that maximizes risk-adjusted portfolio returns while accounting for transaction costs and market dynamics.

This project formulates trading as a Markov Decision Process (MDP) and applies Proximal Policy Optimization (PPO) to learn optimal actions through interaction with a Gym-based trading environment. PPO is chosen due to its training stability, sample efficiency, and robust performance in noisy and stochastic environments, making it a strong baseline for financial reinforcement learning.

The agent learns directly from market feedback by optimizing cumulative rewards, enabling adaptive decision-making under uncertainty rather than relying on fixed predictive signals.

# 🧩 Trading Environment Specification

## Data

The environment is constructed using hourly BTC/USD market data spanning from May 15, 2018 to March 1, 2022.
Each timestep corresponds to one hour and includes standard OHLCV information along with traded volume in USD.

Raw fields:

* Open

* High

* Low

* Close

* Volume

* Volume (USD)

This long historical window enables training and evaluation across multiple market regimes, including bull, bear, and sideways periods.

## Observation Space

At each timestep t, the agent observes a vector of normalized price- and volume-based features, designed to reduce scale sensitivity and improve training stability.

The observation includes the following features:
* Price return -> feature_close[t] = (close[t] - close[t-1]) / close[t-1]

* Relative open price -> feature_open[t] = open[t] / close[t]

* Relative high price -> feature_high[t] = high[t] / close[t]

* Relative low price -> feature_low[t] = low[t] / close[t]

* Normalized volume -> feature_volume[t] = VolumeUSD[t] / max(VolumeUSD[t]-168[t])

The rolling normalization of volume over a 7-day (168-hour) window helps capture relative liquidity changes while preventing extreme magnitude shifts.

These features provide the agent with information about short-term price movement, volatility, and market activity, while avoiding direct exposure to raw price scales.


## Action Space

The agent operates in a discrete action space, representing trading decisions at each timestep:

0 — Hold

1 — Buy

2 — Sell

Actions are executed at the close price of the current timestep
	​
# 📊 Evaluation Metrics

The trained PPO agent is evaluated using portfolio-level performance metrics that reflect both market-relative performance and cumulative decision quality.

## 1️⃣ Market Return

Market Return represents the return obtained by a passive buy-and-hold strategy over the same evaluation period. It serves as a baseline to contextualize the agent’s performance.

Market Return (%) = ((P[end] - P[start]) / P[start]) * 100

Where:
* 𝑃[start] is the initial market price
* 𝑃[end] is the final market price


## 2️⃣ Portfolio Return

Portfolio Return measures the percentage growth of the agent’s trading portfolio over the evaluation episode, accounting for executed trades and transaction costs.

Portfolio Return (%) = ((V[end] - V[start]) / V[start]) * 100

Where:
* V[start] is the initial portfolio value
* V[end] is the final portfolio value

## 3️⃣ Episode Cumulative Reward

Episode Cumulative Reward represents the total reward accumulated by the agent over an episode and reflects the quality and consistency of sequential trading decisions.



## 📂 Project Structure
```
Trader-RL
│── logs                
│── model_logs            
│── notebook               
│── src               
│    |──utility
│    │     │── config.py              # hyper parameter file
│    │     │── data_prep.py           # data preprocessing pipeline
│    │     │── logger.py              # Logger file
│    │── agent.py
│    │──trainer.py
│── main.py
│── requirement.txt
│── README.md              
```

## 📦 Installation & Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/MorningStarTM/Trader-RL.git
   cd Trader-RL
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run training script
```bash
python main.py
   ```

4. Evaluate agent
Call ppo_eval function. 
```bash
if __name__ == "__main__":
    ppo_eval()
   ```

and run main file
```bash
python main.py
```


5. To render Agent's Actions
```bash
from gym_trading_env.renderer import Renderer
renderer = Renderer(render_logs_dir=path_of_rendered_file)
renderer.run()

```


# Future Work
* Explore Sequential Decision making models like Decision Transformer
* Pre-Train Agent without Reward




⚠️ Disclaimer: This project is intended for research and educational purposes only.
It is not financial advice and should not be used for real-world trading.