# Trader-RL
![Candlestick Chart](https://img.freepik.com/free-vector/dynamic-candlestick-chart-financial-analysis_1308-182195.jpg?semt=ais_hybrid&w=740&q=80)



This project involves training a Proximal Policy Optimization (PPO) agent for stock market trading within a Gym-based environment.


## 📂 Project Structure
```
Trade-RL
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