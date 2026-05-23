from jssp_env import JSSPEnv
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
import matplotlib.pyplot as plt

class LogCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self.makespans = []

    def _on_step(self):
        infos = self.locals.get("infos", [])
        for info in infos:
            if "makespan" in info:
                self.makespans.append(info["makespan"])
        return True

# 创建环境
env = JSSPEnv()

# 训练
print("开始训练...")
model = PPO("MlpPolicy", env, verbose=1, learning_rate=3e-4, n_steps=2048)
model.learn(total_timesteps=50000)
model.save("jssp_ppo")
print("训练完成，模型已保存")

# 测试
env = JSSPEnv()
model = PPO.load("jssp_ppo")
obs, _ = env.reset()
total_reward = 0

while True:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = env.step(action)
    total_reward += reward
    if done or truncated:
        break

print(f"\n=== RL结果 ===")
print(f"总完工时间(makespan): {env.get_makespan()}")
