from jssp_env import JSSPEnv
from stable_baselines3 import PPO
import random

# ========== 策略函数 ==========

def random_schedule(jobs_data=None):
    env = JSSPEnv(jobs_data=jobs_data)
    obs, _ = env.reset()
    while True:
        available = [j for j in range(env.n_jobs) if env.job_step[j] < len(env.jobs_data[j])]
        if not available:
            break
        action = random.choice(available)
        obs, reward, done, truncated, info = env.step(action)
        if done or truncated:
            break
    return env.get_makespan()

def spt_schedule(jobs_data=None):
    env = JSSPEnv(jobs_data=jobs_data)
    obs, _ = env.reset()
    while True:
        available = [j for j in range(env.n_jobs) if env.job_step[j] < len(env.jobs_data[j])]
        if not available:
            break
        best_job = min(available, key=lambda j: env.jobs_data[j][env.job_step[j]][1])
        obs, reward, done, truncated, info = env.step(best_job)
        if done or truncated:
            break
    return env.get_makespan()

def train_and_eval(jobs_data, model_name, timesteps=50000):
    """为指定规模训练模型并评估"""
    env = JSSPEnv(jobs_data=jobs_data)

    # 先跑基线
    r_results = [random_schedule(jobs_data) for _ in range(30)]
    r_avg = sum(r_results) / len(r_results)
    s_result = spt_schedule(jobs_data)

    # 训练RL模型
    print(f"  训练 {model_name} 中...")
    model = PPO("MlpPolicy", env, verbose=0, learning_rate=3e-4, n_steps=2048)
    model.learn(total_timesteps=timesteps)
    model.save(model_name)

    # 评估RL模型
    rl_results = []
    for _ in range(10):
        test_env = JSSPEnv(jobs_data=jobs_data)
        obs, _ = test_env.reset()
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = test_env.step(action)
            if done or truncated:
                break
        rl_results.append(test_env.get_makespan())
    rl_avg = sum(rl_results) / len(rl_results)
    rl_best = min(rl_results)

    return r_avg, s_result, rl_avg, rl_best

# ========== 实例数据 ==========
data_2x2 = [
    [(0, 2), (1, 3)],
    [(1, 2), (0, 1)],
]

data_3x3 = [
    [(0, 3), (1, 2), (2, 4)],
    [(1, 3), (0, 2), (2, 3)],
    [(0, 2), (2, 3), (1, 1)],
]

data_5x5 = [
    [(0, 5), (1, 3), (2, 7), (3, 2), (4, 4)],
    [(1, 6), (0, 4), (3, 5), (2, 3), (4, 2)],
    [(2, 3), (3, 8), (0, 2), (4, 6), (1, 4)],
    [(3, 4), (2, 2), (4, 5), (1, 7), (0, 3)],
    [(4, 7), (0, 3), (1, 4), (2, 5), (3, 6)],
]

# ========== 跑实验 ==========
if __name__ == "__main__":
    results = {}

    for name, data, steps in [
        ("2x2", data_2x2, 20000),
        ("3x3", data_3x3, 50000),
        ("5x5", data_5x5, 80000),
    ]:
        print(f"\n{'='*40}")
        print(f"测试规模: {name}")
        print(f"{'='*40}")
        r, s, rl, rl_best = train_and_eval(data, f"model_{name}", steps)
        results[name] = (r, s, rl, rl_best)
        print(f"  随机: {r:.1f}  SPT: {s}  RL平均: {rl:.1f}  RL最优: {rl_best}")
        print(f"  RL vs 随机提升: {(r-rl)/r*100:.1f}%")

    # ========== 汇总输出 ==========
    print(f"\n{'='*50}")
    print("实验结果汇总")
    print(f"{'='*50}")
    print(f"{'规模':<8}{'随机':<10}{'SPT':<10}{'RL平均':<10}{'RL最优':<10}{'提升':<10}")
    print("-" * 58)
    for name in ["2x2", "3x3", "5x5"]:
        r, s, rl, rl_best = results[name]
        improve = (r - rl) / r * 100
        print(f"{name:<8}{r:<10.1f}{s:<10}{rl:<10.1f}{rl_best:<10}{improve:.1f}%")

    # 保存
    with open("results.txt", "w", encoding="utf-8") as f:
        f.write("规模\t随机\tSPT\tRL平均\tRL最优\t提升\n")
        for name in ["2x2", "3x3", "5x5"]:
            r, s, rl, rl_best = results[name]
            improve = (r - rl) / r * 100
            f.write(f"{name}\t{r:.1f}\t{s}\t{rl:.1f}\t{rl_best}\t{improve:.1f}%\n")
    print("\n结果已保存到 results.txt")