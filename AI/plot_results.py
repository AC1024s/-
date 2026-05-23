import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

# ===== 把 baseline.py 跑出来的结果填在这里 =====
# 示例数据，请替换成你的真实结果
random_3x3 = 16.4   # 替换
spt_3x3 = 22         # 替换
rl_3x3 = 14        # 替换

# ========== 图1：3×3 三种方法对比柱状图 ==========
fig, ax = plt.subplots(figsize=(8, 5))
methods = ['随机策略', 'SPT启发式', 'RL(PPO)']
makespans = [random_3x3, spt_3x3, rl_3x3]
colors = ['#ff6b6b', '#feca57', '#48dbfb']
bars = ax.bar(methods, makespans, color=colors, width=0.5)
ax.set_ylabel('总完工时间 (Makespan)', fontsize=12)
ax.set_title('不同调度策略对比 (3×3实例)', fontsize=14)
for bar, val in zip(bars, makespans):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
            f'{val}', ha='center', fontsize=14, fontweight='bold')
ax.set_ylim(0, max(makespans) * 1.15)
plt.tight_layout()
plt.savefig('对比图_3x3.png', dpi=150)
print("已保存: 对比图_3x3.png")

# ========== 图2：不同规模对比 ==========
# 替换成你的真实数据
sizes = ['2×2', '3×3', '5×5']
random_results = [6.8, 5, 5.0]    # 替换
spt_results = [16.4, 22, 14.0]       # 替换
rl_results = [52.4, 84, 39]        # 替换

fig, ax = plt.subplots(figsize=(8, 5))
x = range(len(sizes))
width = 0.25
ax.bar([i - width for i in x], random_results, width, label='随机策略', color='#ff6b6b')
ax.bar(x, spt_results, width, label='SPT启发式', color='#feca57')
ax.bar([i + width for i in x], rl_results, width, label='RL(PPO)', color='#48dbfb')
ax.set_xticks(x)
ax.set_xticklabels(sizes)
ax.set_ylabel('总完工时间 (Makespan)', fontsize=12)
ax.set_title('不同规模实例的调度效果对比', fontsize=14)
ax.legend()
plt.tight_layout()
plt.savefig('对比图_多规模.png', dpi=150)
print("已保存: 对比图_多规模.png")

# ========== 图3：甘特图（RL结果）==========
# 用3×3默认实例，展示RL的调度方案
from jssp_env import JSSPEnv
from stable_baselines3 import PPO

env = JSSPEnv()
model = PPO.load("jssp_ppo")
obs, _ = env.reset()
schedule = []  # [(作业, 工序, 机器, 开始, 结束)]
job_step_tracker = [0] * env.n_jobs

while True:
    available = [j for j in range(env.n_jobs) if env.job_step[j] < len(env.jobs_data[j])]
    if not available:
        break
    action, _ = model.predict(obs, deterministic=True)
    job = int(action)
    if env.job_step[job] >= len(env.jobs_data[job]):
        job = available[0]

    step_idx = env.job_step[job]
    machine, duration = env.jobs_data[job][step_idx]
    start = max(env.job_end_time[job], env.machine_time[machine])
    end = start + duration
    schedule.append((job, step_idx, machine, start, end))

    obs, reward, done, truncated, info = env.step(action)
    if done or truncated:
        break

# 画甘特图
colors_gantt = ['#48dbfb', '#feca57', '#ff6b6b', '#2ecc71', '#9b59b6']
fig, ax = plt.subplots(figsize=(10, 4))
for job, step, machine, start, end in schedule:
    ax.barh(machine, end - start, left=start, height=0.5,
            color=colors_gantt[job % len(colors_gantt)],
            edgecolor='white', linewidth=1)
    ax.text(start + (end - start) / 2, machine, f'J{job+1}',
            ha='center', va='center', fontsize=10, fontweight='bold')

ax.set_xlabel('时间', fontsize=12)
ax.set_ylabel('机器', fontsize=12)
ax.set_yticks(range(env.n_machines))
ax.set_yticklabels([f'机器{i+1}' for i in range(env.n_machines)])
ax.set_title('RL调度方案甘特图 (3×3实例)', fontsize=14)
plt.tight_layout()
plt.savefig('甘特图.png', dpi=150)
print("已保存: 甘特图.png")

plt.show()