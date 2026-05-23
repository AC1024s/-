import gymnasium as gym
from gymnasium import spaces
import numpy as np


class JSSPEnv(gym.Env):
    """车间调度强化学习环境"""

    def __init__(self, jobs_data=None):
        super().__init__()

        if jobs_data is None:
            self.jobs_data = [
                [(0, 3), (1, 2), (2, 4)],
                [(1, 3), (0, 2), (2, 3)],
                [(0, 2), (2, 3), (1, 1)],
            ]
        else:
            self.jobs_data = jobs_data

        self.n_jobs = len(self.jobs_data)
        self.n_machines = max(m for job in self.jobs_data for m, _ in job) + 1

        # 动作空间：选择下一步执行哪个作业
        self.action_space = spaces.Discrete(self.n_jobs)

        # 观察空间
        obs_size = self.n_jobs + self.n_machines
        self.observation_space = spaces.Box(
            low=0, high=1000, shape=(obs_size,), dtype=np.float32
        )

        self.max_steps = self.n_jobs * 10  # 防止死循环
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.job_step = [0] * self.n_jobs
        self.machine_time = [0] * self.n_machines
        self.job_end_time = [0] * self.n_jobs
        self.time = 0
        self.done = False
        self.steps = 0
        return self._get_obs(), {}

    def _get_obs(self):
        obs = []
        max_proc = max(len(job) for job in self.jobs_data)
        for j in range(self.n_jobs):
            remaining = len(self.jobs_data[j]) - self.job_step[j]
            obs.append(remaining / max(max_proc, 1))
        max_mt = max(self.machine_time) if max(self.machine_time) > 0 else 1
        for m in range(self.n_machines):
            obs.append(self.machine_time[m] / max_mt)
        return np.array(obs, dtype=np.float32)

    def _get_available_jobs(self):
        """返回还有未完成工序的作业列表"""
        return [j for j in range(self.n_jobs) if self.job_step[j] < len(self.jobs_data[j])]

    def step(self, action):
        self.steps += 1
        job = int(action)

        # 获取可用作业
        available = self._get_available_jobs()

        # 如果所有作业都完成了
        if not available:
            self.done = True
            return self._get_obs(), 0.0, self.done, False, {}

        # 如果选了已完成的作业，给惩罚并自动换一个
        if self.job_step[job] >= len(self.jobs_data[job]):
            reward = -0.5
            job = available[0]  # 默认选第一个可用的
        else:
            reward = 0.0

        # 取当前工序
        step_idx = self.job_step[job]
        machine, duration = self.jobs_data[job][step_idx]

        # 开始时间 = max(作业上一步完成时间, 机器空闲时间)
        start = max(self.job_end_time[job], self.machine_time[machine])
        end = start + duration

        # 更新状态
        self.machine_time[machine] = end
        self.job_end_time[job] = end
        self.job_step[job] += 1
        self.time = max(self.time, end)

        # 检查是否全部完成
        available = self._get_available_jobs()

        if not available:
            self.done = True
            reward = -self.time / 100.0 + 1.0
        else:
            # 中间奖励：选空闲最久的机器上的作业 → 减少等待
            if not self.done:
                idle_time = max(self.machine_time) - min(
                    self.machine_time[m] for m in range(self.n_machines)
                )
                reward = -idle_time / 100.0  # 惩罚机器负载不均
            else:
                self.done = True
                reward = -self.time / 100.0 + 1.0

        # 防止死循环
        if self.steps >= self.max_steps:
            self.done = True
            reward = -self.time / 100.0

        return self._get_obs(), reward, self.done, False, {}

    def get_makespan(self):
        return self.time