from typing import Optional, Union, Tuple

import gym
import numpy as np
from gym.core import ObsType, ActType


class BaiEnv(gym.Env):
    """
    倒立摆环境模型
    """
    def __init__(self, num_a=200, num_a1=200, ts=0.005):
        """
        :param num_a: 角度的划分区间数
        :param num_a1: 角速度的划分区间数
        :param ts: 采样时间
        """
        # 动作空间
        self.actions = [-3, 0, 3]
        # 采样时间
        self.ts = ts
        # 状态
        self.state = (-np.pi, 0)     # 记录的是连续状态值
        self.num_a = num_a
        self.num_a1 = num_a1
        # 最大角加速度
        self.max_a1 = 15 * np.pi
        # Q table
        self.q = np.zeros([3, num_a, num_a1])   # [action, a, a1]

    def set_state(self, state):
        self.state = state

    def get_state(self):
        return self.state

    def normal_a(self, a):
        """
        规范化角度

        :param a: 角度
        :return: 角度 [-pi, pi)
        """
        return (a + np.pi) % (2 * np.pi) - np.pi

    def d_state(self, c_state):
        """
        根据连续的状态得到离散的状态
        :param c_state: (c_a, c_a1)
        :return: (d_a, d_a1)
        """
        c_a, c_a1 = c_state
        # 规范化角度
        c_a = self.normal_a(c_a)
        # 离散化角度
        block = 2 * np.pi / self.num_a
        index_a = int((c_a + np.pi) / block)
        # 离散化角加速度
        block = 2 * self.max_a1 / self.num_a1
        index_a1 = int((c_a1 + self.max_a1) / block)
        if index_a1 == self.num_a1:     # c_a1 == self.max_a1
            index_a1 -= index_a1
        d_a, d_a1 = index_a, index_a1
        return d_a, d_a1

    def c_state(self, d_state):
        """
        根据离散的状态得到连续的状态，以区间中位数表示
        :param d_state: (d_a, d_a1)
        :return: (c_a, c_a1)
        """
        d_a, d_a1 = d_state
        # 连续化角度
        block = 2 * np.pi / self.num_a
        c_a = (block / 2 + block * d_a) - np.pi
        # 连续化角加速度
        block = 2 * self.max_a1 / self.num_a1
        c_a1 = (block / 2 + block * d_a1) - self.max_a1
        return c_a, c_a1

    def rewards(self, action):
        """
        计算即时回报
        :param action: 动作
        :return: 回报
        """
        a, a1 = self.state
        _action = self.actions[action]
        r = -(5 * a*a + 0.1 * a1*a1) - 1 * _action**2
        return r

    def is_terminate(self):
        """
        what is a terminated state

        :return: 是否处于终止态
        """
        error_a, error_a1 = 0.01 * np.pi, 0.005 * np.pi
        target_a, target_a1 = 0.0, 0.0
        a, a1 = self.state
        if abs(target_a - a) <= error_a and abs(target_a1 - a1) <= error_a1:
            return True
        return False

    def my_step(self, action):
        """
        :param action: 动作
        :return: state_next, reward, is_terminal, information
        """
        # 当前状态
        if self.is_terminate():
            return self.state, 0, True, {}
        # 状态转移
        a, a1 = self.state
        _action = self.actions[action]
        a_next = a + self.ts * a1           # 下一个状态的角度
        a_next = self.normal_a(a_next)      # 角度的规范化
        a1_next = a1 + self.ts * self.get_a2(_action)       # 下一个状态的角加速度
        state_next = (a_next, a1_next)
        r = self.rewards(action)
        if a1_next < -self.max_a1 or a1_next > self.max_a1:     # 角速度出界
            return self.state, r, True, {"end": "a1 out of range"}
        self.state = state_next
        return state_next, r, self.is_terminate(), {}       # (下一个状态, 即时回报, 是否终止, 调试信息)

    def my_reset(self):
        # 每次重置都回到起始状态
        self.state = (-np.pi, 0)

    def my_render(self):
        pass

    def get_a2(self, u):
        """
        计算角加速度

        :param u: 电压
        :return: 角加速度 a2
        """
        m = 0.055
        g = 9.81
        l = 0.042
        J = 1.91e-4
        b = 3e-6
        K = 0.0536
        R = 9.5
        a, a1 = self.state
        a2 = (m * g * l * np.sin(a) - b * a1 - (K ** 2) * a1 / R + K * u / R) / J
        return a2

    def env_close(self):
        self.env.close()

# if __name__ == '__main__':
#     bai = BaiEnv()
#     print(bai.get_state())
#     bai._reset()
#     print(bai.get_state())
#     print(bai.c_state(bai.get_state()))
#     bai._step(2)
#     print(bai.get_state())
#     for i in range(20):
#         print(bai._step(2))
#     print(bai.get_state())
