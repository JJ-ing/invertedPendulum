# invertedPendulum

- 强化学习作业1倒立摆问题，采用时序差分方法的 *SARSA* 和 *QLearning* 算法解决

- td.py 实现了上述两种算法

- visualize.py 实现训练和测试结果的可视化

- bai_env.py 实现了倒立摆的环境，为了使该环境在 gym 中可用：

  - 将 bai_env.py 拷贝到 gym 安装目录 /gym/gym/envs/classic_control 文件夹中

  - 打开 classic_control 下的 __init__.py 文件，在文件末尾加入语句：

    ```python
    from gym.envs.classic_control.bai_env import BaiEnv
    ```

  - 进入 gym 安装目录 /gym/gym/envs，打开该文件夹下的 __init__.py 文件，添加代码：

    ```python
    register(
        id='daolibai-v0',
        entry_point='gym.envs.classic_control:BaiEnv',
        max_episode_steps=200,	# 可省略
        reward_threshold=100.0,     # 可省略
    )
    ```

- 主要环境
  - WSL2 + PyCharm2021.3.3
  - Ubuntu-20.04 + Python3.8 + Gym0.23.1