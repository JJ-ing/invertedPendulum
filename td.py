import gym
import random
import numpy as np


bai = gym.make('daolibai-v0')


def greedy_e(epsilon):
    """
    e贪心策略
    :return: 根据e贪心策略选出的动作
    """
    # 获取q最大的动作
    action = greedy()
    # 计算概率
    p = [0.0 for i in range(len(bai.env.actions))]
    for i in range(len(bai.env.actions)):
        p[i] += epsilon / len(bai.env.actions)
    p[action] += 1 - epsilon
    # 选择动作
    random.seed()
    r = random.random()
    r_sum = 0.0
    for i in range(len(bai.env.actions)):
        r_sum += p[i]
        if r <= r_sum:
            return i
    return len(bai.env.actions) - 1


def greedy():
    """
    贪心策略
    :return: 当前状态下q值最大的动作
    """
    a, a1 = bai.env.d_state(bai.env.get_state())
    q_max = -float('inf')
    index = 0
    for i in range(len(bai.env.actions)):
        if bai.env.q[i][a][a1] > q_max:
            q_max = bai.env.q[i][a][a1]
            index = i
    return index


def TD(num_iter, num_episode, gamma, alpha, _al, epsilon):
    """
    时序差分算法
    :param num_iter: 迭代次数
    :param num_episode: 一幕的最大步数
    :param gamma: 折扣因子
    :param alpha: 学习率
    :param _al: 学习率衰减因子
    :param epsilon: e贪心策略
    :return: None
    """
    r_sum_array = []
    # 已经初始化了Q table
    # 开始迭代
    for i in range(num_iter):
        bai.env.my_reset()
        qTable_last = bai.env.q
        end_out_of_range = False
        count = 0
        r_sum = 0.0     # 该幕的回报
        while (not bai.env.is_terminate()) and count < num_episode:
            state = bai.env.d_state(bai.env.get_state())
            # 先走一步
            a_ge = greedy_e(epsilon)
            state_next, r, is_terminate, information = bai.env.my_step(a_ge)
            state_next = bai.env.d_state(state_next)
            r_sum += r
            # 再看一步
            # ---------------------
            # SARSA
            # a_g = greedy_e(epsilon)
            # QLearning
            a_g = greedy()
            # ----------------------
            # 更新 Q table
            q_old = bai.env.q[a_ge][state[0]][state[1]]
            q_g = bai.env.q[a_g][state_next[0]][state_next[1]]
            q_new = q_old + alpha * (r + gamma * q_g - q_old)
            bai.env.q[a_ge][state[0]][state[1]] = q_new
            if is_terminate and len(information) != 0:
                end_out_of_range = True
                break
            count += 1
        if end_out_of_range:        # 若异常终止，则不更新 Q table
            bai.env.q = qTable_last
            print("iter:%d, episode:%d, last_state:(%spi, %spi)  [a1 out of range]"
                  % (i, count, bai.env.get_state()[0] / np.pi, bai.env.get_state()[1] / np.pi))
        else:
            print("iter:%d, episode:%d, last_state:(%spi, %spi)"
                  % (i, count, bai.env.get_state()[0] / np.pi, bai.env.get_state()[1] / np.pi))
        alpha *= _al
        r_sum_array.append(r_sum)
    print(bai.env.q)
    # np.savez("SARSA_qTable.npz", bai.env.q[0], bai.env.q[1], bai.env.q[2])
    # np.savez("SARSA_r.npz", r_sum_array)
    np.savez("QLearning_qTable.npz", bai.env.q[0], bai.env.q[1], bai.env.q[2])
    np.savez("QLearning_r.npz", r_sum_array)


def test(max_episode):
    bai.env.my_reset()
    data = np.load("SARSA_qTable.npz")
    # data = np.load("QLearning_qTable.npz")
    bai.env.q = np.array([data["arr_0"], data["arr_1"], data["arr_2"]])
    record_sate = []
    record_action = []
    record_reward = []
    count = 0
    while (not bai.env.is_terminate()) and count < max_episode:
        action = greedy_e(0.1)
        # action = greedy()
        state = bai.env.get_state()
        state_next, r, is_terminate, information = bai.env.my_step(action)
        record_sate.append(state)
        record_action.append(bai.env.actions[action])
        record_reward.append(r)
        count += 1
        print("episode:%d, [(%spi, %spi)-%s], (%spi, %spi), %s, %s, %s" % (count,
                                                                           state[0] / np.pi, state[1] / np.pi,
                                                                           bai.env.actions[action],
                                                                           state_next[0] / np.pi, state_next[1] / np.pi,
                                                                           r,
                                                                           is_terminate,
                                                                           information))
        if is_terminate:
            break
    record_sate.append(bai.env.get_state())
    record_action.append(0)
    record_reward.append(0)
    # np.savez("QLearning_test_result.npz", record_sate, record_action, record_reward)
    # np.savez("SARSA_test_result.npz", record_sate, record_action, record_reward)
    return count


# 多次测试SARSA算法, 记录收敛次数
def test_2(n1, n2):
    record_sarsa_count = []
    for i in range(n1):
        count_end = test(n2)
        record_sarsa_count.append(count_end)
    print(record_sarsa_count)
    np.savez("SARSA_test_end_nums", record_sarsa_count)


# 训练
# TD(500, 50000, 0.98, 0.5, 0.999, 0.1)

# 测试
# test(100000)
# test_2(1000, 100000)

