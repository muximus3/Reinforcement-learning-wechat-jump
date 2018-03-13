# -*- coding: utf-8 -*-
import os
import sys

_root = os.path.normpath("%s/" % os.path.dirname(os.path.abspath(__file__)))
sys.path.append(_root)

import numpy as np
from collections import deque
import gym
import subprocess
import cv2
import random
import time
from keras import models


class DQN(object):
    """
    第一版尽量end2end, 没有挂掉就reward，输入是一张压缩后的图像
    """

    def __init__(self, env, model, target_model, epsilon, epsilon_decay, epsilon_min, gamma, tau, max_memory, input_shape):
        """
        args:
            env: 环境
            model: 用于学习的网络
            target_model: 用于预测target，不会学习
            epsilon: 贪婪度
            epsilon_decay: 贪婪度递减
            epsilon_min: 最低贪婪度
            gamma: 奖励递减值
            tau:
            max_memory:
        """
        self.input_shape = input_shape
        self.env = env
        self.model = model
        self.target_model = target_model
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_mini = epsilon_min
        self.gamma = gamma
        self.memory = deque(maxlen=max_memory)
        self.tau = tau

    def choose_act(self, state):
        """根据当前状态选择action """
        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon, self.epsilon_mini)
        if np.random.random() < self.epsilon:
            action = self.env.action_space.sample()
            return False, action
        else:
            action = np.argmax(self.model.predict(np.array([state]))[0])
            print('model action: {}'.format(action))
            return True, action

    def remember(self, state, action, reward, new_state, done):
        self.memory.append([state, action, reward, new_state, done])

    def replay(self):
        """类似人类的复盘，总结、提炼之前的经验"""
        batch_size = 15
        if len(self.memory) < batch_size:
            return
        # 从之前的行为中随机选出几个复盘学习。
        samples = random.sample(self.memory, batch_size)
        states = np.zeros(((batch_size,) + self.input_shape))
        targets = np.zeros((batch_size, action_space))
        for i, sample in enumerate(samples):
            state, action, reward, new_state, done = sample
            # 预测之前的state
            target = self.target_model.predict(np.array([state]))
            if done or reward == 0:
                target[0][action] = reward
            else:
                # 没有结束加入下一个状态的预测
                Q_future = max(self.target_model.predict(np.array([new_state]))[0])
                target[0][action] = reward + Q_future * self.gamma
            states[i] = state
            targets[i] = target
            logging.debug('replay action:{}, reward:{}'.format(action, target[0]))
        if len(states) > 0:
            self.model.fit(np.asarray(states), np.asarray(targets), epochs=1, verbose=0)

    def target_train(self):
        weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        for i in range(len(target_weights)):
            target_weights[i] = weights[i] * self.tau + target_weights[i] * (1 - self.tau)
        self.target_model.set_weights(target_weights)

    def predict(self):
        model = models.load_model('saved_model.h5')
        x = env.feature_trick()
        y = model.predict(np.array([x]))
        return np.argmax(y[0])


class WechatJumpEnv(gym.Env):
    """
    在跳一跳中action_space为按压时间
    """

    def __init__(self, action_space, input_shape):
        self.input_shape = input_shape
        self.action_space = Int(action_space)
        self.model = models.load_model('mnist2.h5')

    def step(self, action):
        cur_score = self.get_score()
        self.jump(action, np.random.randint(100, 300), np.random.randint(1000, 1100))
        # 再等一会看是否挂掉
        time.sleep(2 + action / 10)
        # 记录新的状态
        score = self.get_score()
        reward = max(0, score - cur_score)
        if score == 22 or score == 0:
            self.reset()
            reward = 0
            end = True
        else:
            end = False
        new_state = self.feature_trick()
        logging.debug('action: {}, reward:{}, end:{}'.format(action, reward, end))
        return new_state, reward, end

    def reset(self):
        os.system('adb shell input touchscreen swipe  550 1590 550 1590 10')
        time.sleep(0.5)
        # self.jump(7, 100, 100)
        pass

    def render(self, mode='human'):
        pass

    def jump(self, press_time, left, top):
        press_time *= 100
        os.system('adb shell input touchscreen swipe {} {} {} {} {}'.format(left, top, left, top, press_time))

    # def get_sates(self):
    #     process = subprocess.Popen('adb shell screencap -p', shell=True, stdout=subprocess.PIPE)
    #     img_np = process.stdout.read()
    #     img_np = np.fromstring(img_np, np.uint8)
    #     img_np = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
    #     img_np = cv2.cvtColor(cv2.resize(img_np, (input_shape[0], input_shape[1]), interpolation=cv2.INTER_CUBIC), cv2.COLOR_BGR2GRAY)
    #     img_np = cv2.Canny(img_np, 1, 50)
    #     return img_np[60:160, :].reshape((input_shape[1], input_shape[0], input_shape[2]))

    def match_end(self):
        # 检测分数栏的像素，如果为0或者255表示没有在游戏中
        process = subprocess.Popen('adb shell screencap -p', shell=True, stdout=subprocess.PIPE)
        img_np = process.stdout.read()
        img_np = np.fromstring(img_np, np.uint8)
        img_np = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
        img_np = cv2.cvtColor(cv2.resize(img_np, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC), cv2.COLOR_BGR2GRAY)
        _, img = cv2.threshold(img_np, 127, 255, cv2.THRESH_BINARY_INV)
        img_left = img[100:150, 50:100]
        return np.mean(img_left) == 0 or np.mean(img_left) == 255

    def get_score(self):
        # 用额外训练的mnist模型来识别数字
        process = subprocess.Popen('adb shell screencap -p', shell=True, stdout=subprocess.PIPE)
        img_np = process.stdout.read()
        img_np = np.fromstring(img_np, np.uint8)
        img_np = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
        img_np = cv2.cvtColor(cv2.resize(img_np, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC), cv2.COLOR_BGR2GRAY)
        _, img = cv2.threshold(img_np, 127, 255, cv2.THRESH_BINARY_INV)
        img_left = img[100:150, 50:100]
        img_right = img[148:150, 98:100]
        if np.mean(img_left) == 0 or np.mean(img_left) == 255:
            return 0
        datas_right = cv2.resize(img_right, (28, 28), interpolation=cv2.INTER_CUBIC)
        datas_left = cv2.resize(img_left, (28, 28), interpolation=cv2.INTER_CUBIC)
        datas_left = datas_left.reshape((1, 28, 28, 1))
        datas_right = datas_right.reshape((1, 28, 28, 1))
        datas_pred = np.vstack((datas_left, datas_right))
        a = self.model.predict(datas_pred)
        left = np.argmax(a[0]) if a[0][np.argmax(a[0])] > 0.7 else 0
        right = np.argmax(a[1]) if a[1][np.argmax(a[0])] > 0.7 else -1
        return left * 10 + right if right != -1 else left

    def feature_trick(self):
        # 处理图片，只保留关键特征，即跳跃起点终点
        process = subprocess.Popen('adb shell screencap -p', shell=True, stdout=subprocess.PIPE)
        img_np = process.stdout.read()
        img_np = np.fromstring(img_np, np.uint8)
        img_np = cv2.imdecode(img_np, cv2.IMREAD_GRAYSCALE)
        img_np = cv2.GaussianBlur(img_np, (5, 5), 0)
        img_np = cv2.Canny(img_np, 1, 10)
        image = img_np[650:1350, :]
        res1 = cv2.matchTemplate(image, cv2.Canny(cv2.GaussianBlur(cv2.imread('temp_player.jpg', cv2.IMREAD_GRAYSCALE), (5, 5), 0), 1, 10), cv2.TM_CCOEFF_NORMED)
        min_val1, max_val1, min_loc1, max_loc1 = cv2.minMaxLoc(res1)
        center1_loc = (max_loc1[0] + 39, max_loc1[1] + 189)
        empty_img = np.zeros(image.shape)
        # 消去小跳棋轮廓对边缘检测结果的干扰
        for k in range(max_loc1[1] - 10, max_loc1[1] + 189):
            for b in range(max_loc1[0] - 10, max_loc1[0] + 100):
                image[k][b] = 0
        cv2.circle(empty_img, center1_loc, 4, 255, 4)

        y_top = np.nonzero(image)[0][0]
        # x 中心点就是这几个像素中心点
        x_center = int(np.mean(np.nonzero(image[y_top])))
        y_bottom = y_top + 100
        for row in range(y_bottom, image.shape[0]):
            if image[row, x_center] != 0:
                y_bottom = row
                break
        center = x_center, (y_top + y_bottom) // 2
        cv2.circle(empty_img, center, 4, 255, 4)
        if center1_loc[0] > center[0]:
            empty_img = empty_img[::-1]
        empty_img = cv2.resize(empty_img, (self.input_shape[0], self.input_shape[1]), interpolation=cv2.INTER_CUBIC)
        return np.rot90(empty_img.reshape((self.input_shape[1], self.input_shape[0], self.input_shape[2])))

    def distance(self):
        # 计算需要跳跃的距离
        process = subprocess.Popen('adb shell screencap -p', shell=True, stdout=subprocess.PIPE)
        img_np = process.stdout.read()
        img_np = np.fromstring(img_np, np.uint8)
        img_np = cv2.imdecode(img_np, cv2.IMREAD_GRAYSCALE)

        img_np = cv2.GaussianBlur(img_np, (5, 5), 0)
        img_np = cv2.Canny(img_np, 1, 10)
        image = img_np[650:1350, :]
        res1 = cv2.matchTemplate(image, cv2.Canny(cv2.GaussianBlur(cv2.imread('temp_player.jpg', cv2.IMREAD_GRAYSCALE), (5, 5), 0), 1, 10), cv2.TM_CCOEFF_NORMED)
        min_val1, max_val1, min_loc1, max_loc1 = cv2.minMaxLoc(res1)
        center1_loc = (max_loc1[0] + 39, max_loc1[1] + 189)
        # 消去小跳棋轮廓对边缘检测结果的干扰
        for k in range(max_loc1[1] - 10, max_loc1[1] + 189):
            for b in range(max_loc1[0] - 10, max_loc1[0] + 100):
                image[k][b] = 0
        cv2.circle(image, center1_loc, 10, 127, 10)

        y_top = np.nonzero(image)[0][0]
        # x 中心点就是这几个像素中心点
        x_center = int(np.mean(np.nonzero(image[y_top])))
        y_bottom = y_top + 100
        for row in range(y_bottom, image.shape[0]):
            if image[row, x_center] != 0:
                y_bottom = row
                break
        center = x_center, (y_top + y_bottom) // 2
        cv2.circle(image, center, 10, 127, 10)
        return np.linalg.norm((np.array(center) - np.array(center1_loc)))


class Int(gym.Space):
    """
    依据gym规范做的action_space包装类
    """

    def __init__(self, n):
        self.n = n
        gym.Space.__init__(self, (), np.int64)

    def sample(self):
        return np.random.randint(1, self.n)

    def contains(self, x):
        if isinstance(x, int):
            as_int = x
        elif isinstance(x, (np.generic, np.ndarray)) and (x.dtype.kind in np.typecodes['AllInteger'] and x.shape == ()):
            as_int = int(x)
        else:
            return False
        return as_int >= 0 and as_int < self.n

    def __repr__(self):
        return "Discrete(%d)" % self.n

    def __eq__(self, other):
        return self.n == other.n


if __name__ == '__main__':
    from models import CNNModel
    import logging

    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(name)-2s %(levelname)-5s %(message)s',
                        datefmt='%y-%m-%d %H:%M',
                        filemode='w')
    action_space = 10
    gamma = 0.9
    epsilon = .99
    epsilon_min = 0.01
    epsilon_decay = 0.995
    max_memory = 1000
    tau = .125
    input_dims = (100, 180, 1)
    env = WechatJumpEnv(action_space, input_dims)
    # model = CNNModel((8, 4, 3), (80, 80, 60), input_dims, action_space).build_model()
    model = models.load_model('saved_model.h5')
    target_model = CNNModel((8, 4, 3), (80, 80, 60), input_dims, action_space).build_model()
    dqn_agent = DQN(env, model, target_model, epsilon, epsilon_decay, epsilon_min, gamma, tau, max_memory, input_dims)
    try:
        for eposide in range(30000):
            cur_state = env.feature_trick()
            # 防止没有进入游戏就开始训练
            if env.match_end():
                env.reset()
                continue
            model_pred, action = dqn_agent.choose_act(cur_state)
            new_state, reward, done = env.step(action)
            if done and model_pred:
                cv2.imwrite('fail_states/ep_{}_action_{}.jpg'.format(eposide, action), cur_state)
            dqn_agent.remember(cur_state, action, reward, new_state, done)
            # 核心算法
            dqn_agent.replay()
            dqn_agent.target_train()  # iterates target model
            cur_state = new_state
    except Exception as e:
        logging.exception(e)
        model.save('saved_model.h5')
    # print(dqn_agent.predict())
