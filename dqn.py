# -*- coding: utf-8 -*-
import os
import sys

_root = os.path.normpath("%s/" % os.path.dirname(os.path.abspath(__file__)))
sys.path.append(_root)

import numpy as np
from collections import deque
import gym
from gym import spaces
import subprocess
import cv2
import imutils
import random
import time


class DQN(object):
    """
    第一版尽量end2end, 没有挂掉就reward，输入是一张压缩后的图像
    """

    def __init__(self, env, model, target_model, epsilon, epsilon_decay, epsilon_min, gamma, tau, max_memory):
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
            logging.debug('random action: {}'.format(action))
        else:
            action = np.argmax(self.model.predict(np.array([state]))[0])
            logging.debug('model action: {}'.format(action))
        return action

    def remember(self, state, action, reward, new_state, done):
        self.memory.append([state, action, reward, new_state, done])

    def replay(self):
        """类似人类的复盘，总结、提炼之前的经验"""
        batch_size = 10
        if len(self.memory) < batch_size:
            return
        samples = random.sample(self.memory, batch_size)
        states = []
        targets = []
        for sample in samples:
            state, action, reward, new_state, done = sample
            target = self.target_model.predict(np.array([state]))
            if done:
                target[0][action] = reward
            else:
                Q_future = max(self.target_model.predict(np.array([new_state]))[0])
                target[0][action] = reward + Q_future * self.gamma
            states.append(state)
            targets.append(target[0])
        states = np.asarray(states)
        targets = np.asarray(targets)
        logging.debug('x_shape:{},y_shape:{}'.format(states.shape, targets.shape))
        self.model.fit(states, targets, epochs=1, verbose=0)

    def target_train(self):
        weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        for i in range(len(target_weights)):
            target_weights[i] = weights[i] * self.tau + target_weights[i] * (1 - self.tau)
        self.target_model.set_weights(target_weights)


class WechatJumpEnv(gym.Env):
    """
    在跳一跳中action_space为按压时间
    """

    def __init__(self, action_space):
        self.action_space = Int(action_space)

    def step(self, action):
        self.jump(action, np.random.randint(100, 300), np.random.randint(1000, 1100))
        # 等0.6秒再截图得到
        time.sleep(0.6)
        img = self.screen_shot()
        new_state = self.resize_input(img)
        # 再等一会看是否结束
        time.sleep(np.random.uniform(1.0, 1.5))
        img = self.screen_shot()
        end, btm_loc = self.match_end(img)
        reward = 1 if not end else 0
        logging.debug('env step result end :{}'.format(end))
        return new_state, reward, end

    def reset(self):
        os.system('adb shell input tap 380 1690')

    def render(self, mode='human'):
        pass

    def jump(self, press_time, left, top):
        press_time *= 100
        os.system('adb shell input touchscreen swipe {} {} {} {} {}'.format(left, top, left, top, press_time))

    def screen_shot(self):
        process = subprocess.Popen('adb shell screencap -p', shell=True, stdout=subprocess.PIPE)
        img_np = process.stdout.read()
        img_np = np.fromstring(img_np, np.uint8)
        img_np = cv2.imdecode(img_np, cv2.IMREAD_GRAYSCALE)
        # img_np = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
        # img_np = cv2.cvtColor(cv2.resize(img_np, None, fx=0.2, fy=0.2, interpolation=cv2.INTER_CUBIC), cv2.COLOR_BGR2GRAY)
        # img_np = cv2.GaussianBlur(img_np, (5, 5), 0)
        # img_np = cv2.Canny(img_np, 1, 10)
        return img_np

    def match_end(self, img):
        temp1 = cv2.imread('temp_end.jpg', cv2.IMREAD_GRAYSCALE)
        res1 = cv2.matchTemplate(img, temp1, cv2.TM_CCOEFF_NORMED)
        min_val2, max_val2, min_loc2, max_loc2 = cv2.minMaxLoc(res1)
        return max_val2 > 0.9, max_loc2

    def match_playing(self, img):
        temp1 = cv2.imread('temp_player.jpg', cv2.IMREAD_GRAYSCALE)
        res1 = cv2.matchTemplate(img, temp1, cv2.TM_CCOEFF_NORMED)
        min_val2, max_val2, min_loc2, max_loc2 = cv2.minMaxLoc(res1)
        return max_val2 > 0.75, max_loc2

    def resize_input(self, img):
        img = cv2.resize(img, dsize=(100, 180), interpolation=cv2.INTER_CUBIC)
        img = cv2.Canny(img, 1, 10)
        return img.reshape(100, 180, 1)


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
    action_space = 15
    env = WechatJumpEnv(action_space)
    gamma = 0.9
    epsilon = .95
    epsilon_min = 0.01
    epsilon_decay = 0.995
    learning_rate = 0.01
    max_memory = 3000
    tau = .125
    model = CNNModel((8, 4, 3), (80, 80, 60), (100, 180, 1), action_space).build_model()
    target_model = CNNModel((8, 4, 3), (80, 80, 60), (100, 180, 1), action_space).build_model()
    dqn_agent = DQN(env, model, target_model, epsilon, epsilon_decay, epsilon_min, gamma, tau, max_memory)
    env.reset()
    for eposide in range(3000):
        cur_state = env.resize_input(env.screen_shot())
        action = dqn_agent.choose_act(cur_state)
        new_state, reward, done = env.step(action)
        dqn_agent.remember(cur_state, action, reward, new_state, done)
        dqn_agent.replay()
        dqn_agent.target_train()  # iterates target model
        cur_state = new_state
