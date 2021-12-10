import sys
import os
import random
import time

import tanktrouble.Objects as Objects
import pygame
import pygame.event as GAME_EVENTS
import pygame.locals as GAME_GLOBALS
import pygame.time as GAME_TIME
import numpy as np

import gym
from gym import error, utils
from gym.utils import seeding
from gym.spaces import Discrete, Box
from gym.envs.classic_control import rendering


class TankEnv(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self):

        pygame.init()

        self.play_on = True
        # self.screen_width = 432
        # self.screen_height = 288
        self.screen_width = 600
        self.screen_height = 400
        self.screen = pygame.display.set_mode(
            (self.screen_width, self.screen_height))

        self.world = Objects.world()

        self.greenTank = None
        self.purpleTank = None
        self.generate_tanks()

        # state
        self.initial_obs = self.get_state()

        # action
        # 0 : fire
        # 1 : forward
        # 2 : backward
        # 3 : rotate_left
        # 4 : rotate_right
        self.discrete_actions = [0, 1, 2, 3, 4, 5]
        self.action_space = Discrete(len(self.discrete_actions))
        self.action = np.random.choice(self.discrete_actions)
        self.observation_space = Box(low=0, high=255, shape=(
            self.screen_height, self.screen_width, 3), dtype=np.uint8)
        self.n_step = 0
        self.max_steps = 200
        self.viewer = None

    def redraw(self):
        self.screen.fill((255, 255, 255))

        self.world.drawMap(self.screen)
        if self.greenTank.isWracked or self.purpleTank.isWracked:
            self.generate_tanks(is_restart= True)
            self.greenTank.rotate(self.screen)
            self.purpleTank.rotate(self.screen)

        else:
            self.greenTank.drawTank(self.screen)
            self.purpleTank.drawTank(self.screen)

        # bullets
        for bullet in self.purpleTank.bullets:
            collisionKind = bullet.collision(self.screen)

            if collisionKind == "GREEN TANK COLLISION":
                self.greenTank.isWracked = True
                self.greenTank.wrackTime = GAME_TIME.get_ticks()
                bullet.isExpired = True
                self.purpleTank.bullets.remove(bullet)
            elif collisionKind == "PURPLE TANK COLLISION":
                self.purpleTank.isWracked = True
                self.purpleTank.wrackTime = GAME_TIME.get_ticks()
                bullet.isExpired = True
                self.purpleTank.bullets.remove(bullet)
            bullet.draw(self.screen)
        for bullet in self.greenTank.bullets:
            collisionKind = bullet.collision(self.screen)
            if collisionKind == "GREEN TANK COLLISION":
                self.greenTank.isWracked = True
                self.greenTank.wrackTime = GAME_TIME.get_ticks()
                bullet.isExpired = True
                self.greenTank.bullets.remove(bullet)
            elif collisionKind == "PURPLE TANK COLLISION":
                self.purpleTank.isWracked = True
                self.purpleTank.wrackTime = GAME_TIME.get_ticks()
                bullet.isExpired = True
                self.greenTank.bullets.remove(bullet)
            bullet.draw(self.screen)

        pygame.display.update()

    def generate_tanks(self, is_restart=False):
        r = random.random()
        # 绿在左
        if r < 0.5:
            greenTankPosition = [random.randint(75, 225), random.randint(75, 325)]
            purpleTankPosition = [random.randint(375, 525), random.randint(75, 325)]
        # 紫在右
        else:
            greenTankPosition = [random.randint(375, 525), random.randint(75, 325)]
            purpleTankPosition = [random.randint(75, 225), random.randint(75, 325)]

        if is_restart:
            self.greenTank.restart(greenTankPosition[0], greenTankPosition[1])
            self.purpleTank.restart(purpleTankPosition[0], purpleTankPosition[1])
        else:
            self.greenTank = Objects.tank(greenTankPosition[0], greenTankPosition[1], 'Tanks/greenTank4.png',
                                          self.screen)
            self.purpleTank = Objects.tank(purpleTankPosition[0], purpleTankPosition[1], 'Tanks/candyTank4.png',
                                           self.screen)

    def state(self):
        canvas = np.zeros((self.screen_width, self.screen_height, 3))

    def reset(self):
        self.action = -1
        self.n_step = 0

        self.redraw()

        observation = self.get_state()

        return observation

    def step(self, action):

        self.n_step += 1

        self.play_on = False

        self.screen.fill((255, 255, 255))
        self.world.drawMap(self.screen)
        # let green action first
        if action == 0:
            # fire
            self.greenTank.fire()
        elif action == 1:
            # forward
            self.greenTank.move_forward(self.screen)
        elif action == 2:
            # backward
            self.greenTank.move_backward(self.screen)
        elif action == 3:
            # rotate left
            self.greenTank.rotate_left(self.screen)
        elif action == 4:
            # rotate right
            self.greenTank.rotate_right(self.screen)
        elif action == 5:
            pass

        self.greenTank.drawTank(self.screen)
        self.purpleTank.drawTank(self.screen)
        # bullets
        for bullet in self.purpleTank.bullets:
            collisionKind = bullet.collision(self.screen)

            if collisionKind == "GREEN TANK COLLISION":
                self.greenTank.isWracked = True
                self.greenTank.wrackTime = GAME_TIME.get_ticks()
                bullet.isExpired = True
                self.purpleTank.bullets.remove(bullet)
            elif collisionKind == "PURPLE TANK COLLISION":
                self.purpleTank.isWracked = True
                self.purpleTank.wrackTime = GAME_TIME.get_ticks()
                bullet.isExpired = True
                self.purpleTank.bullets.remove(bullet)
            bullet.draw(self.screen)
        for bullet in self.greenTank.bullets:
            collisionKind = bullet.collision(self.screen)
            if collisionKind == "GREEN TANK COLLISION":
                self.greenTank.isWracked = True
                self.greenTank.wrackTime = GAME_TIME.get_ticks()
                bullet.isExpired = True
                self.greenTank.bullets.remove(bullet)
            elif collisionKind == "PURPLE TANK COLLISION":
                self.purpleTank.isWracked = True
                self.purpleTank.wrackTime = GAME_TIME.get_ticks()
                bullet.isExpired = True
                self.greenTank.bullets.remove(bullet)
            bullet.draw(self.screen)

        pygame.display.update()

        info = {}
        observation = self.get_state()

        # reward
        # 胜利奖励大
        if self.purpleTank.isWracked:
            done = True
            reward = 100
            self.reset()
        # 自己摧毁惩罚要大
        elif self.greenTank.isWracked:
            done = True
            reward = -100
            self.reset()
        # 正常状态 惩罚要小
        else:
            done = False
            reward = 0

        info["green_bullets"] = self.greenTank.bullets

        return observation, reward, done, info

    def sample_action(self):
        return np.random.choice(self.discrete_actions)

    def get_state(self):
        state = np.flip(np.rot90(pygame.surfarray.array3d(
            pygame.display.get_surface()).astype(np.uint8)), axis=0).copy()
        return state

    def render(self, mode='rgb_array', close=False):
        img = self.get_state()
        if mode == 'human':
            if self.viewer is None:
                self.viewer = rendering.SimpleImageViewer()
            self.viewer.imshow(img)
        elif mode == 'rgb_array':
            return img

    def play(self):
        running = True

        while running:
            # pygame.time.delay(50)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                self.keys = pygame.key.get_pressed()
                num_keys = 0
                for key in self.keys:
                    num_keys += 1 if key == 1 else 0

                if event.type == pygame.KEYDOWN and num_keys == 1 and self.play_on:
                    if self.keys[pygame.K_q]:
                        running = False
                        self.reset()
                    elif self.keys[pygame.K_r]:
                        new_ob = self.reset()
                    elif self.keys[pygame.K_e]:
                        self.action = 0
                    elif self.keys[pygame.K_w]:
                        self.action = 1
                    elif self.keys[pygame.K_s]:
                        self.action = 2
                    elif self.keys[pygame.K_a]:
                        self.action = 3
                    elif self.keys[pygame.K_d]:
                        self.action = 4
                    else:
                        self.action = -1
                    if self.action != -1:
                        observation, reward, done, _ = self.step(
                            self.action)

                elif event.type == pygame.KEYUP:
                    if not (self.keys[pygame.K_LEFT] and self.keys[pygame.K_RIGHT] and self.keys[pygame.K_UP] and
                            self.keys[pygame.K_DOWN]):
                        self.play_on = True

            self.redraw()


if __name__ == "__main__":
    tank = TankEnv()
    tank.play()
