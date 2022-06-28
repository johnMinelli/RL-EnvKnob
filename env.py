import os

import numpy as np
import pygame, random
from gym import Env
from skimage.transform import resize
from matplotlib import pyplot as plt

RELATIVE_PATH = "."
skier_images = [RELATIVE_PATH+"/resources/skier_down.png", RELATIVE_PATH+"/resources/skier_right1.png", RELATIVE_PATH+"/resources/skier_right2.png", RELATIVE_PATH+"/resources/skier_left2.png", RELATIVE_PATH+"/resources/skier_left1.png"]
STATE_W = 64  # Atari is 160x192
STATE_H = 64
CNN_STATE_W = 224
CNN_STATE_H = 224
LAYERS_ENTITIES = 3
SCREEN_W = 640
SCREEN_H = 640
SCREEN_MARGIN = 20
MAX_STEPS = 2048
MAX_SPEED = 6


class SkierClass(pygame.sprite.Sprite):

    def __init__(self):
        # Create a skier
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.image.load(RELATIVE_PATH+"/resources/skier_down.png")
        self.rect = self.image.get_rect()
        self.rect.center = [SCREEN_W/2, 100]
        self.angle = 0

    def turn(self, direction):
        # Skier turning method
        self.angle = direction  # self.angle +
        if self.angle < -2:
            self.angle = -2

        if self.angle > 2:
            self.angle = 2

        center = self.rect.center
        self.image = pygame.image.load(skier_images[self.angle])
        self.rect = self.image.get_rect()
        self.rect.center = center
        speed = [self.angle, MAX_SPEED - abs(self.angle) * 2]
        return speed


    def move(self, speed):
        # Skier movement method
        min_x = SCREEN_MARGIN
        self.rect.centerx = self.rect.centerx + speed[0]
        if self.rect.centerx < min_x:
            self.rect.centerx = min_x
        max_x = SCREEN_W-SCREEN_MARGIN
        if self.rect.centerx > max_x:
            self.rect.centerx = max_x


class ObstacleClass(pygame.sprite.Sprite):

    def __init__(self, image_file, location, type):
        pygame.sprite.Sprite.__init__(self)
        # Create trees and flags
        self.image_file = image_file
        self.image = pygame.image.load(image_file)
        self.rect = self.image.get_rect()
        self.rect.center = location
        self.type = type
        self.passed = False

    def update(self, speed):
        self.rect.centery -= speed[1]  # The screen scrolls up
        if self.rect.centery < -32:
            self.kill()  # Remove obstacles rolling down from the top of the screen


class Skiing(Env):
    def __init__(self, opt, generate_map_fn=None):
        super(Skiing, self).__init__()
        self.opt = opt
        self.step_reward = opt.step_reward
        self.generator = generate_map_fn
        self.obs_w = CNN_STATE_W if opt.cnn>0 else STATE_W
        self.obs_h = CNN_STATE_H if opt.cnn>0 else STATE_H
        self.action_space = 5
        self.steps = 0


        self.skier = None
        self.obstacles = None

        # os.environ["SDL_VIDEODRIVER"] = "dummy"
        pygame.init()
        self.screen = pygame.display.set_mode([SCREEN_W, SCREEN_H])
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 50)


    def set_map_generator(self, generate_map_fn = None):
        self.generator = generate_map_fn

    def create_map(self):
        # Create a window with trees and flags
        locations = []

        if self.generator is None:
            # displace randomly the elements
            for i in range(10):
                row = random.randint(0, 9)
                col = random.randint(0, 9)
                location = [col * 64 + SCREEN_MARGIN, row * 64 + SCREEN_MARGIN + SCREEN_W]

                if not (location in locations):
                    locations.append(location)
                    type = np.random.choice(["tree", "flag"], p=[0.3, 0.7])
                    img = RELATIVE_PATH+"/resources/skier_tree.png" if type == "tree" else RELATIVE_PATH+"/resources/skier_flag.png"
                    obstacle = ObstacleClass(img, location, type)
                    self.obstacles.add(obstacle)
        else:
            # let the agent decide about tree's positions
            new_obstacles = self.generator()
            for row, col in new_obstacles:
                location = [col * 64 + SCREEN_MARGIN, row * 64 + SCREEN_MARGIN + SCREEN_W]
                if not (location in locations):
                    locations.append(location)
                    type = random.choice(["flag"])
                    img = RELATIVE_PATH+"/resources/skier_tree.png" if type == "tree" else RELATIVE_PATH+"/resources/skier_flag.png"
                    obstacle = ObstacleClass(img, location, type)
                    self.obstacles.add(obstacle)

    def reset(self):
        self._destroy_sprites()
        self.skier = SkierClass()
        self.obstacles = pygame.sprite.Group()
        self.speed = [0, MAX_SPEED]
        self.map_position = 0
        self.score_points = 0
        self.steps = 0
        self.auxiliary_input = random.choice([-1,-0.5,0.5,1])

        # Preparation screen
        self.create_map()
        return self.render()

    def step(self, action: int, skip: int):
        # Move skier
        # if action < 0:
        #     self.speed = self.skier.turn(-1)
        # if action > 0:
        #     self.speed = self.skier.turn(1)
        self.steps += 1
        reward_over_skipped_steps = self.step_reward
        for _ in range(skip):
            self.speed = self.skier.turn(action-2)  # int(self.action_space/2)
            self.skier.move(self.speed)
            # Scroll scene
            self.map_position += self.speed[1]
            # Create a new scene progression
            if self.map_position >= SCREEN_H:
                self.create_map()
                self.map_position = 0

            # Detect whether it touches trees or small flags
            hit = pygame.sprite.spritecollide(self.skier, self.obstacles, False)
            step_reward = 0
            if hit:
                if hit[0].type == "tree" and not hit[0].passed:
                    self.score_points -= 50
                    step_reward = -2
                    self.skier.image = pygame.image.load(RELATIVE_PATH+"/resources/skier_crash.png")
                    pygame.time.delay(50)
                    self.skier.iamge = pygame.image.load(RELATIVE_PATH+"/resources/skier_down.png")
                    self.skier.angle = 0
                    # self.speed = [0, MAX_SPEED]
                    hit[0].passed = True
                elif hit[0].type == "flag" and not hit[0].passed:
                    self.score_points += 25
                    step_reward = 1
                    hit[0].passed = True
                    # hit[0].kill()

            self.obstacles.update(speed=self.speed)
            # cumulate total_reward
            reward_over_skipped_steps += step_reward
            # Check end of episode
            done = self.steps >= MAX_STEPS
            if done:
                break

        self.state = self.render()

        return self.state, reward_over_skipped_steps, done, {}

    def render(self, mode: str = "human"):
        if mode == "human":
            """Build observation"""
            self.clock.tick(30)  # Graphics are updated 30 times per second
            # Show score
            score_text = self.font.render("Score:" + str(self.score_points), 1, (0, 0, 0))
            difficulty_text = self.font.render("Difficulty:" + str((self.auxiliary_input+1)/2), 1, (0, 0, 0))
            # Redraw the picture
            self.screen.fill([255, 255, 255])
            self.obstacles.draw(self.screen)
            self.screen.blit(self.skier.image, self.skier.rect)
            self.screen.blit(score_text, [10, 10])
            self.screen.blit(difficulty_text, [10, 45])
            pygame.display.flip()

        return self._create_observation()

    def _create_observation(self):
        if self.opt.cnn>0:
            scaled_screen = pygame.transform.smoothscale(self.screen, (self.obs_w, self.obs_h))
            return rgb2gray(np.transpose(np.array(pygame.surfarray.pixels3d(scaled_screen), dtype=np.float32), axes=(1, 0, 2)))/255.
        else:
            observation = np.zeros((SCREEN_W, SCREEN_H, LAYERS_ENTITIES), dtype=np.float32)
            # mid = int(self.obs_w/2)
            # max_score = abs(WIN_SCORE if self.score_points > 0 else LOOSE_SCORE)
            # end = (mid if self.score_points>0 else -mid)+int(abs(self.score_points) * mid / max_score)

            observation[max(0, self.skier.rect.top):self.skier.rect.bottom,
            max(0, self.skier.rect.left):self.skier.rect.right, 0] = 1
            for obstacle in self.obstacles:
                observation[max(0, obstacle.rect.top):obstacle.rect.bottom,
                max(0, obstacle.rect.left):obstacle.rect.right, 1 if obstacle.type == 'tree' else 2] = 1
            scaled_obs = resize(observation, (self.obs_w, self.obs_h), mode='reflect', anti_aliasing=True)
            # add score interface
            # scaled_obs[1, min(mid, end):max(mid, end), 0] = 1
            # add time interface
            scaled_obs[0, 0:int(self.steps * self.obs_w / MAX_STEPS), 0] = 1
            return np.transpose(scaled_obs, axes=(2,0,1)).ravel()



    def close(self):
        self._destroy_sprites()
        if self.screen is not None:
            pygame.display.quit()
            self.isopen = False
            pygame.quit()

    def _destroy_sprites(self):
        if self.skier is not None:
            self.skier.kill()
            [obs.kill() for obs in self.obstacles]


def rgb2gray(rgb: np.ndarray) -> np.ndarray:
    """
    Converts an rgb image array to a grey image array.

    :param rgb: the rgb image array.
    :return: the converted array.
    """
    return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)


# class OneHotEncoding(gym.Wrapper):
#     def __init__(self, game, name, **kwargs):
#         if isinstance(game, str):
#             self.env = gym.make(game)
#         else:
#             self.env = game
#         get_pcgrl_env(self.env).adjust_param(**kwargs)
#         gym.Wrapper.__init__(self, self.env)
# 
#         assert name in self.env.observation_space.spaces.keys(), 'This wrapper only works for representations thave have a {} key'.format(name)
#         self.name = name
# 
#         self.observation_space = gym.spaces.Dict({})
#         for (k,s) in self.env.observation_space.spaces.items():
#             self.observation_space.spaces[k] = s
#         new_shape = []
#         shape = self.env.observation_space[self.name].shape
#         self.dim = self.observation_space[self.name].high.max() - self.observation_space[self.name].low.min() + 1
#         for v in shape:
#             new_shape.append(v)
#         new_shape.append(self.dim)
#         self.observation_space.spaces[self.name] = gym.spaces.Box(low=0, high=1, shape=new_shape, dtype=np.uint8)