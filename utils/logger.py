import time

import numpy as np
import wandb
from progressbar import progressbar


class Logger(object):

    def __init__(self, mode, episodes, batch_size, terminal_print_freq=1, tensorboard=None, wand=None):
        self.episodes = episodes
        self.batch_size = batch_size
        self.print_freq = terminal_print_freq
        self.tensorboard = tensorboard
        self.wandb = wand
        self.total_steps = 0

        s = 10
        e = 1  # episode bar position
        tr = 3  # train bar position
        ts = 6  # valid bar position
        h = 100

        self.progress_bar = None
        self.epoch_bar = None
        self.epoch = None
        self.t = None

        if mode == "valid":
            self.prefix = "Valid"
            self.log_only_at_end = True
            self.writer = Writer(self.t, (0, h - s + ts))
            self.bar_writer = Writer(self.t, (0, h - s + ts + 1))
        elif mode == "train":
            self.prefix = "Train"
            self.log_only_at_end = False
            self.writer = Writer(self.t, (0, h - s + tr))
            self.bar_writer = Writer(self.t, (0, h - s + tr + 1))
            self.progress_bar = progressbar.ProgressBar(maxval=self.episodes, fd=Writer(self.t, (0, h - s + e)))
            [print('') for i in range(2)]
            self.progress_bar.start()

    def set_tensorboard(self, writer):
        self.tensorboard = writer

    def set_wandb(self, writer):
        self.wandb = writer

    def log(self, text):
        self.writer.write(text)

    def episode_start(self, episode):
        self.episode = episode
        self.steps = 0
        self.total_losses = {}
        self.episode_start_time = time.time()
        return self

    def step(self, losses):
        if self.episode is not None:
            self.steps += self.batch_size
            self.total_steps += self.batch_size
            # losses error & metrics
            for single_update_losses in losses:
                for k, v in single_update_losses.items():
                    self.total_losses[k] = (self.total_losses.get(k, 0) + v).numpy()

            avg_losses = np.mean([[np.mean((v).numpy()) for v in l.values()] for l in losses], 0)
            self.log(' * ' + ', '.join(['Avg '+str(k).capitalize()+' : {:.3f}'.format(v) for k, v in zip(self.total_losses.keys(), avg_losses)]))
            self._log_stats_to_dashboards(self.total_steps, {str(k).capitalize():v for k, v in zip(self.total_losses.keys(), avg_losses)})

    def episode_stop(self, total_reward,steps):
        if self.episode is not None:
            episode_time = time.time() - self.episode_start_time
            self.steps = steps
            avg_time = episode_time/self.steps
            avg_reward = total_reward / (self.steps / self.batch_size)
            avg_losses = np.array(list(self.total_losses.values()))/self.steps
            reward_over_time = total_reward / episode_time

            self.log('Ep: %d / %d - Time: %d sec' % (self.episode, self.episodes, episode_time) + '\t' +
                     ' * Avg Reward : {:.3f}'.format(avg_reward) + ', Reward/time : {:.3f}'.format(reward_over_time) +
                     ' - Avg Losses : [' + ', '.join([str(l) for l in avg_losses]) + ']' +
                     ' - Avg Time : {:.3f}'.format(avg_time))
            self._log_stats_to_dashboards(self.episode, {"Avg_reward": avg_reward, "Reward_over_time": reward_over_time, "Avg_time": avg_time})

            if self.progress_bar is not None:
                self.progress_bar.update(self.episode + 1)
                if self.episode + 1 == self.episodes:
                    self.progress_bar.finish()

            return avg_time, avg_losses, avg_reward

    def _log_stats_to_dashboards(self, step, stats):
        for name, value in stats.items():
            namet = self.prefix + "/" + name
            namew = self.prefix + "/" + self.prefix.lower() + "_" + name
            if self.tensorboard is not None:
                self.tensorboard.add_scalar(namet, value, step)
            if self.wandb:
                wandb.log({namew: value}, step)


class Writer(object):
    """Create an object with a write method that writes to a
    specific place on the screen, defined at instantiation.

    This is the glue between blessings and progressbar.
    """

    def __init__(self, t, location):
        """
        Input: location - tuple of ints (x, y), the position
                        of the bar in the terminal
        """
        self.location = location
        self.t = t

    def write(self, string):
        print(string)

    def flush(self):
        return

