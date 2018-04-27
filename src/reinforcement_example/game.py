import numpy as np


class AbstractGame(object):
    def get_state(self):
        raise NotImplementedError("Abstract method")

    def time_step(self, action):
        raise NotImplementedError("Abstract method")


class Game(AbstractGame):
    def __init__(self):
        self.maxheight = 100
        self.mean_hole_height = 0.1 * self.maxheight
        self.min_time_to_hole = 20
        self.fall_rate = 5
        self.climb_rate = 3
        self.mean_time_to_hole = 2 * self.min_time_to_hole
        self.alive = True
        self.height = np.random.rand() * self.maxheight
        self._next_hole()

    def _next_hole(self):
        self.time_to_hole = self.min_time_to_hole + \
                            (self.mean_time_to_hole - self.min_time_to_hole) * np.random.exponential()
        hole_height = self.mean_hole_height * np.random.exponential()
        hole_height /= (1 + hole_height / self.maxheight)
        self.hole_bottom = np.random.rand() * (self.maxheight - hole_height)
        self.hole_top = self.hole_bottom + hole_height

    def get_state(self):
        return np.array([self.height, self.hole_bottom, self.hole_top, self.time_to_hole])

    def time_step(self, action):
        if not self.alive:
            return 0
        dt = np.random.exponential()
        if dt > self.time_to_hole:
            dt = self.time_to_hole
        assert action == 1 or action == 0
        if action == 1:
            self.height += self.climb_rate * dt
        else:
            self.height -= self.fall_rate * dt
        self.time_to_hole -= dt
        if self.time_to_hole <= 0:
            if self.height < self.hole_bottom or self.height > self.hole_top:
                self.alive = False
            else:
                self._next_hole()
        return dt


if __name__ == "__main__":
    # Play the game
    game = Game()
    while True:
        print(game.get_state())
        action = int(input())
        game.time_step(action)
        if not game.alive:
            print("DEAD!")
            break