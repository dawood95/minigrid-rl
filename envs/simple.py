from gym_minigrid.minigrid import MiniGridEnv, Grid, Goal
from gym_minigrid.register import register

class SimpleEnv(MiniGridEnv):
    """
    Simple empty environment where the agent starts in the middle,
    target is randomly generated.
    """

    def __init__(self, size=5):
        assert size % 2 != 0, "Size needs to be odd"
        super().__init__(
            grid_size=size,
            max_steps=4*size*size,
            see_through_walls=False
        )

    def _gen_grid(self, width, height):
        # Create empty grid
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)

        # Agent starts in the center
        self.start_pos = (width // 2, height // 2)
        self.start_dir = 0

        # Goal is anywhere but the center
        self.place_obj(Goal())

        # Set mission string
        self.mission = "GO TO GREEN SQUARE"

class SimpleEnv5x5(SimpleEnv):
    def __init__(self):
        super().__init__(size=5)

register(
    id='MiniGrid-SimpleEnv-5x5-v0',
    entry_point='envs:SimpleEnv5x5',
)

class SimpleEnv9x9(SimpleEnv):
    def __init__(self):
        super().__init__(size=9)

register(
    id='MiniGrid-SimpleEnv-9x9-v0',
    entry_point='envs:SimpleEnv9x9',
)

class SimpleEnv15x15(SimpleEnv):
    def __init__(self):
        super().__init__(size=15)

register(
    id='MiniGrid-SimpleEnv-15x15-v0',
    entry_point='envs:SimpleEnv15x15',
)
