from gym_miniworld.envs import Maze, DEFAULT_PARAMS


class MiniMaze(Maze):
    def __init__(self, num_rows=8, num_cols=8, forward_step=0.7, turn_step=45, max_steps=1000):

        # Parameters for larger movement steps, fast stepping
        params = DEFAULT_PARAMS.no_random()
        params.set('forward_step', forward_step)
        params.set('turn_step', turn_step)

        max_steps = max_steps

        super().__init__(
            num_rows=num_rows,
            num_cols=num_cols,
            params=params,
            max_episode_steps=max_steps,
            domain_rand=False
        )
