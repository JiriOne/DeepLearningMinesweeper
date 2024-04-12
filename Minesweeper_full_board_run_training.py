import numpy as np

import Minesweeper
from NN import Agent
import copy
import matplotlib.pyplot as plt


def main():
    n_games = 1_000_000

    w, h = 10, 10
    obs_w, obs_h = 5, 5
    # initialize the agent
    agent = Agent(
        gamma=0.99,
        epsilon=0.75,
        batch_size=64,
        n_actions=obs_w * obs_h,
        eps_end=0.01,
        input_dims=[obs_w * obs_h],
        lr=0.001,
        eps_dec=(1 / n_games),
        max_mem_size=100000
    )

    win_ratio_list, trailing_wl, trailing_reward, _, _ = Minesweeper.play_games(n_games, w, h, [agent],
                                                                                update_agent=True,
                                                                                n_bombs_low=10,
                                                                                n_bombs_high=11,
                                                                                save_stats=True,
                                                                                obs_w=obs_w,
                                                                                obs_h=obs_h)

    # save final model and stats
    Minesweeper.save_run_stats(win_ratio_list, trailing_wl, trailing_reward)

    agent.save_model()


if __name__ == "__main__":
    main()
