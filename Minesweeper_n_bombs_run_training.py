import numpy as np

import Minesweeper_default_n_bombs as Minesweeper
from NN import Agent
import copy
import matplotlib.pyplot as plt


def main():
    # initialize the board
    w, h = 5, 5

    n_games = 1_000_000

    # initialize the agent
    agent = Agent(gamma=0.99, epsilon=1.0, batch_size=64, n_actions=w * h, eps_end=0.01, input_dims=[w * h], lr=0.001,
                  eps_dec=(1 / n_games), max_mem_size=100000)

    win_ratio_list, trailing_wl, trailing_reward, _, _ = Minesweeper.play_games(n_games, w, h, agent,
                                                                                update_agent=True,
                                                                                n_bombs_low=3,
                                                                                n_bombs_high=4,
                                                                                save_stats=True)

    # save final model and stats
    Minesweeper.save_run_stats(win_ratio_list, trailing_wl, trailing_reward)

    agent.save_model()


if __name__ == "__main__":
    main()
