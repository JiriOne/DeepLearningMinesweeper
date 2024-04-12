import numpy as np

import Minesweeper
from NN import Agent
import copy
import matplotlib.pyplot as plt


def main():
    # initialize the board
    w, h = 5, 5

    n_games = 1000

    # initialize the agent
    agent = Agent(gamma=0.99, epsilon=0.1, batch_size=64, n_actions=w * h, eps_end=0, input_dims=[w * h], lr=0.04,
                  eps_dec=(1 / n_games), max_mem_size=100000)

    # load the model
    iteration = input("Enter the iteration number: ")
    folder = "./Model_3Bombs_5x5_batch64_lr0.001_mem100000"
    agent = agent.load_model(name=f"{folder}/model_{iteration}.pt")

    _, _, _, n_action_list, wins = Minesweeper.play_games(n_games, w, h, agent,
                                                          update_agent=False,
                                                          n_bombs_low=3,
                                                          n_bombs_high=4)

    # print final wr
    print(f"Final win ratio: {wins / n_games:.2f}")
    print(f"Average number of actions: {np.mean(n_action_list):.2f}")

    # plot trailing reward
    # plt.plot(trailing_wl)
    # plt.savefig("winpercentage_trained.png")


if __name__ == "__main__":
    main()
