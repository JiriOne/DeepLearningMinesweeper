import numpy as np

import Minesweeper
from NN import Agent
import copy
import matplotlib.pyplot as plt


def main():
    # initialize the board
    w, h = 10,10

    n_games = 5000

    # initialize the agent
    agent1 = Agent(gamma=0.99, epsilon=0.01, batch_size=64, n_actions=w * h, eps_end=0, input_dims=[w * h], lr=0.04,
                  eps_dec=(1 / n_games), max_mem_size=100000)
    agent2 = Agent(gamma=0.99, epsilon=0.01, batch_size=64, n_actions=w * h, eps_end=0, input_dims=[w * h], lr=0.04,
                  eps_dec=(1 / n_games), max_mem_size=100000)
    agent3 = Agent(gamma=0.99, epsilon=0.01, batch_size=64, n_actions=w * h, eps_end=0, input_dims=[w * h], lr=0.04,
                  eps_dec=(1 / n_games), max_mem_size=100000)
    agent4 = Agent(gamma=0.99, epsilon=0.01, batch_size=64, n_actions=w * h, eps_end=0, input_dims=[w * h], lr=0.04,
                  eps_dec=(1 / n_games), max_mem_size=100000)
    agent5 = Agent(gamma=0.99, epsilon=0.01, batch_size=64, n_actions=w * h, eps_end=0, input_dims=[w * h], lr=0.04,
                  eps_dec=(1 / n_games), max_mem_size=100000)

    # load the model
    # iteration = input("Enter the iteration number: ")
    # folder = "./Model_3Bombs_5x5_batch64_lr0.001_mem100000"
    # agent = agent.load_model(name=f"{folder}/model_{iteration}.pt")

    #load 5 agents
    agent1 = agent1.load_model(name="./Model_3Bombs_5x5_batch64_lr0.001_mem100000/model_final.pt")
    agent2 = agent2.load_model(name="./Model_3Bombs_5x5_batch64_lr0.001_mem100000_eps0.75/model.pt")
    agent3 = agent3.load_model(name="./AnotherModel/model_800000.pt")
    agent4 = agent4.load_model(name="./AnotherModel2/model.pt")
    agent5 = agent5.load_model(name="./Anothermodel3/model.pt")

    agents = [agent1, agent2, agent3, agent4, agent5]

    _, _, _, n_action_list, wins = Minesweeper.play_games(n_games, w, h, agents,
                                                          update_agent=False,
                                                          n_bombs_low=5,
                                                          n_bombs_high=6)

    # print final wr
    print(f"Final win ratio: {wins / n_games:.2f}")
    print(f"Average number of actions: {np.mean(n_action_list):.2f}")

    # plot trailing reward
    # plt.plot(trailing_wl)
    # plt.savefig("winpercentage_trained.png")


if __name__ == "__main__":
    main()
