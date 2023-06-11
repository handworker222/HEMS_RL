# RL_HEMS
Reinforcement Learning for HEMS

This project documents the code used for our RL HEMS experiements using Q Learning and DQN.<br />
In order to run these files successfully, it is necessary to do the following:
1. Download the zip file associated with this project or clone the repository
2. Have Python 3.9 installed on your machine
3. Run `pip install -r requirements.txt` to set up the environment for the files
4. Ensure that the following files are in the same directory as the .py files:
      - home.toml
      - 2019.npy
      - 2020.npy
      - 2021.npy
5. Run `python HEMS_dqn.py` to run the DQN algorithm for HEMS. This file completes 10 runs with explicitly set seeds to gather the results based on the data provided in Step 4. On an  i7-1165G7 @ 2.80GHz machine, this operation takes around 3 hours to complete.
6. Run `python HEMS_q_learning.py` to run the Q Learning algorithm for HEMS. This file completes 10 runs with explicitly set seeds to gather the results based on the data provided in Step 4. On an  i7-1165G7 @ 2.80GHz machine, this operation takes around 1.5 hours to complete.
