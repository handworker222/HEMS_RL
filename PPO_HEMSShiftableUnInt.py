#!/usr/bin/env python
# coding: utf-8




import gym
import os
from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import PPO
from stable_baselines3 import DDPG





import numpy as np
import torch
SEEDLIST = [10129,10353,22373,54284,35519,40046,75647,66957,85409,92451]
DATASEED = 10

import numpy as np
import torch
def set_seeds(seed):
    torch.manual_seed(seed)  # Sets seed for PyTorch RNG
    torch.cuda.manual_seed_all(seed)  # Sets seeds of GPU RNG
    np.random.seed(seed=seed)  # Set seed for NumPy RNG
    random.seed(seed)  # Set seed for random RNG


# Define Classes for Loads




class CriticalLoad():
  def __init__(self, loadName, powerRating):
    self.powerRating = powerRating
    self.loadName = loadName
    self.isOn = 1

  def takeOneTimestep(self):
    return self.powerRating

  def checkStatus(self):
    print(self.loadName)
    print("Load Power Rating:", self.powerRating)
    print("Load Status:", self.isOn)





class AdjustableLoad():
  def __init__(self, loadName, minPowerRating, maxPowerRating, alpha):
    self.minPowerRating = minPowerRating
    self.maxPowerRating = maxPowerRating
    self.powerRating = minPowerRating
    self.alpha = alpha
    self.loadName = loadName

  def setPower(self, powerToBeSet):
    if powerToBeSet > self.maxPowerRating:
      self.powerRating = self.maxPowerRating
    elif powerToBeSet < self.minPowerRating:
      self.powerRating = self.minPowerRating
    else:
      self.powerRating = powerToBeSet

  def takeOneTimestep(self):
    return self.powerRating

  def checkStatus(self):
    print(self.loadName)
    print("Min Power Rating:", self.minPowerRating)
    print("Max Power Rating:", self.maxPowerRating)
    print("Current Power Set:", self.powerRating)
    print("Alpha", self.alpha)





class ShiftableInterruptible():
  def __init__(self, loadName, powerRating, startTime, endTime, requiredHours):
    self.loadName = loadName
    self.powerRating = powerRating
    self.startTime = startTime
    self.endTime = endTime
    self.requiredHours = requiredHours
    self.requiredHoursRemaining = requiredHours
    self.isOn = 0

  def initiateLoad(self):
    if self.requiredHoursRemaining > 0:
      self.isOn = 1 
    else:
      print("Already completed todays load usage")

  def takeOneTimestep(self):
    if self.isOn == 1:
      self.requiredHoursRemaining -= 1
      print(self.loadName, "is running this hour, required hours remaining today:", self.requiredHoursRemaining)
      self.isOn = 0
      return self.powerRating
    else:
      return 0

  def resetDay(self):
    self.requiredHoursRemaining = self.requiredHours

  def checkStatus(self):
    print(self.loadName)
    print("Power Rating:", self.powerRating)
    print("Required Hours per Day:", self.requiredHours)
    print("Required Hours Remaining Today:", self.requiredHoursRemaining)
    print("Time Allotted for Load:", self.startTime, "to", self.endTime)





class ShiftableUninterruptible():
  def __init__(self, loadName, powerRating, startTime, endTime, requiredHours):
    self.loadName = loadName
    self.powerRating = powerRating
    self.startTime = startTime
    self.endTime = endTime
    self.requiredHours = requiredHours
    self.requiredHoursRemaining = requiredHours
    self.isOn = 0

  def initiateLoad(self):
    if self.isOn:
      print(self.loadName, "already on")
    elif self.requiredHoursRemaining > 0:
      self.isOn = 1 
    else:
      print("Already completed todays load usage")
     

  def takeOneTimestep(self):
    if self.isOn == 1:
      self.requiredHoursRemaining -= 1
      print(self.loadName, "is running this hour, required hours remaining today:", self.requiredHoursRemaining)
      if self.requiredHoursRemaining == 0:
        self.isOn = 0
      return self.powerRating
    else:
      return 0
  

  def resetDay(self):
    self.requiredHoursRemaining = self.requiredHours

  def checkStatus(self):
    print(self.loadName)
    print("Power Rating:", self.powerRating)
    print("Required Hours per Day:", self.requiredHours)
    print("Required Hours Remaining Today:", self.requiredHoursRemaining)
    print("Time Allotted for Load:", self.startTime, "to", self.endTime)


# Setting Seeds




import numpy as np
import torch
SEEDLIST = [10129,10353,22373,54284,35519,40046,75647,66957,85409,92451]
DATASEED = 10

def set_seeds(seed):
    torch.manual_seed(seed)  # Sets seed for PyTorch RNG
    torch.cuda.manual_seed_all(seed)  # Sets seeds of GPU RNG
    np.random.seed(seed=seed)  # Set seed for NumPy RNG
    random.seed(seed)  # Set seed for random RNG


# Define Loads



import tomli

with open("./home.toml", "rb") as f:
    toml_dict = tomli.load(f)

for key in toml_dict:
  print(key, toml_dict[key])

cr1 = CriticalLoad(toml_dict["CR"][0]["id"], toml_dict["CR"][0]["P"])
cr2 = CriticalLoad(toml_dict["CR"][1]["id"], toml_dict["CR"][1]["P"])
 
ad1 = AdjustableLoad(toml_dict["AD"][0]["id"],toml_dict["AD"][0]["Pmin"], toml_dict["AD"][0]["Pmax"], toml_dict["AD"][0]["α"])
ad2 = AdjustableLoad(toml_dict["AD"][1]["id"],toml_dict["AD"][1]["Pmin"], toml_dict["AD"][1]["Pmax"], toml_dict["AD"][1]["α"])
ad3 = AdjustableLoad(toml_dict["AD"][2]["id"],toml_dict["AD"][2]["Pmin"], toml_dict["AD"][2]["Pmax"], toml_dict["AD"][2]["α"])

su1 = ShiftableUninterruptible(toml_dict["SU"][0]["id"], toml_dict["SU"][0]["P"], toml_dict["SU"][0]["ts"], toml_dict["SU"][0]["tf"], toml_dict["SU"][0]["L"])
su2 = ShiftableUninterruptible(toml_dict["SU"][1]["id"], toml_dict["SU"][1]["P"], toml_dict["SU"][1]["ts"], toml_dict["SU"][1]["tf"], toml_dict["SU"][1]["L"])

si1 = ShiftableInterruptible(toml_dict["SI"][0]["id"], toml_dict["SI"][0]["P"], toml_dict["SI"][0]["ts"], toml_dict["SI"][0]["tf"], toml_dict["SI"][0]["L"])

loadList = [cr1, cr2, ad1, ad2, ad3, su1, su2, si1]

for load in loadList:
  load.checkStatus()
  print("\n")

"""# Start Loading in Scenarios"""

import numpy as np
data2019 = np.load('./data/learning/scenarios/2019.npy')
data2020 = np.load('./data/learning/scenarios/2020.npy')
data2021 = np.load('./data/learning/scenarios/2021.npy')
data2019and2020 =np.concatenate((data2019, data2020), axis=2)
allData = np.concatenate((data2019and2020, data2021), axis=2)





print("2019", data2019.shape)
print("2020", data2020.shape)
print("2021", data2021.shape)
print("2019 and 2020", data2019and2020.shape)





import random
random.seed(10)
#Extract Validation Set, 10 percent
validation = random.sample(range(365+366), 70)
validationSet = []
trainingSet = []
trainingFullSet = []
pv_validationSet = []
pv_trainingSet = []
pv_trainingFullSet = []
for i in range(365+366):
  trainingFullSet.append(data2019and2020[:,0,i])
  pv_trainingFullSet.append(data2019and2020[:,1,i])
  if i in validation:
    validationSet.append(data2019and2020[:,0,i])
    pv_validationSet.append(data2019and2020[:,1,i])
  else:
    trainingSet.append(data2019and2020[:,0,i])
    pv_trainingSet.append(data2019and2020[:,1,i])

validationData = np.asarray(validationSet)
trainingData = np.asarray(trainingSet)
trainingFullData = np.asarray(trainingFullSet)
print(validationData.shape)
print(trainingData.shape)

testingSet = data2021[:,0,:]
testingData = np.transpose(testingSet)
print(testingData.shape)

allDataTemp = allData[:,0,:]
allDataProcessed = np.transpose(allDataTemp)
print(allDataProcessed.shape)


print('pv:')
pv_validationData = np.asarray(pv_validationSet)
pv_trainingData = np.asarray(pv_trainingSet)
pv_trainingFullData = np.asarray(pv_trainingFullSet)
print(pv_validationData.shape)
print(pv_trainingData.shape)

pv_testingSet = data2021[:,1,:]
pv_testingData = np.transpose(pv_testingSet)
print(pv_testingData.shape)

pv_allDataTemp = allData[:,1,:]
pv_allDataProcessed = np.transpose(pv_allDataTemp)
print(pv_allDataProcessed.shape)





#solar data
solarTestingSet = data2021[:,1,:]
solarTestingData = np.transpose(solarTestingSet)
print(solarTestingData.shape)

solarTrainingSet = data2019and2020[:,1,:]
solarTrainingData = np.transpose(solarTrainingSet)
print(solarTrainingData.shape)





print(np.min(validationData), np.max(validationData))
print(np.min(trainingData), np.max(trainingData))
print(np.min(testingData), np.max(testingData))
print(len(trainingData))





trainingData[1]


# Pipeline Shitable Uninterruptible Load




import gym
from gym import spaces

class CustomEnv(gym.Env):
  """Custom Environment that follows gym interface"""
  # metadata = {'render.modes': ['human']}

  def __init__(self, load, set = None):
    super(CustomEnv, self).__init__()
    self.load = load
    self.solar = None
    # self.price = [0.1061027 , 0.10518237, 0.10348876, 0.1025972 , 0.10004596,
    #    0.10728957, 0.14616401, 0.14855754, 0.13336482, 0.14566902,
    #    0.13109758, 0.13705286, 0.20942042, 0.13099825, 0.12543957,
    #    0.12171211, 0.15030475, 0.15189125, 0.13866458, 0.13601246,
    #    0.13050256, 0.11916902, 0.11551519, 0.11015341]
    self.set = set
    if set == "training":
      self.price = trainingData[0]
      self.pv = pv_trainingData[0]
    elif set == "validation":
      self.price =  validationData[0] 
      self.pv = pv_validationData[0]
    elif set == "testing":
      self.price = testingData[0]
      self.pv = pv_testingData[0]
    else:
      print("No set specified, will use dummy price data")
    self.hour = 0
    self.episodeCount = 0

    self.scenario = None

    #we define the action space: zero or one
    self.action_space = self.action_space = spaces.Discrete(2)
    # we define the state space: price, hoursReqRemaining, hour
    self.observation_space = spaces.Box(low=np.array([0, 0, 0, 0]), high=np.array([999, 999, 999, 999]), dtype=np.float64)

  def checkLegal(self, action):
    # print("hour", str(self.hour))
    # print('load start time', str(self.load.startTime))
    # print('load end time', str(self.load.endTime))
    # print('req hours of work remaining', str(self.load.requiredHoursRemaining))
    # print('is on?', str(self.load.isOn))
    
    #if outside the window, return 0
    if self.hour < self.load.startTime:
      return 0
    elif self.hour >= self.load.endTime:
      return 0

    #if there are no more hours remainig to complete, return 0
    elif self.load.requiredHoursRemaining <= 0:
      return 0
    
    elif self.load.isOn and self.load.requiredHoursRemaining > 0:
      return 1

    elif self.load.requiredHoursRemaining >= self.load.endTime - self.hour:
      return 1

    else:
      return action



    # self.loadName = loadName
    # self.powerRating = powerRating
    # self.startTime = startTime
    # self.endTime = endTime
    # self.requiredHours = requiredHours
    # self.requiredHoursRemaining = requiredHours
    # self.isOn = 0


  def step(self, action):
    legalAction = self.checkLegal(action)
    reward = -1 * (self.price[self.hour] * legalAction * self.load.powerRating)
    #update load state
    self.load.isOn = legalAction
    self.load.requiredHoursRemaining -= legalAction

    #done flagged only when its the last hour of the day
    done = False
    if self.hour >= 23:
      done = True
      self.episodeCount += 1
      
    #info is always set to nothing
    info = {"legalAction": legalAction}
    #update the hour so the next data is corret, but for the last piece of data we dont wanna overflow
    self.hour += 1

    if done:
      self.hour -= 1
    observation = [self.price[self.hour], self.pv[self.hour], self.load.requiredHoursRemaining, self.hour]

    return np.array(observation), reward, done, info
    
  def reset(self):
    if self.set == "training":
      exampleX = random.randint(0,len(trainingData)-1)
      self.price = trainingData[exampleX]
      self.pv = pv_trainingData[exampleX]
    elif self.set == "validation":
      self.price =  validationData[self.episodeCount%len(validationData)] 
      self.pv = pv_validationData[self.episodeCount%len(pv_validationData)] 
    elif self.set == "testing":
      self.price = testingData[self.episodeCount%len(testingData)]
      self.pv = pv_testingData[self.episodeCount%len(pv_testingData)]
    else:
      print("No set specified, will use dummy price data")
    
    self.load.requiredHoursRemaining = self.load.requiredHours
    self.hour = 0
    observation = [self.price[0], self.pv[0], self.load.requiredHoursRemaining, self.hour]
    return np.array(observation)  # reward, done, info can't be included





from stable_baselines3.common.env_checker import check_env

env = CustomEnv(su1, "training")
check_env(env)





# dishWasher = ShiftableUninterruptible(toml_dict["SU"][1]["id"], toml_dict["SU"][1]["P"], toml_dict["SU"][1]["ts"], toml_dict["SU"][1]["tf"], toml_dict["SU"][1]["L"])
# env = CustomEnv(dishWasher)

randomAction = env.action_space.sample()
print("random Action", str(randomAction))
obs, rew, done, info = env.step(randomAction)
print(obs, rew, done, info)


# Dish Washer


load = [su1, su2]

import time
import gym
import os
from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import PPO
from stable_baselines3 import DDPG
from typing import Callable
from stable_baselines3.common.utils import update_learning_rate
from torch.optim.adam import Adam 

# def linear_schedule(initial_value: float) -> Callable[[float], float]:
#     """
#     Linear learning rate schedule.

#     :param initial_value: Initial learning rate.
#     :return: schedule that computes
#       current learning rate depending on remaining progress
#     """
#     def func(progress_remaining) -> float:
#         """
#         Progress will decrease from 1 (beginning) to 0.

#         :param progress_remaining:
#         :return: current learning rate
#         """
#         print("new learning rate:", str(progress_remaining * initial_value))
#         return progress_remaining * initial_value

#     return func

tt1 = 0
tt2 = 0
for i in range(7,9):
  set_seeds(22373)
  t1 = time.time()
  env = CustomEnv(load[i], "training")
  env = Monitor(env, "{}Env".format(load[i].loadName))
  model = PPO("MlpPolicy", env, verbose=0, device='cuda:0')

  validationCostArray = []
  for episode in range(1500):
      #train first on one training episode
      print("training", load[i].loadName, "episode", episode)
      env.reset()
      model.learn(total_timesteps=24)
      validationTotalCost = 0
      #then check cost on validation set
      for validationScenario in range(70):
          envValidation = CustomEnv(load[i], "validation")
          obs = envValidation.reset()
          episodeReward = 0
          while True:
              action, _states = model.predict(obs)
              obs, rewards, dones, info = envValidation.step(action)
              # print(obs, rewards, dones, info)
              episodeReward += rewards
              if dones == True:
                  break
          validationTotalCost += episodeReward
      validationCostArray.append(validationTotalCost/70)
      print("Average Cost on Validation Set:", str(validationTotalCost/70))
      np.savetxt('PPO/PPO_{}_validationCost.txt'.format(load[i].loadName), np.asarray(validationCostArray), delimiter=',')
      model.save("PPO/PPO_{}".format(load[i].loadName))
  
  t2 = time.time()
  
  actionlist = np.zeros((len(testingData), 24))
  for testEpisode in range(len(testingData)):
    envTesting = CustomEnv(load[i], "testing")
    obs = envTesting.reset()
    episodeReward = 0
    h = 0
    while True:
      action, _states = model.predict(obs)
      actionlist[testEpisode, h] = envTesting.checkLegal(action)
      h += 1
      obs, rewards, dones, info = envTesting.step(action)
      # print(obs, rewards, dones, info)
      episodeReward += rewards
      if dones == True:
        break
  np.save("PPO/action/{}.npy".format(load[i].loadName), actionlist)
  
  t3 = time.time()
  tt1 += t2 - t1
  tt2 += t3 - t2

print('Time for all su in PPO for training is ', tt1, 's')
print('Time for all su in PPO for testing is ', tt2, 's')

