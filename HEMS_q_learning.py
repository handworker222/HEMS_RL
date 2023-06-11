# -*- coding: utf-8 -*-
"""Submittable Q Learning.ipynb


# Define Classes for Loads
"""

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

"""# Setting Seeds"""

import numpy as np
import torch
SEEDLIST = [10129,10353,22373,54284,35519,40046,75647,66957,85409,92451]
DATASEED = 10

def set_seeds(seed):
    torch.manual_seed(seed)  # Sets seed for PyTorch RNG
    torch.cuda.manual_seed_all(seed)  # Sets seeds of GPU RNG
    np.random.seed(seed=seed)  # Set seed for NumPy RNG
    random.seed(seed)  # Set seed for random RNG

"""# Start Defining Loads"""



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

from sklearn.preprocessing import KBinsDiscretizer
est = KBinsDiscretizer(n_bins=20, encode='ordinal', strategy='uniform')
est.fit(allDataProcessed)
validationDataDiscrete = est.transform(validationData)
trainingDataDiscrete = est.transform(trainingData)
testingDataDiscrete = est.transform(testingData)
trainingFullDataDiscrete = est.transform(trainingFullData)

pv_est = KBinsDiscretizer(n_bins=20, encode='ordinal', strategy='uniform')
pv_est.fit(pv_allDataProcessed)
pv_validationDataDiscrete = est.transform(pv_validationData)
pv_trainingDataDiscrete = est.transform(pv_trainingData)
pv_testingDataDiscrete = est.transform(pv_testingData)
pv_trainingFullDataDiscrete = est.transform(pv_trainingFullData)

print(validationDataDiscrete.shape)
print(trainingDataDiscrete.shape)
print(testingDataDiscrete.shape)
print(np.min(validationDataDiscrete), np.max(validationDataDiscrete))
print(np.min(trainingDataDiscrete), np.max(trainingDataDiscrete))
print(np.min(testingDataDiscrete), np.max(testingDataDiscrete))

"""# Pipeline for Adjustable Loads
the cost for adjustable loads consist of eletrcity cost: price * power
and also discomfort cost: alpha * (maxPower - currentPower)
"""

import random, time
#setting up world
DISCOUNT_FACTOR = 1 #this is an undiscounted MDP
ACTIONS = [0,1,2,3,4,5,6,7,8,9,10]
PRICEDISCRETIZATION = 20
tt1_ad, tt2_ad, tt1_si, tt2_si, tt1_su, tt2_su = 0, 0, 0, 0, 0, 0


class AdjustableAgent:
    def __init__(self, adjustableLoad):
        self.epsilon = 0.1
        self.learningRate = 0.2
        self.q_table = np.zeros((PRICEDISCRETIZATION, PRICEDISCRETIZATION, 24, len(ACTIONS)))
        self.freq_table = np.zeros((PRICEDISCRETIZATION, PRICEDISCRETIZATION, 24, len(ACTIONS)))
        self.state = [0, 0, 0]
        self.totalRewards = 0.0
        self.minPowerRating = adjustableLoad.minPowerRating
        self.maxPowerRating = adjustableLoad.maxPowerRating
        self.powerRating = adjustableLoad.minPowerRating
        self.alpha = adjustableLoad.alpha
        self.loadName = adjustableLoad.loadName
    
        
    def choose_action(self, state):
        i = int(state[0])
        j = int(state[1])
        k = int(state[2])
        x = random.uniform(0, 1)
        if x <= self.epsilon:
            return np.random.choice(ACTIONS)
        else:
            arr = self.q_table[i, j, k, :]
            best_action = np.where(arr == np.amax(arr)) #there might be ties            
            real_action = np.random.choice(np.asarray(best_action).flatten())
            return real_action

def run_QLEARNING(agent, data, pv_data, dataNonDiscrete):
    hour = 0
    
    agent.totalRewards = 0.0
    auditList = []
    while (hour < 24):
        agent.state = [data[hour], pv_data[hour], hour]
        action = agent.choose_action(agent.state)
        powerSet = action*((agent.maxPowerRating-agent.minPowerRating) /10) +agent.minPowerRating
        reward = -(dataNonDiscrete[hour]*(powerSet) + agent.alpha*(agent.maxPowerRating - powerSet)**2)
        next_state = [data[hour], pv_data[hour], hour]
        agent.totalRewards += reward
        
        #update rule
        target = np.max(agent.q_table[next_state[0], next_state[1], next_state[2], :])

        agent.q_table[agent.state[0],agent.state[1], agent.state[2], action] = (1-agent.learningRate)*(agent.q_table[agent.state[0],agent.state[1], agent.state[2], action]) \
                                                                + (agent.learningRate)*(reward + DISCOUNT_FACTOR*0)

        hour += 1
        auditList.append(action)
    return agent.q_table, agent.totalRewards, auditList

def validate_QLEARNING(agent, data, pv_data, dataNonDiscrete):
    hour = 0
    
    agent.totalRewards = 0.0
    auditList = []
    while (hour < 24):
        agent.state = [data[hour], pv_data[hour], hour]
        action = agent.choose_action(agent.state)
        powerSet = action*((agent.maxPowerRating-agent.minPowerRating) /10) +agent.minPowerRating
        reward = -(dataNonDiscrete[hour]*(powerSet) + agent.alpha*(agent.maxPowerRating - powerSet)**2)
        agent.totalRewards += reward
        
        hour += 1
        auditList.append(action)
    return agent.q_table, agent.totalRewards, auditList

def test_QLEARNING(agent, data, pv_data, dataNonDiscrete):
    hour = 0
    
    agent.totalRewards = 0.0
    auditList = []
    while (hour < 24):
        agent.state = [data[hour], pv_data[hour], hour]
        action = agent.choose_action(agent.state)
        # arr = agent.q_table[agent.state[0],agent.state[1], agent.state[2], :]
        # best_action = np.where(arr == np.amax(arr)) #there might be ties            
        # real_action = min(np.random.choice(np.asarray(best_action).flatten())+2,10) #fix index, tiebreaking
        # action = real_action
        powerSet = action*((agent.maxPowerRating-agent.minPowerRating) /10) +agent.minPowerRating
        reward = -(dataNonDiscrete[hour]*(powerSet) + agent.alpha*(agent.maxPowerRating - powerSet)**2)
        agent.totalRewards += reward
        
        hour += 1
        auditList.append(powerSet)
    return agent.q_table, agent.totalRewards, auditList



def random_NOLEARNING(agent, data, pv_data, dataNonDiscrete):
    hour = 0
    agent.state = [data[0], pv_data[0], hour, 0]
    agent.totalRewards = 0.0
    while (hour < 24):
        action = np.random.choice(ACTIONS)
        powerSet = action*(agent.maxPowerRating/10)
        reward = -(dataNonDiscrete[hour]*(powerSet) + agent.alpha*(agent.maxPowerRating - powerSet)**2)
        next_state = [data[hour], pv_data[hour], hour, action]
        agent.totalRewards += reward
        
        agent.state = next_state
        hour += 1
    return agent.q_table, agent.totalRewards 



def runAdjustableLoad(seedRun, seed, load):
  set_seeds(seed)
  global tt1_ad
  global tt2_ad
  Loadagent = AdjustableAgent(load)
  Loadagent.epsilon = 0.1
  #Training
  convergenceCurve = []
  trainingCurve = []
  q_tableSum = []
  t1_ad = time.time()
  for run in range(1):
    for episode in range(1500):
      print("run", seedRun, "training", load.loadName, "episode", episode)
      exampleX = random.randint(0,len(trainingDataDiscrete)-1)
      qlearning_episode_qtable, qlearning_episode_rewards, auditList = run_QLEARNING(Loadagent, trainingDataDiscrete[exampleX].astype(int), pv_trainingDataDiscrete[exampleX].astype(int), trainingData[exampleX])

      validationReward = 0
      for validationScenario in range(70):
        validationAgent = AdjustableAgent(load)
        validationAgent.epsilon = 0
        validationAgent.q_table = qlearning_episode_qtable
        validation_qlearning_episode_qtable, validation_average_qlearning_episode_rewards, auditList = validate_QLEARNING(validationAgent, validationDataDiscrete[validationScenario].astype(int), pv_validationDataDiscrete[validationScenario].astype(int), validationData[validationScenario])
        validationReward += validation_average_qlearning_episode_rewards
      convergenceCurve.append(validationReward/70)
      trainingCurve.append(qlearning_episode_rewards)
      q_tableSum.append(np.sum(qlearning_episode_qtable))
  t2_ad = time.time()
  TestingResults = []
  TestingAuditList = []
  print("now testing", load.loadName)
  for testEpisode in range(len(testingDataDiscrete)):
    testAgent = AdjustableAgent(load)
    testAgent.epsilon = 0
    testAgent.q_table = qlearning_episode_qtable
    test_qlearning_episode_qtable, test_qlearning_episode_rewards, auditList = test_QLEARNING(testAgent, testingDataDiscrete[testEpisode].astype(int), pv_testingDataDiscrete[testEpisode].astype(int), testingData[testEpisode])
    TestingResults.append(test_qlearning_episode_rewards)
    TestingAuditList.append(auditList)
  t3_ad = time.time()
  tt1_ad += t2_ad - t1_ad
  tt2_ad += t3_ad - t2_ad
  generalisationResults = []
  generalisationAuditList = []
  print("now testing with training set", load.loadName)
  for testEpisode in range(len(trainingFullData)):
    testAgent = AdjustableAgent(load)
    testAgent.epsilon = 0
    testAgent.q_table = qlearning_episode_qtable
    test_qlearning_episode_qtable, test_qlearning_episode_rewards, auditList = test_QLEARNING(testAgent, trainingFullDataDiscrete[testEpisode].astype(int), pv_trainingFullDataDiscrete[testEpisode].astype(int), trainingFullData[testEpisode])
    
    generalisationResults.append(test_qlearning_episode_rewards)
    generalisationAuditList.append(auditList)
  return convergenceCurve, TestingResults, generalisationResults, TestingAuditList, generalisationAuditList


import random
DISCOUNT_FACTOR = 1 #this is an undiscounted MDP
ACTIONSbinary = [0,1]
PRICEDISCRETIZATION = 20

class ShiftableAgent:
    def __init__(self, shiftableLoad):
        self.epsilon = 0.1
        self.learningRate = 0.1
        self.requiredHoursRemaining = shiftableLoad.requiredHours
        self.endTime = shiftableLoad.endTime
        self.startTime = shiftableLoad.startTime
        self.window = shiftableLoad.endTime-shiftableLoad.startTime
        self.q_table = np.zeros((PRICEDISCRETIZATION, PRICEDISCRETIZATION, self.requiredHoursRemaining+1, self.window+1, 24, len(ACTIONSbinary)))
        self.state = [0, 0, 0, 0, 0]
        self.totalRewards = 0.0
        self.powerRating = shiftableLoad.powerRating
        self.startTime = shiftableLoad.startTime
        self.requiredHours = shiftableLoad.requiredHours
        self.requiredHoursRemaining = shiftableLoad.requiredHours
        self.isOn = 0
    
        
    def choose_action(self, state):
        i = int(state[0])
        j = int(state[1])
        k = int(state[2])
        l = int(state[3])
        m = int(state[4])
        x = random.uniform(0, 1)
        interimAction = 0
        if x <= self.epsilon:
            interimAction = np.random.choice(ACTIONSbinary)
        else:
            arr = self.q_table[i, j, k, l, m, :]
            best_action = np.where(arr == np.amax(arr)) #there might be ties            
            interimAction = np.random.choice(np.asarray(best_action).flatten())


        if self.requiredHoursRemaining <= 0:
          return 0

        else:
          if self.isOn == 1:
            return 1

          else:
            if interimAction == 1:
              return interimAction
            #if not, we need to look ahead and check if there is still room for the agent to do its duty by the given time
            else:
              if self.requiredHoursRemaining < l:
                return interimAction
              #there is no choice, you need to turn it on now to finish by the end time
              else:
                return 1    

class randomShiftableAgent:
    def __init__(self, shiftableLoad):
        self.epsilon = 0.1
        self.learningRate = 0.1
        self.requiredHoursRemaining = shiftableLoad.requiredHours
        self.endTime = shiftableLoad.endTime
        self.startTime = shiftableLoad.startTime
        self.window = shiftableLoad.endTime-shiftableLoad.startTime
        self.q_table = np.zeros((PRICEDISCRETIZATION, PRICEDISCRETIZATION, self.requiredHoursRemaining+1, self.window+1, 24, len(ACTIONSbinary)))
        self.state = [0, 0, 0, 0, 0]
        self.totalRewards = 0.0
        self.powerRating = shiftableLoad.powerRating
        self.startTime = shiftableLoad.startTime
        self.requiredHours = shiftableLoad.requiredHours
        self.requiredHoursRemaining = shiftableLoad.requiredHours
        self.isOn = 0
    
        
    def choose_action(self, state):
        i = int(state[0])
        j = int(state[1])
        k = int(state[2])
        l = int(state[3])
        m = int(state[4])
        x = random.uniform(0, 1)
        interimAction = 0
        interimAction = np.random.choice(ACTIONSbinary)

        if self.requiredHoursRemaining <= 0:
          return 0


        else:

          if self.isOn == 1:
            return 1

          else:
            if interimAction == 1:
              return interimAction
            else:
              if self.requiredHoursRemaining < l:
                return interimAction
              #there is no choice, you need to turn it on now to finish by the end time
              else:
                return 1    


def run_ShiftableUnQLEARNING(agent, data, pv_data, dataNonDiscrete):
    hour = 0
    agent.totalRewards = 0.0
    agent.requiredHoursRemaining = agent.requiredHours
    agent.window = agent.endTime - agent.startTime
    agent.state = [data[0], pv_data[0], agent.requiredHoursRemaining, agent.window, hour]
    auditList = []
    while (hour < 24):
        reward = 0
        action = 0
        if hour >= agent.startTime and hour < agent.endTime:
          agent.state = [data[hour], pv_data[hour], agent.requiredHoursRemaining, agent.window, hour]
          action = agent.choose_action(agent.state)
          if action == 1:
            agent.requiredHoursRemaining -= 1
          agent.isOn = action
          agent.window -= 1
          reward = -(dataNonDiscrete[hour]*(action)*agent.powerRating)
          next_hour = 0 if hour==23 else hour+1
          next_state = [data[next_hour], pv_data[next_hour], agent.requiredHoursRemaining, agent.window, next_hour]
          agent.totalRewards += reward
          agent.isOn = action
        
          #update rule
          target = np.max(agent.q_table[next_state[0], next_state[1], next_state[2], next_state[3], next_state[4], :])
          agent.q_table[agent.state[0],agent.state[1], agent.state[2], agent.state[3], agent.state[4], action] = (1-agent.learningRate)*(agent.q_table[agent.state[0],agent.state[1],agent.state[2], agent.state[3], agent.state[4], action]) \
                                                                  + (agent.learningRate)*(reward + DISCOUNT_FACTOR*target)
          agent.state = next_state
        auditList.append(action)
        hour += 1
    return agent.q_table, agent.totalRewards, auditList


def validate_ShiftableUnQLEARNING(agent, data, pv_data, dataNonDiscrete):
    hour = 0
    agent.totalRewards = 0.0
    agent.requiredHoursRemaining = agent.requiredHours
    agent.window = agent.endTime - agent.startTime
    agent.state = [data[0], pv_data[0], agent.requiredHoursRemaining, agent.window, hour]
    auditList = []
    while (hour < 24):
        reward = 0
        action = 0
        if hour >= agent.startTime and hour < agent.endTime:
          agent.state = [data[hour], pv_data[hour], agent.requiredHoursRemaining, agent.window, hour]
          action = agent.choose_action(agent.state)
          if action == 1:
            agent.requiredHoursRemaining -= 1
          agent.isOn = action
          agent.window -= 1

          reward = -(dataNonDiscrete[hour]*(action)*agent.powerRating)
          agent.totalRewards += reward
          agent.isOn = action
        
        auditList.append(action)
        hour += 1
    return agent.q_table, agent.totalRewards, auditList


def test_ShiftableUnQLEARNING(agent, data, pv_data, dataNonDiscrete):
    hour = 0
    agent.totalRewards = 0.0
    agent.requiredHoursRemaining = agent.requiredHours
    agent.window = agent.endTime - agent.startTime
    agent.state = [data[0], pv_data[0], agent.requiredHoursRemaining, agent.window, hour]
    auditList = []
    while (hour < 24):
        reward = 0
        action = 0
        if hour >= agent.startTime and hour < agent.endTime:
          agent.state = [data[hour], pv_data[hour], agent.requiredHoursRemaining, agent.window, hour]
          action = agent.choose_action(agent.state)
          if action == 1:
            agent.requiredHoursRemaining -= 1
          agent.isOn = action
          agent.window -= 1
          reward = -(dataNonDiscrete[hour]*(action)*agent.powerRating)
          agent.totalRewards += reward
          agent.isOn = action
        

        auditList.append(action)
        hour += 1
    return agent.q_table, agent.totalRewards, auditList

def runShiftableUnLoad(seedRun, seed, load):
  global tt1_su
  global tt2_su
  set_seeds(seed)
  loadQagent = ShiftableAgent(load)

  #Training
  convergenceCurve = []
  trainingCurve = []
  qvaluesSum = []
  t1_su = time.time()
  for run in range(1):
    for episode in range(1500):
      print("run", seedRun, "training", load.loadName, "episode", episode)
      loadQagent.epsilon = 0.1
      exampleX = random.randint(0,len(trainingDataDiscrete)-1)
      qlearning_episode_qtable, qlearning_episode_rewards, auditList = run_ShiftableUnQLEARNING(loadQagent, trainingDataDiscrete[exampleX].astype(int), pv_trainingDataDiscrete[exampleX].astype(int), trainingData[exampleX])

      validationReward = 0
      for validationScenario in range(70):
        validationAgent = ShiftableAgent(load)
        validationAgent.q_table = np.copy(qlearning_episode_qtable)
        validationAgent.epsilon = 0
        validation_qlearning_episode_qtable, validation_average_qlearning_episode_rewards, auditList = validate_ShiftableUnQLEARNING(validationAgent, validationDataDiscrete[validationScenario].astype(int), pv_validationDataDiscrete[validationScenario].astype(int), validationData[validationScenario])
        validationReward += validation_average_qlearning_episode_rewards
      convergenceCurve.append(validationReward/70)
      trainingCurve.append(qlearning_episode_rewards)
      qvaluesSum.append(np.sum(qlearning_episode_qtable))

  t2_su = time.time()
  TestingResults = []
  TestingAuditList = []
  print("now testing", load.loadName)
  for testEpisode in range(len(testingDataDiscrete)):
    testAgent = ShiftableAgent(load)
    testAgent.q_table = np.copy(qlearning_episode_qtable)
    testAgent.epsilon = 0
    test_qlearning_episode_qtable, test_qlearning_episode_rewards, auditList = test_ShiftableUnQLEARNING(testAgent, testingDataDiscrete[testEpisode].astype(int), pv_testingDataDiscrete[testEpisode].astype(int), testingData[testEpisode])
    TestingResults.append(test_qlearning_episode_rewards)
    TestingAuditList.append(auditList)
  t3_su = time.time()
  tt1_su += t2_su - t1_su
  tt2_su += t3_su - t2_su
  generalisationResults = []
  generalisationAuditList = []
  print("now testing with training set", load.loadName)
  for testEpisode in range(len(trainingFullData)):
    testAgent = ShiftableAgent(load)
    testAgent.epsilon = 0
    testAgent.q_table = qlearning_episode_qtable
    test_qlearning_episode_qtable, test_qlearning_episode_rewards, auditList = test_ShiftableUnQLEARNING(testAgent, trainingFullDataDiscrete[testEpisode].astype(int), pv_trainingFullDataDiscrete[testEpisode].astype(int), trainingFullData[testEpisode])
    
    generalisationResults.append(test_qlearning_episode_rewards)
    generalisationAuditList.append(auditList)
  return convergenceCurve, TestingResults, generalisationResults, TestingAuditList, generalisationAuditList

import random
DISCOUNT_FACTOR = 1 #this is an undiscounted MDP
ACTIONSbinary = [0,1]
PRICEDISCRETIZATION = 20

class ShiftableInterruptibleAgent:
    def __init__(self, shiftableInterruptibleLoad):
        self.epsilon = 0.1
        self.learningRate = 0.1
        self.requiredHoursRemaining = shiftableInterruptibleLoad.requiredHours
        self.endTime = shiftableInterruptibleLoad.endTime
        self.startTime = shiftableInterruptibleLoad.startTime
        self.window = shiftableInterruptibleLoad.endTime-shiftableInterruptibleLoad.startTime
        self.q_table = np.zeros((PRICEDISCRETIZATION, PRICEDISCRETIZATION, self.requiredHoursRemaining+1, self.window+1, 24, len(ACTIONSbinary)))
        self.state = [0, 0, 0, 0, 0]
        self.totalRewards = 0.0
        self.powerRating = shiftableInterruptibleLoad.powerRating
        self.startTime = shiftableInterruptibleLoad.startTime
        self.requiredHours = shiftableInterruptibleLoad.requiredHours
        self.requiredHoursRemaining = shiftableInterruptibleLoad.requiredHours
        self.isOn = 0
    
        
    def choose_action(self, state):
        i = int(state[0])
        j = int(state[1])
        k = int(state[2])
        l = int(state[3])
        m = int(state[4])
        x = random.uniform(0, 1)
        interimAction = 0
        if x <= self.epsilon:
            interimAction = np.random.choice(ACTIONSbinary)
        else:
            arr = self.q_table[i, j, k, l, m, :]
            best_action = np.where(arr == np.amax(arr)) #there might be ties            
            interimAction = np.random.choice(np.asarray(best_action).flatten())


        if self.requiredHoursRemaining <= 0:
          return 0

        else:
          if interimAction == 1:
            return interimAction
          #if not, we need to look ahead and check if there is still room for the agent to do its duty by the given time
          else:
            if self.requiredHoursRemaining < l:
              return interimAction
            #there is no choice, you need to turn it on now to finish by the end time
            else:
              return 1  
              
class randomShiftableInterruptibleAgent:
    def __init__(self, shiftableInterruptibleLoad):
        self.epsilon = 0.1
        self.learningRate = 0.1
        self.requiredHoursRemaining = shiftableInterruptibleLoad.requiredHours
        self.endTime = shiftableInterruptibleLoad.endTime
        self.startTime = shiftableInterruptibleLoad.startTime
        self.window = shiftableInterruptibleLoad.endTime-shiftableInterruptibleLoad.startTime
        self.q_table = np.zeros((PRICEDISCRETIZATION, PRICEDISCRETIZATION, self.requiredHoursRemaining+1, self.window+1, 24, len(ACTIONSbinary)))
        self.state = [0, 0, 0, 0, 0]
        self.totalRewards = 0.0
        self.powerRating = shiftableInterruptibleLoad.powerRating
        self.startTime = shiftableInterruptibleLoad.startTime
        self.requiredHours = shiftableInterruptibleLoad.requiredHours
        self.requiredHoursRemaining = shiftableInterruptibleLoad.requiredHours
        self.isOn = 0
    
        
    def choose_action(self, state):
        i = int(state[0])
        j = int(state[1])
        k = int(state[2])
        l = int(state[3])
        m = int(state[4])
        x = random.uniform(0, 1)
        interimAction = 0
        interimAction = np.random.choice(ACTIONSbinary)


        if self.requiredHoursRemaining <= 0:
          return 0

        else:
          if interimAction == 1:
            return interimAction
          else:
            if self.requiredHoursRemaining < l:
              return interimAction
            #there is no choice, you need to turn it on now to finish by the end time
            else:
              return 1                 

def run_ShiftableIntQLEARNING(agent, data, pv_data, dataNonDiscrete):
    hour = 0
    agent.totalRewards = 0.0
    agent.requiredHoursRemaining = agent.requiredHours
    agent.window = agent.endTime - agent.startTime
    agent.state = [data[0], pv_data[0], agent.requiredHoursRemaining, agent.window, hour]
    auditList = []
    while (hour < 24):
        reward = 0
        action = 0
        if hour >= agent.startTime and hour < agent.endTime:
          agent.state = [data[hour], pv_data[hour], agent.requiredHoursRemaining, agent.window, hour]
          action = agent.choose_action(agent.state)
          if action == 1:
            agent.requiredHoursRemaining -= 1
          agent.isOn = action
          agent.window -= 1
          reward = -(dataNonDiscrete[hour]*(action)*agent.powerRating)
          next_state = [data[hour+1], pv_data[hour+1], agent.requiredHoursRemaining, agent.window, hour+1]
          agent.totalRewards += reward
        
          #update rule
          target = np.max(agent.q_table[next_state[0], next_state[1], next_state[2], next_state[3], next_state[4], :])
          agent.q_table[agent.state[0],agent.state[1], agent.state[2], agent.state[3], agent.state[4], action] = (1-agent.learningRate)*(agent.q_table[agent.state[0],agent.state[1],agent.state[2], agent.state[3], agent.state[4], action]) \
                                                                  + (agent.learningRate)*(reward + DISCOUNT_FACTOR*0)
          agent.state = next_state
        auditList.append(action)
        hour += 1
    return agent.q_table, agent.totalRewards, auditList


def validate_ShiftableIntQLEARNING(agent, data, pv_data, dataNonDiscrete):
    hour = 0
    agent.totalRewards = 0.0
    agent.requiredHoursRemaining = agent.requiredHours
    agent.window = agent.endTime - agent.startTime
    agent.state = [data[0], pv_data[0], agent.requiredHoursRemaining, agent.window, hour]
    auditList = []
    while (hour < 24):
        reward = 0
        action = 0
        if hour >= agent.startTime and hour < agent.endTime:
          agent.state = [data[hour], pv_data[hour], agent.requiredHoursRemaining, agent.window, hour]
          action = agent.choose_action(agent.state)
          if action == 1:
            agent.requiredHoursRemaining -= 1
          agent.isOn = action
          agent.window -= 1
          reward = -(dataNonDiscrete[hour]*(action)*agent.powerRating)
          next_state = [data[hour+1], pv_data[hour+1], agent.requiredHoursRemaining, agent.window, hour+1]
          agent.totalRewards += reward

          agent.state = next_state
        auditList.append(action)
        hour += 1
    return agent.q_table, agent.totalRewards, auditList

def test_ShiftableIntQLEARNING(agent, data, pv_data, dataNonDiscrete):
    hour = 0
    agent.totalRewards = 0.0
    agent.requiredHoursRemaining = agent.requiredHours
    agent.window = agent.endTime - agent.startTime
    agent.state = [data[0], pv_data[0], agent.requiredHoursRemaining, agent.window, hour]
    auditList = []
    while (hour < 24):
        reward = 0
        action = 0
        if hour >= agent.startTime and hour < agent.endTime:
          agent.state = [data[hour], pv_data[hour], agent.requiredHoursRemaining, agent.window, hour]
          action = agent.choose_action(agent.state)
          if action == 1:
            agent.requiredHoursRemaining -= 1
          agent.isOn = action
          agent.window -= 1
          reward = -(dataNonDiscrete[hour]*(action)*agent.powerRating)
          next_state = [data[hour+1], pv_data[hour+1], agent.requiredHoursRemaining, agent.window, hour+1]
          agent.totalRewards += reward
        
          agent.state = next_state
        auditList.append(action)
        hour += 1
    return agent.q_table, agent.totalRewards, auditList


def runShiftableIntLoad(seedRun, seed, load):
  global tt1_si
  global tt2_si
  set_seeds(seed)
  loadQagent = ShiftableInterruptibleAgent(load)
  #Training
  convergenceCurve = []
  trainingCurve = []
  qvaluesSum = []
  t1_si = time.time()
  for run in range(1):
    for episode in range(1500):
      print("run", seedRun, "training", load.loadName, "episode", episode)
      qlearning_episode_qtable, qlearning_episode_rewards, auditList = run_ShiftableIntQLEARNING(loadQagent, trainingDataDiscrete[episode%len(trainingData)].astype(int), pv_trainingDataDiscrete[episode%len(pv_trainingData)].astype(int), trainingData[episode%len(trainingData)])
      validationReward = 0
      for validationScenario in range(70):
        validationAgent = ShiftableInterruptibleAgent(load)
        validationAgent.q_table = np.copy(qlearning_episode_qtable)
        validationAgent.epsilon = 0
        validation_qlearning_episode_qtable, validation_average_qlearning_episode_rewards, auditList = validate_ShiftableIntQLEARNING(validationAgent, validationDataDiscrete[validationScenario].astype(int), pv_validationDataDiscrete[validationScenario].astype(int), validationData[validationScenario])
        validationReward += validation_average_qlearning_episode_rewards
      convergenceCurve.append(validationReward/70)
      trainingCurve.append(qlearning_episode_rewards)
      qvaluesSum.append(np.sum(qlearning_episode_qtable))
  t2_si = time.time()
  TestingResults = []
  TestingAuditList = []
  print("now testing", load.loadName)
  for testEpisode in range(len(testingDataDiscrete)):
    testAgent = ShiftableInterruptibleAgent(load)
    testAgent.q_table = qlearning_episode_qtable
    testAgent.epsilon = 0
    test_qlearning_episode_qtable, test_qlearning_episode_rewards, auditList = test_ShiftableIntQLEARNING(testAgent, testingDataDiscrete[testEpisode].astype(int), pv_testingDataDiscrete[testEpisode].astype(int), testingData[testEpisode])
    TestingResults.append(test_qlearning_episode_rewards)
    TestingAuditList.append(auditList)
  t3_si = time.time()
  tt1_si += t2_si - t1_si
  tt2_si += t3_si - t2_si
  generalisationResults = []
  generalisationAuditList = []
  print("now testing with training set", load.loadName)
  for testEpisode in range(len(trainingFullData)):
    testAgent = ShiftableAgent(load)
    testAgent.epsilon = 0
    testAgent.q_table = qlearning_episode_qtable
    test_qlearning_episode_qtable, test_qlearning_episode_rewards, auditList = test_ShiftableIntQLEARNING(testAgent, trainingFullDataDiscrete[testEpisode].astype(int), pv_trainingFullDataDiscrete[testEpisode].astype(int), trainingFullData[testEpisode])
    
    generalisationResults.append(test_qlearning_episode_rewards)
    generalisationAuditList.append(auditList)
  return convergenceCurve, TestingResults, generalisationResults, TestingAuditList, generalisationAuditList


def calculateCost(actionListTestad1, actionListTestad2, actionListTestad3,
                  actionListTestsu1, actionListTestsu2,
                  actionListTestsi1,
                  ad1, ad2, ad3, 
                  su1, su2, 
                  si1, 
                  cr1, cr2, 
                  price, solarGeneration):
  
  cost = []
  for i in range(len(price)):
    print("now calculating cost for day", i, "of day", len(price))
    totalElectricCost = 0
    totalDiscomfortCost = 0
    for j in range(24):
      powerUseThisHour = 0
      powerUseThisHour += cr1.powerRating
      powerUseThisHour += cr2.powerRating
      powerUseThisHour += actionListTestad1[i][j]
      powerUseThisHour += actionListTestad2[i][j]
      powerUseThisHour += actionListTestad3[i][j]
      powerUseThisHour += actionListTestsu1[i][j]*su1.powerRating
      powerUseThisHour += actionListTestsu2[i][j]*su2.powerRating
      powerUseThisHour += actionListTestsi1[i][j]*si1.powerRating
      powerUseThisHour -= solarGeneration[i,j]

      if powerUseThisHour >= 0:
        totalElectricCost += price[i,j]*abs(powerUseThisHour)
      else:
        totalElectricCost -= 0.5*price[i,j]*abs(powerUseThisHour)

      totalDiscomfortCost += ad1.alpha*(ad1.maxPowerRating - actionListTestad1[i][j])**2
      totalDiscomfortCost += ad2.alpha*(ad2.maxPowerRating - actionListTestad2[i][j])**2
      totalDiscomfortCost += ad3.alpha*(ad3.maxPowerRating - actionListTestad3[i][j])**2


    cost.append(totalElectricCost + totalDiscomfortCost)  
  return cost

trainingCurvesi1avg = np.zeros((1500))

trainingCurvesu1avg = np.zeros((1500))
trainingCurvesu2avg = np.zeros((1500))

trainingCurvead1avg = np.zeros((1500))
trainingCurvead2avg = np.zeros((1500))
trainingCurvead3avg = np.zeros((1500))
testingCostavg = np.zeros(len(testingData))
generalisationCostavg = np.zeros(len(trainingFullData))


for seed in range(10):

  trainingCurvesi1, testCurvesi1, generalisationCurvesi1, actionListTestsi1, actionListTrainsi1 = runShiftableIntLoad(seed, SEEDLIST[seed], si1)
  trainingCurvesi1avg += trainingCurvesi1

  trainingCurvead1, testCurvead1, generalisationCurvead1, actionListTestad1, actionListTrainad1 = runAdjustableLoad(seed, SEEDLIST[seed], ad1)
  trainingCurvead1avg += trainingCurvead1
  trainingCurvead2, testCurvead2, generalisationCurvead2, actionListTestad2, actionListTrainad2 = runAdjustableLoad(seed, SEEDLIST[seed], ad2)
  trainingCurvead2avg += trainingCurvead2
  trainingCurvead3, testCurvead3, generalisationCurvead3, actionListTestad3, actionListTrainad3 = runAdjustableLoad(seed, SEEDLIST[seed], ad3)
  trainingCurvead3avg += trainingCurvead3

  trainingCurvesu1, testCurvesu1, generalisationCurvesu1, actionListTestsu1, actionListTrainsu1 = runShiftableUnLoad(seed, SEEDLIST[seed], su1)
  trainingCurvesu1avg += trainingCurvesu1
  trainingCurvesu2, testCurvesu2, generalisationCurvesu2, actionListTestsu2, actionListTrainsu2 = runShiftableUnLoad(seed, SEEDLIST[seed], su2)
  trainingCurvesu2avg += trainingCurvesu2

  testingCost = calculateCost(actionListTestad1, actionListTestad2, actionListTestad3, 
                              actionListTestsu1, actionListTestsu2,
                              actionListTestsi1, 
                              ad1, ad2, ad3, 
                              su1, su2,
                              si1,
                              cr1, cr2,
                              testingData, solarTestingData)
  testingCostavg += testingCost
  

print('Time of all ad for training in Q learning is ', tt1_ad, 's')
print('Time of all ad for testing in Q learning is ', tt2_ad, 's')
print('Time of all su for training in Q learning is ', tt1_su, 's')
print('Time of all su for testing in Q learning is ', tt2_su, 's')
print('Time of all si for training in Q learning is ', tt1_si, 's')
print('Time of all si for testing in Q learning is ', tt2_si, 's')
print('The total time for training in Q learning is ', tt1_ad+tt1_si+tt1_su, 's')
print('The total time for testing in Q learning is ', tt2_ad+tt2_su+tt2_si, 's')


np.save("Qlearning/trainingCurvesi1avg.npy", trainingCurvesi1avg)
np.save("Qlearning/trainingCurvesu1avg.npy", trainingCurvesu1avg)
np.save("Qlearning/trainingCurvesu2avg.npy", trainingCurvesu2avg)
np.save("Qlearning/trainingCurvead1avg.npy", trainingCurvead1avg)
np.save("Qlearning/trainingCurvead2avg.npy", trainingCurvead2avg)
np.save("Qlearning/trainingCurvead3avg.npy", trainingCurvead3avg)

np.save("Qlearning/testingCost_ql.npy", testingCostavg)