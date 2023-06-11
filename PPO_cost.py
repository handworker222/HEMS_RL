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

actionListTestad1 = np.load("PPO/action/AD1.npy")
actionListTestad2 = np.load("PPO/action/AD2.npy")
actionListTestad3 = np.load("PPO/action/AD3.npy")

actionListTestsi1 = np.load("PPO/action/SI1.npy")

actionListTestsu1 = np.load("PPO/action/SU1.npy")
actionListTestsu2 = np.load("PPO/action/SU2.npy")

testingCost = calculateCost(actionListTestad1, actionListTestad2, actionListTestad3, 
                            actionListTestsu1, actionListTestsu2,
                            actionListTestsi1, 
                            ad1, ad2, ad3, 
                            su1, su2,
                            si1,
                            cr1, cr2,
                            testingData, solarTestingData)
np.save("PPO/testingCost_ppo.npy", testingCost)