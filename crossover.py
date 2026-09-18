import numpy as np


def crossover(target,mutant,crossover_rate,random_generator,levels= 256):
    #if theres no target or mutant array, or if they are not the same length, raise an error 
    if len(target) ==0:
      raise ValueError("Target array is empty.")
    
    if len(target) != len(mutant):
      raise ValueError("Target and mutant arrays must have the same length.")
    
    if crossover_rate < 0 or crossover_rate > 1:
      raise ValueError("Crossover rate must be between 0 and 1.")
    
    #number of thresholds is the length of the target array
    number_of_thresholds = len(target)
    
    j_rand = random_generator.integers(0, number_of_thresholds)
    
    trial =[]
    
    for j in range (number_of_thresholds):
       random_value = random_generator.random()
       
       use_mutant= (random_value <= crossover_rate) or (j == j_rand)
       
       if use_mutant:
        trial.append(mutant[j])
       else:
         trial.append(target[j])
         
    return repair_thresholds(trial, levels)    
       