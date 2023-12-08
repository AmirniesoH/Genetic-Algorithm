from os import system
import pandas as pd
from random import random, sample, randrange, uniform
import numpy as np
import matplotlib.pyplot as plt

class EvolutionaryComputing:

    def __init__(self, thePopulationSize, theProblemSize, theProbabilityMutation):      ## Constructor  
        
        self.__populationSize = thePopulationSize
        self.__problemSize = theProblemSize
        self.__probabilityMutation = theProbabilityMutation

## Public functions

    def initialize_Population(self):    ## Initializes population

        currentGeneration = pd.DataFrame(columns=['Chromosome', 'Fitness'])

        for i in range(self.__populationSize):
            currentGeneration.loc[i] = {'Chromosome' : np.random.permutation(np.arange(1, self.__problemSize+1)), 'Fitness': None}

        return currentGeneration

    def evaluate_Population(self, thePopulation):      ## Evaluates population
        for i in range(self.__populationSize):
            error =0
            for j in range(self.__problemSize):
                for k in range(j+1,self.__problemSize):
                    ## If two queens are placed at position (i, j) and (k, l) Then they are on same diagonal only if (i - j) = k - l or i + j = k + l.
                    if thePopulation.iat[i,0][j]-j == thePopulation.iat[i,0][k]-k or thePopulation.iat[i,0][j]+j == thePopulation.iat[i,0][k]+k:
                        error +=1
            thePopulation.iat[i,1] = 28-error
        return thePopulation
    
    def get_BestSolution(self, theCurrentGeneration):     ## Gets best solution

        theCurrentGeneration['Fitness'] = pd.to_numeric(theCurrentGeneration['Fitness'])
        return theCurrentGeneration.iloc[theCurrentGeneration['Fitness'].idxmax()]
    
    def randomSelection(self, theCurrentGeneration):        ## Selects parents by random selection method
        
        parents = pd.DataFrame(columns=['Chromosome'])

        for i in range(self.__populationSize):
            randomNumber = randrange(0,self.__populationSize)
            parents = parents.append({'Chromosome' : theCurrentGeneration.iat[randomNumber,0]}, ignore_index=True)
        
        return parents
    
    def proportionalSelection(self, theCurrentGeneration):      ## Selects parents by proportional selection method
        
        parents = pd.DataFrame(columns=['Chromosome'])

        parents['Chromosome']=np.random.choice(theCurrentGeneration.iloc[:,0], self.__populationSize, p=self.__compute_Probability(theCurrentGeneration['Fitness'], sum(theCurrentGeneration['Fitness'])))

        return parents

    def rankBasedSelection(self, theCurrentGeneration):     ## Selects parents by rank based selection method

        parents = pd.DataFrame(columns=['Chromosome'])
        rank = [0]*self.__populationSize

        ## Computes for each chromosome the probability
        buffer = self.__compute_Probability(theCurrentGeneration['Fitness'], sum(theCurrentGeneration['Fitness']))

        for i in range(self.__populationSize):
            index =np.argmin(buffer) 
            rank[index]=i+1
            buffer[index]=1

        parents['Chromosome']=np.random.choice(theCurrentGeneration.iloc[:,0], self.__populationSize, p=self.__compute_Probability(rank, sum(np.arange(1,self.__populationSize+1)) ))

        return parents

    def tournamentSelection(self, theCurrentGeneration, theSize):       ## Selects parents by tounament selection method

        parents = pd.DataFrame(columns=['Chromosome'])
        buffer = pd.DataFrame(columns=['Chromosome', 'Fitness'])

        for i in range(self.__populationSize):
            for j in range(theSize):
                randomNumber=randrange(self.__populationSize)
                buffer=buffer.append(theCurrentGeneration.iloc[randomNumber], ignore_index=True)
            parents = parents.append({'Chromosome': buffer.iat[np.argmax(buffer['Fitness']),0]},ignore_index=True)

        return parents
   
    def trancationSelection(self, theCurrentGeneration, theSize):       ## Selects parents by trancation selection method
        
        parents = pd.DataFrame(columns=['Chromosome'])
        buffer = theCurrentGeneration
        buffer = buffer.sort_values(by='Fitness')

        for i in range(self.__populationSize):
            randomNumber= randrange(theSize)
            parents = parents.append({'Chromosome' : buffer.iat[randomNumber,0]}, ignore_index=True)

        return parents
   
    def reproduce_OnePointCrossover(self, theParent_One, theParent_Two):      ## Reproduces by one point crossover method
        
        ## initializes offspring
        offspring_One = [0]*self.__problemSize
        offspring_Two = [0]*self.__problemSize


        randomNumber=randrange(1,self.__problemSize)

        offspring_One[:randomNumber], offspring_Two[:randomNumber] = theParent_One[:randomNumber], theParent_Two[:randomNumber] 

        for i in range(self.__problemSize):
            if theParent_Two[i] not in offspring_One:
                offspring_One[offspring_One.index(0)]=theParent_Two[i]
            if theParent_One[i] not in offspring_Two:
                offspring_Two[offspring_Two.index(0)]=theParent_One[i]

        return offspring_One, offspring_Two

    def reproduce_MultiPointCrossover(self, theParent_One, theParent_Two):    ## Reproduces by multi point crossover method
        
        ## initializes offspring
        offspring_One = [0]*self.__problemSize
        offspring_Two = [0]*self.__problemSize

        randomNumber = sample(range(1,self.__problemSize-1), 2)
        randomNumber = sorted(randomNumber)

        offspring_One[randomNumber[0]:randomNumber[1]], offspring_Two[randomNumber[0]:randomNumber[1]] = theParent_One[randomNumber[0]:randomNumber[1]], theParent_Two[randomNumber[0]:randomNumber[1]]

        for i in range(self.__problemSize):
            if theParent_Two[i] not in offspring_One:
                offspring_One[offspring_One.index(0)]=theParent_Two[i]
            if theParent_One[i] not in offspring_Two:
                offspring_Two[offspring_Two.index(0)]=theParent_One[i]

        return offspring_One, offspring_Two

    def reproduce_UniformCrossover(self,theParent_One, theParent_Two):    ## Reproduces by uniform crossover method
        
        offspring_One = []
        offspring_Two = []

        for i in range(self.__problemSize):
            
            randomNumber=randrange(2)

            if randomNumber==0:
                offspring_One.append(theParent_One[i])
                offspring_Two.append(theParent_Two[i])
            else:
                offspring_One.append(theParent_Two[i])
                offspring_Two.append(theParent_One[i])
        return offspring_One, offspring_Two

    def repalce_SteadyStateReplacement(self, theCurrentGeneration, theOffspring):   ## Replaces by steady state replacement method
        
        nextGeneration=pd.DataFrame(columns=['Chromosome', 'Fitness'])

        theOffspring = theOffspring.sort_values(by='Fitness')
        
        for i in range(self.__populationSize):

            if (i < self.__populationSize*self.__probabilityReplace):
                randomNumber=randrange(self.__populationSize)
                nextGeneration = nextGeneration.append(theCurrentGeneration.iloc[randomNumber], ignore_index=True)
            else:
                nextGeneration = nextGeneration.append(theOffspring.iloc[i], ignore_index=True)

        nextGeneration = nextGeneration.sample(frac=1).reset_index(drop=True)

        return nextGeneration

    def replace_GenerationalReplacement(self, theCurrentGeneration, theOffspring):      ## Replaces by generational replacement method

        theCurrentGeneration = theCurrentGeneration.sort_values(by='Fitness', ascending=False)
        theOffspring = theOffspring.sort_values(by='Fitness')
        theCurrentGeneration.iloc[1:]=theOffspring.iloc[1:]
        theCurrentGeneration = theCurrentGeneration.sample(frac=1).reset_index(drop=True)

        return theCurrentGeneration

    def mutate(self, theOffspring): ## The offspring  mutates with a certain probability
        for i in range(100):
            randomNumber = uniform(0,1)

            if randomNumber<self.__probabilityMutation:
                theOffspring = np.roll(theOffspring, 1)
        return theOffspring
    
    def plot_SelectivePressure(self, theSpeciesDiversity, theEpochs):
        
        x=np.array(np.arange(theEpochs))
        y=np.array(theSpeciesDiversity)
        

        plt.plot(x, y, color='blue')

        plt.title('Selective Pressure')
        plt.xlabel('Generation', fontdict={'fontsize':10})
        plt.ylabel('Species Diversity', fontdict={'fontsize':10})

        plt.show()

    def compute_SpeciesDiversity(self, theCurrentGeneration):   ## Computs species diversity 
        
        flag =False
        speciesDiversity = self.__populationSize

        for i in range(self.__populationSize):
            for j in range(i,self.__populationSize-1):
                flag = False
                for k in range(self.__problemSize):
                    if theCurrentGeneration.iat[i,0][k]!=theCurrentGeneration.iat[j+1,0][k]:
                        flag=False
                        break
                    else:
                        flag =True
                if flag:
                    speciesDiversity-=1

        return speciesDiversity

## Private function

    def __compute_Probability(self, theFavourableEvent, theTotalEvent):     ## computes the probability

        probability = [0]*self.__populationSize
        for i in range(self.__populationSize):
            probability[i] = theFavourableEvent[i]/theTotalEvent
        
        return probability
    

## Private Variables

    __populationSize = 0
    __problemSize = 0
    __probabilityMutation = 0.01
    __probabilityReplace = 0.8

def GeneticAlgorithm(thePopulationSize, theProblemSize, theProbabilityMutation):
   
    evolutionaryComputing = EvolutionaryComputing(thePopulationSize, theProblemSize, theProbabilityMutation)
    epochs=0
    speciesDiversity = np.array([])

    ## Initializes current Generation
    currentGeneration = evolutionaryComputing.initialize_Population()

    ## Evaluates current Generation
    currentGeneration = evolutionaryComputing.evaluate_Population(currentGeneration)

    ## Gets best Solution
    bestSolution = evolutionaryComputing.get_BestSolution(currentGeneration)

    for i in range(1000):

        ## Selects Parents
        parents = evolutionaryComputing.rankBasedSelection(currentGeneration)

        ## Initializes offspring
        offsprings = pd.DataFrame(columns=['Chromosome', 'Fitness'], index=np.arange(thePopulationSize))

        ## Reproduces
        for i in range(0,parents.shape[0],2):
            offspring_One, offspring_Two = evolutionaryComputing.reproduce_MultiPointCrossover(parents.iat[i,0], parents.iat[i+1,0]) 

        ## The offspring  mutates with a certain probability 
            offsprings.iat[i,0]=evolutionaryComputing.mutate(offspring_One)
            offsprings.iat[i+1,0]=evolutionaryComputing.mutate(offspring_Two)
        
        ## Evaluates offsprings
        offsprings = evolutionaryComputing.evaluate_Population(offsprings)
        
        ## Replaces
        currentGeneration = evolutionaryComputing.replace_GenerationalReplacement(currentGeneration,offsprings)

        ## Gets best solution
        bestSolution = evolutionaryComputing.get_BestSolution(currentGeneration)

        epochs+=1

        ## Compute Species Diversity
        speciesDiversity = np.insert(speciesDiversity, np.size(speciesDiversity),evolutionaryComputing.compute_SpeciesDiversity(currentGeneration))

        if max(currentGeneration['Fitness'])==28:
            break

    ## Plot Selective Pressure
    evolutionaryComputing.plot_SelectivePressure(speciesDiversity, epochs)

    return bestSolution,epochs


## Main Code

bestSolution, epoches = GeneticAlgorithm(20, 8, 0.01)

system('cls')
print('Epoches = {0} \n\nBest Solution = Chromosome : {1} , Fitness : {2}\n\n'.format(epoches, bestSolution['Chromosome'], bestSolution['Fitness']))