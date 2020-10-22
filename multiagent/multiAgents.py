# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util
import sys

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        (newScaredTimes) # unused variable warning
        foodValue = 1.0 / (1 + min([sys.maxsize] + [manhattanDistance(newPos, food) for food in newFood.asList()]))

        return successorGameState.getScore() + foodValue

def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        return self.minimaxValueAndAction(gameState)[1]

    def minimaxValueAndAction(self, gameState, agentIndex=0, depth=0):
        if agentIndex == gameState.getNumAgents(): # next cycle of turns
            agentIndex = 0
            depth += 1

        if agentIndex == 0: # case pacman
            return self.maxValueAndAction(gameState, agentIndex, depth)
        else: # case ghost
            return self.minValueAndAction(gameState, agentIndex, depth)

    def maxValueAndAction(self, gameState, agentIndex, depth):
        if depth == self.depth or gameState.isLose() or gameState.isWin():
            return (self.evaluationFunction(gameState), None)

        bestValueAndAction = (-sys.maxsize, None)
        for action in gameState.getLegalActions(agentIndex):
            bestValueAndAction = max(bestValueAndAction, 
                                     (self.minimaxValueAndAction(gameState.generateSuccessor(agentIndex, action), 
                                                                 agentIndex + 1, 
                                                                 depth)[0], action))
                                      
        return bestValueAndAction

    def minValueAndAction(self, gameState, agentIndex, depth):
        if depth == self.depth or gameState.isLose() or gameState.isWin():
            return (self.evaluationFunction(gameState), None)
        
        bestValueAndAction = (sys.maxsize, None)
        for action in gameState.getLegalActions(agentIndex):
            bestValueAndAction = min(bestValueAndAction, 
                                     (self.minimaxValueAndAction(gameState.generateSuccessor(agentIndex, action), 
                                                                 agentIndex + 1, 
                                                                 depth)[0], action))
        
        return bestValueAndAction


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        return self.minimaxValueAndAction(gameState)[1]

    def minimaxValueAndAction(self, gameState, agentIndex=0, depth=0, alpha=-sys.maxsize, betha=sys.maxsize):
        if agentIndex == gameState.getNumAgents(): # next cycle of turns
            agentIndex = 0
            depth += 1

        if agentIndex == 0: # case pacman
            return self.maxValueAndAction(gameState, agentIndex, depth, alpha, betha)
        else: # case ghost
            return self.minValueAndAction(gameState, agentIndex, depth, alpha, betha)

    def maxValueAndAction(self, gameState, agentIndex, depth, alpha, betha):
        if depth == self.depth or gameState.isLose() or gameState.isWin():
            return (self.evaluationFunction(gameState), None)
            
        bestValueAndAction = (-sys.maxsize, None)
        for action in gameState.getLegalActions(agentIndex):
            bestValueAndAction = max(bestValueAndAction, 
                                     (self.minimaxValueAndAction(gameState.generateSuccessor(agentIndex, action), 
                                                                 agentIndex + 1, 
                                                                 depth,
                                                                 alpha,
                                                                 betha)[0], action))
            
            if bestValueAndAction[0] > betha:
                break
            else:
                alpha = max(alpha, bestValueAndAction[0])

        return bestValueAndAction

    def minValueAndAction(self, gameState, agentIndex, depth, alpha, betha):
        if depth == self.depth or gameState.isLose() or gameState.isWin():
            return (self.evaluationFunction(gameState), None)
        
        bestValueAndAction = (sys.maxsize, None)
        for action in gameState.getLegalActions(agentIndex):
            bestValueAndAction = min(bestValueAndAction, 
                                     (self.minimaxValueAndAction(gameState.generateSuccessor(agentIndex, action), 
                                                                 agentIndex + 1, 
                                                                 depth,
                                                                 alpha,
                                                                 betha)[0], action))

            if bestValueAndAction[0] < alpha:
                break
            else:
                betha = min(betha, bestValueAndAction[0])
            
        return bestValueAndAction


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        return self.expectimaxValueAndAction(gameState)[1]

    def expectimaxValueAndAction(self, gameState, agentIndex=0, depth=0):
        if agentIndex == gameState.getNumAgents(): # next cycle of turns
            agentIndex = 0
            depth += 1

        if agentIndex == 0: # case pacman
            return self.maxValueAndAction(gameState, agentIndex, depth)
        else: # case ghost
            return self.expectedValueAndAction(gameState, agentIndex, depth)

    def maxValueAndAction(self, gameState, agentIndex, depth):
        if depth == self.depth or gameState.isLose() or gameState.isWin():
            return (self.evaluationFunction(gameState), None)

        bestValueAndAction = (-sys.maxsize, None)
        for action in gameState.getLegalActions(agentIndex):
            bestValueAndAction = max(bestValueAndAction, 
                                     (self.expectimaxValueAndAction(gameState.generateSuccessor(agentIndex, action), 
                                                                    agentIndex + 1, 
                                                                    depth)[0], action))    

        return bestValueAndAction

    def expectedValueAndAction(self, gameState, agentIndex, depth):
        if depth == self.depth or gameState.isLose() or gameState.isWin():
            return (self.evaluationFunction(gameState), None)
        
        actions = gameState.getLegalActions(agentIndex)
        prob    = 1.0 / len(actions)
        value   = sum([prob * self.expectimaxValueAndAction(gameState.generateSuccessor(agentIndex, action), agentIndex + 1, depth)[0] 
                       for action in actions])

        return (value, actions[0])

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <evaluation function depends on ->
        distance to nearest food,
        distance to nearest ghost,
        number of food left,
        number of capsules left,
        gameState is either win or lose,
    >
    """
    "*** YOUR CODE HERE ***"
    if currentGameState.isWin():
        return +10000000
    elif currentGameState.isLose():
        return -10000000

    foodLeftWeight     = 1000000.0
    capsulesLeftWeight = 10000.0
    minFoodWeight      = 1000.0

    newPos  = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()

    minFoodValue  = minFoodWeight      / (1 + min([sys.maxsize] + [manhattanDistance(newPos, food) for food in newFood.asList()]))
    foodLeftValue = foodLeftWeight     / (1 + currentGameState.getNumFood())
    capsulesLeft  = capsulesLeftWeight / (1 + len(currentGameState.getCapsules()))
    
    minGhostValue = min([sys.maxsize] + [manhattanDistance(newPos, ghostPos) for ghostPos in currentGameState.getGhostPositions()])
    
    return minFoodValue + foodLeftValue + capsulesLeft + minGhostValue


# Abbreviation
better = betterEvaluationFunction
