import random, util
from game import Agent

#     ********* Reflex agent- sections a and b *********
from HW2.pacman.util import *


class ReflexAgent(Agent):
  """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.
  """
  def __init__(self):
    self.lastPositions = []
    self.dc = None


  def getAction(self, gameState):
    """
    getAction chooses among the best options according to the evaluation function.

    getAction takes a GameState and returns some Directions.X for some X in the set {North, South, West, East, Stop}
    ------------------------------------------------------------------------------
    """
    # Collect legal moves and successor states
    legalMoves = gameState.getLegalActions()

    # Choose one of the best actions
    scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
    bestScore = max(scores)
    bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
    chosenIndex = random.choice(bestIndices) # Pick randomly among the best


    return legalMoves[chosenIndex]

  def evaluationFunction(self, currentGameState, action):
    """
    The evaluation function takes in the current GameState (pacman.py) and the proposed action
    and returns a number, where higher numbers are better.
    """
    successorGameState = currentGameState.generatePacmanSuccessor(action)
    #return scoreEvaluationFunction(successorGameState)
    return betterEvaluationFunction(successorGameState)


#     ********* Evaluation functions *********

def scoreEvaluationFunction(gameState):
  """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.
  """
  return gameState.getScore()

######################################################################################
# b: implementing a better heuristic function
def betterEvaluationFunction(gameState):
  """

  The betterEvaluationFunction takes in a GameState (pacman.py) and should return a number, where higher numbers are better.

  A GameState specifies the full game state, including the food, capsules, agent configurations and more.
  Following are a few of the helper methods that you can use to query a GameState object to gather information about
  the present state of Pac-Man, the ghosts and the maze:

  gameState.getLegalActions():
  gameState.getPacmanState():
  gameState.getGhostStates():
  gameState.getNumAgents():
  gameState.getScore():
  The GameState class is defined in pacman.py and you might want to look into that for other helper methods.
  """

  # TODO: question: is it possible to run the game without ghosts?
  # TODO: if so, check if list is null

  # find closest ghost:
  closestGhost = ClosestGhostToPacman(gameState)


  pacmanPos = gameState.getPacmanPosition()
  closestGhostPos = closestGhost.configuration.pos
  distanceFromClosestGhost = manhattanDistance(closestGhostPos, pacmanPos)

  # heuristic parameters:

  # game score:
  gameScore = gameState.getScore()

  # distance from closest ghost:
  # distanceFromClosestGhost

  # distance from closest food:
  listOfFood = getListOfFood(gameState)
  closestFood = getClosestElementToPacman(gameState, listOfFood)
  if closestFood is None:
    distanceFromClosestFood = 0
  else:
    distanceFromClosestFood = manhattanDistance(closestFood, pacmanPos)

  # amount of food on grid:
  foodAmount = gameState.getNumFood()

  # distance from closest capsule:
  capsulesPos = gameState.getCapsules()
  if len(capsulesPos) == 0:
    distanceFromClosestCapsule = 0
  else:
    closestCapsule = getClosestElementToPacman(gameState, capsulesPos)
    distanceFromClosestCapsule = manhattanDistance(closestCapsule, pacmanPos)



  # decide which state are we in: Danger, Chase, Safe, Normal
  # TODO: Change radius according to try and error:
  dangerRadius = 10
  safeRadius = 100

  if distanceFromClosestGhost < dangerRadius:
    # check if closest ghost is scared:
    if closestGhost.scaredTimer > 0:
      # ghost is scared, Chase mode
      a = 100
      b = 200
      c = 30
      d = -40
      e = 0
    else:
      # ghost is not scared, Danger
      a = 20
      b = -100
      c = 10
      d = -10
      e = 90
  elif distanceFromClosestGhost < safeRadius:
    # Normal mode
    a = 50
    b = -30
    c = 40
    d = -40
    e = 20
  else:
    # Safe mode
    a = 50
    b = -10
    c = 50
    d = -40
    e = 0

  score = a*gameScore + b*distanceFromClosestGhost + 0*distanceFromClosestFood + d*foodAmount + 0*distanceFromClosestCapsule
  return score


def getListOfFood(gameState):
  foodGrid = gameState.getFood()
  foodPos = []
  for x, a in enumerate(foodGrid):
    for y, aa in enumerate(a):
      if aa == True:
        foodPos.append([x, y])
  return foodPos

def getClosestElementToPacman(gameState, listOfPos):
  pacmanPos = gameState.getPacmanPosition()
  closestElementPos = listOfPos[0]
  for pos in listOfPos:
    if manhattanDistance(pos, pacmanPos) < manhattanDistance(closestElementPos, pacmanPos):
      closestElementPos = pos
  return closestElementPos

def ClosestGhostToPacman(gameState):
  pacmanPos = gameState.getPacmanPosition()
  ghostStates = gameState.getGhostStates()
  closestGhost = gameState.getGhostState(1)

  # find closest ghost to pacman:
  for ghostState in ghostStates:
    ghostPos = ghostState.configuration.pos
    closestGhostPos = closestGhost.configuration.pos
    if manhattanDistance(ghostPos, pacmanPos) < manhattanDistance(closestGhostPos, pacmanPos):
      closestGhost = ghostState

  return closestGhost


#     ********* MultiAgent Search Agents- sections c,d,e,f*********

class MultiAgentSearchAgent(Agent):
  """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxAgent, AlphaBetaAgent & both ExpectimaxAgents.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
  """

  def __init__(self, evalFn = 'betterEvaluationFunction', depth = '2'):
    self.index = 0 # Pacman is always agent index 0
    self.evaluationFunction = util.lookup(evalFn, globals())
    self.depth = int(depth)

######################################################################################
# c: implementing minimax

class MinimaxAgent(MultiAgentSearchAgent):
  """
    Your minimax agent
  """

  def getAction(self, gameState):
    """
      Returns the minimax action from the current gameState using self.depth
      and self.evaluationFunction. Terminal states can be found by one of the following:
      pacman won, pacman lost or there are no legal moves.

      Here are some method calls that might be useful when implementing minimax.

      gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

      Directions.STOP:
        The stop direction

      gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

      gameState.getNumAgents():
        Returns the total number of agents in the game

      gameState.getScore():
        Returns the score corresponding to the current state of the game

      gameState.isWin():
        Returns True if it's a winning state

      gameState.isLose():
        Returns True if it's a losing state

      self.depth:
        The depth to which search should continue

    """
    agent_idx = 0
    cur_max = -float('inf')
    for action in gameState.getLegalActions(agent_idx):
      v = self.minMaxRecursion(self.depth, agent_idx+1, gameState.generateSuccessor(agent_idx, action))
      if v > cur_max:
        cur_max = v
        max_action = action
    return max_action

  def minMaxRecursion(self, depth, agent_idx, game_state):
    if depth == 0 or game_state.isWin() or game_state.isLose():
      return self.evaluationFunction(game_state)
    if agent_idx == 0: #pacman
      cur_max = -float('inf')
      for action in game_state.getLegalActions(agent_idx):
        v = self.minMaxRecursion(depth, agent_idx + 1, game_state.generateSuccessor(agent_idx, action))
        if v > cur_max:
          cur_max = v
      return cur_max
    else: #ghost
      cur_min = float('inf')
      next_agent_idx = agent_idx + 1
      if agent_idx >= (game_state.getNumAgents() - 1):
        depth -= 1
        next_agent_idx = 0
      for action in game_state.getLegalActions(agent_idx):
        v = self.minMaxRecursion(depth, next_agent_idx, game_state.generateSuccessor(agent_idx, action))
        if v < cur_min:
          cur_min = v
      return cur_min




######################################################################################
# d: implementing alpha-beta

class AlphaBetaAgent(MultiAgentSearchAgent):
  """
    Your minimax agent with alpha-beta pruning
  """

  def getAction(self, gameState):
    """
      Returns the minimax action using self.depth and self.evaluationFunction
    """

    # BEGIN_YOUR_CODE
    raise Exception("Not implemented yet")
    # END_YOUR_CODE

######################################################################################
# e: implementing random expectimax

class RandomExpectimaxAgent(MultiAgentSearchAgent):
  """
    Your expectimax agent
  """

  def getAction(self, gameState):
    """
      Returns the expectimax action using self.depth and self.evaluationFunction
      All ghosts should be modeled as choosing uniformly at random from their legal moves.
    """

    # BEGIN_YOUR_CODE
    raise Exception("Not implemented yet")
    # END_YOUR_CODE

######################################################################################
# f: implementing directional expectimax

class DirectionalExpectimaxAgent(MultiAgentSearchAgent):
  """
    Your expectimax agent
  """

  def getAction(self, gameState):
    """
      Returns the expectimax action using self.depth and self.evaluationFunction
      All ghosts should be modeled as using the DirectionalGhost distribution to choose from their legal moves.
    """

    # BEGIN_YOUR_CODE
    raise Exception("Not implemented yet")
    # END_YOUR_CODE


######################################################################################
# I: implementing competition agent

class CompetitionAgent(MultiAgentSearchAgent):
  """
    Your competition agent
  """

  def getAction(self, gameState):
    """
      Returns the action using self.depth and self.evaluationFunction

    """

    # BEGIN_YOUR_CODE
    raise Exception("Not implemented yet")
    # END_YOUR_CODE



