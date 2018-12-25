import random, util, game
from game import *
from util import *

#     ********* Reflex agent- sections a and b *********


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

  #print(distanceFromClosestGhost)
  # heuristic parameters:

  # game score:
  gameScore = gameState.getScore()

  # distance from closest ghost:
  # distanceFromClosestGhost

  # distance from closest food:
  listOfFood = getListOfFood(gameState)
  if len(listOfFood) == 0:
    closestFood = None
  else:
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
  dangerRadius = 3
  safeRadius = 7

  if distanceFromClosestGhost < dangerRadius:
    # Danger mode
      a = 0.1
      b = 0.6
      c = 0
      d = 0
      e = 0.3
      score_factor = 1
  elif distanceFromClosestGhost < safeRadius:
    # Normal mode
      a = 0.2
      b = 0.2
      c = 0.2
      d = 0.2
      e = 0.2
      score_factor = 1
  else:
    # Safe mode
      a = 0.3
      b = 0
      c = 0.4
      d = 0.3
      e = 0
      score_factor = 1

  c = b = d = e = 1
  a = 0.1
  b = 2

  if closestGhost.scaredTimer > 0:
    score = a * gameScore \
            + b * (1 / (distanceFromClosestGhost+1))\
            + c * (1 / (distanceFromClosestFood+1)) \
            + e * (1 / (distanceFromClosestCapsule+1)) \
            + d * (1 / (foodAmount+1))
  else:
    score = a * gameScore \
            - b * (1 / (distanceFromClosestGhost+1))\
            + c * (1 / (distanceFromClosestFood+1)) \
            + e * (1 / (distanceFromClosestCapsule+1)) \
            + d * (1 / (foodAmount+1))
  score *= score_factor
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
        cur_max = max(v, cur_max)
      return cur_max
    else: #ghost
      cur_min = float('inf')
      next_agent_idx = agent_idx + 1
      if agent_idx >= (game_state.getNumAgents() - 1):
        depth -= 1
        next_agent_idx = 0
      for action in game_state.getLegalActions(agent_idx):
        v = self.minMaxRecursion(depth, next_agent_idx, game_state.generateSuccessor(agent_idx, action))
        cur_min = min(v, cur_min)
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
    agent_idx = 0
    cur_max = -float('inf')
    next_actions = gameState.getLegalActions(agent_idx)
    next_actions.sort()
    for action in next_actions:
      v = self.minMaxRecursion(self.depth, agent_idx + 1, gameState.generateSuccessor(agent_idx, action), cur_max, float('inf'))
      if v > cur_max:
        cur_max = v
        max_action = action
    return max_action

  def minMaxRecursion(self, depth, agent_idx, game_state, a, b):
    if depth == 0 or game_state.isWin() or game_state.isLose():
      return self.evaluationFunction(game_state)
    if agent_idx == 0:  # pacman
      cur_max = -float('inf')
      for action in game_state.getLegalActions(agent_idx):
        v = self.minMaxRecursion(depth, agent_idx + 1, game_state.generateSuccessor(agent_idx, action), a, b)
        cur_max = max(cur_max, v)
        a = max(a, cur_max)
        if cur_max >= b:
          return float('inf')
      return cur_max
    else:  # ghost
      cur_min = float('inf')
      next_agent_idx = agent_idx + 1
      if agent_idx >= (game_state.getNumAgents() - 1):
        depth -= 1
        next_agent_idx = 0
      for action in game_state.getLegalActions(agent_idx):
        v = self.minMaxRecursion(depth, next_agent_idx, game_state.generateSuccessor(agent_idx, action), a, b)
        cur_min = min(v, cur_min)
        b = min(b, cur_min)
        if cur_min <= a:
          return -float('inf')
      return cur_min


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
    agent_idx = 0
    cur_max = -float('inf')
    for action in gameState.getLegalActions(agent_idx):
      v = self.minMaxRecursion(self.depth, agent_idx + 1, gameState.generateSuccessor(agent_idx, action))
      if v > cur_max:
        cur_max = v
        max_action = action
    return max_action

  def minMaxRecursion(self, depth, agent_idx, game_state):
    if depth == 0 or game_state.isWin() or game_state.isLose():
      return self.evaluationFunction(game_state)
    if agent_idx == 0:  # pacman
      cur_max = -float('inf')
      for action in game_state.getLegalActions(agent_idx):
        v = self.minMaxRecursion(depth, agent_idx + 1, game_state.generateSuccessor(agent_idx, action))
        cur_max = max(v, cur_max)
      return cur_max
    else:  # ghost
      next_agent_idx = agent_idx + 1
      if agent_idx >= (game_state.getNumAgents() - 1):
        depth -= 1
        next_agent_idx = 0
      legal_actions = game_state.getLegalActions(agent_idx)
      total_sum = 0.0
      for action in legal_actions:
        total_sum += self.minMaxRecursion(depth, next_agent_idx, game_state.generateSuccessor(agent_idx, action))
      return total_sum / len(legal_actions)


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
    agent_idx = 0
    cur_max = -float('inf')
    for action in gameState.getLegalActions(agent_idx):
      v = self.minMaxRecursion(self.depth, agent_idx + 1, gameState.generateSuccessor(agent_idx, action))
      if v > cur_max:
        cur_max = v
        max_action = action
    return max_action

  def minMaxRecursion(self, depth, agent_idx, game_state):
    if depth == 0 or game_state.isWin() or game_state.isLose():
      return self.evaluationFunction(game_state)
    if agent_idx == 0:  # pacman
      cur_max = -float('inf')
      for action in game_state.getLegalActions(agent_idx):
        v = self.minMaxRecursion(depth, agent_idx + 1, game_state.generateSuccessor(agent_idx, action))
        cur_max = max(v, cur_max)
      return cur_max
    else:  # ghost
      next_agent_idx = agent_idx + 1
      if agent_idx >= (game_state.getNumAgents() - 1):
        depth -= 1
        next_agent_idx = 0
      legal_actions = game_state.getLegalActions(agent_idx)
      total_sum = 0.0
      p = self.get_directional_ghost_dist(game_state, agent_idx)
      for action in legal_actions:
        total_sum += p[action] * self.minMaxRecursion(depth, next_agent_idx, game_state.generateSuccessor(agent_idx, action))
      return total_sum

  def get_directional_ghost_dist(self, game_state, agent_idx):
    # TODO: check if constants
    prob_attack = 0.8
    prob_scaredFlee = 0.8

    ghostState = game_state.getGhostState(agent_idx)
    legalActions = game_state.getLegalActions(agent_idx)
    pos = game_state.getGhostPosition(agent_idx)
    isScared = ghostState.scaredTimer > 0
    speed = 1
    if isScared: speed = 0.5
    actionVectors = [Actions.directionToVector(a, speed) for a in legalActions]
    newPositions = [(pos[0] + a[0], pos[1] + a[1]) for a in actionVectors]
    pacmanPosition = game_state.getPacmanPosition()
    # Select best actions given the state
    distancesToPacman = [manhattanDistance(pos, pacmanPosition) for pos in newPositions]
    if isScared:
      bestScore = max(distancesToPacman)
      bestProb = prob_scaredFlee
    else:
      bestScore = min(distancesToPacman)
      bestProb = prob_attack
    bestActions = [action for action, distance in zip(legalActions, distancesToPacman) if distance == bestScore]

    # Construct distribution
    dist = util.Counter()
    for a in bestActions: dist[a] = bestProb / len(bestActions)
    for a in legalActions: dist[a] += (1 - bestProb) / len(legalActions)
    dist.normalize()
    return dist


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



