# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def getDirectionFromAction(action):
  from game import Directions
  if action == 'North':
    return Directions.NORTH
  if action == 'South':
    return Directions.SOUTH  
  if action == 'West':
    return Directions.WEST  
  if action == 'East':
    return Directions.EAST

def getUnvisitedChildren(node, expandedMatrix, problem):
  from util import Queue
  children = problem.getSuccessors(node)

  if len(children) == 0:
    return [];

  unvisited = Queue()

  for child in children:
    x = child[0][0]
    y = child[0][1]

    if (expandedMatrix[x][y] != True):
      unvisited.push(child)
    
  return unvisited

def depthPreview(path, solution, node, action, expandedMatrix, problem, count, debug = False):
  if debug and count % 200 == 0:
    raw_input("Press Enter to continue...")
  
  if count > 2e12:
    return 'STACK_OVERFLOW'
  count = count + 1

  print "\n\npath = ", path, " node = ", node
  expandedMatrix[node[0]][node[1]] = True
  if problem.isGoalState(node):
    print "\t!!!!!!FOUND GOAL!!!!!!!!"
    return ([node] + path, [action] + solution)
  else:
    unvisited = getUnvisitedChildren(node, expandedMatrix, problem)
    print "\tunvisited = ", unvisited.list

    if (unvisited.isEmpty()):
      print "\tThere's no unvisited....Rollback"
      return 'DEADEND'
    else:
      while not(unvisited.isEmpty()):
        child = unvisited.pop()
        print "\t\tvisiting child = ", child
        
        search = depthPreview(path, solution, child[0], getDirectionFromAction(child[1]), expandedMatrix, problem, count, debug)
        if search == 'STACK_OVERFLOW':
          print "STACK OVERFLOW, count = ", count
        elif search != 'DEADEND':
          print "\t\t\t child is NOT dead end: ", node, " -> ", child[0]
          return ([node] + search[0] + path, [action] + search[1] + solution)
        else:
          print "\t\t\t child leads to DEADEND ", node, " -> ", child[0]
      print "\t\t ALL CHILDS ARE DEADEND - ", node
      return 'DEADEND'
  
def depthFirstSearch(problem):
  """
  Search the deepest nodes in the search tree first.

  Your search algorithm needs to return a list of actions that reaches the
  goal. Make sure to implement a graph search algorithm.

  To get started, you might want to try some of these simple commands to
  understand the search problem that is being passed in:

  print "Start:", problem.getStartState()
  print "Is the start a goal?", problem.isGoalState(problem.getStartState())
  print "Start's successors:", problem.getSuccessors(problem.getStartState())
  """
  "*** YOUR CODE HERE ***"
  print "Start:", problem.getStartState()
  print "Is the start a goal?", problem.isGoalState(problem.getStartState())
  print "Start's successors:", problem.getSuccessors(problem.getStartState())

  from util import Stack, Queue
  w, h = 1024, 1024;
  expanded = [[0 for x in range(w)] for y in range(h)] 
  path = []
  solution = []
  action = 'Start'

  print '\n\nStarting search...........................'
  
  search = depthPreview(path, solution, problem.getStartState(), action, expanded, problem, 0)

  print '\n\n****************************************\n'
  print search
  path = search[0]
  solution = search[1]
  solution.remove('Start')

  print '\n\n SOLUTION is = ', solution, '\n\n\n'
  return solution


  
def depthFirstSearchBACKUP(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    """
    "*** YOUR CODE HERE ***"
    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())

    from util import Stack, Queue
    fringe = Stack()
    actions = Stack()
    expanded = Stack()
    solution = []

    foundRoute = False
    fringe.push(problem.getStartState())

    print("\n\n CURRENT NODE:  -")
    print("fringe: " + str(fringe.list))
    print("expanded: " + str(expanded.list))
    print("solution: " + str(solution))
    print("actions: " + str(actions.list))
    
    maxiterations = 50
    while not(foundRoute) and not(fringe.isEmpty()) and maxiterations > 0:
      maxiterations = maxiterations - 1
      currentNode = fringe.pop()
      if not(currentNode in expanded.list):
        expanded.push(currentNode)
      print("\n\n CURRENT NODE:  " + str(currentNode))
      print("fringe: " + str(fringe.list))
      print("expanded: " + str(expanded.list))
      print("solution: " + str(solution))
      print("actions: " + str(actions.list))
      if not(actions.isEmpty()):
        action = actions.pop()
        solution.append(getDirectionFromAction(action))
      if problem.isGoalState(currentNode):
        foundRoute = True
        print "Yuuuuum!!!"
      else:
        print "Not goal :("
        children = problem.getSuccessors(currentNode)
        print ("chidren: ", children)
        # if there's not only one child
        if not(len(children) == 1 and not(expanded.isEmpty()) and children[0][0] == expanded.list[len(expanded.list) - 1]):
          # solution.push(currentNode)
          print "Can have unvisited children"
          numChildrenExpanded = 0
          for child in children:
            if child[0] not in expanded.list:
              fringe.push(child[0])
              actions.push(child[1])
            else:
              numChildrenExpanded = numChildrenExpanded + 1
          
          if numChildrenExpanded == len(children): # se todos os filhos ja foram visitados
            print "ALL CHILDREN WERE VISITED, RETURNING..."
            solution.pop() # cancels action
            expanded.pop()
            fringe.push(expanded.pop()) # set the previous node as next
        else: # if there is no unvisited children, goes back to the previous node, until there is
          solution.pop() # cancels action
          expanded.pop()
          fringe.push(expanded.pop())# set the previous node as next
        print("newFringe: " + str(fringe.list))
    print solution
    return solution


def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


def aStarSearch(problem, heuristic=nullHeuristic):
  """Search the node that has the lowest combined cost and heuristic first."""
  "*** YOUR CODE HERE ***"
  print "Start:", problem.getStartState()
  print "Is the start a goal?", problem.isGoalState(problem.getStartState())
  print "Start's successors:", problem.getSuccessors(problem.getStartState())

  w, h = 1024, 1024;
  expanded = [[0 for x in range(w)] for y in range(h)] 
  path = []
  solution = []
  action = 'Start'

  print '\n\nStarting search...........................'
  
  search = aStarPreview(path, solution, 0, problem.getStartState(), action, expanded, problem, heuristic, 0)

  print '\n\n****************************************\n'
  print search
  path = search[0]
  solution = search[1]
  solution.remove('Start')

  print '\n\n SOLUTION is = ', solution, '\n\n\n'
  return solution

def aStarPreview(path, solution, currentCost, node, action, expandedMatrix, problem, heuristic, count, debug = False):
  if debug and count % 200 == 0:
    raw_input("Press Enter to continue...")
  
  if count > 2e12:
    return 'STACK_OVERFLOW'
  count = count + 1

  print "\n\npath = ", path, " node = ", node
  expandedMatrix[node[0]][node[1]] = True
  if problem.isGoalState(node):
    print "\t!!!!!!FOUND GOAL!!!!!!!!"
    return ([node] + path, [action] + solution, currentCost)
  else:
    unvisited = getUnvisitedChildrenPriorityQueue(node, expandedMatrix, problem, heuristic, currentCost)
    print "\tunvisited = ", unvisited.heap

    if (unvisited.isEmpty()):
      print "\tThere's no unvisited....Rollback"
      return 'DEADEND'
    else:
      while not(unvisited.isEmpty()):
        child = unvisited.pop()
        childCost = child[2]
        print "\t\tvisiting child = ", child
        
        search = aStarPreview(path, solution, currentCost + childCost, child[0], getDirectionFromAction(child[1]), expandedMatrix, problem, heuristic, count, debug)
        if search == 'STACK_OVERFLOW':
          print "STACK OVERFLOW, count = ", count
        elif search != 'DEADEND':
          print "\t\t\t child is NOT dead end: ", node, " -> ", child[0], "   child cost = ", childCost
          return ([node] + search[0] + path, [action] + search[1] + solution, currentCost + childCost)
        else:
          print "\t\t\t child leads to DEADEND ", node, " -> ", child[0]
      print "\t\t ALL CHILDS ARE DEADEND - ", node
      return 'DEADEND'

def getUnvisitedChildrenPriorityQueue(node, expandedMatrix, problem, heuristic, currentCost):
  from util import PriorityQueue
  children = problem.getSuccessors(node)

  print '\t\tcurrent cost = ', currentCost
  if len(children) == 0:
    return [];

  unvisited = PriorityQueue()

  for child in children:
    x = child[0][0]
    y = child[0][1]
    f = currentCost + child[2] + heuristic((x,y), problem) # f(n) = g(n) + h(n)

    if (expandedMatrix[x][y] != True):
      print '\t\tchild ', child[0], '  h(n) =',  heuristic((x,y), problem), ' + g(n) = ', child[2], ' =>  f(n) = ', f
      unvisited.push(child, f)
    
  return unvisited


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
