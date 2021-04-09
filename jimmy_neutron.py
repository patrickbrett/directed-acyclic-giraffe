from aiarena21.client.classes import Player, Map

import random
import math

def sign(the_int):
    if the_int <= -1:
        return -1
    elif the_int >= 1:
        return 1
    else:
        return 0


def get_value_at(items, me_x, me_y, x, y):
    points_at = items[y][x]
    manhattan_distance = abs(me_y - y) + abs(me_x - x)
    value_at = points_at / max(manhattan_distance, 1)
    return value_at



def play_powerup(game_map: Map, me: Player, opponent: Player, items: list, new_items: list, heatmap, remaining_turns):
    # powerups are for the weak
    return ''


def play_turn(game_map: Map, me: Player, opponent: Player, items: list, new_items: list, heatmap, remaining_turns):
    me_y, me_x = me.location

    max_value_y = 0
    max_value_x = 0
    # find the coordinates of the highest-value square on the board, and move toward it
    for y in range(game_map.rows):
        for x in range(game_map.cols):
            value_at = get_value_at(items, me_x, me_y, x, y)
            value_at_max = get_value_at(items, me_x, me_y, max_value_x, max_value_y)

            if value_at > value_at_max:
                max_value_y = y
                max_value_x = x

    # Calculate path from current square to this square
    astar_graph = process_map_to_astar_graph(game_map)
    best_path = aStarAlgorithm(me_y, me_x, max_value_y, max_value_x, astar_graph)
    print(best_path)

    if len(best_path) < 2:
        return f'{me.location[0]},{me.location[1]}'

    # Move towards square with max points

    dy = best_path[1][0] - me.location[0]
    dx = best_path[1][1] - me.location[1]

    new_row = me.location[0] + dy
    new_col = me.location[1] + dx

    print(me.location, dy, dx, max_value_y, max_value_x, new_row, new_col)

    return f'{new_row},{new_col}'


def play_auction(game_map: Map, me: Player, opponent: Player, items: list, new_items: list, heatmap, remaining_turns):
    # auctions are also for the weak
    return 0


def play_transport(game_map: Map, me: Player, opponent: Player, items: list, new_items: list, heatmap, remaining_turns):
    return f'{random.randint(0, game_map.rows-1)},{random.randint(0, game_map.cols-1)}'










################ A STAR ALGORITHM #######################

class Node:
  def __init__(self, row, col, value):
    self.id = str(row) + "-" + str(col)
    self.row = row
    self.col = col
    self.value = value
    self.distanceFromStart = float('inf')
    self.estimatedDistanceToEnd = float('inf')
    self.cameFrom = None

# O(w * h * log(w * h)) time | O(w * h) space
def aStarAlgorithm(startRow, startCol, endRow, endCol, graph):
  nodes = initialiseNodes(graph)

  startNode = nodes[startRow][startCol]
  endNode = nodes[endRow][endCol]

  startNode.distanceFromStart = 0
  startNode.estimatedDistanceToEnd = calculateManhattanDistance(startNode, endNode)

  nodesToVisit = MinHeap([startNode])

  while not nodesToVisit.isEmpty():
    currentMinDistanceNode = nodesToVisit.remove()

    if currentMinDistanceNode == endNode:
      break

    neighbours = getNeighbouringNodes(currentMinDistanceNode, nodes)
    for neighbour in neighbours:
      if neighbour.value == 1:
        continue

      tentativeDistanceToNeighbour = currentMinDistanceNode.distanceFromStart + 1

      if tentativeDistanceToNeighbour >= neighbour.distanceFromStart:
        continue

      neighbour.cameFrom = currentMinDistanceNode
      neighbour.distanceFromStart = tentativeDistanceToNeighbour
      neighbour.estimatedDistanceToEnd = tentativeDistanceToNeighbour + calculateManhattanDistance(neighbour, endNode)

      if not nodesToVisit.containsNode(neighbour):
        nodesToVisit.insert(neighbour)
      else:
        nodesToVisit.update(neighbour)
      
  return reconstructPath(endNode)


def initialiseNodes(graph):
  nodes = []

  for i, row in enumerate(graph):
    nodes.append([])
    for j, value in enumerate(row):
      nodes[i].append(Node(i, j, value))
    
  return nodes
  

def calculateManhattanDistance(currentNode, endNode):
  currentRow = currentNode.row
  currentCol = currentNode.col
  endRow = endNode.row
  endCol = endNode.col

  return abs(currentRow - endRow) + abs(currentCol - endCol)


def getNeighbouringNodes(node, nodes):
  neighbours = []

  numRows = len(nodes)
  numCols = len(nodes[0])
  
  row = node.row
  col = node.col

  if row < numRows - 1: # DOWN
    neighbours.append(nodes[row + 1][col])
  
  if row > 0: # UP
    neighbours.append(nodes[row - 1][col])

  if col < numCols - 1: # RIGHT
    neighbours.append(nodes[row][col + 1])

  if col > 0: # LEFT
    neighbours.append(nodes[row][col - 1])
  
  return neighbours


def reconstructPath(endNode):
  if not endNode.cameFrom:
    return []
  
  currentNode = endNode
  path = []

  while currentNode is not None:
    path.append([currentNode.row, currentNode.col])
    currentNode = currentNode.cameFrom
  
  return path[::-1] # reverse path


class MinHeap:
  def __init__(self, array):
    # Holds position in each heap that each node is at
    self.nodePositionsInHeap = {node.id: idx for idx, node in enumerate(array)}
    self.heap = self.buildHeap(array)
  
  def isEmpty(self):
    return len(self.heap) == 0

  # O(n) time | O(1) space
  def buildHeap(self, array):
    firstParentIdx = (len(array) - 2) // 2
    for currentIdx in reversed(range(firstParentIdx + 1)):
      self.siftDown(currentIdx, len(array) - 1, array)
    return array

  # O(log(n)) time | O(1) space
  def siftDown(self, currentIdx, endIdx, heap):
    childOneIdx = currentIdx * 2 + 1
    while childOneIdx <= endIdx:
      childTwoIdx = currentIdx * 2 + 2 if currentIdx * 2 + 2 <= endIdx else -1
      if (
        childTwoIdx != -1
        and heap[childTwoIdx].estimatedDistanceToEnd < heap[childOneIdx].estimatedDistanceToEnd
      ):
        idxToSwap = childTwoIdx
      else:
        idxToSwap = childOneIdx
      
      if heap[idxToSwap].estimatedDistanceToEnd < heap[currentIdx].estimatedDistanceToEnd:
        self.swap(currentIdx, idxToSwap, heap)
        currentIdx = idxToSwap
        childOneIdx = currentIdx * 2 + 1
      else:
        return
  
  # O(log(n)) time | O(1) space
  def siftUp(self, currentIdx, heap):
    parentIdx = (currentIdx - 1) // 2
    while currentIdx > 0 and heap[currentIdx].estimatedDistanceToEnd < heap[parentIdx].estimatedDistanceToEnd:
      self.swap(currentIdx, parentIdx, heap)
      currentIdx = parentIdx
      parentIdx = (currentIdx - 1) // 2
  
  # O(log(n)) time | O(1) space
  def remove(self):
    if self.isEmpty():
      return
    
    self.swap(0, len(self.heap) - 1, self.heap)
    node = self.heap.pop()
    del self.nodePositionsInHeap[node.id]
    self.siftDown(0, len(self.heap) - 1, self.heap)
    return node

  # O(log(n)) time | O(1) space
  def insert(self, node):
    self.heap.append(node)
    self.nodePositionsInHeap[node.id] = len(self.heap) - 1
    self.siftUp(len(self.heap) - 1, self.heap)
  
  def swap(self, i, j, heap):
    self.nodePositionsInHeap[heap[i].id] = j
    self.nodePositionsInHeap[heap[j].id] = i
    heap[i], heap[j] = heap[j], heap[i]
  
  def containsNode(self, node):
    return node.id in self.nodePositionsInHeap
  
  def update(self, node):
    self.siftUp(self.nodePositionsInHeap[node.id], self.heap)


def process_map_to_astar_graph(game_map):
    a_star_graph = [[0 for _ in range(game_map.cols)] for _ in range(game_map.rows)]

    for y in range(game_map.rows):
        for x in range(game_map.cols):
            is_free = game_map.is_free(y, x)
            a_star_graph[y][x] = 0 if is_free else 1
    
    return a_star_graph


# Run

if __name__ == '__main__':
  print('hello')


  startRow = 0
  startCol = 1
  endRow = 4
  endCol = 3
  graph = [
    [0, 0, 0, 0, 0],
    [0, 1, 1, 1, 0],
    [0, 0, 0, 0, 0],
    [1, 0, 1, 1, 1],
    [0, 0, 0, 0, 0]
  ]

  response = aStarAlgorithm(startRow, startCol, endRow, endCol, graph)
  print(response)
