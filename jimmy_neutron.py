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


def manhattan_distance(r1, c1, r2, c2):
  return abs(r2-r1) + abs(c2-c1)


def items_around(game_map, items, x, y):
    # returns total value of items adjacent

    left = items[y][x-1] if x > 0 else 0
    right = items[y][x+1] if x+1 < game_map.cols else 0
    up = items[y-1][x] if y > 0 else 0
    down = items[y+1][x] if y+1 < game_map.rows else 0

    return left + right + up + down


def get_value_at(array, me_x, me_y, x, y, distance_penalty_factor=1):
    points_at = array[y][x]
    manhattan_distance = abs(me_y - y) + abs(me_x - x)
    value_at = points_at / math.pow(max(manhattan_distance, 1), distance_penalty_factor)
    return value_at


def get_cluster_value(game_map, items, y, x, cluster_size):
    total_value = 0
    for row in range(max(0, y - cluster_size), min(game_map.rows, y + cluster_size + 1)):
        for col in range(max(0, x - cluster_size), min(game_map.cols, x + cluster_size + 1)):
            if manhattan_distance(x, y, col, row) <= cluster_size:
                total_value += items[row][col]
    
    return total_value



def play_powerup(game_map: Map, me: Player, opponent: Player, items: list, new_items: list, heatmap, remaining_turns):
    global teleport_to
    me_y, me_x = me.location
    op_y, op_x = opponent.location
    
    ## HUNTING MODE - finding clusters of powerups that are well worth teleporting to, that are far away from the opponent

    min_cluster_value_threshold = 150
    cluster_size = 2 # max manhattan distance from central element
    min_distance_threshold = 10 # if the player is closer than this, then there is no point teleporting

    possible_clusters = []
    # find all dragonfruits on the map
    for y in range(game_map.rows):
        for x in range(game_map.cols):
            dist_to_opponent = manhattan_distance(x, y, op_x, op_y)
            dist_to_player = manhattan_distance(x, y, me_x, me_y)
            if max(dist_to_opponent, dist_to_player) > min_distance_threshold:
                continue

            cluster_value = get_cluster_value(game_map, items, y, x, cluster_size)

            if cluster_value > min_cluster_value_threshold:
                possible_clusters.append((y, x, cluster_value))
    
    possible_clusters.sort(key = lambda x: -x[2]) # cluster value, sorted descending

    if len(possible_clusters) > 0:
        # let's go there
        teleport_to = f'{possible_clusters[0][0]},{possible_clusters[0][1]}'
        return teleport_to



    # TODO: this would only be worth it if the max exploitation within cluster size of cluster_size+1
    # around the player's current location is less than min_cluster_value_threshold-100
    # (as that is the cost of teleporting there).
    # Implement that value calculation or similar.

    ## -------------
    ## THIEF MODE - stealing powerups off opponent

    # If opponent's distance to a high value square/area (defined as any square + surrounding squares
    # with total value >= 60) is less than the player's distance to a dragonfruit,
    # and the opponent's distance to said dragon fruit is <= 8 squares (manhattan distance),
    # then teleport to the dragonfruit.

    # TODO - decide if a bike would allow the same thing to be accomplished more cheaply.
    min_value_threshold = 60 # including cell and all surrounding
    opponent_max_distance_threshold = 8
    opponent_min_distance_threshold = 3

    value_coords = []
    # find all dragonfruits on the map
    for y in range(game_map.rows):
        for x in range(game_map.cols):
            cell_value = items[y][x]
            surrounding_cell_values = items_around(game_map, items, x, y)
            dist_to_opponent = manhattan_distance(x, y, op_x, op_y)
            dist_to_player = manhattan_distance(x, y, me_x, me_y)
            if cell_value + surrounding_cell_values >= min_value_threshold:
                value_coords.append((y, x, dist_to_opponent, dist_to_player))

    if len(value_coords) == 0:
        # no squares worth using a powerup for
        return ''

    players_closest = sorted(value_coords, key=lambda x: x[3])[0]
    opponents_closest = sorted(value_coords, key=lambda x: x[2])[0]

    # if we don't meet the conditions above, don't use a powerup
    dist_to_opp = opponents_closest[2] # opponent's nearest dragonfruit distance to opponent
    dist_to_me = players_closest[3] # my nearest dragonfruit distance to me
    if dist_to_opp > opponent_max_distance_threshold or dist_to_opp < opponent_min_distance_threshold or dist_to_me < dist_to_opp:
        print('decided not to teleport', dist_to_me, dist_to_opp)
        return ''

    print('vc', value_coords, players_closest, opponents_closest)

    teleport_to = f'{opponents_closest[0]},{opponents_closest[1]}'

    print('teleport to: ', teleport_to)

    return 'portal gun'


def play_turn(game_map: Map, me: Player, opponent: Player, items: list, new_items: list, heatmap, remaining_turns):
    # if we bought the teleporter, use it!
    if me.portal_gun:
        print('teleporting to: ' + teleport_to)
        return teleport_to

    me_y, me_x = me.location

    max_value_y = 0
    max_value_x = 0
    heatmap_max_y = 0
    heatmap_max_x = 0
    # find the coordinates of the highest-value square on the board, and move toward it
    for y in range(game_map.rows):
        for x in range(game_map.cols):
            value_at = get_value_at(items, me_x, me_y, x, y)
            value_at_max = get_value_at(items, me_x, me_y, max_value_x, max_value_y)

            heatmap_at = get_value_at(heatmap, me_x, me_y, x, y, 0.5)
            heatmap_at_max = get_value_at(heatmap, me_x, me_y, heatmap_max_x, heatmap_max_y, 0.5)

            if value_at > value_at_max:
                max_value_y = y
                max_value_x = x

            if heatmap_at >= heatmap_at_max:
                heatmap_max_y = y
                heatmap_max_x = x

    if items[max_value_y][max_value_x] == 0:
        max_value_y = heatmap_max_y
        max_value_x = heatmap_max_x
    
    print(max_value_x, max_value_y)

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
    # too much lag!
    return 0

    global value_map
    value_map = calculate_value_map(game_map, items, heatmap)
    row, col = me.location[0], me.location[1]
    current_value = eval_tile(min(SEARCH_DEPTH, remaining_turns), row, col, game_map, items, heatmap)
    worst_tile = np.unravel_index(np.argmin(value_map), value_map.shape)
    worst_value = eval_tile(min(SEARCH_DEPTH, remaining_turns), worst_tile[0], worst_tile[1], game_map, items, heatmap)
    r =  math.floor((current_value - worst_value))
    return r


def play_transport(game_map: Map, me: Player, opponent: Player, items: list, new_items: list, heatmap, remaining_turns):
    # one way we could possibly find the worst square is to calculate the number of fruits within 
    # a given manhattan distance threshold of it.
    
    worst_tile = np.unravel_index(np.argmin(value_map), value_map.shape)
    return f'{worst_tile[0], worst_tile[1]}'










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
