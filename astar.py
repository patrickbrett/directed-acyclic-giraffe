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
        self.swap(curretnIdx, idxToSwap, heap)
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


def process_map_to_astar_graph(map):
  pass


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