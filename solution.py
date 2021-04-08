from aiarena21.client.classes import Player, Map
import math
import random
import numpy as np

HEATMAP_COEFF = 0.01
DISTANCE_COEFF = 0.1
SEARCH_DEPTH = 5

dirs = [(-1, 0), (0, 1), (1, 0), (0, -1)]
value_map = None

def play_powerup(game_map: Map, me: Player, opponent: Player, items: list, new_items: list, heatmap, remaining_turns):
    return ''

def eval_tile(depth, row, col, game_map, items, heatmap, value=0, visited=[]):
  # Simulate the best path from the tile and add to base point value of tile
  # TODO: avoid fruits that are closer to the opponent
    if not (0 <= row<game_map.size[0] and 0 <= col<game_map.size[1] and game_map.is_free(row, col)): return -1
    item_value = items[row][col] if (row, col) not in visited else 0
    value += (item_value + HEATMAP_COEFF*heatmap[row][col])*(depth+1)*DISTANCE_COEFF
    if depth == 0: return value
    return max([eval_tile(depth-1, row+dy, col+dx, game_map, items, heatmap, value, visited[:] + [(row, col)]) for dx, dy in dirs])

def get_next_tile(row, col, game_map: Map, me: Player, opponent: Player, items: list, new_items: list, heatmap, remaining_turns):

  best_dir = dirs[0]
  max_score = -1
  row, col = me.location[0], me.location[1]
  for dx, dy in dirs:
    score = eval_tile(min(SEARCH_DEPTH, remaining_turns), row+dy, col+dx, game_map, items, heatmap)
    if score > max_score:
      best_dir = (dx, dy)
      max_score = score
  return (row+best_dir[1], col+best_dir[0]), max_score

def manhattan_distance(r1, c1, r2, c2):
  return abs(r2-r1) + abs(c2-c1)

def calculate_value_map(game_map, items, heatmap):
  rows, cols = game_map.size[0], game_map.size[1]
  m = np.zeros((rows, cols))
  for r in range(rows):
    for c in range(cols):
      for i in range(rows):
        for j in range(cols):
          if i != r or j != c:
            # Corner cells are at a disadvantage here however.
            m[r, c] += (items[i][j] + HEATMAP_COEFF*heatmap[i][j])/manhattan_distance(r, c, i, j)
  return m

def play_turn(game_map: Map, me: Player, opponent: Player, items: list, new_items: list, heatmap, remaining_turns):
  print(me.name)
  row, col = me.location[0], me.location[1]
  next_tile, _ = get_next_tile(row, col, game_map, me, opponent, items, new_items, heatmap, remaining_turns)
  new_row, new_col = next_tile
  return f'{new_row},{new_col}'


def play_auction(game_map: Map, me: Player, opponent: Player, items: list, new_items: list, heatmap, remaining_turns):
  global value_map
  value_map = calculate_value_map(game_map, items, heatmap)
  row, col = me.location[0], me.location[1]
  current_value = eval_tile(min(SEARCH_DEPTH, remaining_turns), row, col, game_map, items, heatmap)
  worst_tile = np.unravel_index(np.argmin(value_map), value_map.shape)
  worst_value = eval_tile(min(SEARCH_DEPTH, remaining_turns), worst_tile[0], worst_tile[1], game_map, items, heatmap)
  r =  math.floor((current_value - worst_value))
  return r

def play_transport(game_map: Map, me: Player, opponent: Player, items: list, new_items: list, heatmap, remaining_turns):
  worst_tile = np.unravel_index(np.argmin(value_map), value_map.shape)
  return f'{worst_tile[0], worst_tile[1]}'
