from aiarena21.client.classes import Player, Map
import math
import random

HEATMAP_COEFF = 0.01
DISTANCE_COEFF = 0.1

def play_powerup(game_map: Map, me: Player, opponent: Player, items: list, new_items: list, heatmap, remaining_turns):
    return ''

def evaluate_tile_value(value, depth, row, col, game_map: Map, me: Player, opponent: Player, items: list, new_items: list, heatmap, remaining_turns):
  # Simulate the best path from the tile and add to base point value of tile
  if not (0 <= row<game_map.size[0] and 0 <= col<game_map.size[1] and game_map.is_free(row, col)): return -math.inf
  value += (items[row][col] + HEATMAP_COEFF*heatmap[row][col])*(depth+1)*DISTANCE_COEFF
  items_copy = [row[:] for row in items]
  items_copy[row][col] = 0
  if depth == 0: return value

  dirs = [(-1, 0), (0, 1), (1, 0), (0, -1)]
  a = [evaluate_tile_value(value, min(remaining_turns, depth-1), row+dy, col+dx, game_map, me, opponent, items_copy, new_items, heatmap, remaining_turns) for (dx, dy) in dirs]
  #print(a)
  return max(a)

def find_best_tile(game_map: Map, me: Player, opponent: Player, items: list, new_items: list, heatmap, remaining_turns):
  dirs = [(1, 0), (0, 1), (-1, 0), (0, -1)]
  best_dir = dirs[0]
  max_score = -math.inf
  row, col = me.location[0], me.location[1]
  for dx, dy in dirs:
    score = evaluate_tile_value(0, 5, row+dy, col+dx, game_map, me, opponent, items, new_items, heatmap, remaining_turns)
    if score > max_score:
      best_dir = (dx, dy)
      max_score = score
  return (row+best_dir[1], col+best_dir[0])
    

def play_turn(game_map: Map, me: Player, opponent: Player, items: list, new_items: list, heatmap, remaining_turns):
    new_row, new_col = find_best_tile(game_map, me, opponent, items, new_items, heatmap, remaining_turns)
    return f'{new_row},{new_col}'


def play_auction(game_map: Map, me: Player, opponent: Player, items: list, new_items: list, heatmap, remaining_turns):
    return 0#random.randint(0, min(opponent.score, me.score))


def play_transport(game_map: Map, me: Player, opponent: Player, items: list, new_items: list, heatmap, remaining_turns):
    return f'{random.randint(0, game_map.rows-1)},{random.randint(0, game_map.cols-1)}'
