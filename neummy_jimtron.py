from aiarena21.client.classes import Player, Map


import random
from timeit import default_timer as timer


def play_powerup(game_map: Map, me: Player, opponent: Player, items: list, new_items: list, heatmap, remaining_turns):
    return random.choice([''])


def play_turn(game_map: Map, me: Player, opponent: Player, items: list, new_items: list, heatmap, remaining_turns):
    """
    head example:
        (row, col)
    """
    start = timer()
    max_move, max_value = path_search(game_map, me, opponent, items, new_items, heatmap, remaining_turns,
                                      depth=50, vehicle=None)
    print(timer() - start)
    return f'{max_move[0]},{max_move[1]}'




def play_auction(game_map: Map, me: Player, opponent: Player, items: list, new_items: list, heatmap, remaining_turns):
    return 0


def play_transport(game_map: Map, me: Player, opponent: Player, items: list, new_items: list, heatmap, remaining_turns):
    return f'{random.randint(0, game_map.rows-1)},{random.randint(0, game_map.cols-1)}'


def generate_moves(game_map, origin, vehicle=None):
    if vehicle is None:
        moves = []
        for dir in [1, -1]:
            for axis in [0, 1]:
                move = [0, 0]
                move[axis] = dir
                new_pos = tuple([origin[i] + move[i] for i in range(2)])
                if new_pos[1] in range(game_map.cols) and \
                        new_pos[0] in range(game_map.rows) and \
                        game_map.is_free(*new_pos):
                    moves.append(new_pos)
    elif vehicle == "bike":
        moves = {origin}
        for i in range(3):
            new_moves = set()
            for move in moves:
                for position in generate_moves(game_map, move):
                    new_moves.add(position)
            moves = new_moves
    return moves


def path_search(game_map: Map, me: Player, opponent: Player, items: list, new_items: list,
                heatmap, remaining_turns, depth=50, vehicle=None):

    available = {tuple(me.location)}
    rows, cols = game_map.rows, game_map.cols
    paths = [[[] for _ in range(game_map.cols)] for _ in range(game_map.rows)]
    paths[me.location[0]][me.location[1]] = [tuple(me.location)]
    value = [[-1]*game_map.cols for _ in range(game_map.rows)]
    value[me.location[0]][me.location[1]] = 0
    depth = min(50, remaining_turns)
    value_per_item = 10

    max_move, max_value = None, -1
    for d in range(depth):
        new_available = set()
        for head in available:
            pre_r, pre_c = head
            moves = generate_moves(game_map, head, vehicle)
            for move in moves:
                r, c = move
                # calculate new value
                new_value = value[pre_r][pre_c]
                if move not in paths[pre_r][pre_c]:
                    new_value += value_per_item * items[r][c]
                    new_value += heatmap[r][c] / 100
                # consider update cell
                if new_value > value[r][c]:
                    value[r][c], paths[r][c] = new_value, paths[pre_r][pre_c] + [move]
                # consider update max
                if new_value > max_value:
                    max_value, max_move = new_value, paths[r][c][1]
                new_available.add(move)
        available = new_available
    if max_move is None: max_move = me.location
    return max_move, max_value
