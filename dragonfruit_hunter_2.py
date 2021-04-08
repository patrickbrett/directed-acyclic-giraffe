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
    # every square is awarded a value function = number of points / manhattan distance from player
    
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

    # Move towards square with max points

    dy = sign(max_value_y - me.location[0])
    dx = sign(max_value_x - me.location[1])

    if dx != 0 and dy != 0:
        # don't move diag
        if random.choice([1, 2]) == 1:
            dy = 0
        else:
            dx = 0

    new_row = me.location[0] + dy
    new_col = me.location[1] + dx

    return f'{new_row},{new_col}'


def play_auction(game_map: Map, me: Player, opponent: Player, items: list, new_items: list, heatmap, remaining_turns):
    # auctions are also for the weak
    return 0


def play_transport(game_map: Map, me: Player, opponent: Player, items: list, new_items: list, heatmap, remaining_turns):
    return f'{random.randint(0, game_map.rows-1)},{random.randint(0, game_map.cols-1)}'
