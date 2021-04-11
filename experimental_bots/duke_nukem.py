from aiarena21.client.classes import Player, Map

## Moves towards the opponent then spends all its points teleporting them to a random location

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
    # move towards the opponent

    dy = sign(opponent.location[0] - me.location[0])
    dx = sign(opponent.location[1] - me.location[1])

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
    # spend as much as humanly possible to win every auction
    return min(opponent.score + 1, me.score)


def play_transport(game_map: Map, me: Player, opponent: Player, items: list, new_items: list, heatmap, remaining_turns):
    return f'{random.randint(0, game_map.rows-1)},{random.randint(0, game_map.cols-1)}'
