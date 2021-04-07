from aiarena21.client.classes import Player, Map

import random


def play_powerup(game_map: Map, me: Player, opponent: Player, items: list, new_items: list, heatmap, remaining_turns):
    return random.choice(['bike', 'portal gun', ''])


def play_turn(game_map: Map, me: Player, opponent: Player, items: list, new_items: list, heatmap, remaining_turns):
    dx, dy = 1, 1
    while dx*dy != 0 or dx+dy == 0:
        dx = random.randint(-1, 1)
        dy = random.randint(-1, 1)
    new_row = me.location[0] + dx
    new_col = me.location[1] + dy
    return f'{new_row},{new_col}'


def play_auction(game_map: Map, me: Player, opponent: Player, items: list, new_items: list, heatmap, remaining_turns):
    return random.randint(0, min(opponent.score, me.score))


def play_transport(game_map: Map, me: Player, opponent: Player, items: list, new_items: list, heatmap, remaining_turns):
    return f'{random.randint(0, game_map.rows-1)},{random.randint(0, game_map.cols-1)}'
