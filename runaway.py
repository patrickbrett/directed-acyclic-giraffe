from aiarena21.client.classes import Player, Map

import random


def dist(a, b):
    return abs(a[0]-b[0]) + abs(a[1]-b[1])


def play_powerup(game_map: Map, me: Player, opponent: Player, items: list, new_items: list, heatmap, remaining_turns):
    if dist(me.location, opponent.location) <= 3:
        return 'bike'
    return ''


def play_turn(game_map: Map, me: Player, opponent: Player, items: list, new_items: list, heatmap, remaining_turns):
    dx = 1 if me.location[0] > opponent.location[0] else -1
    dy = 1 if me.location[1] > opponent.location[1] else -1
    if me.bike:
        return f'{me.location[0] + dx},{me.location[1] + dy}'
    else:
        if random.randint(0, 1) == 0:
            return f'{me.location[0] + dx}, {me.location[1]}'
        else:
            return f'{me.location[0]},{me.location[1] + dy}'


def play_auction(game_map: Map, me: Player, opponent: Player, items: list, new_items: list, heatmap, remaining_turns):
    return random.randint(0, min(opponent.score, me.score))


def play_transport(game_map: Map, me: Player, opponent: Player, items: list, new_items: list, heatmap, remaining_turns):
    if dist((0, 0), me.location) > dist((game_map.rows - 1, game_map.cols - 1), me.location):
        return '0,0'
    else:
        return f'{game_map.rows-1},{game_map.cols-1}'