from aiarena21.client.classes import Player, Map
from aiarena21.server.game import Game
from aiarena21.server.player import Player as ServerPlayer
import aiarena21.server.settings as settings

import copy
import random
import sys, os



# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')


# Restore
def enablePrint():
    sys.stdout = sys.__stdout__


class Mock(Game):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def s_hash(self):
        s = ""
        s += str(self.map) + \
             str(self.current_round) + \
             ",".join(list(map(lambda x: str(x.score), self.players)))
        return s


class State:
    def __init__(self, game: Mock):
        self.game = game
        all_states = dict()
        self.children = []

    def simulate_round(self, game, players, moves):
        game.deploy_items()
        game.update_heatmap()
        ask_powerups(game)
        for i in range(2):
            blockPrint()
            players[i].play_move(game, moves[i])
            enablePrint()

        locations_before_wager = [game.players[i].location for i in range(2)]
        # This is a while instead of an if so if the random location after transport is still the same location
        # auction would happen again
        wagers = None
        # while players[0].location == players[1].location or round_counter - last_auction == 30:
        #     wagers = start_auction(game)
        #     last_auction = round_counter

        for player in game.players:
            player.pickup_items(game)
            player.update_powerups()

        if wagers is not None:
            wagers = {
                'wagers': wagers,
                'before_positions': locations_before_wager
            }
        game.current_round += 1

    def simulate(self, game, players, moves_ahead, initial_move, initial_score):
        for round_counter in range(min(game.total_rounds, moves_ahead)):
            moves = get_moves(game)
            if round_counter == 0:
                moves[0] = initial_move
            self.simulate_round(game, players, moves)
        print(players[0].score, players[1].score)
        return (players[0].score - initial_score[0]) - (players[1].score - initial_score[1])


def log(*args, **kwargs):
    pass


def ask_powerups(game):
    for player in game.players:
        powerup_request = play_random_powerup()
        if powerup_request not in ['bike', 'portal gun', '']:
            log(f'{player.name} requested invalid powerup: {powerup_request}')
        else:
            if powerup_request == 'bike':
                if player.equip_bike(game):
                    log(f'{player.name} got bike powerup.')
                else:
                    log(f'{player.name} did not have enough scores for bike powerup.')
            elif powerup_request == 'portal gun':
                if player.equip_portal_gun(game):
                    log(f'{player.name} got portal gun powerup.')
                else:
                    log(f'{player.name} did not have enough scores for portal gun powerup.')
            else:
                log(f'{player.name} did not purchase a powerup.')


def get_moves(game):
    moves = []
    for player in game.players:
        move = play_random_turn(player)
        log(f'{player.name} played {move} for their turn.')
        moves.append(move)
    return moves


# def auction_transport(game, winner, cost):
#     loser = 1 - winner
#     game.players[winner].score -= cost
#     waiting_id = network.send_player(game.players[winner], {'type': 'transport'})
#     location = network.recv_player(game.players[winner], waiting_id)
#     transport_reg = re.compile(f'([0-9]+),([0-9]+)')
#     if not transport_reg.fullmatch(location):
#         game.transport_random(game.players[loser])
#     else:
#         row, col = map(int, transport_reg.fullmatch(location).groups())
#         if not game.cell_available(row, col):
#             game.transport_random(game.players[loser])
#         else:
#             game.players[loser].update_location(row, col)


# def start_auction(game):
#     amounts = []
#     for player in game.players:
#         amount = play_random_auction(game.players[0], game.players[1])
#         log(f'{player.name} wagered {amount} for their auction.')
#         amounts.append(amount)
#
#     for i in range(2):
#         try:
#             amounts[i] = int(amounts[i])
#             if not 0 <= amounts[i] <= game.players[i].score:
#                 log(f'Invalid wager from {game.players[i].name}: {amounts[i]}')
#                 amounts[i] = 0
#         except (ValueError, TypeError):
#             log(f'Non-numeric wager from {game.players[i].name}: {amounts[i]}')
#             amounts[i] = 0
#
#     avg_wager = sum(amounts) // 2
#     if amounts[0] > amounts[1]:
#         auction_transport(game, 0, avg_wager)
#     elif amounts[1] > amounts[0]:
#         auction_transport(game, 1, avg_wager)
#     else:
#         game.transport_random(game.players[0])
#         game.transport_random(game.players[1])
#     return amounts

def generate_moves(game_map, me):
    moves = []
    for dir in [1, -1]:
        for axis in [0, 1]:
            move = [0, 0]
            move[axis] = dir
            new_pos = [me.location[i] + move[i] for i in range(2)]
            if new_pos[1] in range(game_map.cols) and \
                    new_pos[0] in range(game_map.rows) and \
                    game_map.is_free(*new_pos):
                moves.append(f'{new_pos[0]},{new_pos[1]}')
    return moves


def generate_s_players(me, opponent):
    s_players = [ServerPlayer("0", "0"), ServerPlayer("1", "1")]
    c_players = [me, opponent]
    for i in range(len(s_players)):
        s_players[i].name = c_players[i].name
        s_players[i].score = copy.deepcopy(c_players[i].score)
        s_players[i].location = copy.deepcopy(c_players[i].location)
    return s_players


def generate_items_map(items):
    class Item:
        def __init__(self, points):
            self.points = points
    empty = True
    for i in range(len(items)):
        for j in range(len(items[i])):
            if items[i][j] > 1:
                empty = False
            items[i][j] = [Item(10)]*items[i][j]
    return items, empty


def output(*args, **kwargs):
    print("MCTS OUTPUT: ", *args, **kwargs)

def play_powerup(game_map: Map, me: Player, opponent: Player, items: list, new_items: list, heatmap, remaining_turns):
    return random.choice([''])


def play_turn(game_map: Map, me: Player, opponent: Player, items: list, new_items: list, heatmap, remaining_turns):
    sims = 30
    moves = generate_moves(game_map, me)
    output(me.location, moves)
    settings.TOTAL_ROUNDS = remaining_turns
    items_map, empty = generate_items_map(items)
    visits = [0] * len(moves)
    wins = [0] * len(moves)
    best_result, best_move = -float("inf"), moves[0]
    if not empty:
        for sim in range(sims):
            output("Starting sim", sim)
            for i, move in enumerate(moves):
                s_players = generate_s_players(me, opponent)
                state = State(Mock(s_players))
                state.game.map = copy.deepcopy(game_map._map)
                state.game.items_map = copy.deepcopy(items_map)

                sim_res = state.simulate(state.game, s_players, 10, move, (me.score, opponent.score))
                output("location")
                visits[i] += 1
                wins[i] += sim_res
                win_rate = wins[i] / visits[i]
                if win_rate > best_result:
                    best_result, best_move = win_rate, move
    output(wins)
    output(me.location, moves)
    return best_move


def play_auction(game_map: Map, me: Player, opponent: Player, items: list, new_items: list, heatmap, remaining_turns):
    return random.randint(0, min(opponent.score, me.score))


def play_transport(game_map: Map, me: Player, opponent: Player, items: list, new_items: list, heatmap, remaining_turns):
    return f'{random.randint(0, game_map.rows-1)},{random.randint(0, game_map.cols-1)}'








def play_random_powerup():
    # return random.choice(['bike', 'portal gun', ''])
    return random.choice([''])


def play_random_turn(me: Player):
    dx, dy = 1, 1
    while dx * dy != 0 or dx + dy == 0:
        dx = random.randint(-1, 1)
        dy = random.randint(-1, 1)
    new_row = me.location[0] + dx
    new_col = me.location[1] + dy
    return f'{new_row},{new_col}'


def play_random_auction(me: Player, opponent: Player):
    return random.randint(0, min(opponent.score, me.score))


def play_random_transport(game_map: Map):
    return f'{random.randint(0, game_map.rows - 1)},{random.randint(0, game_map.cols - 1)}'



if __name__ == "__main__":
    ps = [ServerPlayer("0", "0"), ServerPlayer("1", "1")]
    settings.TOTAL_ROUNDS = 100
    s = State(Mock(ps))
    print(s.simulate(s.game, ps, 0))
