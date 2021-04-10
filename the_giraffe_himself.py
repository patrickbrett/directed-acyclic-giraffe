from aiarena21.client.classes import Player, Map

import random
from timeit import default_timer as timer

total_rounds = None

def play_powerup(game_map: Map, me: Player, opponent: Player, items: list, new_items: list, heatmap, remaining_turns):
    global total_rounds
    if total_rounds is None: total_rounds = remaining_turns
  
    vehicles = ['bike', 'portal gun', '']
    cost = [30, 100, 0]
    best_veh, best_score = '', -float("inf")
    for i, vehicle in enumerate(vehicles):
        start = timer()
        next_move, value = path_search(game_map, me, opponent, items, new_items, heatmap,
                                          remaining_turns, depth=35, vehicle=vehicle)
        value -= cost[i]
        #print(timer() - start)
        if value > best_score:
            best_veh, best_score = vehicle, value

    if remaining_turns < 15:
        best_veh = ''

    return best_veh


def parse_vehicle(player):
    vehicle = ""
    if player.bike:
        vehicle = "bike"
    elif player.portal_gun:
        vehicle = 'portal gun'
    return vehicle


def play_turn(game_map: Map, me: Player, opponent: Player, items: list, new_items: list, heatmap, remaining_turns):
    #if remaining_turns < 80:
        #print(play_auction(game_map, me, opponent, items, new_items, heatmap, remaining_turns))
    """
    head example:
        (row, col)
    """
    print("The Giraffe Himself: "+me.name)
    start = timer()

    me_vehicle = parse_vehicle(me)
    opp_vehicle = parse_vehicle(opponent)
    
    # find opp best move
    opp_max_path, opp_max_move, opp_max_value = path_search(game_map, opponent, me, items, new_items, heatmap, remaining_turns,
                                      depth=50, vehicle=opp_vehicle, ret_path=True)

    max_move, max_value = path_search(game_map, me, opponent, items, new_items, heatmap, remaining_turns,
                                      depth=50, vehicle=me_vehicle, opp_max_path=opp_max_path)
    #print(timer() - start)
    return f'{max_move[0]},{max_move[1]}'


def play_auction(game_map: Map, me: Player, opponent: Player, items: list, new_items: list, heatmap, remaining_turns):
    # find the value of player's square

    player_max_move, m = path_search(game_map, me, opponent, items, new_items, heatmap, remaining_turns,
                                      depth=20)

    #print('player m', player_max_move, m)

    # find the value of the opponent's square

    # notice that 'me' and 'opponent' args are swapped here
    opp_max_move, g = path_search(game_map, opponent, me, items, new_items, heatmap, remaining_turns,
                                      depth=20)

    #print('opp g', opp_max_move, g)

    # find the value of the worst spot on the map

    # check the four corners plus the centre
    max_r = game_map.rows - 1
    max_c = game_map.cols - 1
    places = [(0, 0), (max_r, 0), (0, max_c), (max_r, max_c), (max_r // 2, max_c // 2)]
    # if any are in walls, replace them with somewhere random
    for i in range(len(places)):
        r, c = places[i]
        while not game_map.is_free(r, c):
            places[i] = (random.randint(0, max_r), random.randint(0, max_c))
            r, c = places[i]
        
    # add more random places
    num_random_to_add = 10
    for i in range(num_random_to_add):
        places.append((random.randint(0, max_r), random.randint(0, max_c)))
    
    #print('places: ', places)
    
    # find value of each place
    x_loc = places[0]
    x = float('inf')
    for place in places:
        place_formatted = PlaceFormatted(place)
        place_loc, place_value = path_search(game_map, place_formatted, opponent, items, new_items, heatmap, remaining_turns, depth=20)
        if place_value < x:
            x = place_value
            x_loc = place_loc
    
    #print('worst square', x_loc, x)
    global worst_square
    worst_square = x_loc
    
    # amount to bid
    required_profit = 0.2 # 20% profit on bid
    bid = ((g - x) - (x - m)) * (1 - required_profit)
    bid_rounded = min(max(int(bid), 0), 100)

    #print('bidding: ', bid_rounded, ' (unrounded):', bid)

    return bid_rounded



def play_transport(game_map: Map, me: Player, opponent: Player, items: list, new_items: list, heatmap, remaining_turns):
    if worst_square is not None:
        #print('sending opponent to the worst square: ', worst_square)
        return f'{worst_square[0]},{worst_square[1]}'
    
    return f'{random.randint(0, game_map.rows - 1)},{random.randint(0, game_map.cols - 1)}'


def generate_moves(game_map, origin, vehicle=""):
    if vehicle == "":
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
    elif vehicle == "portal gun":
        moves = set()
        for r in range(game_map.rows):
            for c in range(game_map.cols):
                if game_map.is_free(r, c):
                    moves.add((r, c))
    return moves


def path_search(game_map: Map, me: Player, opponent: Player, items: list, new_items: list,
                heatmap, remaining_turns, depth=40, vehicle="", discount=0.9, ret_path=False, opp_max_path=None):
    available = {tuple(me.location)}
    paths = [[[] for _ in range(game_map.cols)] for _ in range(game_map.rows)]
    paths[me.location[0]][me.location[1]] = [tuple(me.location)]
    value = [[-1] * game_map.cols for _ in range(game_map.rows)]
    value[me.location[0]][me.location[1]] = 0
    depth = min(depth, remaining_turns)
    value_per_item = 10

    max_move, max_value = None, -1

    max_path = None
    vehicle_moves = {'bike': 3, 'portal gun': 1, '': 0}[vehicle]
    for d in range(depth):
        new_available = set()
        next_fruit_spawn = (remaining_turns - 1 - total_rounds)% 20
        if d >= vehicle_moves:
            vehicle = ""
        for head in available:
            pre_r, pre_c = head
            moves = generate_moves(game_map, head, vehicle)
            for move in moves:
                r, c = move
                # calculate new value
                new_value = value[pre_r][pre_c]
                if move not in paths[pre_r][pre_c]:
                    square_value = items[r][c]
                    if d >= next_fruit_spawn:
                        square_value += heatmap[r][c] * 0.9 ** d
                    new_value += square_value * discount ** d

                    multiplier = get_multiplier(opponent, move, d, opp_max_path)
                    new_value *= multiplier
                # consider update cell
                if new_value > value[r][c]:
                    value[r][c], paths[r][c] = new_value, paths[pre_r][pre_c] + [move]
                # consider update max
                if new_value > max_value and len(paths[r][c]) >= 2:
                        max_value, max_move = new_value, paths[r][c][1]
                        max_path = paths[r][c]
                new_available.add(move)
        available = new_available
    if max_move is None: max_move = me.location

    if ret_path:
        return max_path, max_move, max_value

    return max_move, max_value

class PlaceFormatted:
    def __init__(self, place_tuple):
        self.location = place_tuple


def get_multiplier(opponent, move, d, opp_max_path, z=1):
    if opp_max_path == None or True:
        return 1 # no multiplier

    # check if move is in the path
    dest_r, dest_c = move
    if move in opp_max_path:
        g = opp_max_path.index(move)
    else:
        return 1
    
    m = d
    if g > m and g < 5:
        multiplier = 1 + ((g - m) ** (-z)) * 0.5
    elif g < m:
        multiplier = 0.5 * g / m
    else: # g == m
        multiplier = 1
    
    return multiplier
