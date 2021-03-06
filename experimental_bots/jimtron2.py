from aiarena21.client.classes import Player, Map

import random
from timeit import default_timer as timer

Total_Rounds = None

def play_powerup(game_map: Map, me: Player, opponent: Player, items: list, new_items: list, heatmap, remaining_turns):
  
    global Total_Rounds
    if Total_Rounds is None: Total_Rounds = remaining_turns
    vehicles = ['bike', 'portal gun', '']
    cost = [30, 100, 0]
    best_veh, best_score = '', -float("inf")
    for i, vehicle in enumerate(vehicles):
        start = timer()
        next_move, value = path_search(game_map, me, opponent, items, new_items, heatmap,
                                          remaining_turns, depth=35, vehicle=vehicle, discount=0.7)
        value -= cost[i]
        print(value, timer() - start)
        if value > best_score:
            best_veh, best_score = vehicle, value

    return best_veh


def play_turn(game_map: Map, me: Player, opponent: Player, items: list, new_items: list, heatmap, remaining_turns):
    """
    head example:
        (row, col)
    """
    print(Total_Rounds)
    print('jimtron2: '+me.name+', '+str(me.score))
    start = timer()
    vehicle = ""
    if me.bike:
        vehicle = "bike"
    elif me.portal_gun:
        vehicle = 'portal gun'
    print("remaining_turns", remaining_turns)
    max_move, max_value = path_search(game_map, me, opponent, items, new_items, heatmap, remaining_turns,
                                      depth=50, vehicle=vehicle)
    print(timer() - start)
    return f'{max_move[0]},{max_move[1]}'


def play_auction(game_map: Map, me: Player, opponent: Player, items: list, new_items: list, heatmap, remaining_turns):
    # find the value of player's square

    player_max_move, m = path_search(game_map, me, opponent, items, new_items, heatmap, remaining_turns,
                                      depth=20)

    print('player m', player_max_move, m)

    # find the value of the opponent's square

    # notice that 'me' and 'opponent' args are swapped here
    opp_max_move, g = path_search(game_map, opponent, me, items, new_items, heatmap, remaining_turns,
                                      depth=20)

    print('opp g', opp_max_move, g)

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
    
    print('places: ', places)
    
    # find value of each place
    x_loc = places[0]
    x = float('inf')
    for place in places:
        place_formatted = PlaceFormatted(place)
        place_loc, place_value = path_search(game_map, place_formatted, opponent, items, new_items, heatmap, remaining_turns, depth=20)
        if place_value < x:
            x = place_value
            x_loc = place_loc
    
    print('worst square', x_loc, x)
    global worst_square
    worst_square = x_loc
    
    # amount to bid
    required_profit = 0.2 # 20% profit on bid
    bid = ((g - x) - (x - m)) * (1 - required_profit)
    bid_rounded = max(int(bid), 0)

    print('bidding: ', bid_rounded, ' (unrounded):', bid)

    return bid_rounded



def play_transport(game_map: Map, me: Player, opponent: Player, items: list, new_items: list, heatmap, remaining_turns):
    if worst_square is not None:
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
                heatmap, remaining_turns, depth=40, vehicle="", discount=0.9):

    available = {tuple(me.location)}
    paths = [[[] for _ in range(game_map.cols)] for _ in range(game_map.rows)]
    paths[me.location[0]][me.location[1]] = [tuple(me.location)]
    value = [[-1] * game_map.cols for _ in range(game_map.rows)]
    value[me.location[0]][me.location[1]] = 0
    depth = min(50, remaining_turns)
    value_per_item = 10

    max_move, max_value = None, -1

    vehicle_moves = {'bike': 3, 'portal gun': 1, '': 0}[vehicle]
    for d in range(depth):
        new_available = set()
        next_fruit_spawn = (remaining_turns - 1 - Total_Rounds) % 20
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




if __name__ == "__main__":
    class StubMap:
        def __init__(self):
            self.rows = 10
            self.cols = 10

        def is_free(self, *args):
            return True


    g = StubMap()
    print(generate_moves(g, (5,5), "bike"))
