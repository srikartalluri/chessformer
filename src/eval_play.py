from stockfish import Stockfish
import chess
from engine import Engine
from tokenizer import ChessTokenizer
import json
import random


stockfish = Stockfish(path = "../stockfish/stockfish", depth=1, parameters={"Threads": 1, "Minimum Thinking Time": 30})
stockfish.set_skill_level(1)

# stockfish.set_position(["e2e4", "e7e6", "d2d4"])

# best_move = stockfish.get_best_move()

# print(best_move)

def play_once(engine: Engine, model_first=True, print_moves=False):
    board = chess.Board()
    while not board.is_game_over():
        if model_first:
            move_list = [str(move) for move in board.move_stack]
            move = engine.get_next_move(move_list)
            board.push(chess.Move.from_uci(move))
            if print_moves:
                print(move)
            if board.is_game_over():
                return 1
            
            move_list = [str(move) for move in board.move_stack]
            stockfish.set_position(move_list)
            move = stockfish.get_best_move()
            board.push(chess.Move.from_uci(move))
            if print_moves:
                print(move)
            if board.is_game_over():
                return 0
            

        else:
            move_list = [str(move) for move in board.move_stack]
            stockfish.set_position(move_list)
            move = stockfish.get_best_move()
            board.push(chess.Move.from_uci(move))
            if print_moves:
                print(move)
            if board.is_game_over():
                return 0
            
            move_list = [str(move) for move in board.move_stack]
            move = engine.get_next_move(move_list)
            board.push(chess.Move.from_uci(move))
            if print_moves:
                print(move)
            if board.is_game_over():
                return 1
    return 2

def play_n(engine: Engine, n, print_moves=False):
    games = 0
    wins = 0
    model_first = random.choice([True, False])
    for i in range(n):
        ret = play_once(engine, model_first=model_first, print_moves=print_moves)
        if ret == 1:
            wins += 1
        games += 1
    
    print(f"Wins: {wins}/{games}")

    return wins / games, wins



# with open('config.json', 'r') as config_file:
#         config = json.load(config_file)

# tok = ChessTokenizer()
# engine = Engine(model_path="../models/model_10.pt", tokenizer=tok, model_config=config)

# # ret = play_once(engine, stockfish, model_first=False)
# ret = play_n(engine, stockfish, 10, print_moves=False)
# print(ret)