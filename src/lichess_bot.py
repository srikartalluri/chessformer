import requests
import json
import time
import chess
from engine import Engine

import berserk
import threading

from tokenizer import ChessTokenizer

from transformers import GPT2Config
from utils import setup

from dotenv import load_dotenv
import os
load_dotenv(override=True)

# Set up API token and Lichess API URL

API_TOKEN = os.getenv('LICHESS_KEY')

model, tok, config, gpu_device = setup()

engine = Engine(model_path = "../models/model_10.pt", model_config = config, tokenizer = tok, model = model)

session = berserk.TokenSession(API_TOKEN)
client = berserk.Client(session)


def play_game(game_id):
    board = chess.Board()
    is_bot_white = None
    is_bot_black = None

    for event in client.bots.stream_game_state(game_id):

        if event['type'] == 'gameFull':
            print("game full")
            moves = event['state']['moves']
            print("moves", moves)
            is_bot_white = int(event['white']['id'] == "srikar_bot")
            is_bot_black = 1 - is_bot_white
            print("set bot white and black", is_bot_white)

            if is_bot_white:
                next_move = engine.get_next_move([])
                print("first move", next_move)
                client.bots.make_move(game_id, next_move)


        if event['type'] == 'gameState':
            print("game state")
            moves = event['moves']
            print("moves", moves)
            moves = moves.split(" ")
            print("split moves", moves)

            if is_bot_white and len(moves) % 2 == 0:
                next_move = engine.get_next_move(moves)
                print("next move", next_move)
                client.bots.make_move(game_id, next_move)
            
            if is_bot_black and len(moves) % 2 == 1:
                next_move = engine.get_next_move(moves)
                print("next move", next_move)
                client.bots.make_move(game_id, next_move)


for event in client.bots.stream_incoming_events():
    print("---")
    print(event)

    if event['type'] == 'challenge':

        client.bots.accept_challenge(event['challenge']['id'])
        thread = threading.Thread(target=play_game, args=(f"{event['challenge']['id']}", ))
        thread.start()