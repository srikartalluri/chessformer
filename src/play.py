import requests
import json
import time
import chess
import chess.engine

# Set up API token and Lichess API URL
API_TOKEN = "lip_9J9hHE4WhMf5AvIVexDf"
API_URL = "https://lichess.org/api"

headers = {
    "Authorization": f"Bearer {API_TOKEN}"
}

# Initialize the chess engine (e.g., Stockfish)
engine = chess.engine.SimpleEngine.popen_uci("/opt/homebrew/bin/stockfish")  # Update path to your engine

# Function to accept and play a game
def play_game(game_id):
    board = chess.Board()
    game_url = f"{API_URL}/bot/game/stream/{game_id}"

    # Connect to the game stream
    response = requests.get(game_url, headers=headers, stream=True)
    
    for line in response.iter_lines():
        if line:
            event = json.loads(line.decode("utf-8"))

            # If it's our turn, make a move
            if event.get("type") == "gameState" and event.get("status") == "started":
                moves = event["moves"].split()
                for move in moves:
                    board.push(chess.Move.from_uci(move))
                
                if board.turn:
                    result = engine.play(board, chess.engine.Limit(time=0.1))
                    move = result.move
                    board.push(move)

                    # Send move to Lichess
                    move_url = f"{API_URL}/bot/game/{game_id}/move/{move.uci()}"
                    requests.post(move_url, headers=headers)
                    
            # Handle game end events
            if event.get("type") == "gameFinish":
                print("Game over")
                break

# Function to accept challenges
def accept_challenges():
    while True:
        response = requests.get(f"{API_URL}/bot/account/playing", headers=headers)

        # response = requests.get(f"{API_URL}/challenge", headers=headers)

        # response = response.json()

        # print(response)

        games = response.json().get("nowPlaying", [])


        print(games)
        if not games:
            # Accept new challenges
            challenge_response = requests.get(f"{API_URL}/challenge", headers=headers)
            challenges = challenge_response.json().get("in", [])

            for challenge in challenges:
                if challenge["variant"]["key"] == "standard":  # Accept only standard games
                    challenge_id = challenge["id"]
                    requests.post(f"{API_URL}/bot/accept/{challenge_id}", headers=headers)
                    print(f"Accepted challenge {challenge_id}")

        # Play ongoing games
        for game in games:
            play_game(game["gameId"])

        time.sleep(1)

# Start accepting challenges
accept_challenges()

# Cleanup
engine.quit()
