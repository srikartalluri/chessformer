import boto3
from dotenv import load_dotenv
import os
import torch
from torch.utils.data import IterableDataset
import chess
import chess.pgn
import tokenizer
import s3fs


# load_dotenv()


# s3 = boto3.client('s3')
# response = s3.list_objects_v2(Bucket='chessformerdata', Prefix='')
# files = [obj["Key"] for obj in response.get('Contents', [])]
# print(files)


class ChessDataset(IterableDataset):
    def __init__(self, tokenizer: tokenizer.ChessTokenizer, from_local = True, prefix = "", max_seq_len=256):
        self.tokenizer = tokenizer
        self.vocab_size = tokenizer.vocabulary_size()
        self.max_seq_len = max_seq_len

        self.bucket = "chessformerdata"
        self.prefix = prefix
        self.fs = s3fs.S3FileSystem()
        
        self.files = self.fs.ls(f"{self.bucket}/{self.prefix}")
        print(self.files)
    

    def get_legal_moves(self, board):
        """
        Board is a chess.Board object
        return mask of legal moves 1 where legal 0 where illegal across entire vocab size
        """
        legal_moves = list(board.legal_moves)
        legal_moves_str = [str(move) for move in legal_moves]
        legal_mask = [0] * self.vocab_size
        if len(legal_moves_str) == 0:
            return legal_mask
        legal_moves_ids = self.tokenizer.tokens_to_ids_vect(legal_moves_str)
        for id in legal_moves_ids:
            legal_mask[id] = 1
        
        return legal_mask

    def get_data_from_game(self, game):
        move_list = list(game.mainline_moves())
        move_list_str = [str(move) for move in move_list]
        game_len = len(move_list_str)
        if game_len == 0 or game_len > self.max_seq_len - 1:
            return None

        token_ids = [0] * self.max_seq_len
        attn_mask = [0] * self.max_seq_len
        labels = [0] * self.max_seq_len
        legal_masks = [[0] * self.vocab_size] * self.max_seq_len

        board = game.board()
        token_ids[0] = self.tokenizer.get_bos()
        attn_mask[0] = 1
        legal_masks[0] = self.get_legal_moves(board)
        labels[0] = self.tokenizer.tokens_to_ids_single(move_list_str[0])

        for i in range(1, game_len):

                cur_id = self.tokenizer.tokens_to_ids_single(move_list_str[i - 1])
                token_ids[i] = cur_id
                attn_mask[i] = 1
                labels[i - 1] = cur_id
                board.push(move_list[i - 1])
                legal_masks[i] = self.get_legal_moves(board)
            
        labels[game_len - 1] = self.tokenizer.tokens_to_ids_single(move_list_str[game_len - 1])

        return torch.tensor(token_ids), torch.tensor(attn_mask), torch.tensor(legal_masks, dtype=torch.bool), torch.tensor(labels)

    def is_good_game(self, game):
        # required_headers = ["White", "Black", "Result"]
        # for header in required_headers:
        #     if header not in game.headers or not game.headers[header]:
        #         return False
        
        valid_results = {"1-0", "0-1", "1/2-1/2"}
        if game.headers["Result"] not in valid_results:
            return False
        moves = list(game.mainline_moves())
        if len(moves) < 10 or len(moves) > 200:
            return False

        termination = game.headers.get("Termination", "").lower()
        if termination and ("abandoned" in termination or "forfeit" in termination):
            return False

        return True        

    def __iter__(self):
        
        for file_key in self.files:
            if "training" not in file_key:
                continue
            print(file_key)
            
            with self.fs.open(file_key, 'r') as f:
                while True:
                    game = chess.pgn.read_game(f)
                    if game is None:
                        break
                    if self.is_good_game(game):
                        yield self.get_data_from_game(game)

        # return self.generate2()