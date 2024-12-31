import torch
import numpy as np
from torch.utils.data import IterableDataset
import chess
import chess.pgn

class ChessDataset(IterableDataset):
    def __init__(self, file_name, tokenizer, max_seq_len=256):
        self.file_name = file_name
        self.tokenizer = tokenizer
        self.vocab_size = tokenizer.vocabulary_size()
        self.fd = open(file_name, 'r')
        self.max_seq_len = max_seq_len
        self.game = chess.pgn.read_game(self.fd)




    def generate(self):
        """
        Yields token_ids (seq_len)
        attn_mask (seq_len)
        legal_mask (seq_len x vocab_size)
        """
        while self.game:
            move_list = self.game.mainline_moves()
            move_list_str = [str(move) for move in list(move_list)]
            game_len = len(move_list_str)
            if game_len == 0 or game_len > self.max_seq_len:
                self.game = chess.pgn.read_game(self.fd)
                continue


            random_index = np.random.randint(0, game_len)
            random_move = move_list_str[random_index]

            random_move_id = self.tokenizer.tokens_to_ids_single(random_move)

            board = self.game.board()
            counter = 0
            for move in move_list:
                if counter >= random_index:
                    break
                board.push(move)
                counter += 1
            
            legal_moves = list(board.legal_moves)
            legal_moves_str = [str(move) for move in legal_moves]
            legal_moves_ids = self.tokenizer.tokens_to_ids_vect(legal_moves_str)


            legal_mask = [0] * self.vocab_size

            for id in legal_moves_ids:
                legal_mask[id] = 1
            
            token_ids = [0] * self.max_seq_len
            attn_mask = [0] * self.max_seq_len

            move_list_str = move_list_str[:random_index]

            for i in range(len(move_list_str)):
                token_ids[i] = self.tokenizer.tokens_to_ids_single(move_list_str[i])
                attn_mask[i] = 1
            
            yield torch.tensor(token_ids), torch.tensor(attn_mask), torch.tensor(legal_mask).bool(), torch.tensor(random_index - 1), torch.tensor(random_move_id)

                
                
            self.game = chess.pgn.read_game(self.fd)

    def __iter__(self):
        return iter(self.generate())

