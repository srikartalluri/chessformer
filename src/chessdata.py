import torch
from torch.utils.data import IterableDataset
import chess
import chess.pgn
import tokenizer

class ChessDataset(IterableDataset):
    def __init__(self, file_name, tokenizer: tokenizer.ChessTokenizer, max_seq_len=256):
        self.file_name = file_name
        self.tokenizer = tokenizer
        self.vocab_size = tokenizer.vocabulary_size()
        self.fd = open(file_name, 'r')
        self.max_seq_len = max_seq_len
        self.game = chess.pgn.read_game(self.fd)
    

    def get_legal_moves(self, board):
        """
        Board is a chess.Board object
        return mask of legal moves 1 where legal 0 where illegal across entire vocab size
        """
        legal_moves = list(board.legal_moves)
        legal_moves_str = [str(move) for move in legal_moves]
        # print(legal_moves_str)
        legal_mask = [0] * self.vocab_size
        if len(legal_moves_str) == 0:
            return legal_mask
        legal_moves_ids = self.tokenizer.tokens_to_ids_vect(legal_moves_str)


        

        for id in legal_moves_ids:
            legal_mask[id] = 1
        
        return legal_mask

    def generate(self):
        """
        Yields 
        token_ids (seq_len)
        attn_mask (seq_len)
        legal_mask (seq_len x vocab_size)
        labels (seq_len)
        """
        while self.game:
            move_list = self.game.mainline_moves()
            move_list = list(move_list)
            move_list_str = [str(move) for move in move_list]
            game_len = len(move_list_str)
            if game_len == 0 or game_len > self.max_seq_len - 1:
                self.game = chess.pgn.read_game(self.fd)
                continue

            token_ids = [0] * self.max_seq_len
            attn_mask = [0] * self.max_seq_len
            labels = [0] * self.max_seq_len
            legal_masks = [[0] * self.vocab_size] * self.max_seq_len

            board = self.game.board()
            token_ids[0] = self.tokenizer.get_bos()
            attn_mask[0] = 1
            legal_masks[0] = self.get_legal_moves(board)
            labels[0] = self.tokenizer.tokens_to_ids_single(move_list_str[0])

            
            for i in range(1, game_len):

                cur_id = self.tokenizer.tokens_to_ids_single(move_list_str[i - 1])
                token_ids[i] = cur_id
                attn_mask[i] = 1
                labels[i - 1] = cur_id
                # print(move_list_str[i])
                board.push(move_list[i - 1])
                legal_masks[i] = self.get_legal_moves(board)
            
            labels[game_len - 1] = self.tokenizer.tokens_to_ids_single(move_list_str[game_len - 1])

            yield torch.tensor(token_ids), torch.tensor(attn_mask), torch.tensor(legal_masks, dtype=torch.bool), torch.tensor(labels)
            self.game = chess.pgn.read_game(self.fd)
        

    def __iter__(self):
        return self.generate()