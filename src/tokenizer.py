import numpy as np

class ChessTokenizer():

    def __init__(self):
        with open('./moves.txt', 'r') as file:
            lines = file.readlines()
        
        moves = [line.strip() for line in lines]
        vocab = ['', "[BOS]"] + moves + ["[UNK]"]
        ids_to_tokens = dict(enumerate(vocab))
        tokens_to_ids = {v: k for k, v in ids_to_tokens.items()}

        self.ids_to_tokens = ids_to_tokens
        self.tokens_to_ids = tokens_to_ids

    def get_bos(self):
        return self.tokens_to_ids["[BOS]"]

    def vocabulary_size(self):
        return len(self.ids_to_tokens)

    def tokens_to_ids_single(self, move):
        if move in self.tokens_to_ids:
            return self.tokens_to_ids[move]
        else:
            return self.tokens_to_ids["[UNK]"]

    def ids_to_tokens_single(self, id):
        if id in self.ids_to_tokens:
            return self.ids_to_tokens[id]
        else:
            return "[UNK]"
    
    def tokens_to_ids_vect(self, moves):
        return np.vectorize(self.tokens_to_ids_single)(moves)
    
    def ids_to_tokens_vect(self, ids):
        return np.vectorize(self.ids_to_tokens_single)(ids)

    def get_mask_id(self):
        return self.tokens_to_ids["[MASK]"]

    def vocabulary_size(self):
        return len(self.ids_to_tokens)