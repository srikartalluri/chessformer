import chess
from tokenizer import ChessTokenizer
import torch
import torch.nn as nn
from model import ChessGPTModel
from transformers import GPT2Config, GPT2LMHeadModel
from chessdata import ChessDataset


class Engine():
    
    def __init__(self, model_path, model_config: GPT2Config):
        # self.model = ChessGPTModel(model_config)
        self.model = GPT2LMHeadModel(model_config)
        self.model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        self.model.eval()
        self.tok = ChessTokenizer()
        self.dataset = ChessDataset('./data/test.pgn', self.tok)
        self.tokenizer = ChessTokenizer()
        self.seq_len = model_config.n_positions
        self.vocab_size = model_config.vocab_size
        self.mps_device = torch.device("mps")

        self.model.to(self.mps_device)
    
    def get_next_move(self, move_list):
        board = chess.Board()

        token_ids = [0] * self.seq_len
        attn_mask = [0] * self.seq_len


        move_list_w_bos = ['[BOS]'] + move_list
        move_list_id = self.tokenizer.tokens_to_ids_vect(move_list_w_bos)

        for i in range(1, len(move_list_id)):
            token_ids[i] = move_list_id[i]
            attn_mask[i] = 1

        token_ids = torch.tensor(token_ids)
        attn_mask = torch.tensor(attn_mask)

        # print(token_ids.shape)
        # print(attn_mask.shape)

        token_ids = torch.reshape(token_ids, (1, -1)).to(self.mps_device)
        attn_mask = torch.reshape(attn_mask, (1, -1)).to(self.mps_device)


        legal_mask = [0] * self.vocab_size
        for move in move_list:
            board.push(chess.Move.from_uci(move))
        legal_moves = list(board.legal_moves)
        legal_moves = [str(move) for move in legal_moves]
        for move in legal_moves:
            legal_mask[self.tokenizer.tokens_to_ids_single(move)] = 1
        
        # print(legal_mask)
        # print("legal mask sum", sum(legal_mask))
        legal_mask = torch.tensor(legal_mask, dtype = torch.bool).to(self.mps_device)

        # print("legal mask", legal_mask.shape)
        outputs = self.model(input_ids = token_ids, attention_mask = attn_mask).logits
        # print("model outputs", outputs.shape)

        wanted_outputs = outputs[0][len(move_list)]
        # print("wanted outputs", wanted_outputs.shape)

        masked_logits = wanted_outputs.masked_fill(~legal_mask, float('-inf'))
        # print("masked logits", masked_logits.shape)


        softmax = nn.Softmax(dim=-1)
        probs = softmax(masked_logits)
        # print("probs", probs.shape)
        probs = probs.cpu().detach().numpy()

        to_sort = [(probs[i], i, self.tok.ids_to_tokens_single(i)) for i in range(len(probs))]
        to_sort.sort(reverse=True)

        # print(to_sort)

        for i in range(10):
            print(to_sort[i])
        
        return self.tokenizer.ids_to_tokens_single(to_sort[0][1])

        # for i in range(len(move_list)):
        #     move = move_list[i]
        #     new_move = chess.Move.from_uci(move)
        #     board.push(new_move)
        #     token_ids[i + 1] = self.tokenizer.tokens_to_ids_single(move)
        #     attn_mask[i + 1] = 1


        # legal_moves = list(board.legal_moves)
        # legal_moves = [str(move) for move in legal_moves]
        # legal_mask = [0] * self.tokenizer.vocabulary_size()

        # legal_moves_ids = self.tokenizer.tokens_to_ids_vect(legal_moves)

        # for id in legal_moves_ids:
        #     legal_mask[id] = 1
        
        # token_ids = torch.tensor(token_ids)
        # attn_mask = torch.tensor(attn_mask)
        # legal_mask = torch.tensor(legal_mask)

        # token_ids = torch.reshape(token_ids, (1, -1))
        # attn_mask = torch.reshape(attn_mask, (1, -1))
        # legal_mask = torch.reshape(legal_mask, (1, -1))

        # token_ids = token_ids.to(self.mps_device)
        # attn_mask = attn_mask.to(self.mps_device)
        # legal_mask = legal_mask.to(self.mps_device)

        # outputs = self.model(token_ids, attn_mask, legal_mask)

        # softmax = nn.Softmax(dim=-1)
        # outputs = softmax(outputs)

        # outputs = outputs.cpu().detach().numpy()
        # outputs = outputs[0]

        # to_sort = [(outputs[i], i) for i in range(len(outputs))]
        # to_sort.sort(reverse=True)

        # print(to_sort)

        # for i in range(10):
        #     print(self.tokenizer.ids_to_tokens_single(to_sort[i][1]), to_sort[i][0])

        # return self.tokenizer.ids_to_tokens_single(to_sort[0][1])