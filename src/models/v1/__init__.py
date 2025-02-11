"""

every move consissts of 4 tokens. 1 for the side, 1 for the start square, 1 for the end square, and 1 for a potential promotion piece

Side Tokens: 0 for white, 1 for black -> 2 total
Square Tokens: 0-63 for a1-h8 -> 64 total
Promotion Tokens: 0 for no promotion, 1-4 for queen, rook, bishop, knight -> 5 total

Total Tokens: 2 + 64 + 5 = 71

Add two special tokens for start and end of sequence -> 73 total

"""