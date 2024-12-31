This is a transformer-based model to play chess.

Inspiration from:
https://arxiv.org/pdf/2008.04057



Trained on dataset of chess games from https://www.ficsgames.org/

I'm not uploading the pgn files to github because they are too big, but they can easily be downloaded from the website above.

I'm training on gpus on google colab on free tier so not much compute power is available to train a good model on millions of games.

v0:
The first model I trained is on the first 1000 games from dataset so it's not very good at playing chess. It does openings pretty well but bad at everything else.

