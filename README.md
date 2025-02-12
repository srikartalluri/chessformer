This is a transformer-based model to play chess.

Inspiration from:
https://arxiv.org/pdf/2008.04057



Trained on dataset of chess games from https://www.ficsgames.org/

I'm not uploading the pgn files to github because they are too big, but they can easily be downloaded from the website above.

# Project Structure

```
.
├── data
├── models
├── notebooks
├── src
│   ├── models
│   │   └── v0
│   │       ├── chess_dataset.py # dataset class, handles reading pgn files
│   │       ├── model_config.json # model configurations (n_embed, n_head, etc)
│   │       ├── model.py # model class
│   │       ├── tokenizer.py # tokenizer class, handles tokenizing chess moves
│   │       └── engine.py # for inference, to play chess
│   ├── train.py # training script
│   ├── metrics.py
│   ├── config.json # training cofigurations (bs, lr, etc)
│   └── utils.py
```


# Training Models

There are several versions of models and training possibilities.
All models and their corresponding model configs can be found in the models folder.

These configs have parameters for n_embed, n_head, n_layer, etc. If you want to train bigger/smaller model, change it here

In the top level config.json, you can pick which version of model you want. 
After that, ```python train.py``` takes care of all training processes


Every model version has tokenizer, dataset, model, and engine



# Version History

v0:
Model: Bert Base, very small model. The first model I trained is on the first 1000 games from dataset so it's not very good at playing chess. It does openings pretty well but bad at everything else.

I'm training on gpus on google colab on free tier so not much compute power is available to train a good model on millions of games.


v1:
Model: GPT2LMHeadModel

Didn't mess around with the model itself, but rather tokenization, move legality, and training processes. Uses an auto-regressive model to predict next move.

Also massively increased size of model and dataset. Approx ~100 million params. Trained on 4 million games (all games from 2024 and 2023)

Performs much better and beats me in chess (I'm rated ~1600)


