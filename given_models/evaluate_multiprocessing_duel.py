import multiprocessing as mp
import time

import pandas as pd
import torch
import yaml
from chess import Board

import chess_gameplay as chg
from model import Model
from chess_gameplay import evaluate_position, sans_to_pgn


def fast_evaluate(agents, max_rounds=float('inf'), eval_time_limit=2, eval_depth_limit=25):
    board = Board()
    move_sans = [] # for constructing the pgn
    scores = {'white': [], 'black': [], 'round': []}
    rounds = 0
    
    while True:
        player = 'white' if board.turn else 'black'
        opponent = 'black' if player == 'white' else 'white'
        # print(f"round {rounds}: processing {player}")

        # generate legal move sans
        legal_moves = list(board.legal_moves)
        if legal_moves == []:
            print(f"legal_moves is empty with pgn: {pgn}", flush=True)
            print("board:", flush=True)
            print(board, flush=True)
            break
        legal_move_sans = [board.san(move) for move in legal_moves]

        # agent selects move
        pgn = sans_to_pgn(move_sans)
        # print("pgn:", pgn)
        selected_move_san = agents[player].select_move(pgn, legal_move_sans)
        # print("selected_move_san:", selected_move_san)
        move_sans.append(selected_move_san)

        # push move to the board
        board.push_san(selected_move_san)
        # print("board:")
        # print(board)

        # evaluate the board:
        score = evaluate_position(board, time_limit=eval_time_limit, depth_limit=eval_depth_limit, STOCKFISH_PATH='stockfish')
        # print(f"score for: {player}", -score)
        # print(f'score for: {opponent}', score)
        scores[player].append(-score)
        scores[opponent].append(score)
        scores['round'].append(rounds)

        if board.is_checkmate() or board.is_stalemate():
            break
        
        if player == "black":
            rounds += 1
            if rounds >= max_rounds:
                break
        # print()
    return scores


def process_game(game, agents, max_rounds, eval_time_limit, eval_depth_limit):
    print(f"processing game {game}")
    start = time.perf_counter()
    scores = pd.DataFrame(fast_evaluate(
        agents,
        max_rounds=max_rounds,
        eval_time_limit=eval_time_limit,
        eval_depth_limit=eval_depth_limit
    ))
    scores['game'] = game
    end = time.perf_counter()
    print(f"game {game} processed. Time spent: {end - start} seconds\n")
    return scores

if __name__ == "__main__":
    # Number of games to process
    games = 1000  # For example
    model_config = yaml.safe_load(open("model_config.yaml"))

    model_v2 = Model(**model_config)
    checkpoint_v2 = torch.load("submission_202411/checkpoint.pt", map_location="cpu")
    model_v2.load_state_dict(checkpoint_v2["model"])

    model_v1 = Model(**model_config)
    checkpoint_v1 = torch.load("submission/checkpoint.pt", map_location="cpu")
    model_v1.load_state_dict(checkpoint_v1["model"])

    # agents = {'white': chg.Agent(model), 'black': chg.Agent()}
    agents = {'white': chg.Agent(model_v1), 'black': chg.Agent(model_v2)}

    start = time.perf_counter()
    # Multiprocessing
    num_workers = mp.cpu_count()  # Use all available CPUs, or set a custom number
    with mp.Pool(num_workers) as pool:
        results = pool.starmap(
            process_game,
            [(game, agents, 50, 2, 25) for game in range(games)]  # Arguments for each game
        )

    end = time.perf_counter()
    print(f"Total time spent: {end - start} seconds", )
    print("\nsaving results\n")
    concat_results = pd.concat(results, ignore_index=True)
    concat_results.to_csv("results.csv", index=False)

    for player in ("white", "black"):
        print(f"{player} mean: {concat_results[player].mean()}")
        print(f"{player} std: {concat_results[player].std()}")
