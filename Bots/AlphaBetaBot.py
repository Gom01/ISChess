import collections
import copy
from collections import deque
from time import process_time
from turtledemo.penrose import inflatedart

import numpy as np
from PyQt6 import QtCore
from Bots.ChessBotList import register_chess_bot

import random

def chess_bot(player_sequence, board, time_budget, **kwargs):



    ##FUNCTION OF MOVEMENTS (pawn,...)
    def get_pawn_moves(x, y, color, current_board, direction):
        moves = []
        board_size = current_board.shape[0]  # Assuming a square board (e.g., 8x8)

        # Single step forward
        next_x = x + direction
        if 0 <= next_x < board_size and current_board[next_x, y] == '':
            moves.append(((x, y), (next_x, y)))

        # Capture diagonally
        for dy in [-1, 1]:
            next_y = y + dy
            if 0 <= next_x < board_size and 0 <= next_y < board_size:
                target = current_board[next_x, next_y]
                if target != '' and target[-1] != color:  # Opponent's piece
                    moves.append(((x, y), (next_x, next_y)))

        return moves

    def get_rook_moves(x, y, color, current_board):
        moves = []
        directions = [(0,1), (0,-1), (1,0), (-1,0)]  # right, left, down, up
        for dx, dy in directions:
            current_x, current_y = x + dx, y + dy
            # move +1 in the direction step by step until reaching end OR hitting piece
            while 0 <= current_x < 8 and 0 <= current_y < 8:
                # empty is possible move
                if current_board[current_x,current_y] == '':
                    moves.append(((x,y), (current_x,current_y)))
                # detected enemy piece => can catch
                elif current_board[current_x,current_y][-1] != color:
                    moves.append(((x,y), (current_x,current_y)))
                    break
                # own piece = can not go there
                else:
                    break
                current_x, current_y = current_x + dx, current_y + dy
        return moves

    def get_bishop_moves(x, y, color, current_board):
        moves = []
        directions = [(1,1), (1,-1), (-1,1), (-1,-1)]  # diagonals
        for dx, dy in directions:
            current_x, current_y = x + dx, y + dy
            # move in any diagonal direction step by step until reaching end OR hitting piece SAME AS ROOK
            while 0 <= current_x < 8 and 0 <= current_y < 8:
                if current_board[current_x,current_y] == '':
                    moves.append(((x,y), (current_x,current_y)))
                elif current_board[current_x,current_y][-1] != color:
                    moves.append(((x,y), (current_x,current_y)))
                    break
                else:
                    break
                current_x, current_y = current_x + dx, current_y + dy
        return moves

    def get_queen_moves(x, y, color, current_board):
        # Queen combines rook and bishop moves
        moves = []
        directions_b = [(1,1), (1,-1), (-1,1), (-1,-1)]  # diagonals
        for dx, dy in directions_b:
            current_x, current_y = x + dx, y + dy
            # move in any diagonal direction step by step until reaching end OR hitting piece SAME AS ROOK
            while 0 <= current_x < 8 and 0 <= current_y < 8:
                if current_board[current_x,current_y] == '':
                    moves.append(((x,y), (current_x,current_y)))
                elif current_board[current_x,current_y][-1] != color:
                    moves.append(((x,y), (current_x,current_y)))
                    break
                else:
                    break
                current_x, current_y = current_x + dx, current_y + dy
        directions_r = [(0,1), (0,-1), (1,0), (-1,0)]  # right, left, down, up
        for dx, dy in directions_r:
            current_x, current_y = x + dx, y + dy
            # move +1 in the direction step by step until reaching end OR hitting piece
            while 0 <= current_x < 8 and 0 <= current_y < 8:
                # empty is possible move
                if current_board[current_x,current_y] == '':
                    moves.append(((x,y), (current_x,current_y)))
                # detected enemy piece => can catch
                elif current_board[current_x,current_y][-1] != color:
                    moves.append(((x,y), (current_x,current_y)))
                    break
                # own piece = can not go there
                else:
                    break
                current_x, current_y = current_x + dx, current_y + dy
        return moves

    def get_knight_moves(x, y, color, current_board):
        moves = []
        # 8 vectors in L shape
        knight_moves = [(-2,-1), (-2,1), (-1,-2), (-1,2),
                        (1,-2), (1,2), (2,-1), (2,1)]
        for dx, dy in knight_moves:
            new_x, new_y = x + dx, y + dy
            if 0 <= new_x < 8 and 0 <= new_y < 8:
                if current_board[new_x,new_y] == '' or current_board[new_x,new_y][-1] != color:
                    moves.append(((x,y), (new_x,new_y)))
        return moves

    def get_king_moves(x, y, color, current_board):
        moves = []
        directions = [(0,1), (0,-1), (1,0), (-1,0),
                      (1,1), (1,-1), (-1,1), (-1,-1)]
        for dx, dy in directions:
            new_x, new_y = x + dx, y + dy
            if 0 <= new_x < 8 and 0 <= new_y < 8:
                if current_board[new_x,new_y] == '' or current_board[new_x,new_y][-1] != color:
                    moves.append(((x,y), (new_x,new_y)))
        return moves


    ##CALCULATING POSSIBLE MOVES
    def getAllPossibleMoves(current_board, player_color, my_color):
        all_possible_moves = []

        if my_color != player_color:
            direction = -1
        else:
            direction = 1

        for x in range(current_board.shape[0]):
            for y in range(current_board.shape[1]):
                moves = []
                if current_board[x,y] != '':
                    piece = current_board[x,y][0]
                    color = current_board[x,y][1]
                    if color != player_color:
                        continue
                    match piece:
                        case "p":
                            moves = get_pawn_moves(x, y, color, current_board, direction)
                        case "n":
                            moves = get_knight_moves(x, y, color, current_board)
                        case "b":
                            moves = get_bishop_moves(x, y, color, current_board)
                        case "q":
                            moves = get_queen_moves(x, y, color, current_board)
                        case "k":
                            moves = get_king_moves(x, y, color, current_board)
                        case "r":
                            moves = get_rook_moves(x, y, color, current_board)
                    if moves and( moves != -1):
                        all_possible_moves.append(moves)

        print(all_possible_moves)
        return all_possible_moves

    #MOVE PIECE (VIRTUALLY)
    def movePiece(move, current_board):
        oldX = move[0][0]
        oldY = move[0][1]
        new_x = move[1][0]
        new_y = move[1][1]
        piece = current_board[oldX,oldY]
        current_board[oldX,oldY] = ''
        current_board[new_x,new_y] = piece
        return current_board

    #CALCULATING ALL POSSIBLE BOARDS
    def allpossibleBoards(board:Board, currentColor, my_color):
        boards = []
        possible_moves = getAllPossibleMoves(board.board, currentColor, my_color)
        possible_moves = [item for sublist in possible_moves for item in sublist]
        for move in possible_moves:
            new_board = copy.deepcopy(board.board)
            new_board = movePiece(move, new_board)
            new_board = Board(
                new_board,
                move,
                board.depth + 1,
                board,
                calculate_score(new_board, currentColor)
            )
            boards.append(new_board)
        return boards

    # CALCULATING SCORE BASED ON THE BOARD
    def calculate_score(board, my_color):
        piece_values = {
            'p': 1, 'n': 3, 'b': 3, 'r': 5, 'q': 9, 'k': 1000  # Material values for pieces
        }

        # Positional table (applies to all pieces)
        positional_table = [
            [-5, -4, -3, -3, -3, -3, -4, -5],
            [-4, -2, -1, -1, -1, -1, -2, -4],
            [-3, -1, 1, 1, 1, 1, -1, -3],
            [-3, -1, 1, 2, 2, 1, -1, -3],
            [-3, -1, 1, 2, 2, 1, -1, -3],
            [-3, -1, 1, 1, 1, 1, -1, -3],
            [-4, -2, -1, -1, -1, -1, -2, -4],
            [-5, -4, -3, -3, -3, -3, -4, -5]
        ]

        opponent_color = 'b' if my_color == 'w' else 'w'
        score = 0

        attack_bonus = 0  # Variable to track attacking bonuses

        # Function to check if a piece is attacking an opponent piece
        def is_attacking(square, color):
            attacked_squares = set()
            for row_index in range(8):
                for col_index in range(8):
                    piece = board[row_index][col_index]
                    if piece == '':
                        continue

                    piece_type, piece_color = piece[0], piece[1]
                    if piece_color == color:
                        if piece_type == 'p':
                            # Pawns attack diagonally
                            if (color == 'w' and row_index - 1 == square[0] and abs(col_index - square[1]) == 1) or \
                                    (color == 'b' and row_index + 1 == square[0] and abs(col_index - square[1]) == 1):
                                attacked_squares.add((square[0], square[1]))
                        elif piece_type == 'n':
                            # Knights move in "L" shape
                            knight_moves = [(2, 1), (2, -1), (-2, 1), (-2, -1), (1, 2), (1, -2), (-1, 2), (-1, -2)]
                            for move in knight_moves:
                                if (row_index + move[0] == square[0]) and (col_index + move[1] == square[1]):
                                    attacked_squares.add((square[0], square[1]))
                        elif piece_type in ['r', 'b', 'q']:
                            # Rooks, bishops, and queens move in straight or diagonal lines
                            directions = []
                            if piece_type == 'r' or piece_type == 'q': directions += [(1, 0), (-1, 0), (0, 1), (0, -1)]
                            if piece_type == 'b' or piece_type == 'q': directions += [(1, 1), (-1, -1), (1, -1),
                                                                                      (-1, 1)]

                            for direction in directions:
                                r, c = row_index, col_index
                                while True:
                                    r += direction[0]
                                    c += direction[1]
                                    if 0 <= r < 8 and 0 <= c < 8:
                                        if (r, c) == square:
                                            attacked_squares.add((r, c))
                                        if board[r][c] != '': break
                                    else:
                                        break
                        elif piece_type == 'k':
                            # King moves one square in any direction
                            for dr in [-1, 0, 1]:
                                for dc in [-1, 0, 1]:
                                    if (row_index + dr == square[0]) and (col_index + dc == square[1]):
                                        attacked_squares.add((square[0], square[1]))

            return attacked_squares

        # Iterate over the board and calculate the score
        for row_index, row in enumerate(board):
            for col_index, piece in enumerate(row):
                if piece == '':
                    continue

                piece_type = piece[0]  # e.g., 'p' for pawn
                piece_color = piece[1]  # 'w' or 'b'
                piece_value = piece_values.get(piece_type, 0)  # Material value

                # Positional value from the table
                position_value = positional_table[row_index][col_index]

                # Check if the piece is attacking or under attack
                attacked_squares = is_attacking((row_index, col_index), opponent_color)

                # Attack reward: if a piece is attacking an opponent's piece, reward it
                if piece_color == my_color:
                    # Reward for attacking an opponent's piece
                    for (r, c) in attacked_squares:
                        target_piece = board[r][c]
                        if target_piece and target_piece[1] == opponent_color:
                            attack_bonus += piece_values.get(target_piece[0], 0)

                if piece_color == opponent_color:
                    score -= (piece_value + position_value)
                else:
                    score += (piece_value + position_value)

        # Add the attack bonus to the score
        score += attack_bonus

        return score

    # CHANGE COLOR
    def changeColor(color):
        if color == 'w':
            return 'b'
        else:
            return 'w'

    #ALPHA BETA PRUNING
    def minimax(b:Board, depth, maximizingPlayer, alpha, beta, my_color, opponent_color):

        if depth == 0:
            return b

        bestBoard = None

        if maximizingPlayer:
            print(f"All the moves I can do : Current Depth {b.depth}")
            maxEval = -1000
            boards = allpossibleBoards(b, my_color, my_color)
            for board in boards:
                resultBoard = minimax(board, depth - 1, False, alpha, beta, my_color, opponent_color)
                resultBoardEval = resultBoard.score


                if resultBoardEval > maxEval:

                    maxEval = resultBoardEval
                    bestBoard = resultBoard

                alpha = max(alpha, resultBoardEval)

                if beta <= alpha:
                    break

            bestBoard.score = maxEval
            return(bestBoard)


        else:
            print(f"Possible moves of my opponent : {b.depth}")
            minEval = 1000
            boards = allpossibleBoards(b, opponent_color, my_color)
            print(f"Number of moves : {len(boards)}")
            for board in boards:
               # print(f"Score : {board.score}")
               # print(f"{np.flipud(board.board)}\n")
                resultBoard = minimax(board, depth - 1, True, alpha, beta, my_color, opponent_color)
                resultBoardEval = -resultBoard.score

               # print(f"MinEval : {minEval}")

                if resultBoardEval < minEval:
                    minEval = resultBoardEval
                    bestBoard = resultBoard

                beta = min(beta, resultBoardEval)

                if beta <= alpha:
                    break

            bestBoard.score = minEval
            return(bestBoard)



    player_color = player_sequence[1]
    startTime = process_time()

    current_board:Board = Board(board, None, 0, None, 0)

    bestBoard = None
    bestValue = -1000
    alpha = -1000
    beta = 1000
    depth = 2
    boards = allpossibleBoards(current_board, player_color, player_color)
    for board in boards:
        print("***************************************************")
        print("Beginning: ")
        print(f"{board.parent.board} \n ")
        print("My First Move: ")
        print(f"{board.board} \n ")
        print("***************************************************")

        resultBoard = minimax(board, depth - 1, False, alpha, beta, player_color, changeColor(player_color))
        childEval = resultBoard.score
        if childEval > bestValue:
            bestValue = childEval
            bestBoard = resultBoard

    try :
        print("-------------------------------------------------------")
        print(f"Beginning SCORE = {bestBoard.parent.parent.score} : \n {bestBoard.parent.parent.board}")
        print(f"{player_color} choice 1 SCORE = {bestBoard.parent.score}: \n {bestBoard.parent.board}")
        print(f"{changeColor(player_color)} choice 1 SCORE = {bestBoard.score}: \n {bestBoard.board}")
        print("-------------------------------------------------------")
    except Exception as e:
        print("Cannot display parents ! ")

    finaleTime = process_time() - startTime


    print(f"Calculated time : {finaleTime}")
    return bestBoard.parent.move


    # default for DEBUG
    return (0,0), (0,0)

register_chess_bot("alphaBeta", chess_bot)

class Board:
    def __init__(self, board, move, depth, parent, score):
        self.parent = parent
        self.board = board
        self.move = move
        self.depth = depth
        self.score = score