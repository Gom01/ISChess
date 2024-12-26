import collections
import copy
from collections import deque
from time import process_time
import numpy as np
from PyQt6 import QtCore
from Bots.ChessBotList import register_chess_bot

import random

def chess_bot(player_sequence, board, time_budget, **kwargs):

    ##FUNCTION OF MOVEMENTS (pawn,...)
    def get_pawn_moves(x, y, color, current_board):
        moves = []
        direction = 1

        # tout droit
        next_x = x + direction
        if 0 <= next_x < 8 and current_board[next_x,y] == '':
            moves.append(((x,y), (next_x,y)))

        # diago
        for dy in [-1, 1]:
            next_y = y + dy
            if 0 <= next_x < 8 and 0 <= next_y < 8:
                if current_board[next_x,next_y] != '' and current_board[next_x,next_y][-1] != color:
                    moves.append(((x,y), (next_x,next_y)))

        for move in moves:
            print(move)
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
    def getAllPossibleMoves(current_board, player_color):
        all_possible_moves = []
        for x in range(current_board.shape[0]):
            for y in range(current_board.shape[1]):
                moves = []
                if current_board[x,y] != '':
                    #print(f"Position ({x},{y}): {current_board[x,y]}")
                    piece = current_board[x,y][0]
                    color = current_board[x,y][1]
                    if color != player_color:
                        continue
                    match piece:
                        case "p":
                            moves = get_pawn_moves(x, y, color, current_board)
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
        possible_moves = getAllPossibleMoves(board.board, currentColor)
        possible_moves = [item for sublist in possible_moves for item in sublist]
        print(f"All possible moves : {possible_moves}, TOTAL = {len(possible_moves)}")

        for move in possible_moves:
            new_board = copy.deepcopy(board.board)
            new_board = movePiece(move, new_board)
            new_board = Board(
                new_board,
                move,
                board.depth + 1,
                board,
                calculate_score(new_board, board.score, my_color)
            )
            boards.append(new_board)
            print(f"My current score :  {new_board.score} - DEPTH {new_board.depth} : \n {new_board.board} ")
        return boards

    # CALCULATING SCORE BASED ON THE BOARD
    def calculate_score(board, parent_score, my_color):
        piece_values = {
            'p': 10, 'n': 30, 'b': 30, 'r': 50, 'q': 90, 'k': 10000  # Material values for pieces
        }
        # Single positional table (applies to all pieces)
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
        score = parent_score

        for row_index, row in enumerate(board):
            for col_index, piece in enumerate(row):
                if piece == '':
                    continue

                piece_type = piece[0]  # e.g., 'p' for pawn
                piece_color = piece[1]  # 'w' or 'b'
                piece_value = piece_values.get(piece_type, 0)  # Material value

                # Positional value from the unified table
                position_value = positional_table[row_index][col_index]

                if piece_color == opponent_color:
                    score -= (piece_value + position_value)
                else:
                    score += (piece_value + position_value)

        return score

    # CHANGE COLOR
    def changeColor(color):
        if color == 'w':
            return 'b'
        else:
            return 'w'

    #ALPHA BETA PRUNING
    def minimax(board:Board, depth, maximizingPlayer, alpha, beta, my_color, opponent_color):

            if board.depth == depth:
                return board

            if maximizingPlayer:
                print(f"Should move all my pieces : Current Depth {board.depth}")
                best = MIN
                ##Go through all the children of my color
                boards = allpossibleBoards(board, my_color, my_color)
                for b in boards:
                    finalBoard = minimax(b, depth,False, alpha, beta, my_color, opponent_color)

                    if finalBoard.score > best.score:
                        best = finalBoard

                    alpha = max(alpha, best.score)

                    # Alpha Beta Pruning
                    if beta <= alpha:
                        break
                return best
            else:
                best = MAX
                print(f"Should move all my opponent's pieces : Current Depth {board.depth}")
                board.board = np.flipud(np.fliplr(board.board))
                boards = allpossibleBoards(board, opponent_color, my_color)
                for b in boards:
                    b.board = np.flipud(b.board)
                    finalBoard:Board = minimax(b, depth, True , alpha, beta, my_color, opponent_color)

                    if finalBoard.score < best.score:
                        best:Board = finalBoard

                    beta = min(beta, best.score)

                    # Alpha Beta Pruning
                    if beta <= alpha:
                        break
                return best

    player_color = player_sequence[1]
    startTime = process_time()

    MAX = Board(None, None, None, None, float('inf'))
    MIN = Board(None, None, None, None, float('-inf'))
    current_board = Board(board, None, 0, None, 0)
    bestTable:Board = minimax(current_board, 3, True, 0, 1000, player_color, changeColor(player_color))


    try :
        print("-------------------------------------------------------")
        print(f"Beginning SCORE = {bestTable.parent.parent.parent.score} : \n {bestTable.parent.parent.parent.board}")
        print(f"{player_color} choice 1 SCORE = {bestTable.parent.parent.score} : \n {bestTable.parent.parent.board}")
        print(f"{changeColor(player_color)} choice 1 SCORE = {bestTable.parent.score}: \n {bestTable.parent.board}")
        print(f"{player_color} choice 2 SCORE = {bestTable.score}: \n {bestTable.board}")
        print("-------------------------------------------------------")
    except Exception as e:
        print("Cannot display parents ! ")

    finaleTime = process_time() - startTime


    print(f"Calculated time : {finaleTime}")
    return bestTable.parent.parent.move


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