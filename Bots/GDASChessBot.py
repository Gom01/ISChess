import collections
from collections import deque

from PyQt6 import QtCore
from Bots.ChessBotList import register_chess_bot

import random

def chess_bot(player_sequence, board, time_budget, **kwargs):


    def print_board_content():
        print("Board contents and coordinates:")
        for x in range(board.shape[0]):
            for y in range(board.shape[1]):
                if board[x,y] != '':
                    print(f"Position ({x},{y}): {board[x,y]}")

    def get_pawn_moves(x, y, color, current_board):
        moves = []
        direction = 1 # ou 2 dans le jeu normal ?

        # TOUT DROIT YEAAAH
        next_x = x + direction
        if 0 <= next_x < 8 and current_board[next_x,y] == '':
            moves.append(((x,y), (next_x,y)))

        # en diago
        for dy in [-1, 1]:
            next_y = y + dy
            if 0 <= next_x < 8 and 0 <= next_y < 8:
                if current_board[next_x,next_y] != '' and current_board[next_x,next_y][-1] != color:
                    moves.append(((x,y), (next_x,next_y)))
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
                        #print("appending moves: ")
                        #print(moves)

                        all_possible_moves.append(moves)
                        #print("There are " + str(len(moves)))
        return all_possible_moves


    def movePiece(move, current_board):
        oldX = move[0][0]
        oldY = move[0][1]
        new_x = move[1][0]
        new_y = move[1][1]
        piece = current_board[oldX,oldY]
        current_board[oldX,oldY] = ''
        current_board[new_x,new_y] = piece
        return current_board


    player_color = player_sequence[1]

    def bfs(board:Board, depth, player_color):
        # Initialize the BFS queue
        queue = collections.deque([board])  # (current_board_state, current_depth)
        explored_boards = []# To store all explored board states
        final_scores = []

        while queue:
            current_board:Board = queue.popleft()
            current_depth = current_board.depth
            # If the maximum depth is reached, skip further exploration
            if current_depth == depth:
                explored_boards.append(current_board)
                final_scores.append(calculate_score(current_board.board, player_color))
                continue

            # Get all possible moves for the current player
            #print(current_board)
            possible_moves = getAllPossibleMoves(current_board.board, player_color)
            possible_moves = [item for sublist in possible_moves for item in sublist]  # Flatten
            # Generate new board states and add to the queue
            for move in possible_moves:
                new_board = current_board.board.copy()
                new_board = movePiece(move, new_board)
                queue.append(Board(new_board, move, current_depth + 1, current_board))
        return explored_boards, final_scores  # Return all the boards explored up to the given depth

    def calculate_score(board, my_color):
        piece_values = {
            'pw': 1, 'nw': 3, 'bw': 3, 'rw': 5, 'qw': 9, 'kw': 0,  # White pieces
            'pb': 1, 'nb': 3, 'bb': 3, 'rb': 5, 'qb': 9, 'kb': 0  # Black pieces
        }
        if my_color == 'w':
            opponent_prefix = 'b'
        else:
            opponent_prefix = 'w'

        score = 0
        for row in board:
            for piece in row:
                if piece in piece_values:
                    if piece[1] == opponent_prefix:
                        score -= piece_values[piece]
                    else:
                        score += piece_values[piece]
        return score

    ##Applying BFS and returning all the Boards possible depth = 3
    final_tables, final_scores = bfs(Board(board, None, 0, None), 3, player_color)
    max_index = final_scores.index(max(final_scores))
    best_table = final_tables[max_index]

    print(best_table.board)
    print(best_table.parent.board)
    print(best_table.parent.parent.board)
    print(best_table.parent.parent.move)

    return best_table.parent.parent.move


    # default for DEBUG
    return (0,0), (0,0)



register_chess_bot("GDAS", chess_bot)


class Board:
    def __init__(self, board, move, depth, parent):
        self.parent = parent
        self.board = board
        self.move = move
        self.depth = depth
