import time
from time import *
import hashlib
import numpy as np
from PyQt6 import *

from Bots.ChessBotList import register_chess_bot

def chess_bot(player_sequence, board, time_budget, **kwargs):
    start_time = time()
    time_limit = time_budget - 0.005

    #Memoization
    def hash_board(b, current_color):
        hash = hashlib.sha256(b.flatten()).hexdigest()
        return hash + current_color


    ##FUNCTION OF MOVEMENTS (pawn,...)
    def get_pawn_moves(x, y, color, current_board, direction):
        moves = []
        board_size = current_board.shape[0]

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
        return all_possible_moves


    def sort_moves(moves, b:Board, color):

        def evaluate(move, b, color):
            score = 0
            score += king_safety_score(move,b,color)
            score += base_capture_score(move, b, color)
            score += control_center_score(move)

            return score

        return sorted(moves, key=lambda move:evaluate(move, b, color), reverse=True)

    def base_capture_score(move, b, my_color):
        oldX = move[0][0]
        oldY = move[0][1]
        x = move[1][0]
        y = move[1][1]
        piece = b.board[x,y]
        if piece == '' or piece[1] == my_color:
            return 0
        piece_values = {'p': 10, 'n': 30, 'b': 30, 'r': 50, 'q': 90, 'k': 1000}
        return piece_values.get(piece[0])

    def control_center_score(move):
        oldX = move[0][0]
        oldY = move[0][1]
        x = move[1][0]
        y = move[1][1]
        center_squares = [(3, 3), (3, 4), (4, 3), (4, 4)]

        if (x, y) in center_squares:
            return 5

        if 2 <= x <= 5 and 2 <= y <= 5:
            return 2

        return 0

    def king_safety_score(move, b:Board, my_color):

        new_board = movePiece(move, b.board.copy())
        king_position = None

        opponent_color = 'b' if my_color == 'w' else 'w'

        for x in range(8):
            for y in range(8):
                if new_board[x, y] == f'k{my_color}':
                    king_position = (x, y)
                    break

        if king_position is None:
            return -1000

        if is_king_in_check(new_board, king_position, my_color):
            return -40  # Negative score for exposing the king

        if is_king_protected(new_board, king_position, my_color):
            return 10  # Positive score for keeping the king protected

        return 0  # Neutral score otherwise

    def is_king_in_check(b, king_position, my_color):
        x, y = king_position
        opponent_color = 'b' if my_color == 'w' else 'w'
        all_opponent_moves = getAllPossibleMoves(b, opponent_color, my_color)
        all_opponent_moves = [item for sublist in all_opponent_moves for item in sublist]

        for move in all_opponent_moves:
            end_x = move[1][0]
            end_y = move[1][1]
            if b[end_x, end_y] == b[x,y]:
                return True
        return False

    def is_king_protected(b, king_position, my_color):
        x, y = king_position
        # Define adjacent squares around the king
        adjacent_squares = [
            (x + dx, y + dy) for dx, dy in [(-1, -1), (-1, 0), (-1, 1),
                                            (0, -1), (0, 1),
                                            (1, -1), (1, 0), (1, 1)]
        ]
        # Check if there are friendly pieces around the king
        for nx, ny in adjacent_squares:
            if 0 <= nx < 8 and 0 <= ny < 8:
                piece = b[nx, ny]
                if piece != '' and piece[1] == my_color:
                    return True
        return False


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
    def allpossibleBoards(b:Board, currentColor, my_color):
        count = 0
        boards = []
        possible_moves = getAllPossibleMoves(b.board, currentColor, my_color)
        possible_moves = [item for sublist in possible_moves for item in sublist]

        ##Improved sorting moves
        possible_moves = sort_moves(possible_moves, b, my_color)

        for move in possible_moves:
            new_board = movePiece(move, b.board.copy())

            board_hash = hash_board(new_board, currentColor)
            board_hash_opp = hash_board(new_board, changeColor(currentColor))

            score = None

            if board_hash in transposition_table:
                score = transposition_table[board_hash]
                count += 1
            elif board_hash_opp in transposition_table:
                score = -transposition_table[board_hash_opp]
                count += 1
            else:
                score = calculate_score(new_board, currentColor)
                transposition_table[board_hash] = score

            new_board = Board(
                new_board,
                move,
                b.depth + 1,
                b,
                score
            )
            boards.append(new_board)


        print(f"Number of boards : {len(boards)}")
        print(f"Not calculated board : {count}")
        return boards

    def calculate_score(board, my_color):
        piece_values = {
            'p': 10, 'n': 30, 'b': 30, 'r': 50, 'q': 90, 'k': 1000  # Material values for pieces
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
        score = 0

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
    def minimax(b:Board, depth, maximizingPlayer, alpha, beta, my_color, opponent_color, branchCount, time_limit, start_time):
        current_time = time()-start_time
        #print(f"Current time : {current_time}")

        if (depth == 0) or (current_time >= time_limit):
            return b, branchCount

        branchCount += 1
        bestBoard = b


        if maximizingPlayer:
            #print(f"All the moves I can do : Current Depth {b.depth}")
            maxEval = -float('inf')
            boards = allpossibleBoards(b, my_color, my_color)
            for bo in boards:
                resultBoard, branchCount = minimax(bo, depth - 1, False, alpha, beta, my_color, opponent_color, branchCount, time_limit, start_time)
                resultBoardEval = resultBoard.score
                #print(f"{bo.board}\n")
                if resultBoardEval > maxEval:
                    maxEval = resultBoardEval
                    bestBoard = resultBoard

                alpha = max(alpha, resultBoardEval)

                if beta <= alpha:
                    break
            return bestBoard, branchCount


        else:
            #print(f"Possible moves of my opponent : {b.depth}")
            minEval = float('inf')
            boards = allpossibleBoards(b, opponent_color, my_color)
            #print(f"Number of moves : {len(boards)}")
            for bo in boards:
                #print(f"Score : {bo.score}")
                #print(f"{np.flipud(bo.board)}\n")
                resultBoard, branchCount = minimax(bo, depth - 1, True, alpha, beta, my_color, opponent_color, branchCount, time_limit, start_time)
                resultBoardEval = resultBoard.score


                if resultBoardEval < minEval:
                    minEval = resultBoardEval
                    bestBoard = resultBoard



                beta = min(beta, resultBoardEval)

                if beta <= alpha:
                    break

            return bestBoard, branchCount


    ###MAIN###
    player_color = player_sequence[1]
    current_board: Board = Board(board, None, 0, None, 0)
    bestBoard = None
    bestValue = -float('inf')
    alpha = -float('inf')
    beta = float('inf')
    depth = 3
    branchCount = 0
    transposition_table = {}
    boards = allpossibleBoards(current_board, player_color, player_color)
    length = len(boards)
    evaluated_count = 0
    batch_size = 3

    for i,b in enumerate(boards):
        if i % batch_size == 0:
            if time() - start_time >= time_limit:
                print("Time limit exceeded.")
                break

        resultBoard, branchCount = minimax(b, depth-1, False, alpha, beta, player_color, changeColor(player_color), branchCount, time_limit, start_time)
        evaluated_count += 1
        totalScore = resultBoard.score
        if (totalScore > bestValue) and (resultBoard.depth == depth):
            bestValue = totalScore
            bestBoard = resultBoard



    print(f"Number of boards successfully evaluated: {evaluated_count}/{length}")

    for i in range(depth-1):
        if bestBoard == None:
            bestBoard = boards[0]
        if bestBoard.parent.move == None:
            break
        bestBoard = bestBoard.parent
    if(bestBoard.depth == 2):
        bestBoard = bestBoard.parent

    #print("*********")
    #print(bestBoard.score)
    #print(bestBoard.board)
    return bestBoard.move

register_chess_bot("WithMemozation", chess_bot)

class Board:
    def __init__(self, board, move, depth, parent, score):
        self.parent = parent
        self.board = board
        self.move = move
        self.depth = depth
        self.score = score