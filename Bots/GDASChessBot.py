from PyQt6 import QtCore
from Bots.ChessBotList import register_chess_bot

import random

# My bot
def chess_bot(player_sequence, board, time_budget, **kwargs):

    def print_board_content():
        print("Board contents and coordinates:")
        for x in range(board.shape[0]):
            for y in range(board.shape[1]):
                if board[x,y] != '':
                    print(f"Position ({x},{y}): {board[x,y]}")

    def get_pawn_moves(x, y, color):
        moves = []
        direction = 1 # ou 2 dans le jeu normal ?

        # TOUT DROIT YEAAAH
        next_x = x + direction
        if 0 <= next_x < 8 and board[next_x,y] == '':
            moves.append(((x,y), (next_x,y)))

        # en diago
        for dy in [-1, 1]:
            next_y = y + dy
            if 0 <= next_x < 8 and 0 <= next_y < 8:
                if board[next_x,next_y] != '' and board[next_x,next_y][-1] != color:
                    moves.append(((x,y), (next_x,next_y)))

        return moves

    def get_rook_moves(x, y, color):
        moves = []
        directions = [(0,1), (0,-1), (1,0), (-1,0)]  # right, left, down, up

        for dx, dy in directions:
            current_x, current_y = x + dx, y + dy

            # move +1 in the direction step by step until reaching end OR hitting piece
            while 0 <= current_x < 8 and 0 <= current_y < 8:

                # empty is possible move
                if board[current_x,current_y] == '':
                    moves.append(((x,y), (current_x,current_y)))

                # detected enemy piece => can catch
                elif board[current_x,current_y][-1] != color:
                    moves.append(((x,y), (current_x,current_y)))
                    break

                # own piece = can not go there
                else:
                    break
                current_x, current_y = current_x + dx, current_y + dy

        return moves

    def get_bishop_moves(x, y, color):
        moves = []
        directions = [(1,1), (1,-1), (-1,1), (-1,-1)]  # diagonals

        for dx, dy in directions:
            current_x, current_y = x + dx, y + dy

            # move in any diagonal direction step by step until reaching end OR hitting piece SAME AS ROOK
            while 0 <= current_x < 8 and 0 <= current_y < 8:
                if board[current_x,current_y] == '':
                    moves.append(((x,y), (current_x,current_y)))
                elif board[current_x,current_y][-1] != color:
                    moves.append(((x,y), (current_x,current_y)))
                    break
                else:
                    break
                current_x, current_y = current_x + dx, current_y + dy

        return moves

    def get_queen_moves(x, y, color):
        # Queen combines rook and bishop moves
        moves = []

        directions_b = [(1,1), (1,-1), (-1,1), (-1,-1)]  # diagonals

        for dx, dy in directions_b:
            current_x, current_y = x + dx, y + dy

            # move in any diagonal direction step by step until reaching end OR hitting piece SAME AS ROOK
            while 0 <= current_x < 8 and 0 <= current_y < 8:
                if board[current_x,current_y] == '':
                    moves.append(((x,y), (current_x,current_y)))
                elif board[current_x,current_y][-1] != color:
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
                if board[current_x,current_y] == '':
                    moves.append(((x,y), (current_x,current_y)))

                # detected enemy piece => can catch
                elif board[current_x,current_y][-1] != color:
                    moves.append(((x,y), (current_x,current_y)))
                    break

                # own piece = can not go there
                else:
                    break
                current_x, current_y = current_x + dx, current_y + dy

        return moves

    def get_knight_moves(x, y, color):
        moves = []

        # 8 vectors in L shape
        knight_moves = [(-2,-1), (-2,1), (-1,-2), (-1,2),
                        (1,-2), (1,2), (2,-1), (2,1)]

        for dx, dy in knight_moves:
            new_x, new_y = x + dx, y + dy
            if 0 <= new_x < 8 and 0 <= new_y < 8:
                if board[new_x,new_y] == '' or board[new_x,new_y][-1] != color:
                    moves.append(((x,y), (new_x,new_y)))

        return moves

    def get_king_moves(x, y, color):
        moves = []
        directions = [(0,1), (0,-1), (1,0), (-1,0),
                      (1,1), (1,-1), (-1,1), (-1,-1)]

        for dx, dy in directions:
            new_x, new_y = x + dx, y + dy
            if 0 <= new_x < 8 and 0 <= new_y < 8:
                if board[new_x,new_y] == '' or board[new_x,new_y][-1] != color:
                    moves.append(((x,y), (new_x,new_y)))

        return moves

    # the default data
    print(board)
    print(player_sequence)
    print(time_budget)
    print("my color: " + player_sequence[1])

    player_color = player_sequence[1]

    # printing the board content with coords
    #print_board_content()

    # traverse boad and add moves to list

    # list w al the moves
    all_possible_moves = []

    for x in range(board.shape[0]):
        for y in range(board.shape[1]):
            moves = []

            if board[x,y] != '':

                print(f"Position ({x},{y}): {board[x,y]}")

                piece = board[x,y][0]
                color = board[x,y][1]

                if color != player_color:
                    continue

                match piece:
                    case "p":
                        moves = get_pawn_moves(x, y, color)
                    case "n":
                        moves = get_knight_moves(x, y, color)
                    case "b":
                        moves = get_bishop_moves(x, y, color)
                    case "q":
                        moves = get_queen_moves(x, y, color)
                    case "k":
                        moves = get_king_moves(x, y, color)
                    case "r":
                        moves = get_rook_moves(x, y, color)

                if moves and( moves != -1):
                    print("appending moves: ")
                    print(moves)

                    all_possible_moves.append(moves)
                    print("There are " + str(len(moves)))


    # random choices
    print("All moves: \n")
    print(all_possible_moves)
    selected_move = random.choice(random.choice(all_possible_moves))

    # print("selected move is : " + selected_move[0] + ", " + selected_move[1])

    # print("we are trying to move the piece " + board[selected_move[0][0], selected_move[0][1]])

    print(selected_move)

    return selected_move


    # default for DEBUG
    return (0,0), (0,0)

register_chess_bot("NewBot", chess_bot)
