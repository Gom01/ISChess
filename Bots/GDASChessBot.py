from PyQt6 import QtCore

#   Be careful with modules to import from the root (don't forget the Bots.)
from Bots.ChessBotList import register_chess_bot

#   Simply move the pawns forward and tries to capture as soon as possible
def chess_bot(player_sequence, board, time_budget, **kwargs):
    print("GDAS's turn:")
    color = player_sequence[1]
    for x in range(board.shape[0]-1):
        for y in range(board.shape[1]):
            if board[x,y] != "p"+color:
                continue
            if y > 0 and board[x+1,y-1] != '' and board[x+1,y-1][-1] != color:
                return (x,y), (x+1,y-1)
            if y < board.shape[1] - 1 and board[x+1,y+1] != '' and board[x+1,y+1][1] != color:
                return (x,y), (x+1,y+1)
            elif board[x+1,y] == '':
                return (x,y), (x+1,y)

    return (0,0), (0,0)

#   Example how to register the function
register_chess_bot("GDAS", chess_bot)