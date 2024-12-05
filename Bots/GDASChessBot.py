from PyQt6 import QtCore
import numpy as np
import random

#   Be careful with modules to import from the root (don't forget the Bots.)
from Bots.ChessBotList import register_chess_bot

#   Simply move the pawns forward and tries to capture as soon as possible
def chess_bot(player_sequence, board, time_budget, **kwargs):
    print("GDAS's turn:")
    print(board)
    color = player_sequence[1]

    pieces: list[Piece] = []
    
    for y in range(board.shape[0]-1):
        for x in range(board.shape[1]):
            # print(x,y)
            if board[y,x] == "p"+color:
                p: Piece = Piece(x,y,'p')
                pieces.append(p)
                # print(p.type,p.pos.x,p.pos.y)

    newPos: Position = Position(0,0)
    i = random.randint(0, len(pieces)-1)
    p: Piece = pieces[i]

    while newPos.getPosTuple() == (0,0):
        i = random.randint(0, len(pieces)-1)
        p = pieces[i]

        match p.type:
            case 'p':
                newPos = movementPawn(board, p.pos)
            case 'r':
                newPos = movementRook(board, p.pos)
            case 'n':
                newPos = movementKnight(board, p.pos)
            case 'b':
                newPos = movementBishop(board, p.pos)
            case 'q':
                newPos = movementQueen(board, p.pos)
            case 'k':
                newPos = movementKing(board, p.pos)

    return (p.pos.y, p.pos.x), (newPos.y,newPos.x)

class Piece:
    def __init__(self, x: int, y: int, type: str) -> None:
        self.pos: Position = Position(x,y)
        self.type: str = type

class Position:
    def __init__(self, x, y) -> None:
        self.x = x
        self.y = y

    def getPosTuple(self) -> tuple:
        return (self.x, self.y)

def movementPawn(board, pos: Position) -> Position:
    if board[pos.y+1, pos.x] == '':
        print("can move")
        return Position(pos.x, pos.y+1)
    else:
        print("can't move")
        return Position(0,0)

def movementRook(board, pos: Position) -> Position:
    None
def movementKnight(board, pos: Position) -> Position:
    None
def movementBishop(board, pos: Position) -> Position:
    None
def movementQueen(board, pos: Position) -> Position:
    None
def movementKing(board, pos: Position) -> Position:
    None

#   Example how to register the function
register_chess_bot("GDAS", chess_bot)