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
    boards = chessBot(color, board, 0)
    #Calculate scores based on boards
    #Find old position to new position
    return((y, x), (y, x))


def chessBot(color, board, profondeur):
    finalBoards:list[Board] = []

    for y in range(board.shape[0] - 1):
        for x in range(board.shape[1]):
            if board[y, x][1] == color:
                match board[y, x][0]:
                    case 'p':
                         finalBoards += movementPawn(board, Position(x, y), color)

    for board in finalBoards:
        chessBot(color, board.board, profondeur + 1)

    if profondeur == 3:
        return ##



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

class Board:
    def __init__(self, board, score) -> None:
        self.board = board
        self.score = score


def movementPawn(board:Board, pos: Position, color) -> list[Board]:
    score = board.score
    boards:list[Board] = []
    if board[pos.y+1, pos.x] == '':
        board[pos.y+1, pos.x] = "p" + color
        board[pos.y, pos.x] = ''
        boards.append(Board(board, score))
        score += 1

    if board[pos.y+1, pos.x + 1][1] != color:
        score += 3
        board[pos.y+1, pos.x+1] = "p" + color
        board[pos.y, pos.x] = ''
        boards.append(Board(board, score))
    if board[pos.y+1, pos.x - 1][1] != color:
        score += 3
        board[pos.y+1, pos.x-1] = "p" + color
        board[pos.y, pos.x] = ''
        boards.append(Board(board, score))
    return(boards)


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