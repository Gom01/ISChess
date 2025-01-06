# ==========================================
# Chess Bot Implementation
# AI using minimax algorithm with alpha-beta pruning
# Gomez Flavien - Antonietti Nathan - Savioz Pierre-Yves
# ==========================================

import numpy as np
from time import process_time
from Bots.ChessBotList import register_chess_bot

class ChessBoard:
    """
    Represents a chess board state with evaluation capabilities.
    """

    total_moves = 0
    max_depth_reached = 0
    branching_factors = []

    def __init__(self, board, move, depth, parent, score):
        self.parent = parent  # Parent board state
        self.board = board    # Current board configuration
        self.move = move      # Move that led to this state
        self.depth = depth    # Current search depth
        self.score = score    # Evaluation score

    # @classmethod
    # def reset_stats(cls):
    #     """Reset the statistics counters"""
    #     cls.total_moves = 0
    #     cls.max_depth_reached = 0
    #     cls.branching_factors = []

class PieceMovement:
    """
    Contains all piece movement logic with clear movement patterns and validation.
    Each method implements specific chess piece movement rules.
    """
    @staticmethod
    def get_pawn_moves(x, y, color, current_board, direction):
        """Calculate valid pawn moves including forward steps and captures"""
        moves = []
        board_size = current_board.shape[0]

        # Forward movement
        next_x = x + direction
        if 0 <= next_x < board_size and current_board[next_x, y] == '':
            moves.append(((x, y), (next_x, y)))

        # Diagonal captures
        for dy in [-1, 1]:
            next_y = y + dy
            if 0 <= next_x < board_size and 0 <= next_y < board_size:
                target = current_board[next_x, next_y]
                if target != '' and target[-1] != color:
                    moves.append(((x, y), (next_x, next_y)))

        return moves

    @staticmethod
    def get_sliding_piece_moves(x, y, color, current_board, directions):
        """Method for calculating sliding piece moves"""
        moves = []
        for dx, dy in directions:
            current_x, current_y = x + dx, y + dy
            while 0 <= current_x < 8 and 0 <= current_y < 8:
                target = current_board[current_x, current_y]
                if target == '':
                    moves.append(((x, y), (current_x, current_y)))
                elif target[-1] != color:
                    moves.append(((x, y), (current_x, current_y)))
                    break
                else:
                    break
                current_x, current_y = current_x + dx, current_y + dy
        return moves

    @staticmethod
    def get_rook_moves(x, y, color, current_board):
        """Calculate rook moves using sliding piece logic"""
        directions = [(0,1), (0,-1), (1,0), (-1,0)]
        return PieceMovement.get_sliding_piece_moves(x, y, color, current_board, directions)

    @staticmethod
    def get_bishop_moves(x, y, color, current_board):
        """Calculate bishop moves using sliding piece logic"""
        directions = [(1,1), (1,-1), (-1,1), (-1,-1)]
        return PieceMovement.get_sliding_piece_moves(x, y, color, current_board, directions)

    @staticmethod
    def get_queen_moves(x, y, color, current_board):
        """Calculate queen moves using sliding piece logic"""
        rook_dirs = [(0,1), (0,-1), (1,0), (-1,0)]
        bishop_dirs = [(1,1), (1,-1), (-1,1), (-1,-1)]
        return PieceMovement.get_sliding_piece_moves(x, y, color, current_board, rook_dirs + bishop_dirs)

    @staticmethod
    def get_knight_moves(x, y, color, current_board):
        """Calculate knight moves"""
        moves = []
        knight_moves = [(-2,-1), (-2,1), (-1,-2), (-1,2),
                        (1,-2), (1,2), (2,-1), (2,1)]
        for dx, dy in knight_moves:
            new_x, new_y = x + dx, y + dy
            if 0 <= new_x < 8 and 0 <= new_y < 8:
                if current_board[new_x,new_y] == '' or current_board[new_x,new_y][-1] != color:
                    moves.append(((x,y), (new_x,new_y)))
        return moves

    @staticmethod
    def get_king_moves(x, y, color, current_board):
        """Calculate king moves"""
        moves = []
        directions = [(0,1), (0,-1), (1,0), (-1,0),
                      (1,1), (1,-1), (-1,1), (-1,-1)]
        for dx, dy in directions:
            new_x, new_y = x + dx, y + dy
            if 0 <= new_x < 8 and 0 <= new_y < 8:
                if current_board[new_x,new_y] == '' or current_board[new_x,new_y][-1] != color:
                    moves.append(((x,y), (new_x,new_y)))
        return moves

class BoardManager:
    """Manages board state and move generation"""

    @staticmethod
    def get_all_possible_moves(current_board, player_color, my_color, current_depth):  # Added current_depth parameter
        """Generate all possible moves for the current player"""
        all_possible_moves = []
        direction = -1 if my_color != player_color else 1

        for x in range(current_board.shape[0]):
            for y in range(current_board.shape[1]):
                if current_board[x,y] != '':
                    piece = current_board[x,y][0]
                    color = current_board[x,y][1]
                    if color != player_color:
                        continue

                    moves = []
                    match piece:
                        case "p":
                            moves = PieceMovement.get_pawn_moves(x, y, color, current_board, direction)
                        case "n":
                            moves = PieceMovement.get_knight_moves(x, y, color, current_board)
                        case "b":
                            moves = PieceMovement.get_bishop_moves(x, y, color, current_board)
                        case "q":
                            moves = PieceMovement.get_queen_moves(x, y, color, current_board)
                        case "k":
                            moves = PieceMovement.get_king_moves(x, y, color, current_board)
                        case "r":
                            moves = PieceMovement.get_rook_moves(x, y, color, current_board)

                    if moves and (moves != -1):
                        all_possible_moves.append(moves)

        # Count total moves at this position
        total_moves = sum(len(moves) for moves in all_possible_moves)
        ChessBoard.total_moves += total_moves

        # print(f"Depth {current_depth}: Examining {total_moves} moves")

        if current_depth >= len(ChessBoard.branching_factors):
            ChessBoard.branching_factors.append(total_moves)

        return all_possible_moves

    @staticmethod
    def move_piece(move, current_board):
        """Execute a move on the board"""
        old_x, old_y = move[0]
        new_x, new_y = move[1]
        piece = current_board[old_x,old_y]
        current_board[old_x,old_y] = ''
        current_board[new_x,new_y] = piece
        return current_board

class MoveEvaluation:
    """
    Handles move evaluation and scoring.
    Includes comprehensive evaluation features:
    - Material evaluation
    - Position evaluation
    - King safety
    - Center control
    - Capture evaluation
    """

    """Values chosen to balance material importance vs positional play"""
    PIECE_VALUES = {'p': 100, 'n': 320, 'b': 330, 'r': 500, 'q': 900, 'k': 20000}

    """Key squares that give gives tactical advantages"""
    CENTER_SQUARES = [(3, 3), (3, 4), (4, 3), (4, 4)]

    """Position scores for general piece placement and development"""
    POSITION_TABLE = [
        [-5, -4, -3, -3, -3, -3, -4, -5],
        [-4, -2, -1, -1, -1, -1, -2, -4],
        [-3, -1,  1,  1,  1,  1, -1, -3],
        [-3, -1,  1,  2,  2,  1, -1, -3],
        [-3, -1,  1,  2,  2,  1, -1, -3],
        [-3, -1,  1,  1,  1,  1, -1, -3],
        [-4, -2, -1, -1, -1, -1, -2, -4],
        [-5, -4, -3, -3, -3, -3, -4, -5]
    ]

    """Position scores for the King gameplay"""
    KING_POSITION_TABLE = [
        [ 2,  3,  0,  0,  0,  0,  3,  2],
        [ 2,  2, -2, -3, -3, -2,  2,  2],
        [-2, -3, -4, -5, -5, -4, -3, -2],
        [-3, -4, -5, -6, -6, -5, -4, -3],
        [-4, -5, -6, -7, -7, -6, -5, -4],
        [-4, -5, -6, -7, -7, -6, -5, -4],
        [-4, -5, -6, -7, -7, -6, -5, -4],
        [ 2,  3,  1, -1, -1,  1,  3,  2]  # Back rank preferred
    ]

    @staticmethod
    def is_endgame(board):
        """Determine if the position is in endgame based on piece count"""
        piece_count = 0
        for x in range(8):
            for y in range(8):
                if board[x,y] != '' and board[x,y][0] != 'k' and board[x,y][0] != 'p':
                    piece_count += 1
        return piece_count <= 6

    @staticmethod
    def calculate_piece_count(board, color):
        """Count remaining pieces for a given color"""
        count = 0
        for x in range(8):
            for y in range(8):
                if board[x,y] != '' and board[x,y][1] == color:
                    count += 1
        return count

    @staticmethod
    def sort_moves(moves, board, color):
        """
        Sort moves based on multiple evaluation criteria for better pruning efficiency.
        Higher scored moves are tried first in the search tree.
        """
        def evaluate_move(move):
            score = 0
            # Fix: Access the numpy array through board.board
            moving_piece = board.board[move[0][0], move[0][1]]  # Add .board here
            target_piece = board.board[move[1][0], move[1][1]]  # Add .board here

            if target_piece != '':  # If it's a capture
                moving_value = MoveEvaluation.PIECE_VALUES.get(moving_piece[0], 0)
                target_value = MoveEvaluation.PIECE_VALUES.get(target_piece[0], 0)
                if moving_value > target_value:
                    score -= (moving_value - target_value) * 2  # Penalize bad trades
                else:
                    score += target_value - moving_value + 10

            score += MoveEvaluation.king_safety_score(move, board, color)
            score += MoveEvaluation.base_capture_score(move, board, color)
            score += MoveEvaluation.control_center_score(move)
            #score += MoveEvaluation.enemy_king_danger_score(move, board, color)
            return score

        return sorted(moves, key=evaluate_move, reverse=True)


        # return sorted(moves, key=evaluate_move, reverse=True)

    @staticmethod
    def base_capture_score(move, board, my_color):
        """Calculate score for capturing moves based on piece values"""
        x, y = move[1]
        piece = board.board[x,y]
        if piece == '' or piece[1] == my_color:
            return 0
        return MoveEvaluation.PIECE_VALUES.get(piece[0], 0)

    @staticmethod
    def control_center_score(move):
        """Evaluate move's influence on center control"""
        x, y = move[1]

        # Bonus for controlling center squares
        if (x, y) in MoveEvaluation.CENTER_SQUARES:
            return 5

        # Smaller bonus for controlling extended center
        if 2 <= x <= 5 and 2 <= y <= 5:
            return 2

        return 0


    @staticmethod
    def king_safety_score(move, board, my_color):
        """Enhanced king safety evaluation combined with existing checks"""
        # Make a temporary move
        temp_board = np.copy(board.board)
        new_board = BoardManager.move_piece(move, temp_board)

        # Find king position
        king_position = None
        for x in range(8):
            for y in range(8):
                if new_board[x, y] == f'k{my_color}':
                    king_position = (x, y)
                    break
            if king_position:
                break

        if not king_position:
            return -1000  # Serious problem if king not found

        score = 0
        x, y = king_position

        # Check if king is in check after move (from existing code)
        if MoveEvaluation.is_king_in_check(new_board, king_position, my_color):
            return -10000
            # score -= 40

        # Check for piece protection (from existing code)
        if MoveEvaluation.is_king_protected(new_board, king_position, my_color):
            score += 1

        # Add new positional evaluation
        if not MoveEvaluation.is_endgame(new_board):
            # Use king position table for scoring
            score += MoveEvaluation.KING_POSITION_TABLE[x][y] * 0.5

            # Penalty for leaving back rank in early/middle game
            if my_color == 'w' and x < 7:  # White king leaving back rank
                score -= 20
            elif my_color == 'b' and x > 0:  # Black king leaving back rank
                score -= 20

        # Check for pawn shield - only in front
        if my_color == 'w':
            shield_x = x - 1
            shield_y = y
        else:
            shield_x = x + 1
            shield_y = y

        if 0 <= shield_x < 8 and 0 <= shield_y < 8:
            if new_board[shield_x, shield_y] == f'p{my_color}':
                score += 2  # Slightly higher bonus since it's just one critical pawn

        return score

    @staticmethod
    def is_king_in_check(board, king_position, my_color):
        """Check if the king is being attacked"""
        x, y = king_position
        opponent_color = 'b' if my_color == 'w' else 'w'
        all_opponent_moves = BoardManager.get_all_possible_moves(board, opponent_color, my_color, 0)  # Added depth parameter
        all_opponent_moves = [item for sublist in all_opponent_moves for item in sublist]

        for move in all_opponent_moves:
            end_x, end_y = move[1]
            if (end_x, end_y) == (x, y):
                return True
        return False

    @staticmethod
    def enemy_king_danger_score(move, b, color):
        opponent_color = 'b' if color == 'w' else 'w'
        """Give bonus if the enemy king is being attacked"""
        new_board = BoardManager.move_piece(move, b.board.copy())
        king_position = None
        for x in range(8):
            for y in range(8):
                if new_board[x, y] == f'k{color}':
                    king_position = (x, y)
                    break

        if king_position is None:
            return 15000 # Positive score for defeating the king

        if MoveEvaluation.is_king_in_check(new_board, king_position, opponent_color):
            return 80 # Less than a pawn to avoid sacrifice

        if MoveEvaluation.is_king_protected(new_board, king_position, color):
            return -2  # Negative score for keeping the king protected
        return 0

    @staticmethod
    def is_king_protected(board, king_position, my_color):
        """Check if the king has some pieces protecing him ( cover )"""
        x, y = king_position
        adjacent_squares = [
            (x + dx, y + dy) for dx, dy in [
                (-1, -1), (-1, 0), (-1, 1),
                (0, -1), (0, 1),
                (1, -1), (1, 0), (1, 1)
            ]
        ]

        for nx, ny in adjacent_squares:
            if 0 <= nx < 8 and 0 <= ny < 8:
                piece = board[nx, ny]
                if piece != '' and piece[1] == my_color:
                    return True
        return False

    @staticmethod
    def calculate_score(board, my_color):
        score = 0
        opponent_color = 'b' if my_color == 'w' else 'w'

        # Material evaluation should be primary
        for row_idx, row in enumerate(board):
            for col_idx, piece in enumerate(row):
                if piece == '':
                    continue

                piece_type = piece[0]
                piece_color = piece[1]
                piece_value = MoveEvaluation.PIECE_VALUES.get(piece_type, 0)
                # Position value should be scaled down
                position_value = MoveEvaluation.POSITION_TABLE[row_idx][col_idx] * 5  # Scale position influence

                if piece_color == opponent_color:
                    score -= (piece_value + position_value)
                else:
                    score += (piece_value + position_value)

        # Additional positional bonuses should be smaller relative to material
        return score

def perform_search(board, max_depth, player_color, time_budget, start_time):
    """Main search function implementing minimax with alpha-beta pruning
    Processes moves in small batches and tracks time usage
    Returns best move found within time constraints"""
    moves_at_depth = [0] * (max_depth + 1)
    batch_size = 3
    time_reserve = 0.05
    adjusted_budget = time_budget - time_reserve

    def minimax(b, current_depth, maximizing_player, alpha, beta, my_color):
        if process_time() - start_time >= time_budget:
            return None

        # Base case - reached max depth, return current board state
        if current_depth == 0:
            return b

        # Track actual search depth and set up initial values
        actual_depth = max_depth - current_depth
        opponent_color = 'b' if my_color == 'w' else 'w'
        best_board = None

        if maximizing_player:
            # Maximizing player tries to get highest score
            max_eval = float('-inf')
            possible_moves = BoardManager.get_all_possible_moves(b.board, my_color, my_color, actual_depth)
            # Flatten list of moves for processing
            possible_moves = [item for sublist in possible_moves for item in sublist]
            moves_at_depth[actual_depth] = len(possible_moves)

            for move in possible_moves:
                # Alpha-beta pruning cutoff
                if beta <= alpha:
                    break
                # Check if we're out of time
                if process_time() - start_time >= time_budget:
                    return best_board if best_board else b

                # Create new board state after move
                new_board = np.copy(b.board)
                new_board = BoardManager.move_piece(move, new_board)
                new_board_state = ChessBoard(
                    new_board, move, actual_depth, b,
                    MoveEvaluation.calculate_score(new_board, my_color)
                )

                # Recursively evaluate position
                result_board = minimax(new_board_state, current_depth - 1, False, alpha, beta, my_color)
                if result_board and result_board.score > max_eval:
                    max_eval = result_board.score
                    best_board = new_board_state
                alpha = max(alpha, max_eval)

            return best_board if best_board else b
        else:
            # Minimizing player tries to get lowest score
            min_eval = float('inf')

            # Move generation for miinimizing player
            possible_moves = BoardManager.get_all_possible_moves(b.board, opponent_color, my_color, actual_depth)
            possible_moves = [item for sublist in possible_moves for item in sublist]
            moves_at_depth[actual_depth] = len(possible_moves)

            for move in possible_moves:
                if process_time() - start_time >= time_budget:
                    return None

                new_board = np.copy(b.board)
                new_board = BoardManager.move_piece(move, new_board)
                new_board_state = ChessBoard(
                    new_board, move, actual_depth, b,
                    MoveEvaluation.calculate_score(new_board, my_color)
                )

                result_board = minimax(new_board_state, current_depth - 1, True, alpha, beta, my_color)
                if result_board and result_board.score < min_eval:
                    min_eval = result_board.score
                    best_board = new_board_state
                beta = min(beta, min_eval)
                if beta <= alpha:
                    break
            return best_board if best_board else b

    # Get initial set of possible moves
    possible_moves = BoardManager.get_all_possible_moves(board.board, player_color, player_color, 0)
    possible_moves = [item for sublist in possible_moves for item in sublist]
    if not possible_moves:
        return ((0,0), (0,0))

    # Track statistics and set up initial best move
    moves_at_depth[0] = len(possible_moves)
    best_move = possible_moves[0]  # Fallback move
    best_value = float('-inf')

    # Generate all possible board states
    all_boards = []
    for move in possible_moves:
        if process_time() - start_time >= time_budget:
            return best_move

        # Create new board state for evaluation
        new_board = np.copy(board.board)
        new_board = BoardManager.move_piece(move, new_board)
        new_board_state = ChessBoard(
            new_board, move, 0, board,
            MoveEvaluation.calculate_score(new_board, player_color)
        )
        all_boards.append(new_board_state)

    # Sort boards by initial evaluation score
    sorted_boards = sorted(all_boards, key=lambda b: b.score, reverse=True)

    # Process moves in batches
    for i in range(0, len(sorted_boards), batch_size):
        # Time check before each batch*
        if process_time() - start_time >= adjusted_budget:
            # print(f"Time limit reached after {i} moves, using best move found")
            return best_move

        # Get current batch
        batch = sorted_boards[i:i + batch_size]

        # Process all moves in current batch
        for board_state in batch:
            result_board = minimax(board_state, max_depth - 1, False, float('-inf'), float('inf'), player_color)
            if result_board and result_board.score > best_value:
                best_value = result_board.score
                best_move = board_state.move

        # Early exit if we found a winning move
        if best_value >= 10000:  # Winning position found
            # print(f"Found winning move after {i + batch_size} moves")
            return best_move

    # remaining_time = time_budget - (process_time() - start_time)
    # print(f"Exiting search with {remaining_time:.3f}s remaining")

    return best_move

def chess_bot(player_sequence, board, time_budget, **kwargs):
    """Main bot function that manages the game play
    Evaluates positions, manages search depth, and returns chosen move
    Uses time management to work within given budget"""
    player_color = player_sequence[1]
    start_time = process_time()

    # Initialize search
    current_board = ChessBoard(board, None, 0, None, 0)

    # Get all possible moves
    all_possible_moves = BoardManager.get_all_possible_moves(board, player_color, player_color, 0)
    all_possible_moves = [item for sublist in all_possible_moves for item in sublist]

    if not all_possible_moves:
        print("No valid moves available!")
        return ((0,0), (0,0))

    # Create board states for all possible moves
    all_boards = []
    for move in all_possible_moves:
        new_board = np.copy(board)
        new_board = BoardManager.move_piece(move, new_board)
        board_state = ChessBoard(
            new_board,
            move,
            0,
            current_board,
            MoveEvaluation.calculate_score(new_board, player_color)
        )
        all_boards.append(board_state)

    # Sort moves based on initial evaluation
    sorted_boards = sorted(all_boards, key=lambda b: b.score, reverse=True)

    # Perform deeper search
    depth = 3
    best_move = perform_search(current_board, depth, player_color, time_budget, start_time)

    if best_move is None:
        print("Warning: Search returned None, using best scored move from initial evaluation")
        best_move = sorted_boards[0].move

    return best_move

# Register the bot
register_chess_bot("GAS_BOT", chess_bot)