import numpy as np
import sys
import itertools

class Board(object):
    """The board will be a numpy 3x3 numpy array when being used with this class"""
    
    def __init__(self,board = 'None',active_board = (1,1),big_board = 'None',player = 1):
        self.player = player
        self.log = []
        self.alog = [active_board]
        self.over = False
        self.result = 0
        
        self.active_board = active_board
        
        if board == 'None':
            board = np.zeros((9,9))
            self.board = np.copy(board)
        else:
            self.board = np.copy(board)

        if big_board == 'None':
            big_board = np.zeros((3,3))
            self.big_board = big_board
        else:
            self.big_board = big_board
            
        
            

    def update_move(self,move):
        assert(move in self.legal_moves())
        active_board = (move[0]/3,move[1]/3)
#         assert(active_board == self.active_board)
        self.board[move] = self.player
        self.player *= -1
        if self.big_board[active_board]==0:
            self.big_board[active_board] = check_board_win(self.small_board(self.active_board))
            if self.big_board[active_board]:
                self.result = check_board_win(self.big_board)
                if self.result:
                    self.over = True
                
            
        
        self.active_board = (move[0]%3,move[1]%3)
        self.log.append(move)
        self.alog.append(self.active_board)
        return None
    

        
    def erase_move(self):
        move = self.log.pop()
        active_board = self.alog.pop()
        self.board[move] = 0
        self.player *= -1
        
        active_board_board = self.small_board(active_board)
        
        if check_board_win(active_board_board):
            None
        else:
            self.big_board[active_board] = 0
            
        self.active_board = self.alog[-1]
        self.over = False
        self.result = 0
            
        return None
    
    def is_full(self):
        return np.prod(self.board)
    
    def game_over(self):
        self.over = bool(self.winner() or np.prod(self.board))
        return self.over
    
    def legal_moves(self):
        if self.over:
            return []
#        board = self.small_board(self.active_board)
        board_moves = itertools.product(range(self.active_board[0]*3,self.active_board[0]*3+3) , 
                                      range(self.active_board[1]*3,self.active_board[1]*3+3))
        move_list = [move for move in board_moves if self.board[move] == 0]
        
        if len(self.alog)>1 and len(move_list) > 1:
            last_active = self.alog[-2]
            illegal_move = (self.active_board[0]*3 + last_active[0],self.active_board[1]*3 + last_active[1])
            if illegal_move in move_list:
                move_list.remove(illegal_move)
                
        if len(move_list) == 0:
            move_list = [move for move in itertools.product(range(9),range(9)) if self.board[move] == 0]
            
        return move_list
                
            
        
        
    def small_board(self,location):
        return self.board[location[0]*3:location[0]*3+3,location[1]*3:location[1]*3+3]
    
def check_board_win(board):
    #check player 1 win
    dummy = np.copy(board)
    dummy = np.sign(dummy)
    p1_win = 0
    p2_win = 0
    
    if np.any(np.sum(dummy,axis=0)==3):
        p1_win = 1
    elif np.any(np.sum(dummy,axis=1)==3):
        p1_win = 1
    elif dummy[0,0]==dummy[1,1]==dummy[2,2]==1:
        p1_win = 1
    elif dummy[0,2]==dummy[1,1]==dummy[2,0]==1:
        p1_win = 1
        
    #check player 2 win
    dummy = np.copy(board)
    dummy = (((dummy == -1).astype(int) + (dummy == 2).astype(int)) == 1).astype(int)
    
    if np.any(np.sum(dummy,axis=0)==3):
        p2_win = 1
    elif np.any(np.sum(dummy,axis=1)==3):
        p2_win = 1
    elif dummy[0,0]==dummy[1,1]==dummy[2,2]==1:
        p2_win = 1
    elif dummy[0,2]==dummy[1,1]==dummy[2,0]==1:
        p2_win = 1   
        
#     legend
#      2: board over, no win
#      1: player 1 wins
#      0: undecided
#     -1: player 2 wins
#     -2: draw (both players win)
        
    if p1_win and p2_win:
        return -2
    if p1_win:
        return 1
    if p2_win:
        return -1
    
    if np.prod(board):
        return 2
    return 0