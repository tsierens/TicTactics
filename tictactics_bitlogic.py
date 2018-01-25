import numpy as np
import sys
import itertools

from matplotlib import pyplot as plt
import matplotlib

CONVERT = 1L << (2*np.arange(12).reshape((3,4))[:,:-1])
'''
         01 01 01 00            10 10 10 00            11 11 11 00
MASK01 = 01 01 01 00   MASK10 = 10 10 10 00   MASK11 = 11 11 11 00   
         01 01 01 00            10 10 10 00            11 11 11 00
'''
MASK01 = 1381653L #a board full of 01
MASK10 = MASK01 << 1 #a board full of 10
MASK11 = MASK01 | MASK10
MASK_OVERFLOW = 12632256L #a mask turning on all overflow regions for purposes of generating moves

class Board(object):
    """The board object will be a 3x3 array of bitboards and a single mega-board bitboard.
    
    The bitboards will be such that the bit representing each location is given by the map
    
    00 02 04
    08 10 12
    16 18 20
    
    Each bit represents one of the player's pieces occupying a cell.
    00 -> empty cell
    01 -> player 1
    10 -> player 2
    11 -> wild card
    
    """
    
    def __init__(self,board = 'None',active_board = (1,1),big_board = 'None',player = 1):
        self.player = 0L if player == 1 else 1L
        self.log = []
        self.alog = [active_board]
        self.over = False
        self.result = 0L
        self.title = 'tic tactics'
        
        self.active_board = active_board
        
        if board == 'None':
            self.board = np.zeros((3,3)).astype(long)
            self.moves = np.zeros((3,3)) # number of moves played in that cell
        else:
            self.board = np.zeros((3,3)).astype(long)
            board = np.copy(board)
            played_board = (board==1)+ (board == -1)
            self.moves = np.zeros((3,3)).astype(long)
            
            for i in range(3):
                for j in range(3):
                    self.moves[i,j] = np.sum(played_board[3*i:3*i+3,3*j:3*j+3])
                    
                    p1 = (board[3*i:3*i+3,3*j:3*j+3]==1)
                    p2 = (board[3*i:3*i+3,3*j:3*j+3]== - 1)
                    self.board[i,j] = np.sum( p1 * CONVERT + p2 * (CONVERT << 1))
        
                    

        if big_board == 'None':
            self.big_board = 0L
            self.big_moves = 0L
        else:
            self.big_board = long(np.sum(big_board.astype(long) * CONVERT))
            self.big_moves = np.sum(big_board != 0)
        self.key = self.get_key()
            
        
            

    def update_move(self,move,safe = False):
        if safe:
            assert(move in self.legal_moves())
        active_board = (move[0]/3,move[1]/3)
        index = (move[0]%3)*8 + (move[1]%3)*2 #gives the bit-location for the move
        key_index = 24*(active_board[0]*3+active_board[1])
        self.key ^= long(self.board[active_board]) << key_index
        self.board[active_board] ^= (1L << index) << self.player
        self.moves[active_board] +=1
        self.key ^= long(self.board[active_board]) << key_index
        active_index = active_board[0]*8 + active_board[1]*2
        if not (self.big_board >>active_index & 3): #if the board isn't over yet, check to see if it is   
            winner = check_board_win(self.board[active_board],self.moves[active_board])
        else:
            winner = 0L #psuedo no winner to avoid fiddling with anything else
        if winner:
            self.key ^= self.big_board << 216L
            self.big_board ^= winner << active_index
            self.key ^= self.big_board << 216L
            self.big_moves +=1
            result = check_board_win(self.big_board,self.big_moves)
            if result:
                if result==1:
                    self.result = 1
                elif result == 2:
                    self.result = -1
                elif result == 3:
                    self.result = -2
                else:
                    sys.exit("result not 0 1 -1 or 2")
                self.over = True        
        
        self.player ^=1
        self.active_board = (move[0]%3,move[1]%3)
        self.log.append(move)
        self.alog.append(self.active_board)
        return None
    

        
    def erase_move(self):
        move = self.log.pop()
        active_board = (move[0]/3,move[1]/3)
        index = (move[0]%3)*8 + (move[1]%3)*2 #gives the bit-location for the move    
        self.alog.pop()
        key_index = 24*(active_board[0]*3+active_board[1])
        self.player ^=1
        self.key ^= long(self.board[active_board]) << key_index
        self.board[active_board] ^= (1L << index )<< self.player
        self.moves[active_board] -= 1
        self.key ^= long(self.board[active_board]) << key_index
        
        active_index = active_board[0]*8 + active_board[1]*2
        if (self.big_board >> active_index) & 3:
            winner = check_board_win(self.board[active_board],self.moves[active_board])
            if not winner:
                self.big_moves -= 1
                self.key ^= self.big_board << 216L
                self.big_board &= ~(3 << active_index) # turn off 11 at location active_index   problems?
                self.key ^= self.big_board << 216L
                
        

        self.active_board = self.alog[-1]
        self.over = False
        self.result = 0
            
        return None
    
#     def is_legal(self,move):
#         active_board = self.board[self.active_board]
    

        
    def legal_moves(self):
        if self.over:
            return []
        
        if self.log == []:
            #returns all empty cells
            return [(i,j) for i,j in itertools.product(xrange(9),xrange(9)) 
                    if not (self.board[i/3,j/3]>>(8*(i%3)+2*(j%3)) & 3)]
            
        
        board = self.board[self.active_board] #relevant bitboard
        mp = self.moves[self.active_board]#need to check if the active board has space
        
        if mp == 0: #just a tiny speedup if we know the cell is empty, we can skip the empty-check
            last_active = (self.log[-1][0]/3,self.log[-1][1]/3)
            moves = [(self.active_board[0]*3+i,self.active_board[1]*3+j)
                     for i,j in itertools.product(xrange(3),xrange(3)) if (i,j) != last_active]
            
        elif mp<8:#board has space, don't return a move to previous board
            last_active = (self.log[-1][0]/3,self.log[-1][1]/3)
            
            moves = [(self.active_board[0]*3+i,self.active_board[1]*3+j) 
                     for i,j in itertools.product(xrange(3),xrange(3)) if not (board>>(8*i+2*j) & 3) and
                     (i,j) != last_active]#checks if cell is empty
            
              
        elif mp==8:
            for i,j in itertools.product(xrange(3),xrange(3)):
                if not (board>>(8*i+2*j) & 3):
                    move = (self.active_board[0]*3+i,self.active_board[1]*3+j)
                    break #saves time
            moves = [move]
            
            
        elif mp == 9:
            last_active = (self.log[-1][0]/3,self.log[-1][1]/3)
            moves = []
            illegal = []
            for i,j in itertools.product(xrange(9),xrange(9)):
                if not (self.board[i/3,j/3] >> (8*(i%3) + 2*(j%3))&3):#cell is empty
                    if (i%3,j%3) == last_active:
                        illegal.append((i,j))
                    else:
                        moves.append((i,j))
            if len(moves)==0:
                moves = illegal            
            
        else:
            print board
            print 'moves played : ',mp
            sys.exit('board has more than 9 moves played???')
        return moves
        
        

    def get_key(self):
        board = self.board.reshape(-1)
        key = 0L
        for i in range(9):
            key |= long(board[i]) << (24*i)
        key |= long(self.big_board) << (216) #24*9
        return key
        #return tuple(self.board.reshape(-1))
    
    def print_board(self):
        #returns a board and big_board arrays
        def recon(bitboard):
            answer = np.zeros((3,3))
            dummy = bitboard
            for i in range(3):
                for j in range(3):
                    val = dummy%4
                    if val == 1 or val == 0:
                        answer[i,j] = val
                    elif val == 2:
                        answer[i,j] = -1
                    elif val == 3:
                        answer[i,j] = 2
                    else:
                        sys.exit('impossible')
                    dummy /= 4
                dummy /=4
            return answer
        col_stacks = (np.concatenate((recon(self.board[0,0]),recon(self.board[1,0]),recon(self.board[2,0]))),
                      np.concatenate((recon(self.board[0,1]),recon(self.board[1,1]),recon(self.board[2,1]))),
                      np.concatenate((recon(self.board[0,2]),recon(self.board[1,2]),recon(self.board[2,2]))))
        board = np.concatenate(col_stacks,axis=1)
        return board , recon(self.big_board)
        
def get_moves(board,exclude = None):
    # generate legal moves belonging to board, but excluding index exclude
    board |= MASK_OVERFLOW # artificially fill in overflow areas
    if exclude is None:
        exclude = -2 #neat trick to make the rest of the trick flow smoothly
        
    #splitting into two for loops might be the fastest method
#    moves = []
#    for index in xrange(0,exclude,2):
#        if not (board >> index)&3:
#            # yield index
#            moves.append(index)
#        
#        
#    for index in xrange(exclude+2,24,2):
#        if not (board >> index)&3:
#            moves.append(index)
    return [index for index in xrange(0,24,2) if not (board>>index)&3 and index != exclude]
        
    
    
def moves_played(board):
    mp = 0
    for i in xrange(12):
        if board & 3:
            mp+=1
        board >>=2
    return mp
#     return bin(board).count('1') #seems to be very fast but screws up when needing to look at '11's are possibilities
    
def check_board_win(board,moves):
    if moves < 3: #easy return
        return 0L
    winner = 0L
    '''
    legend
     3: both players win! tie
     2: player 2 wins
     1: player 1 wins
     0: game ongoing!
    '''


#     check player 1 win
    board01 = board & MASK01
    if ((board01 & board01<<2 & board01<<4) or  #horizontal check
        (board01 & board01<<6 & board01<<12) or #diagonal / check
        (board01 & board01<<8 & board01<<16) or #vertical check
        (board01 & board01<<10& board01<<20)):  #diagonal \ check

        winner ^=1L

    #check player 2 win
    board10 = board & MASK10
    if ((board10 & board10<<2 & board10<<4) or  #horizontal check
        (board10 & board10<<6 & board10<<12) or #diagonal / check
        (board10 & board10<<8 & board10<<16) or #vertical check
        (board10 & board10<<10& board10<<20)):  #diagonal \ check

        winner ^=2L

    if moves==9 and not winner:
        winner = 3L
        
        
    return winner

def fancy_board(game):
    board,big_board = game.print_board()
    print_board = np.copy(board).astype(float)
    for x,y in itertools.product(range(9),range(9)):
        if big_board[x/3,y/3] != 0:
            if print_board[x,y] == 0:
                print_board[x,y] = 0.5
            else:
                print_board[x,y] = 0
            print_board[x,y] += big_board[x/3,y/3]
        if (x,y) in game.legal_moves():
            print_board[x,y] = -2




    plt.matshow(print_board,vmin = -2,vmax = 2)
    plt.plot([-0.5,8.5],[2.5,2.5],'k')
    plt.plot([-0.5,8.5],[5.5,5.5],'k')
    plt.plot([2.5,2.5],[-0.5,8.5],'k')
    plt.plot([5.5,5.5],[-0.5,8.5],'k')