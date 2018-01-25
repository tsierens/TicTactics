from libc.stdint cimport uint64_t
import numpy as np
cimport numpy as np
import sys
import itertools
import random

from matplotlib import pyplot as plt
import matplotlib

# cdef extern from *:
#     ctypedef int int256 "__int256_t"

CONVERT = 1L << (2*np.arange(12).reshape((3,4))[:,:-1])
'''
         01 01 01 00            10 10 10 00            11 11 11 00
MASK01 = 01 01 01 00   MASK10 = 10 10 10 00   MASK11 = 11 11 11 00   
         01 01 01 00            10 10 10 00            11 11 11 00
'''
cdef unsigned long long MASK01,MASK10,MASK11,MASK_OVERFLOW
MASK01 = 0x151515 #a board full of 01
MASK10 = 0x2a2a2a #a board full of 10
MASK11 = 0x3f3f3f #a board full of 11
MASK_OVERFLOW = 0xc0c0c0 #a mask turning on all overflow regions for purposes of generating moves

random.seed(42)
cdef unsigned long long hash_table[2][90]
hash_table = get_zob()

cdef class Board(object):
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

    def __cinit__(self,board = 'None',big_board = 'None',player = 1):
        cdef int i,j
        if player ==1:
            self.player = 0
        else:
            self.player = 1
        self.log = []
        self.alog = []
        self.over = 0
        self.result = 0
        self.key = 0
        self.title = "tic tactics"
        
#         self.active_board = active_board
        '''
        board indexed :   0 1 2
                          3 4 5  
                          6 7 8
                          
        '''
        if board == "None":
            self.board = [0]*9
            self.moves = [0]*9
        else:
            self.board = [0]*9
            self.moves = [0]*9
            played_board = (board==1)+ (board == -1)


            for i in range(3):
                for j in range(3):
                    self.moves[3*i+j] = np.sum(played_board[3*i:3*i+3,3*j:3*j+3])

                    p1 = (board[3*i:3*i+3,3*j:3*j+3]==1)
                    p2 = (board[3*i:3*i+3,3*j:3*j+3]== - 1)
                    self.board[3*i+j] = np.sum( p1 * CONVERT + p2 * (CONVERT << 1))
        
                    
            for i in xrange(9):
                for j in xrange(9):
                    if board[i,j]==1:
                        self.key ^= hash_table[0][9*i+j]
                    if board[i,j]==-1:
                        self.key ^= hash_table[1][9*i+j]
                        
                        
        if big_board == 'None':
            self.big_board =0
            self.big_moves = 0

        else:
            self.big_board = np.sum(big_board.astype(long) * CONVERT)
            self.big_moves = np.sum(big_board != 0)

            for i in xrange(3):
                for j in xrange(3):
                    if big_board[i,j]==1 or big_board[i,j]==2:
                        self.key ^= hash_table[0][81+3*i+j]
                    if big_board[i,j]==-1 or big_board[i,j]==2:
                        self.key ^= hash_table[1][81+3*i+j]
        
            

    cpdef update_move(self,object move,int safe = 0):
        cdef int index, winner,active_index,result,active_board_loc,#key_index
        cdef int[2] cmove
        cmove = move
        if safe:
            assert(move in self.legal_moves())
        active_board_loc = (cmove[0]/3)*3 + cmove[1]/3
        index = (cmove[0]%3)*8 + (cmove[1]%3)*2 #gives the bit-location for the move
        self.key ^= hash_table[self.player][9*cmove[0]+cmove[1]]
        self.board[active_board_loc] ^= (1L << index) << self.player
        self.moves[active_board_loc] +=1
        active_index = (cmove[0]/3)*8 + (cmove[1]/3)*2
        if not (self.big_board >>active_index & 3): #if the board isn't over yet, check to see if it is   
            winner = check_board_win(self.board[active_board_loc],self.moves[active_board_loc])
        else:
            winner = 0L #psuedo no winner to avoid fiddling with anything else
        if winner:
            if winner == 1 or winner == 3:
                self.key ^= hash_table[0][81+active_board_loc]
            if winner == 2 or winner == 3:
                self.key ^= hash_table[1][81+active_board_loc]
            self.big_board ^= winner << active_index
            self.big_moves +=1
            result = check_board_win(self.big_board,self.big_moves)
            if result != 0:
                if result==1:
                    self.result = 1
                elif result == 2:
                    self.result = -1
                elif result == 3:
                    self.result = -2
                else:
                    sys.exit("result not 0 1 -1 or 2")
                self.over = 1     
        
        self.player ^=1
        self.active_board = [cmove[0]%3,cmove[1]%3]
        self.log.append(move)
        self.alog.append(self.active_board)
        return None
    

        
    cpdef erase_move(self):
        cdef int index,active_index,winner,active_board_loc#,key_index
        cdef int cmove[2]
        move = self.log.pop()
        cmove = move
        active_board_loc = (cmove[0]/3)*3+(cmove[1]/3)
        index = (cmove[0]%3)*8 + (cmove[1]%3)*2 #gives the bit-location for the move    
        self.alog.pop()
#         key_index = 24*(active_board_loc)
        self.player ^=1
#         self.key ^= long(self.board[active_board_loc]) << key_index
        self.key ^= hash_table[self.player][9*cmove[0]+cmove[1]]
        self.board[active_board_loc] ^= (1L << index )<< self.player
        self.moves[active_board_loc] -= 1
#         self.key ^= long(self.board[active_board_loc]) << key_index
        
        active_index = (cmove[0]/3)*8 + (cmove[1]/3)*2
        if (self.big_board >> active_index) & 3:
            winner = check_board_win(self.board[active_board_loc],self.moves[active_board_loc])
            if not winner:
                self.big_moves -= 1
                if (self.big_board >> active_index) & 3 == 1 or (self.big_board >> active_index) & 3 == 3:
                    self.key ^= hash_table[0][81+active_board_loc]
                if (self.big_board >> active_index) & 3 == 2 or (self.big_board >> active_index) & 3 == 3:
                    self.key ^= hash_table[1][81+active_board_loc]
                self.big_board &= ~(3 << active_index) # turn off 11 at location active_index   problems?

                
        

        if self.alog:
            self.active_board = self.alog[-1]
        self.over = 0
        self.result = 0
        return None
    
    

        
    cpdef object legal_moves(self):
        cdef int i,j
        moves = []
        if self.over:
            return []
        
        if self.log == []:
            #returns all empty cells
            for i in xrange(9):
                for j in xrange(9):
                    if (self.board[(i/3)*3+j/3]>>(8*(i%3)+2*(j%3)) & 3 == 0 and (i!=4 or j!=4) 
                        and not check_board_win(self.board[(i/3)*3+j/3]|((1L<<8*(i%3)+2*(j%3))<<self.player),
                                                                         self.moves[(i/3)*3+j/3]+1)):
                        moves.append((i,j))
                        
            
            return moves
#         print 'starting legal moves function'
        cdef int active_board_loc
        cdef unsigned long long board
        active_board_loc = self.active_board[0]*3+  self.active_board[1]
        board = self.board[active_board_loc] #relevant bitboard
        cdef int mp , break_loop = 0
        cdef int last_active[2]
        mp = self.moves[active_board_loc]#need to check if the active board has space
#         print 'moves played on active board',mp
        if mp == 0: #just a tiny speedup if we know the cell is empty, we can skip the empty-check
            last_active = [self.log[-1][0]/3,self.log[-1][1]/3]
            for i in xrange(3):
                for j in xrange(3):
                    if i != last_active[0] or j!= last_active[1]:
                        moves.append(((active_board_loc/3)*3+i,(active_board_loc%3)*3+j))
            
        elif mp<8:#board has space, don't return a move to previous board
            last_active = [self.log[-1][0]/3,self.log[-1][1]/3]
            for i in xrange(3):
                for j in xrange(3):

                    if board >> (8*i+2*j) & 3 == 0 and (i != last_active[0] or j!= last_active[1]):
                        moves.append(((active_board_loc/3)*3+i,(active_board_loc%3)*3+j))
        
        elif mp==8:
            for i in xrange(3):
                for j in xrange(3):
                    if board >> (8*i+2*j) & 3 == 0:
                        moves = [((active_board_loc/3)*3+i,(active_board_loc%3)*3+j)]
                        break_loop = 1
                        break
                if break_loop:
                    break
                        
            
            
        elif mp == 9:
            last_active = [self.log[-1][0]/3,self.log[-1][1]/3]
            moves = []
            illegal = []
            for i in xrange(9):
                for j in xrange(9):
                    if self.board[(i/3)*3+j/3] >> (8*(i%3) + 2*(j%3))&3 == 0:
                        if i%3 == last_active[0] and j%3 == last_active[1]:
                            illegal.append((i,j))
                        else:
                            moves.append((i,j))
                            
            if len(moves)==0:
                moves = illegal            
            
        else:
            print board
            print 'moves played : ',mp
            self.print_board()
            sys.exit('board has more than 9 moves played???')
        return moves
        
        

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
        col_stacks = (np.concatenate((recon(self.board[0]),recon(self.board[3]),recon(self.board[6]))),
                      np.concatenate((recon(self.board[1]),recon(self.board[4]),recon(self.board[7]))),
                      np.concatenate((recon(self.board[2]),recon(self.board[5]),recon(self.board[8]))))
        board = np.concatenate(col_stacks,axis=1)
        return board , recon(self.big_board) 
    
    
cdef int moves_played(unsigned long long board):
    cdef int mp
    mp = 0
    for i in xrange(12):
        if board & 3:
            mp+=1
        board >>=2
    return mp
#     return bin(board).count('1') #seems to be very fast but screws up when needing to look at '11's are possibilities
    
cdef int check_board_win(unsigned long long board,int moves):
    if moves < 3: #easy return
        return 0L
    cdef int winner
    winner = 0
    '''
    legend
     3: both players win! tie
     2: player 2 wins
     1: player 1 wins
     0: game ongoing!
    '''


#     check player 1 win
    cdef unsigned long long board01
    board01 = board & MASK01
    if ((board01 & board01<<2L & board01<<4L) or  #horizontal check
        (board01 & board01<<6L & board01<<12L) or #diagonal / check
        (board01 & board01<<8L & board01<<16L) or #vertical check
        (board01 & board01<<10L& board01<<20L)):  #diagonal \ check

        winner ^=1L

    #check player 2 win
    cdef unsigned long long board10
    board10 = board & MASK10
    if ((board10 & board10<<2 & board10<<4) or  #horizontal check
        (board10 & board10<<6 & board10<<12) or #diagonal / check
        (board10 & board10<<8 & board10<<16) or #vertical check
        (board10 & board10<<10& board10<<20)):  #diagonal \ check

        winner ^=2L

    if moves==9 and not winner:
        winner = 3L
        
        
    return winner


cdef get_zob():

    cdef:
        unsigned long long zob[2][90]
        int i,j
    for i in range(2):
        for j in range(90):
            zob[i][j] = random.getrandbits(64)
    return zob


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