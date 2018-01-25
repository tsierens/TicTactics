#cython: cdivision = True,profile = False
from cpython cimport array
import array

cimport tictactics_cython
from tictactics_cython cimport Board
import numpy as np
import itertools
import matplotlib
import cProfile
from matplotlib import pyplot as plt

import random
import time
from IPython import display
import weakref
from cpython.exc cimport PyErr_CheckSignals

ctypedef unsigned long long int64
cdef enum:
    MAX_MOVES = 81
    MAX_LENGTH = 81

cdef int64 MASK01,MASK10,MASK11
MASK01 = 0x151515 #a board full of 01
MASK10 = 0x2a2a2a #a board full of 10
MASK11 = 0x3f3f3f #a board full of 11

cpdef object tictactics_simulation_policy(Board game):
    cdef:
        list actions
        int cmove[2]
        int scores[MAX_MOVES]
        int bscores[9],bigbscores[9],bigbtranscore[9]
        int n_acitons,board_index,pos_index,active_board_loc,mp
        int total_score
        long r
        int i,j
        int64 board,bigbtran
    actions = game.legal_moves()
    n_actions = len(actions) #maybe not needed
    
    #first, lets get the importance of the big_board
    total_score = 0
    
    active_board_loc = game.active_board[0]*3+  game.active_board[1]
    mp = game.moves[active_board_loc]
    if mp == 0:
        for i in xrange(n_actions):
            scores[i] = 1
    elif mp<9:
        move = actions[0]
        cmove = move
        
        if game.big_board >> ((cmove[0]/3)*8 + (cmove[1]/3)*2) & 3:
            for i in xrange(n_actions):
                scores[i] = 0
        else:
            board_index = (cmove[0]/3)*3+(cmove[1]/3)
            board = game.board[board_index]

            if game.player == 1:
                #swap board
                board = ((board&MASK01) << 1) | ((board&MASK10)>>1)
            board_score(board,bscores)

            for i in xrange(n_actions):
                move = actions[i]
                cmove = move
                scores[i] = bscores[(cmove[0]%3)*3 + (cmove[1]%3)]
    else:
        for i in xrange(n_actions):
            move = actions[i]
            cmove = move
            if game.big_board >> ((cmove[0]/3)*8 + (cmove[1]/3)*2) & 3:
                for i in xrange(n_actions):
                    scores[i] = 0
            else:
                board_index = (cmove[0]/3)*3+(cmove[1]/3)
                board = game.board[board_index]

                if game.player == 1:
                    #swap board
                    board = ((board&MASK01) << 1) | ((board&MASK10)>>1)
                board_score(board,bscores) #calculating board_score several times can be improved
                scores[i] = bscores[(cmove[0]%3)*3 + (cmove[1]%3)]
    board = game.big_board
    if game.player == 1:
        board = ((board&MASK01) << 1) | ((board&MASK10)>>1)
    board_score(board,bigbscores)
    board = ((board&MASK01) << 1) | ((board&MASK10)>>1)
    board_score(board,bigbtranscore)
    
    for i in xrange(9):
        if game.moves[i]==9:
            bigbtranscore[i] = 10
    
#     print 'scores'
#     for i in xrange(n_actions):
#         print scores[i]
#     print 'bigbscores'
#     for i in xrange(9):
#         print bigbscores[i]
#     print 'bigbtranscore'
#     for i in xrange(9):
#         print bigbtranscore[i]  
    
    total_score = 0
    for i in xrange(n_actions):
        move = actions[i]
        cmove = move
        board_index = (cmove[0]/3)*3+(cmove[1]/3)
        pos_index =   (cmove[0]%3)*3+(cmove[1]%3)
        scores[i] *= bigbscores[board_index]
        if scores[i] == 64:
            return actions[i]#win!!!
        scores[i] -= bigbtranscore[pos_index]*bigbtranscore[pos_index]
        scores[i] += 10 #initial relative chance of being chosen
        if scores[i] <0:
            scores[i] = 0
    
        total_score += scores[i]
        
    if total_score == 0:
        for i in xrange(n_actions):
            scores[i] = 1
            total_score+=1
            
    r = (rng()%total_score) + 1
#     print r,total_score
    for i in xrange(n_actions):
        r -= scores[i]
        if r <= 0:
            j=i
            break
            
#     for i in xrange(n_actions):
#         print actions[i] , scores[i]
    return actions[j]
        

                

            

    
cdef board_score(int64 board,int* scores):
    '''
    gives points based on the threats of the board, assumed the board isn't over
    _xx = 8 pts for _
    _oo = 4 pts for _
    __x = 3 pts for _
    __o = 2 pts for _
    ___ = 1 pt  for _
    ox? = 0pt  for any combination of o and x
    
    '''
    
    cdef:
        int64 mut_board,dummy
        int64 x,o,_
        int i,j
        int win = 8
        int block = 4
        int extend = 3
        int prevent = 2
        int start = 1
    for i in xrange(9):
        scores[i] = 0
    
    x = board&MASK01 #includes wild
    o = board&MASK10 #includes wild
    _ = MASK01&(~x)&(~o>>1) #fills 01 place with correct result
    _ |= _<<1 #dilates the result to the 10 place
    #___
    mut_board = 0
    dummy = _ & _ <<  2 & _ <<  4 #horizontal ___
    mut_board |= dummy | dummy >>  2 | dummy >>  4 
    dummy = _ & _ <<  6 & _ << 12 #diagonal / ___
    mut_board |= dummy | dummy >>  6 | dummy >> 12
    dummy = _ & _ <<  8 & _ << 16 #vertical   ___
    mut_board |= dummy | dummy >>  8 | dummy >> 16
    dummy = _ & _ << 10 & _ << 20 #diagonal \ ___
    mut_board |= dummy | dummy >> 10 | dummy >> 20
        
    for i in xrange(3):
        for j in xrange(3):
            if mut_board&3:
                scores[3*i+j] = start
            mut_board >>= 2
        mut_board >>= 2
        
    #__o
    mut_board = ((_ & _ <<  2 & o <<  4)       |  #horizontal __o
                 (o & _ <<  2 & _ <<  4) >> 2  |  #horizontal o__ shifted back 2 (to populate the blank spot)
                 (_ & o <<  2 & _ <<  4) >> 4  |  #horizontal _o_ shifted back 4
                 (_ & _ <<  6 & o << 12)       |  #diagonal / __o
                 (o & _ <<  6 & _ << 12) >> 6  |  #diagonal / o__ shifted back 6
                 (_ & o <<  6 & _ << 12) >> 12 |  #diagonal / _o_ shifted back 12
                 (_ & _ <<  8 & o << 16)       |  #vertical   __o
                 (o & _ <<  8 & _ << 16) >> 8  |  #vertical   o__ shifted back 8
                 (_ & o <<  8 & _ << 16) >> 16 |  #vertical   _o_ shifted back 16
                 (_ & _ << 10 & o << 20)       |  #diagonal \ __o
                 (o & _ << 10 & _ << 20) >> 10 |  #diagonal \ o__ shifted back 10
                 (_ & o << 10 & _ << 20) >> 20)   #diagonal \ _o_ shifted back 20
    
    #_o_ #need to count twice since there are two open spots
    
    mut_board|= ((_ & o <<  2 & _ <<  4)       |  #horizontal _o_
                 (_ & _ <<  2 & o <<  4) >> 2  |  #horizontal __o shifted back 2 (to populate the blank spot)
                 (o & _ <<  2 & _ <<  4) >> 4  |  #horizontal o__ shifted back 4
                 (_ & o <<  6 & _ << 12)       |  #diagonal / _o_
                 (_ & _ <<  6 & o << 12) >> 6  |  #diagonal / __o shifted back 6
                 (o & _ <<  6 & _ << 12) >> 12 |  #diagonal / o__ shifted back 12
                 (_ & o <<  8 & _ << 16)       |  #vertical   _o_
                 (_ & _ <<  8 & o << 16) >> 8  |  #vertical   __o shifted back 8
                 (o & _ <<  8 & _ << 16) >> 16 |  #vertical   o__ shifted back 16
                 (_ & o << 10 & _ << 20)       |  #diagonal \ _o_
                 (_ & _ << 10 & o << 20) >> 10 |  #diagonal \ __o shifted back 10
                 (o & _ << 10 & _ << 20) >> 20)   #diagonal \ o__ shifted back 20
    for i in xrange(3):
        for j in xrange(3):
            if mut_board&3:
                scores[3*i+j] = prevent
            mut_board >>= 2
        mut_board >>= 2

    #__x
    mut_board = ((_ & _ <<  2 & x <<  4)       |  #horizontal __x
                 (x & _ <<  2 & _ <<  4) >> 2  |  #horizontal x__ shifted back 2 (to populate the blank spot)
                 (_ & x <<  2 & _ <<  4) >> 4  |  #horizontal _x_ shifted back 4
                 (_ & _ <<  6 & x << 12)       |  #diagonal / __x
                 (x & _ <<  6 & _ << 12) >> 6  |  #diagonal / x__ shifted back 6
                 (_ & x <<  6 & _ << 12) >> 12 |  #diagonal / _x_ shifted back 12
                 (_ & _ <<  8 & x << 16)       |  #vertical   __x
                 (x & _ <<  8 & _ << 16) >> 8  |  #vertical   x__ shifted back 8
                 (_ & x <<  8 & _ << 16) >> 16 |  #vertical   _x_ shifted back 16
                 (_ & _ << 10 & x << 20)       |  #diagonal \ __x
                 (x & _ << 10 & _ << 20) >> 10 |  #diagonal \ x__ shifted back 10
                 (_ & x << 10 & _ << 20) >> 20)   #diagonal \ _x_ shifted back 20
    
    #_x_ #need to count twice since there are two open spots
    
    mut_board|= ((_ & x <<  2 & _ <<  4)       |  #horizontal _x_
                 (_ & _ <<  2 & x <<  4) >> 2  |  #horizontal __x shifted back 2 (to populate the blank spot)
                 (x & _ <<  2 & _ <<  4) >> 4  |  #horizontal x__ shifted back 4
                 (_ & x <<  6 & _ << 12)       |  #diagonal / _x_
                 (_ & _ <<  6 & x << 12) >> 6  |  #diagonal / __x shifted back 6
                 (x & _ <<  6 & _ << 12) >> 12 |  #diagonal / x__ shifted back 12
                 (_ & x <<  8 & _ << 16)       |  #vertical   _x_
                 (_ & _ <<  8 & x << 16) >> 8  |  #vertical   __x shifted back 8
                 (x & _ <<  8 & _ << 16) >> 16 |  #vertical   x__ shifted back 16
                 (_ & x << 10 & _ << 20)       |  #diagonal \ _x_
                 (_ & _ << 10 & x << 20) >> 10 |  #diagonal \ __x shifted back 10
                 (x & _ << 10 & _ << 20) >> 20)   #diagonal \ x__ shifted back 20
    for i in xrange(3):
        for j in xrange(3):
            if mut_board&3:
                scores[3*i+j] = extend
            mut_board >>= 2
        mut_board >>= 2

 
    #_oo
    mut_board = ((_ & o <<  2 & o <<  4)       |  #horizontal _oo
                 (o & _ <<  2 & o <<  4) >> 2  |  #horizontal o_o shifted back 2 (to populate the blank spot)
                 (o & o <<  2 & _ <<  4) >> 4  |  #horizontal oo_ shifted back 4
                 (_ & o <<  6 & o << 12)       |  #diagonal / _oo
                 (o & _ <<  6 & o << 12) >> 6  |  #diagonal / o_o shifted back 6
                 (o & o <<  6 & _ << 12) >> 12 |  #diagonal / oo_ shifted back 12
                 (_ & o <<  8 & o << 16)       |  #vertical   _oo
                 (o & _ <<  8 & o << 16) >> 8  |  #vertical   o_o shifted back 8
                 (o & o <<  8 & _ << 16) >> 16 |  #vertical   oo_ shifted back 16
                 (_ & o << 10 & o << 20)       |  #diagonal \ _oo
                 (o & _ << 10 & o << 20) >> 10 |  #diagonal \ o_o shifted back 10
                 (o & o << 10 & _ << 20) >> 20)   #diagonal \ oo_ shifted back 20
    for i in xrange(3):
        for j in xrange(3):
            if mut_board&3:
                scores[3*i+j] = block
            mut_board >>= 2
        mut_board >>= 2
        
        
    #_xx
    mut_board = ((_ & x <<  2 & x <<  4)       |  #horizontal _xx
                 (x & _ <<  2 & x <<  4) >> 2  |  #horizontal x_x shifted back 2 (to populate the blank spot)
                 (x & x <<  2 & _ <<  4) >> 4  |  #horizontal xx_ shifted back 4
                 (_ & x <<  6 & x << 12)       |  #diagonal / _xx
                 (x & _ <<  6 & x << 12) >> 6  |  #diagonal / x_x shifted back 6
                 (x & x <<  6 & _ << 12) >> 12 |  #diagonal / xx_ shifted back 12
                 (_ & x <<  8 & x << 16)       |  #vertical   _xx
                 (x & _ <<  8 & x << 16) >> 8  |  #vertical   x_x shifted back 8
                 (x & x <<  8 & _ << 16) >> 16 |  #vertical   xx_ shifted back 16
                 (_ & x << 10 & x << 20)       |  #diagonal \ _xx
                 (x & _ << 10 & x << 20) >> 10 |  #diagonal \ x_x shifted back 10
                 (x & x << 10 & _ << 20) >> 20)   #diagonal \ xx_ shifted back 20
    for i in xrange(3):
        for j in xrange(3):
            if mut_board&3:
                scores[3*i+j] = win
            mut_board >>= 2
        mut_board >>= 2
        
    
cdef int64 seed[2]     
seed[1] = 42

cdef int64 rng():
    cdef int64 x,y
    x=seed[0]
    y=seed[1]
    seed[0]=y
    x^=x<<23
    seed[1]=x^y^(x>>17)^(y>>26)
    return seed[1]+y
    