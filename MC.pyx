#cython: cdivision = True,profile = False
from cpython cimport array
import array

cimport tictactics_cython
from tictactics_cython cimport Board
import sys
import time
import numpy as np
import itertools
import matplotlib
import cProfile
from matplotlib import pyplot as plt
from IPython import display
import random
from cpython.exc cimport PyErr_CheckSignals
#import time

cdef class MC_node(object):
#    cdef:
#        int64 key
#        int is_root
#        int leaf,solved,result,player,n_children,sib_index
#        long long N[MAX_MOVES],V[MAX_MOVES]
#        double Q[MAX_MOVES]
#        object actions
#        MC_node child
#        MC_node sib
#        MC_node parent

    
    
    def __init__(self,Board game,int is_root=0):
        self.player = 1 if game.player==0 else -1 #tictactics
        self.is_root = is_root
        self.leaf = 1
        self.n_children = 0
        self.key = game.key
        self.solved = game.over
        self.result = game.result
        for i in xrange(MAX_MOVES):
            self.children_solved[i] = 0
            self.children_result[i] = 0


cpdef select(MC_node node,Board game,list moves,double puct_constant = 1): 
    #point node to the selected leaf and append the list of moves to 'moves'

    cdef int i
    cdef object move,best_move
    cdef double q,u,max_puct
    cdef double sqrt_sum_children
    cdef MC_node best_child,child
    
    while not node.leaf and (node.is_root or not node.solved):
        
        max_puct = -10000. #just a very bad number
        best_move = None
        sum_children = 0
        for i in xrange(node.n_children):
            sqrt_sum_children += node.N[i]
        sqrt_sum_children = sqrt_sum_children**0.5
            
        child = node.child
        for i in xrange(node.n_children):
            move = node.actions[i]

            q = node.player * node.Q[i]
        
            u = puct_constant * sqrt_sum_children / (1 + node.N[i])
            if q+u > max_puct:
                max_puct = q+u
                best_move = move
                best_child = child
            child = child.sib #on the last run, this points to unknown location, probably ok
                            
        node = best_child
        game.update_move(best_move)
        moves.append(best_move)
        
        
    return node 
    

cdef int evaluate(MC_node node,Board game,list moves,object eval_fun):
    #MC rollout, maybe a short PN search
    cdef int result,is_eval_none
    cdef object actions,move
    cdef int n_actions,i
    cdef int64 r
    if eval_fun is None:
        is_eval_none = 1
    else:
        is_eval_none = 0
        
    if node.solved:
        return node.result
    while not game.over:
        if is_eval_none:
            actions = game.legal_moves()
            n_actions = len(actions)
            r = rng()
            move = actions[r%n_actions]
        else:
            move = eval_fun(game)
        moves.append(move)
        game.update_move(move)


    if game.result == 1:
        result = 1
    elif game.result == -1:
        result = -1
    else:
        result = 0
    return result


cdef backprop(MC_node node, Board game, int result , int prune = 0):
    #MC backprop
    cdef int index
    cdef MC_node child

    #incoming node should be leaf or solved
    
    while not node.is_root:
        # MC stuff
        index = node.sib_index
        node = node.parent
        
        #check solved
        
        if not node.solved and node.children_solved[index]:
            node.solved = 1
            node.result = -2 * node.player
            for i in xrange(node.n_children):
                #if node.children_solved[i]: #negamax
                if node.result * node.player < node.children_result[i] * node.player:
                    node.result = node.children_result[i]
                if not node.children_solved[i]:
                    node.solved = 0 #turns the solved flag back off if one of the children is unsolved
                    
            if node.result:
                node.solved = 1
                if not node.is_root:
                    node.parent.Q[node.sib_index] = node.result
                    node.parent.children_solved[node.sib_index] = 1
                    node.parent.children_result[node.sib_index] = node.result
                    
        
        
        node.N[index]+=1
        node.V[index]+=result
        if not node.children_solved[index]:
            node.Q[index]=<double>(node.V[index])/<double>node.N[index]
           
        
        

cdef expand(MC_node node,Board game):
    # Add Children to tree
#     for move in game.legal_moves():
    cdef int i
    cdef MC_node sib
    node.actions = game.legal_moves()
    node.n_children = len(node.actions)

    for i in xrange(node.n_children):
        node.N[i] = 0
        node.V[i] = 0
        node.Q[i] = 0.
        game.update_move(node.actions[i])

        child = MC_node(game)
        child.sib_index = i
        child.parent = node
        if game.over:
            child.solved = 1
            node.children_solved[i] = 1
            if game.result == 1:
                node.Q[i] = 1
                node.children_result[i] = 1
                child.result = 1
            if game.result == -1:
                node.Q[i] = -1
                node.children_result[i] = -1
                child.result = -1
            else:
                node.Q[i] = 0
                node.children_result[i] = 0
                child.result = 0
        
                
        if i == 0:
            node.child = child
            sib = child
        else:
            sib.sib = child
            sib = child

        game.erase_move()
    if node.n_children == 0:
        node.leaf = 1
    else:
        node.leaf = 0
        
        
cpdef mc_pass(MC_node root,Board game,int branch = -1,object eval_fun = None):
    cdef double puct_constant
    cdef MC_node node
    cdef list moves
    moves = []
    node = select(root,game,moves,puct_constant = 0.75)
    
    
    if node.leaf and not node.solved and (node.is_root or node.parent.N[node.sib_index]>branch):
        expand(node,game)
    result = evaluate(node,game,moves,eval_fun)
    backprop(node,game,result)
    
    


cpdef mc_sim(MC_node root,Board game,int branch = -1,double max_time = 0.,long passes = 1000000,object eval_fun = None):
    cdef int i,count
    cdef int log_len
    cdef double t0
    count = 0
    log_len = len(game.log)
    t0 = time.clock()
    for i in xrange(passes):
        if max_time != 0. and time.clock()-t0 > max_time:
            break
        for _ in xrange(len(game.log)-log_len):
            game.erase_move()
        mc_pass(root,game,branch,eval_fun)
        count += 1
        if count%1000==0:
            PyErr_CheckSignals()
            print_node(root)
        #if root.solved:
        #    print_node(root)
        #    break
            

            
def print_node(MC_node node):
    display.clear_output(wait=True)
    print node.key
    print 'this node has {} children'.format(node.n_children)
    print_data = []
    for i in xrange(node.n_children):
        print_data.append((node.actions[i],node.N[i],node.Q[i],get_principal(node,i)[:66]))
    
    print_data = sorted(print_data,key = lambda x: - x[1])
    
    
    
    for data in print_data:
        print 'move:{:5}, runs:{:9}, eval:{:+1.4f}, principal:{:15}'.format(*data)
    print
    print get_principal(node)
    sys.stdout.flush()

def get_principal(MC_node node,int child = -1):
    cdef object s,dummy
    cdef int best_i,best_N,i
    s = ''
    if child != -1:
        dummy = node.actions[child]
        if type(dummy) == tuple:
            dummy = ''.join(map(str,dummy))
        s += str(dummy) + ' '
        
        node = node.child
        for i in xrange(child):
            node = node.sib
            
        
        
    while not node.leaf:
        best_i = -1
        best_N = -1
        for i in xrange(node.n_children):
            if node.N[i] > best_N:
                best_N = node.N[i]
                best_i = i
        dummy = node.actions[best_i]
        if type(dummy) == tuple:
            dummy = ''.join(map(str,dummy))
        s += str(dummy) + ' '
    
        node = node.child
        for i in xrange(best_i):
            node = node.sib
    return s

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
    