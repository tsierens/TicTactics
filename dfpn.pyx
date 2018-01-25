#cython: cdivision = True,profile = False
from libc.stdlib cimport malloc, free

cimport cython
import sys
cimport tictactics_cython
from tictactics_cython cimport Board
import numpy as np
import itertools
import matplotlib
import cProfile
from IPython import display
from matplotlib import pyplot as plt
import random
game = Board()

ctypedef unsigned long long int64


# cache = []

cdef enum:
#     PRIME = 33333331
    PRIME = 15000017
#     PRIME = 4999999
#     PRIME = 1000003
#     PRIME = 555557
    NODE_SIZE = 6
    MAX_MOVES = 81
#     PRIME = 49999
#     PRIME = 51

cdef int log_len
cdef int64 INF = 1 << 50
cdef int64 INF_LOW = 1<<40
cdef int64 INF_HIGH = 1 << 60
cdef int64 KEY_INIT = 0-1

cdef int64 cache[2*PRIME+2][NODE_SIZE]
# cache = [[0]*6 for _ in xrange(2*PRIME)]

# def change_cache10(unsigned long long num):
#     cache[1][0] = num
# def print_cache10():
#     print cache[1][0]
cdef long long FORGOTTEN,STORED,NODES_VISITED
# cdef object game.actions()

def init():
    global FORGOTTEN,STORED,NODES_VISITED
    cdef int i,j
    for i in xrange(2*PRIME+2):
        cache[i][0] = KEY_INIT
        for j in xrange(1,NODE_SIZE):
            cache[i][j] = 0
    FORGOTTEN=0
    STORED=0
    NODES_VISITED=0
            
init()
def get_cache(int64 key = 0):
    if key == 0:
        return cache[2*PRIME+1]
    cdef int64 node[NODE_SIZE]
    cache_pull(key,node)
    return node



#the cache should save infomation on all of the children  (maybe?) the lock will still be huge
#keeping child info saves on going up and down the tree to find child information
#it should also remember how many nodes were searched to get these values
        
def default_callback(game,role,root_key,children,grandchildren,clear = True):
    
    if clear:
        display.clear_output(wait = True)
    print 'nodes visited  ',NODES_VISITED
    print 'cache space    ',2*PRIME
    print 'nodes stored   ', STORED
    print 'nodes forgotten', FORGOTTEN
    print 'current depth  ',len(game.log)-log_len
    print 'current node   ',' '.join(map(lambda x: ''.join(map(str,x)),game.log[log_len:log_len+30]))
    
    #cdef int64 node[NODE_SIZE]
    #cache_pull(game.key,node)
    #print node
    
    print_dfpn(game,role,root_key,children,grandchildren)
    sys.stdout.flush()
    
def print_dfpn(game,role,root_key,children,grandchildren):
    info = cache[2*PRIME+1]
    info = [info[1] if info[1] < INF_LOW else 'INF',
            info[2] if info[2] < INF_LOW else 'INF',
            info[3] if info[3] < INF_LOW else 'INF',
            info[4] if info[4] < INF_LOW else 'INF',
            info[5] if info[5] < INF_LOW else 'INF']
    cdef int64 node[NODE_SIZE]

    print 'role: '+ role
    print '{:6}: work: {:9}, {:6} won, {:6} not lost, {:6} not won, {:6} lost'.format('root',*info)
    print
    print 'role: '+ ('max' if role=='min' else 'min')
    
    for move in children:
        key = children[move]
        cache_pull(key,node)
        if node[0] == KEY_INIT:
            print_node = ['????']*5   
        else:
            print_node =   [node[1] if node[1] < INF_LOW else 'INF',
                            node[2] if node[2] < INF_LOW else 'INF',
                            node[3] if node[3] < INF_LOW else 'INF',
                            node[4] if node[4] < INF_LOW else 'INF',
                            node[5] if node[5] < INF_LOW else 'INF']
       
        print '{:6}: work: {:9}, {:6} won, {:6} not lost, {:6} not won, {:6} lost'.format(move,*print_node)
    print

    if len(game.log) > log_len:
        move = game.log[log_len]
        print 'expanding move ' + str(move)
        print 'role: ' + role

        for move2 in grandchildren[move]:
            key = grandchildren[move][move2]
            cache_pull(key,node)
            if node[0] == KEY_INIT:
                print_node = ['????']*5   
            else:
                print_node =   [node[1] if node[1] < INF_LOW else 'INF',
                                node[2] if node[2] < INF_LOW else 'INF',
                                node[3] if node[3] < INF_LOW else 'INF',
                                node[4] if node[4] < INF_LOW else 'INF',
                                node[5] if node[5] < INF_LOW else 'INF']
            print '{:6}: work: {:9}, {:6} won, {:6} not lost, {:6} not won, {:6} lost'.format(move2,*print_node)
        print
    
cpdef print_tree(Board game,int result,int current_depth=0,int max_depth = 3):
    if current_depth>max_depth:
        return ''
    s=''
    if game.result == result:
        return 'p1win' if result == 1 else 'p2win'
    
    cdef int64 node[NODE_SIZE]
    cdef int64 child[NODE_SIZE]
    cdef int64 key
    cdef int role
    key = game.key
    cache_pull(game.key,node)
    if node[0] == KEY_INIT:
        return '??'
    
    if game.title == 'tic tactics':
        role = 1 if game.player == 0 else -1 #tictactics
    else:
        role = 1 if game.player == 1 else -1 #connect four, tictactoe, mnk
        
    if result * role == 1: #refutation

        for move in game.legal_moves():
            game.update_move(move)
            if game.result == result:
                break
            cache_pull(game.key,child)
            
            if child[0] == KEY_INIT:
                game.erase_move()
                continue
            if result ==1 and child[2]==0 or result == -1 and child[5] == 0:
                break
            game.erase_move()
        if game.key == key: #if no move worked
            s += "??"
            
        else:
            s += str(move)
            if current_depth<max_depth:
                if game.over:
                    if game.result == 1:
                        s+= ' p1win'
                    elif game.result == 0:
                        s+= ' draw'
                    else:
                        s+= ' p2win'
                else:
                    s += print_tree(game,result,current_depth = current_depth+1,max_depth=max_depth)
            game.erase_move()

    else:#branch
        for move in game.legal_moves():
            s += '\n' + '\t'*(current_depth)
            s += str(move)
            game.update_move(move)
            if game.over:
                if game.result == 1:
                    s+= ' p1win'
                elif game.result == 0:
                    s+= ' draw'
                else:
                    s+= ' p2win'
            else:
                s += print_tree(game,result,current_depth = current_depth,max_depth=max_depth)
            game.erase_move()
    return s

@cython.profile(False)
cdef inline void get_2_smallest(int64* array,int size,int* f,int* s):
    cdef:
        int64 first = INF_HIGH
        int64 second = INF_HIGH
        int i
    f[0] = size
    s[0] = size+1
    first = INF_HIGH
    first = INF_HIGH
    for i in xrange(size):
        if array[i]<first:
            second = first
            first  = array[i]
            s[0]=f[0]
            f[0]=i
        elif array[i]<second:
            second = array[i]
            s[0]=i
    
        
    
    
    
# def get_2_smallest(l):
#     # get two smallest elements from a list
#     smallest ,small = None,None
    
#     for item in l:
#         if item < smallest or smallest is None:
#             small = smallest
#             smallest = item
#         elif item < small or small is None:
#             small = item
#     return smallest,small
'''
node will have format [key,work,GW,GT,ST,SL]
'''
cdef void cache_push(int64 key,int64* node):
    global FORGOTTEN
    global STORED
    
    cdef:
        long index
        int64 lock,biglock,newlock,swap_dummy
        int i
        
    index = key%PRIME
    lock = key
    biglock = cache[2*index][0]
    newlock = cache[2*index+1][0]

    if biglock == lock:
        for i in xrange(NODE_SIZE):
            cache[2*index][i] = node[i]
    else:
        if biglock == KEY_INIT:
            STORED += 1
            for i in xrange(NODE_SIZE):
                cache[2*index][i] = node[i]
        else:
        
            if newlock == lock:
                None
            elif newlock == KEY_INIT:
                STORED += 1
            else:
                FORGOTTEN += 1

            for i in xrange(NODE_SIZE):
                cache[2*index+1][i] = node[i]
        
    if cache[2*index+1][1] > cache[2*index][1]:
        
        for i in xrange(NODE_SIZE):
            swap_dummy = cache[2*index][i]
            cache[2*index][i] = cache[2*index+1][i]
            cache[2*index+1][i] = swap_dummy
#####@cython.profile(False)
cdef void cache_pull(int64 key,int64* node):
    cdef:
        int64 lock,biglock,newlock
        long index
        int i
        
    index = key%PRIME
    lock = key
    biglock = cache[2*index][0]
    newlock = cache[2*index+1][0]
    node[0] = KEY_INIT
    for i in xrange(1,NODE_SIZE):
        node[i] = 0

    if biglock == lock:
        for i in xrange(NODE_SIZE):
            node[i] = cache[2*index][i]
            
        
    if newlock == lock:
        for i in xrange(NODE_SIZE):
            node[i] = cache[2*index+1][i]

cdef void dfpn_expand(int64* node,int64 key , int game_over , int game_result):
    cdef int i
    
    node[0] = key
    node[1] = 1
    node[2] = 1
    node[3] = 1
    node[4] = 1
    node[5] = 1
        
    if game_over:
        if game_result == 1:
            node[2] = 0
            node[3] = 0
            node[4] = INF
            node[5] = INF
        elif game_result == -1:
            node[2] = INF
            node[3] = INF
            node[4] = 0
            node[5] = 0
        else:
            node[2] = INF
            node[3] = 0
            node[4] = 0
            node[5] = INF

DEBUG = [0]*100  
DEBUG_INDEX = 0L

def solve(Board game,int64 GWT = INF,int64 GTT = INF,int64 STT = INF,int64 SLT = INF,new_callback=None,
               long callback_freq = 100000,double epsilon = 1.25,int root = 0):
    '''
    A python wrapper used to set up the callback function and call the c-function search
    '''
    global callback
    global log_len
    cdef int64 node[NODE_SIZE]
    cdef int64 root_key
    cdef object children,grandchildren
    cdef int result
    log_len = len(game.log)
    children = {}
    grandchildren = {}
    if game.title == 'tic tactics':
        role = 'max' if game.player == 0 else 'min'
    else:
        role = 'max' if game.player == 1 else 'min'
        
    if new_callback is None:
        root_key = game.key
        for move in game.legal_moves():
            game.update_move(move)
            children[move] = game.key
            grandchildren[move] = {}
            for move2 in game.legal_moves():
                game.update_move(move2)
                grandchildren[move][move2] = game.key
                game.erase_move()
            game.erase_move()
        def callback(game):
            default_callback(game,role,root_key,children,grandchildren,clear = True)
        
        
    else:
        callback = new_callback
        
    
    dfpn_search(game,node,GWT,GTT,STT,SLT,epsilon,root,callback_freq)
    callback(game)
    while len(game.log)>log_len:
        game.erase_move()
    result = 0
    if cache[2*PRIME+1][2]==0:
        result = 1
    elif cache[2*PRIME+1][5]==0:
        result = -1
    tree = print_tree(game,result,0,4)
    return node,tree

cdef dfpn_search(Board game,int64* node,int64 GWT = INF,int64 GTT = INF,int64 STT = INF,int64 SLT = INF,
               double epsilon = 1.25,int root = 0,long callback_freq = 0):
    cdef:
        int64 key,work
        int64 GW,GT,ST,SL
        int role,i,solved,na
        double ep = epsilon

    if GWT > INF_LOW:
        GWT = INF_HIGH
    if GTT > INF_LOW:
        GTT = INF_HIGH
    if STT > INF_LOW:
        STT = INF_HIGH
    if SLT > INF_LOW:
        SLT = INF_HIGH
        
    global NODES_VISITED
    NODES_VISITED +=1

    
    if callback_freq:        
        if NODES_VISITED % callback_freq==0:
            callback(game) #global callback set by dfpn_run

    #init values for proof numbers of this node
    key = game.key
    cache_pull(key,node)
    if node[0] == KEY_INIT:
        dfpn_expand(node,key,game.over,game.result) #returns node
    key2 = node[0]
    work = node[1]
    GW   = node[2]
    GT   = node[3]
    ST   = node[4]
    SL   = node[5]
    
    #global DEBUG_INDEX
    #DEBUG[DEBUG_INDEX%100] = {'where':'outside','threshholds':(GWT,GTT,STT,SLT) , 'pn' : (GW,GT,ST,SL)}
    #DEBUG_INDEX +=1

    if game.title == 'tic tactics':#this should be standardized
        role = 1 if game.player == 0 else -1 #tictactics
    else:
        role = 1 if game.player == 1 else -1 #connect four, tictactoe, mnk
    
    if root:#init the root for callback purposes
        for i in xrange(NODE_SIZE):
            cache[2*PRIME+1][i] = node[i]
        
    solved = 0
    if GW==0 or SL==0 or (GT==ST==0):
        solved = 1
    if solved:
        cache_push(key,node)
        return
    
    if GW > GWT or ST > STT or solved:
        return 
    
    if ST == 0 and GT > GTT or ST == 0 and SL > SLT: #multiple value pn search break
        return 
    
    #initialize
    #slow python stuff
    actions = game.legal_moves()
    na = len(actions)

    cdef:
        int64 GWc[MAX_MOVES]
        int64 GTc[MAX_MOVES]
        int64 STc[MAX_MOVES]
        int64 SLc[MAX_MOVES]
        int64 solvedc[MAX_MOVES]
        int64 workc[MAX_MOVES]
        int64 child[NODE_SIZE]
        int64 child_GT,child_GW,child_ST,child_SL
        int64 child_GTT,child_GWT,child_STT,child_SLT
        int64 other_GW,other_GT,other_ST,other_SL
        int m_index,f_index,s_index
        
    for i in xrange(na):
        GWc[i] = 1
        GTc[i] = 1
        STc[i] = 1
        SLc[i] = 1
        solvedc[i] = 0
        workc[i] = 1
    
    for i in xrange(na):
        move = actions[i]
        game.update_move(move)
        cache_pull(game.key,child)

        if child[0] == KEY_INIT:
            if game.over:
                #tictactics stuff
                if game.result == 1:
                    GWc[i],GTc[i],STc[i],SLc[i] = 0,0,INF,INF
                elif game.result == -1:
                    GWc[i],GTc[i],STc[i],SLc[i] = INF,INF,0,0
                else:
                    GWc[i],GTc[i],STc[i],SLc[i] = INF,0,0,INF
                solvedc[i] = 1

        else:
            
            workc[i] = child[1]
            GWc[i]   = child[2]
            GTc[i]   = child[3]
            STc[i]   = child[4]
            SLc[i]   = child[5]
        game.erase_move()

    if role == 1: #max node
        GW,GT,ST,SL = INF,INF,0,0 #init for min,min,sum,sum 
        for i in xrange(na):
            if GW > GWc[i]:
                GW = GWc[i]
            if GT > GTc[i]:
                GT = GTc[i]
            ST += STc[i]
            SL += SLc[i]
            
        if ST > INF:
            ST = INF
        if SL > INF:
            SL = INF
    if role == -1: #min node
        GW,GT,ST,SL = 0,0,INF,INF #init for sum,sum,min,min
        for i in xrange(na):
            GW += GWc[i]
            GT += GTc[i]
            if ST > STc[i]:
                ST = STc[i]
            if SL > SLc[i]:
                SL = SLc[i]
                
        if GW > INF:
            GW = INF
        if GT > INF:
            GT = INF
    if GW == 0 or SL ==0 or (ST==GT==0):
        solved = 1


    while GW <= GWT and ST<= STT and not solved:
        if (ST == 0 and GT > GTT) or (ST == 0 and SL>SLT): #multiple value pn search break
            break
            
            
        #DEBUG[DEBUG_INDEX%100] = {'where':'inside',
        #                          'threshholds':(GWT,GTT,STT,SLT) , 
        #                          'pn' : (GW,GT,ST,SL),
        #                          'number of children':na,
        #                          'GWc':[GWc[i] for i in xrange(na)],
        #                          'GTc':[GTc[i] for i in xrange(na)],
        #                          'STc':[STc[i] for i in xrange(na)],
        #                          'SLc':[SLc[i] for i in xrange(na)]}
        #DEBUG_INDEX += 1
        
        if root: #update for callback purposes
            cache[2*PRIME+1][0] = key
            cache[2*PRIME+1][1] = work
            cache[2*PRIME+1][2] = GW
            cache[2*PRIME+1][3] = GT
            cache[2*PRIME+1][4] = ST
            cache[2*PRIME+1][5] = SL

        if na == 1:

            move = actions[0]


            child_GTT = GTT
            child_GWT = GWT
            child_STT = STT #- node.ST + child.ST 
            child_SLT = SLT #- node.SL + child.SL

            game.update_move(move)
            dfpn_search(game,child,child_GWT,child_GTT,child_STT,child_SLT,
                        epsilon=ep,root = 0,callback_freq=callback_freq)

            game.erase_move()
            
            work = child[1]
            GW=child[2]
            GT=child[3]
            ST=child[4]
            SL=child[5]
            
            if GW ==0 or SL ==0 or (ST==GT==0):
                solved = 1

        else:
            if role == 1:
                if ST == 0:
                    get_2_smallest(GTc,na,&f_index,&s_index)
                else:
                    get_2_smallest(GWc,na,&f_index,&s_index)
                child_GW = GWc[f_index]
                child_GT = GTc[f_index]
                child_ST = STc[f_index]
                child_SL = SLc[f_index]
                
                move = actions[f_index]
                other_GW = GWc[s_index]
                other_GT = GTc[s_index]
                
                

                
                #if ST != 0: #don't start the threshhold until GW is INF
                #    child_GTT = INF
                if GTT < ep*other_GT:
                    child_GTT = GTT
                else:
                    child_GTT = <int64>(ep*other_GT)
                
                #if ST == 0:
                #    child_GWT = INF
                if GWT < other_GW*ep:
                    child_GWT = GWT
                else:
                    child_GWT = <int64>(other_GW*ep)
                
                #if ST == 0:
                #    child_STT = INF
                if STT > INF_LOW:
                    child_STT = INF
                else:
                    child_STT = STT - ST + child_ST
                    
                #if ST != 0:
                #    child_SLT = INF
                if SLT > INF_LOW:
                    child_SLT = INF
                else:
                    child_SLT = SLT - SL + child_SL
                
                game.update_move(move)
                dfpn_search(game,child,child_GWT,child_GTT,child_STT,child_SLT,
                            epsilon=ep,root = 0,callback_freq=callback_freq)
                game.erase_move()
            
                workc[f_index] = child[1]
                GWc[f_index] = child[2]
                GTc[f_index] = child[3]
                STc[f_index] = child[4]
                SLc[f_index] = child[5]
                
                GW = INF
                GT = INF
                ST = 0
                SL = 0
                work = 1
                
                for i in xrange(na):
                    if GW > GWc[i]:
                        GW = GWc[i]
                    if GT > GTc[i]:
                        GT = GTc[i]
                    work += workc[i]
                    ST += STc[i]
                    SL += SLc[i]
                    
                if ST > INF:
                    ST = INF
                if SL > INF:
                    SL = INF
                    
                if GW ==0 or SL ==0 or (ST==GT==0):
                    solved = 1            
            else:
                if ST == 0:
                    get_2_smallest(SLc,na,&f_index,&s_index)
                else:
                    get_2_smallest(STc,na,&f_index,&s_index)
                child_SL = SLc[f_index]
                child_ST = STc[f_index]
                child_GT = GTc[f_index]
                child_GW = GWc[f_index]
                move = actions[f_index]
                other_SL = SLc[s_index]
                other_ST = STc[s_index]


                if ST == 0:
                    child_STT = INF
                elif STT < other_ST*ep:
                    child_STT = STT
                else:
                    child_STT =  <int64>(other_ST*ep)
                    
                #if ST != 0:
                #    child_SLT = INF
                if SLT < other_SL*ep:
                    child_SLT = SLT
                else:
                    child_SLT =  <int64>(other_SL*ep)
                
                #if ST != 0: #don't start the threshhold until GW is INF
                #    child_GTT = INF
                if GTT > INF_LOW:
                    child_GTT = INF
                else:
                    child_GTT = GTT - GT + child_GT
                #if ST == 0:
                #    child_GWT = INF
                if GWT > INF_LOW:
                    child_GWT = INF
                else:
                    child_GWT = GWT - GW + child_GW
                
                game.update_move(move)
                dfpn_search(game,child,child_GWT,child_GTT,child_STT,child_SLT,
                            epsilon=ep,root = 0,callback_freq=callback_freq)
                game.erase_move()
            
                workc[f_index] = child[1]
                GWc[f_index] = child[2]
                GTc[f_index] = child[3]
                STc[f_index] = child[4]
                SLc[f_index] = child[5]
                
                GW = 0
                GT = 0
                ST = INF
                SL = INF
                work = 1
                
                for i in xrange(na):
                    GW += GWc[i]
                    GT += GTc[i]
                    work += workc[i]
                    if ST > STc[i]:
                        ST = STc[i]
                    if SL > SLc[i]:
                        SL = SLc[i]
                        
                if ST > INF:
                    ST = INF
                if SL > INF:
                    SL = INF
                
                if GW ==0 or SL ==0 or (ST==GT==0):
                    solved = 1
                    
    node[0] = key2
    node[1] = work
    node[2] = GW
    node[3] = GT
    node[4] = ST
    node[5] = SL
    cache_push(key,node)


    if root:
        cache[2*PRIME+1][0] = key
        cache[2*PRIME+1][1] = work
        cache[2*PRIME+1][2] = GW
        cache[2*PRIME+1][3] = GT
        cache[2*PRIME+1][4] = ST
        cache[2*PRIME+1][5] = SL
    return
   