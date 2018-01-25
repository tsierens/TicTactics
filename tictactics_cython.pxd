cdef class Board(object):
    cdef:
        public int player,result,big_moves,over
        public unsigned long long big_board,key
        public unsigned long long board[9]
        public int moves[9]
        public int active_board[2]
        public object log,alog,title
        
    cpdef object legal_moves(self)
    
    cpdef erase_move(self)
    
    cpdef update_move(self,object move,int safe=?)  
    
    