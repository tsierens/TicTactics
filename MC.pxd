ctypedef unsigned long long int64
cdef enum:
    MAX_MOVES = 81
    MAX_LENGTH = 81
    
cdef class MC_node(object):
    cdef:
        int64 key
        public int is_root
        public int leaf,solved,result,player,sib_index
        public long long N[MAX_MOVES],V[MAX_MOVES]
        public double Q[MAX_MOVES]
        public object actions
        public int n_children
        public int children_solved[MAX_MOVES]
        public int children_result[MAX_MOVES]
        MC_node child
        MC_node sib
        MC_node parent