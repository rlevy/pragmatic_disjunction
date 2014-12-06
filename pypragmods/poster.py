import itertools
import cPickle as pickle
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import lsa


exp = lsa.Experiment(
    n=3, 
    baselexicon={'A': ['1'], 'B': ['2'], 'X':['1', '2']}, 
    lexical_costs={'A':0.0, 'B':0.0, 'X':0.0},
    temperature=1.0, 
    disjunction_cost=0.01, 
    beta=7.0, 
    alpha=5.0, 
    null_cost=5.0)

exp.build()

def lis2latex(mat, msgs):
    msgs = [exp.messages.index(msg) for msg in msgs]    
    output = ""
    mat = np.round(mat, 2)
    for msg in msgs:
        row = mat[msg]
        strs = map(str, row)
        s = " & ".join(strs)
        s = "{%s}" % s
        output += s
    return "\lismat%s" % output


def spk2latex(mat, msgs):
    msgs = [exp.messages.index(msg) for msg in msgs]    
    output = ""
    mat = np.round(mat, 2)
    for msg in msgs:
        row = mat[:, msg]
        strs = map(str, row)
        s = " & ".join(strs)
        s = "{%s}" % s
        output += s
    return "\spkmat%s" % output

msgs = ['A', 'X', 'A v X']

for lex in exp.lexica:
    print "======================================================================"
    print lis2latex(lex, msgs)
    print lis2latex(exp.model.l0(lex), msgs)
    print spk2latex(exp.model.s1(lex), msgs)        
    print lis2latex(exp.model.l1(lex), msgs)    
                    
    

    
