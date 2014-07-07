#!/usr/bin/env python

import numpy as np
from pragmods import Pragmod, display_matrix
from lexica import Lexica

def display_all_models(mod, n=2, display=True):        
    print '**********************************************************************'
    print 'Lexical uncertainty model'
    mod.run_uncertainty_model(n=n, display=display)
    print '**********************************************************************'        
    print 'Social anxiety model'        
    mod.run_anxiety_model(n=n, display=display)
    print '**********************************************************************'
    print 'Anxiety/expertise model'        
    mod.run_expertise_model(n=n, display=display)
        
def scalars(n=2):
    """Scalar example without and with disjunction; compare with Bergen et al.'s figure 5 and figure 9"""
    baselexicon = {'some': ['w_SOMENOTALL', 'w_ALL'], 'all': ['w_ALL']}
    basic_lexica = Lexica(baselexicon=baselexicon)
    disjunctive_lexica = Lexica(baselexicon=baselexicon, join_closure=True)
    for lexica, label in ((basic_lexica, 'basic_lexica'), (disjunctive_lexica, 'disjunctive_lexica')):
        print "\n\n", label    
        mod = Pragmod(lexica=lexica.lexica2matrices(),
                      messages=lexica.messages,
                      meanings=lexica.states,
                      costs=lexica.cost_vector(),                     
                      prior=np.repeat(1.0/len(lexica.states), len(lexica.states)),
                      lexprior=np.repeat(1.0/len(lexica), len(lexica)),
                      temperature=1.0)
        display_all_models(mod, n=n)

def m_implicature_bergen_etal2014(n=2):
    """Settings for Bergen et al.'s figure 6. Seems to reproduce the effects they report."""
    lexica = Lexica(baselexicon={'expensive': ['w_RARE', 'w_FREQ'], 'cheap': ['w_RARE', 'w_FREQ']},
                    costs={'expensive':2.0, 'cheap':1.0},
                    null_cost=5.0)    
    mod = Pragmod(lexica=lexica.lexica2matrices(),
                  messages=lexica.messages,
                  meanings=lexica.states,
                  costs=lexica.cost_vector(),
                  prior=np.array([2.0/3.0, 1.0/3.0]),
                  lexprior=np.repeat(1.0/len(lexica), len(lexica)),
                  temperature=10.0)
    display_all_models(mod, n=n)  

def m_implicature_smith_etal(n=2):
    """Settings for Smith et al.'s 2013 ex from p. 5, though the code does 
    not reproduce the numbers they provide, for unknown reasons."""
    lexica = Lexica(baselexicon={'expensive': ['w_RARE', 'w_FREQ'], 'cheap': ['w_RARE', 'w_FREQ']},
                    costs={'expensive':1.0, 'cheap':0.5},
                    nullsem=False,
                    block_ineffability=True)
    mod = Pragmod(lexica=lexica.lexica2matrices(),
                  messages=lexica.messages,
                  meanings=lexica.states,
                  costs=lexica.cost_vector(),
                  prior=np.array([0.8, 0.2]),
                  lexprior=np.repeat(1.0/len(lexica), len(lexica)),
                  temperature=3.0)
    display_all_models(mod, n=n)


def explore_expert_disjunction(n=25,
                               baselexicon=None,
                               disjunction_cost=0.001,
                               null_cost=4.0,
                               temperature=2.0,
                               alpha=1.0,
                               beta=1.0,
                               unknown_word='X',
                               unknown_disjunction='A v X',
                               unknown_word_has_atomic_meaning=True,
                               target_lexicon_index=0):
    
    lexica = Lexica(baselexicon=baselexicon,
                    join_closure=True,
                    disjunction_cost=disjunction_cost,
                    null_cost=null_cost)

    mats = lexica.lexica2matrices()

    unk_index = lexica.messages.index(unknown_word)

    unknown_disjunction_index = lexica.messages.index(unknown_disjunction)

    if unknown_word_has_atomic_meaning:        
        mats = [mat for mat in mats if np.sum(mat[unk_index]) == 1]

    mod = Pragmod(lexica=mats,
                  messages=lexica.messages,
                  meanings=lexica.states,
                  costs=lexica.cost_vector(),
                  prior=np.repeat(1.0/len(lexica.states), len(lexica.states)),
                  lexprior=np.repeat(1.0/len(mats), len(mats)),
                  temperature=temperature,
                  alpha=alpha,
                  beta=beta)

    langs = mod.run_expertise_model(n=n, display=True)

    print "======================================================================"
    print """Reproduce Roger's "key speaker result" """
    print "".join([x.rjust(10) for x in ["Spk", "A", "X", "A v X", "NULL"]])
    index = 2
    for i in range(1, len(langs), 2):
        vals = [langs[i][j][target_lexicon_index][0] for j in [0, unk_index, unknown_disjunction_index, -1]]
        print ('S%i' % index).rjust(10), "".join([str(round(x, 4)).rjust(10) for x in vals])
        index += 1

    print "======================================================================"
    print 'Corresponding key listener result (same, mutatis mutandis, for <Lex1, B>, <Lex2, C>, <Lex3, D>)'
    print "".join([x.rjust(10) for x in ["Lis", "A", "X", "A v X", "NULL"]])
    index = 1
    for i in range(0, len(langs), 2):
        vals = [langs[i][target_lexicon_index][j][0] for j in [0, unk_index, unknown_disjunction_index, -1]]
        print ('L%i, Lex0' % index).rjust(10), "".join([str(round(x, 4)).rjust(10) for x in vals])
        index += 1

def expert_disjunction1():
    baselexicon = {
        'A': ['w1'], 
        'B': ['w2'], 
        'C': ['w3'],
        'X': ['w1', 'w2', 'w3']}

    explore_expert_disjunction(n=25,
                               baselexicon=baselexicon,
                               disjunction_cost=0.05,
                               null_cost=4.0,
                               temperature=3.0,
                               alpha=1.0,
                               beta=5.0,
                               unknown_word_has_atomic_meaning=True)

def expert_disjunction2():
    baselexicon = {
        'A': ['w1'], 
        'B': ['w2'], 
        'C': ['w3'],
        'D': ['w4'],
        'X': ['w1', 'w2', 'w3', 'w4']}

    explore_expert_disjunction(n=25,
                               baselexicon=baselexicon,
                               disjunction_cost=0.001,
                               null_cost=4.0,
                               temperature=2.0,
                               alpha=1.0,
                               beta=1.0,
                               unknown_word_has_atomic_meaning=True)
                                                  
                                                      
if __name__ == '__main__':    

    #scalars(n=3)
    #m_implicature_bergen_etal2014()
    #m_implicature_smith_etal(n=3)
    expert_disjunction2()

