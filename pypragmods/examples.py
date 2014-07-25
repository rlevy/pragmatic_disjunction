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

    
def expert_disjunction(n=2):
    lexica = Lexica(
        baselexicon={
            'A': ['w1'], 
            'B': ['w2'], 
            'C': ['w3'],
            'D': ['w4'],
            'X': ['w1', 'w2', 'w3', 'w4']},
        join_closure=True,
        disjunction_cost=0.001,
        null_cost=4.0)

    # The unknown word has an atomic meaning:
    mats = lexica.lexica2matrices()    
    # mats = [mat for mat in mats if np.sum(mat[4]) == 1]
    #print len(mats)
    #for mat in mats:
    #    display_matrix(mat, rnames=lexica.messages, cnames=lexica.states)
            
    mod = Pragmod(lexica=mats,
                  messages=lexica.messages,
                  meanings=lexica.states,
                  costs=lexica.cost_vector(),
                  prior=np.repeat(1.0/len(lexica.states), len(lexica.states)),
                  lexprior=np.repeat(1.0/len(mats), len(mats)),
                  temperature=2.0,
                  alpha=1.0,
                  beta=1.0)
    langs = mod.run_expertise_model(n=n, display=False)

    # Reproduce Roger's "key speaker result":
    print "======================================================================"
    print """Reproduce Roger's "key speaker result" """
    print "".join([x.rjust(10) for x in ["Spk", "A", "X", "A v X"]])
    index = 2
    for i in range(1, len(langs), 2):
        vals = [langs[i][j][0][0] for j in [0,4,5]]
        print ('S%i' % index).rjust(10), "".join([str(round(x, 3)).rjust(10) for x in vals])
        index += 1

    # Corresponding key listener result:
    print "======================================================================"
    print 'Corresponding key listener result (same, mutatis mutandis, for <Lex1, B>, <Lex2, C>, <Lex3, D>)'
    print "".join([x.rjust(10) for x in ["Lis", "A", "X", "A v X"]])
    index = 2
    for i in range(0, len(langs), 2):
        vals = [langs[i][0][j][0] for j in [0,4,5]]
        print ('L%i, Lex0' % index).rjust(10), "".join([str(round(x, 3)).rjust(10) for x in vals])
        index += 1
        
                                      
if __name__ == '__main__':    

    #scalars(n=3)
    #m_implicature_bergen_etal2014()
    #m_implicature_smith_etal(n=3)
    expert_disjunction(n=25)

