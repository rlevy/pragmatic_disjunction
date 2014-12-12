
## Understanding the relationship between definitional and Hurfordian readings

# In[1]:

import itertools
import random
import cPickle as pickle
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import lsa
import hyperparameterexplore

paramexplore_lex5 = pickle.load(file('paramexplore-defin-lex5.pickle'))


base_lexica = [{'A': ['1'], 'B': ['2'], 'X':['1', '2']},
               {'A': ['1'], 'B': ['2'], 'C': ['3'], 'X': ['1', '2', '3']},
               {'A': ['1'], 'B': ['2'], 'C': ['3'], 'D': ['4'], 'X': ['1', '2', '3', '4']}]


def classify_lexica(base_lexicon):
    classified = []
    experiment = lsa.Experiment(baselexicon=base_lexicon, lexical_costs={key:0.0 for key in base_lexicon.keys()})
    experiment.build()
    for i, lex in enumerate(experiment.lexica):
        a_index = experiment.messages.index('A')
        a_sem = set([s for j, s in enumerate(experiment.states) if lex[a_index][j] > 0.0])
        x_index = experiment.messages.index('X')
        x_sem = set([s for j, s in enumerate(experiment.states) if lex[x_index][j] > 0.0])
        cls = 'O'
        if len(a_sem & x_sem) == 0:
            cls = 'H'            
        elif a_sem == x_sem:
            cls = 'D'
        classified.append((i, experiment.lex2str(i), cls))                              
    return classified                        

def view_lex_classes(cls):
    for base_lexicon in base_lexica:
        print "=" * 70
        print base_lexicon
        for i, lex, c in classify_lexica(base_lexicon):        
            if c == cls:
                print i, lex

def get_cls_params(base_lexicon, cls, state, paramexplore_lex):
    cls_lex = []
    for lexindex, lex, c in classify_lexica(base_lexicon):
        if c == cls:
            cls_lex += paramexplore_lex[(lexindex, state)]
    return cls_lex

hc_lex5 = get_cls_params(base_lexica[2], 'H', '1 v 2', paramexplore_lex5)
defin_lex5 = get_cls_params(base_lexica[2], 'D', '1', paramexplore_lex5)


param_pairs = [(hc_lex5, defin_lex5)]

hyperparameterexplore.alpha_beta_gamma_scatterplot(param_pairs, logx=True, title=False, output_filename='paramexplore.pdf')
