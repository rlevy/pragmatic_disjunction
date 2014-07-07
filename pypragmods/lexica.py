#!/usr/bin/env python

import numpy as np
from copy import copy
from itertools import combinations, product
from collections import defaultdict
import pragmods


NULL_MSG = 'NULL'
DISJUNCTION_SIGN = ' v '
CONJUNCTION_SIGN = ' & '

class Lexica:
    def __init__(self,
                 baselexicon=None,
                 nullsem=True,
                 join_closure=False,
                 meet_closure=False,
                 block_ineffability=False,
                 costs=defaultdict(float),
                 disjunction_cost=0.01,
                 conjunction_cost=0.01,
                 null_cost=5.0
                 ):
        self.baselexicon = baselexicon
        self.messages = sorted(self.baselexicon.keys())
        self.atomic_states = sorted(list(set(reduce((lambda x,y : x + y), self.baselexicon.values()))))
        self.states = copy(self.atomic_states)
        self.nullsem = nullsem
        self.join_closure = join_closure
        self.meet_closure = meet_closure
        self.block_ineffability = block_ineffability
        self.costs = costs
        self.disjunction_cost = disjunction_cost
        self.conjunction_cost = conjunction_cost
        self.null_cost = null_cost
        self.lexica = self.get_lexica()

    def cost_vector(self):
        return np.array([self.costs[msg] for msg in self.messages])
        
    def get_lexica(self):
        lexica = []        
        enrichments = [self.powerset(self.baselexicon[msg]) for msg in self.messages]        
        for x in product(*enrichments):
            lexica.append(dict(zip(self.messages, x)))
        if self.join_closure:
            lexica = self.add_join_closure(lexica)
        if self.meet_closure:
            lexica = self.add_meet_closure(lexica)
        # Update the canonical message set and state set; has to be done AFTER all closures!
        if self.join_closure:
            self.messages += [DISJUNCTION_SIGN.join(sorted(cm)) for cm in self.powerset(self.baselexicon.keys(), minsize=2)]
            self.states += [DISJUNCTION_SIGN.join(sorted(sem)) for sem in self.powerset(self.atomic_states, minsize=2)]
        if self.meet_closure:
            self.messages += [CONJUNCTION_SIGN.join(sorted(cm)) for cm in self.powerset(self.baselexicon.keys(), minsize=2)]
            self.states += [CONJUNCTION_SIGN.join(sorted(sem)) for sem in self.powerset(self.atomic_states, minsize=2)]
        # Add nullsem last so that it doesn't participate in any closures (and displays last in matrices):
        if self.nullsem:
            lexica = self.add_nullsem(lexica)
            self.messages.append(NULL_MSG)
            self.costs[NULL_MSG] = self.null_cost
        return lexica

    def add_join_closure(self, lexica):
        return self.add_closure(lexica=lexica, connective=DISJUNCTION_SIGN, combo_func=(lambda x,y : x | y), cost_value=self.disjunction_cost)   

    def add_meet_closure(self, lexica):
        return self.add_closure(lexica=lexica, connective=CONJUNCTION_SIGN, combo_func=(lambda x,y : x & y), cost_value=self.conjunction_cost)    

    def add_closure(self, lexica=None, connective=None, combo_func=None, cost_value=None):
        complex_msgs = [connective.join(sorted(cm)) for cm in self.powerset(self.messages, minsize=2)]
        for i, lex in enumerate(lexica):
            for cm in complex_msgs:
                cm_parts = cm.split(connective)
                # Get all the worlds consistent with the complex message:
                vals = reduce(combo_func, [set(lex[word]) for word in cm_parts])
                # Get the powerset of that set of worlds:
                vals = self.powerset(vals, minsize=1)
                # Create the new value, containing worlds and "disjoined worlds":
                lex[cm] = [connective.join(sorted(sem)) for sem in vals]
                self.costs[cm] = (cost_value*(len(cm_parts)-1)) + sum(self.costs[word] for word in cm_parts)
            for msg in self.messages:
                lex[msg] += [connective.join(sorted(sem)) for sem in self.powerset(lex[msg], minsize=2)]
            lexica[i] = lex
        return lexica

    def add_nullsem(self, lexica):
        for i, lex in enumerate(lexica):
            lex[NULL_MSG] = self.states
            lexica[i] = lex
        return lexica   

    def lexica2matrices(self):
        mats = []
        m = len(self.messages)
        n = len(self.states)                
        for lex in self.lexica:
            mat = np.zeros((m, n))
            for i, msg in enumerate(self.messages):
                for j, d in enumerate(self.states):
                    if d in lex[msg]:
                        mat[i,j] = 1.0
            minval = 1 if self.nullsem else 0
            if not (self.block_ineffability and minval in np.sum(mat, axis=0)):                
                mats.append(mat)
        return mats    
    
    def powerset(self, x, minsize=1, maxsize=None):
        result = []
        if maxsize == None: maxsize = len(x)
        for i in range(minsize, maxsize+1):
            for val in combinations(x, i):
                result.append(list(val))
        return result

    def display(self, digits=4):
        for i, mat in enumerate(self.lexica2matrices()):      
            pragmods.display_matrix(mat, rnames=self.messages, cnames=self.states, title="Lex%s" % i, digits=digits)     

    def __len__(self):
        return len(self.lexica2matrices())
            
if __name__ == '__main__':

    def scalarlex():
        lexica = Lexica(baselexicon={'some': ['w_SOMENOTALL', 'w_ALL'], 'all': ['w_ALL']},
                        join_closure=True,
                        meet_closure=True,
                        block_ineffability=False
                        )            
        lexica.display()
        print lexica.cost_vector()
        
    scalarlex()
    
        
                 
