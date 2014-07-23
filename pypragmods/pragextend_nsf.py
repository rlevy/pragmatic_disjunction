#!/usr/bin/env python

import numpy as np
from pragmods import Pragmod, display_matrix
from lexica import Lexica
import sys

def complex():

    m = np.array([[0.,0.,1.],[0.,1.,1.], [1.,1.,0.]])
    
    mod = Pragmod(messages=['hat', 'glasses', 'mustache'],
                  meanings=['r1', 'r2', 'r3'],
                  costs=np.repeat(0.0, 3),
                  prior=np.repeat(1/3., 3),
                  temperature=1.0)

    mod.run_base_model(m, n=11)
                


#complex()

def context():

    eps = sys.float_info.epsilon

    eps = 0.0
     
    nullsem = [1.0, 1.0]
    
    mod = Pragmod(lexica=[
        np.array([[eps, eps], [1.0, 1.0]]),
        np.array([[eps, 1.0], [1.0, 1.0]])],
        messages=['hat', 'glasses'],
        meanings=['r1', 'r2'],
        costs=np.array([0.0, 0.0]),
        prior=np.array([0.5, 0.5]),
        lexprior=np.array([0.5, 0.5]),
        temperature=2.8)

    langs = mod.run_expertise_model(n=10, display=True)

    target_lexicon_index = 1

    print "======================================================================"
    print """Speaker"""
    print "".join([x.rjust(12) for x in ["", "hat", "glasses"]])
    index = 2
    for i in range(1, len(langs), 2):
        vals = [langs[i][j][0][target_lexicon_index] for j in [0, 1]]
        print ('Spk%s observes <r1,house2>' % index).rjust(28), "".join([str(round(x, 4)).rjust(12) for x in vals])
        index += 1

    print "======================================================================"
    print "Listener"
    print "".join([x.rjust(12) for x in ["", "<house1,r1>", "<house1,r2>", "<house2,r1>", "<house2,r2>"]])
    index = 1
    for i in range(0, len(langs), 2):
        lang = langs[i]
        vals = [lang[0][1][0],  lang[0][1][1], lang[1][1][0], lang[1][1][1]]
        print ('L%i hears glasses' % index).rjust(28), "".join([str(round(x, 4)).rjust(15) for x in vals])
        index += 1

context()
