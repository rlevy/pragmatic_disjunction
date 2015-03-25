import numpy as np
import sys
from collections import defaultdict
import cPickle as pickle
from itertools import product
import matplotlib
import matplotlib.pyplot as plt
sys.path.append('pypragmods/')
from lexica import Lexica, NULL_MSG, DISJUNCTION_SIGN
from pragmods import Pragmod

#plt.style.use('ggplot')

w1 = '1'; w2 = '2'; w3 = '3'

general = 'general'
specific = 'specific'

# matplotlib.rc('text', usetex=True)
# matplotlib.rc('font', family='serif', serif='times')
# matplotlib.rcParams['text.latex.preamble'] = [r"\usepackage{amsmath}", r'\boldmath']
# matplotlib.rcParams['xtick.major.pad']='1'
# matplotlib.rcParams['ytick.major.pad']='0'

def Q_implicature(output_filename="qimplicature-modeling.pdf", dcosts=(0,1,2,3,4,5,6,7,8)):
    cost_pairs = zip(np.repeat(0.0, 16), np.arange(15.0, -1.0, -1))
    #cost_pairs += zip(np.arange(0.0, 5.0, 1), np.repeat(0.0, 5))
    fig, axarray = plt.subplots(nrows=3, ncols=3)
    fig.set_figheight(14)
    fig.set_figwidth(14)
    axindices = list(product((0,1,2), (0,1,2)))
    for axind, dcost in zip(axindices, dcosts):
        print dcost
        speaker_vals = []
        listener_vals = []
        ax = axarray[axind]
        for c1, c2 in cost_pairs:
            print "\t", c1, c2
            lexica = Lexica(
                baselexicon={general: [w1, w2], specific: [w1]},
                costs={general:c1, specific:c2},
                join_closure=True,
                nullsem=True,
                nullcost=5.0,
                disjunction_cost=dcost)    
            mod = Pragmod(
                lexica=lexica.lexica2matrices(),
                messages=lexica.messages,
                states=lexica.states,
                costs=lexica.cost_vector(),
                prior=np.repeat(1.0/len(lexica.states), len(lexica.states)),
                lexprior=np.repeat(1.0/len(lexica), len(lexica)),
                temperature=1.0,
                alpha=3.0,
                beta=1.0)
            langs = mod.run_expertise_model(n=3, display=False, digits=2)
            print "======================================================================"
            print "Cost(general) = %s; Cost(specific) = %s; disjunction cost: %s" % (c1, c2, dcost)
            # Agents:
            speaker =  mod.speaker_lexical_marginalization(langs[-2])
            listener = mod.listener_lexical_marginalization(langs[-3])
            mod.display_listener_matrix(listener)
            mod.display_speaker_matrix(speaker)
            # Values:            
            general_msg_index = lexica.messages.index(general)
            general_only_state = lexica.states.index(w2)
            disj_state_index = lexica.states.index('1 v 2')
            disj_msg_index = lexica.messages.index(general + DISJUNCTION_SIGN + specific)
            speaker_val = speaker[disj_state_index, disj_msg_index]
            listener_val = listener[general_msg_index, general_only_state]
            # Store the values:
            speaker_vals.append(speaker_val)
            listener_vals.append(listener_val)
        ax.plot(listener_vals, speaker_vals)
        ax.tick_params(axis='both', which='both', bottom='off', left='off', top='off', right='off')
        ax.set_title("C(or) = %s" % dcost)
    xlab = r'Probability of \textit{general} implicating \textit{not specific}'
    ylab = r'Probability of saying \textit{general or specific} given a disjunctive state'    
    fig.text(0.5, 0.04, xlab, ha='center', va='center', fontsize=30)
    fig.text(0.04, 0.5, ylab, ha='center', va='center', rotation='vertical', fontsize=30)
    plt.savefig(output_filename, bbox_inches='tight')

superkind = 'boat'
common = 'motor'
uncommon = 'canoe'
disj = "%s%s%s" % (superkind, DISJUNCTION_SIGN, uncommon)


COMMON_REF = 'r_MOTOR'
UNCOMMON_REF = 'r_CANOE'
DISJ = "%s%s%s" % (UNCOMMON_REF, DISJUNCTION_SIGN, COMMON_REF)

def I_implicature(output_filename="I-implicature-modeling.pdf", dcost=1.0):
    fig, ax = plt.subplots(nrows=1, ncols=1)
    speaker_vals = []
    listener_vals = []
    maxspk = []
    lexica = Lexica(
        baselexicon={superkind: [UNCOMMON_REF, COMMON_REF], common: [COMMON_REF], uncommon: [UNCOMMON_REF]},
        costs={superkind:0.0, common:0.0, uncommon:0.0},
        join_closure=True,
        nullsem=True,
        nullcost=5.0,
        disjunction_cost=dcost)
    for prob in np.arange(1.0/3.0, 1.0/1.0, 0.01):
        prob2 = (1.0-prob)/2.0
        prob3 = (1.0-prob)/2.0           
        mod = Pragmod(
            lexica=lexica.lexica2matrices(),
            messages=lexica.messages,
            states=lexica.states,
            costs=lexica.cost_vector(),
            prior=np.array([prob2, prob, prob3]),
            lexprior=np.repeat(1.0/len(lexica), len(lexica)),
            temperature=1.0,
            alpha=1.06,
            beta=1.0)
        print "P(%s) = %s; P(%s) = %s; P(%s) = %s" % (COMMON_REF, prob, UNCOMMON_REF, prob2, DISJ, prob3)
        langs = mod.run_expertise_model(n=3, display=False, digits=2)
        speaker =  mod.speaker_lexical_marginalization(langs[-2])
        listener = mod.listener_lexical_marginalization(langs[-3])
        #mod.display_listener_matrix(listener)
        mod.display_speaker_matrix(speaker)
        superkind_term_index = mod.messages.index(superkind)
        common_state_index = mod.states.index(COMMON_REF)
        disj_term_index = mod.messages.index(disj)
        disj_state_index = mod.states.index(DISJ)

        listener_val = listener[superkind_term_index, common_state_index]

        speaker_val = speaker[disj_state_index, disj_term_index]

        maxspkval = np.max(speaker[disj_state_index])
        print speaker_val, maxspkval
        if speaker_val == maxspkval:
            maxspk.append((listener_val, speaker_val))                
        print 'L1(%s | %s) = %s' % (COMMON_REF, superkind, listener_val)
        print 'S2(%s | %s) = %s' % (disj, DISJ, speaker_val)
        speaker_vals.append(speaker_val)
        listener_vals.append(listener_val)
    ax.plot(listener_vals, speaker_vals)
    maxx, maxy = zip(*maxspk)
    ax.plot(maxx, maxy, linestyle='', markersize=6, marker='o')
    ax.set_xlabel('L1(%s | %s)' % (COMMON_REF, superkind))
    ax.set_ylabel('S2(%s | %s)' % (disj, DISJ))
    plt.savefig(output_filename, bbox_inches='tight')

        

            
I_implicature()
