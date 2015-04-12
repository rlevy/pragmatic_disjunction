import numpy as np
import sys
from collections import defaultdict
import cPickle as pickle
from itertools import product
import matplotlib
import matplotlib.pyplot as plt
sys.path.append('/Volumes/CHRIS/Documents/research/pypragmods/pypragmods/')
from lexica import Lexica, NULL_MSG, DISJUNCTION_SIGN
from pragmods import Pragmod

plt.style.use('bls41.mplstyle')
COLORS = ['#1B9E77', '#D95F02', '#7570B3', '#E7298A', '#66A61E', '#E6AB02', '#A6761D', '#666666']

def Q_implicature_simulation(output_filename="fig/Q-implicature-simulation"):
    # Messages:
    GENERAL_MSG = 'general'
    SPECIFIC_MSG = 'specific'
    DISJ_MSG = GENERAL_MSG + DISJUNCTION_SIGN + SPECIFIC_MSG
    # States:
    GENERAL_ONLY_REF = r'w_{\textsc{general-only}}'
    SPECIFIC_REF = r'w_{\textsc{specific}}'
    DISJ_REF = r'%s v %s' % (GENERAL_ONLY_REF, SPECIFIC_REF)
    # Common structures:
    BASELEXICON = {GENERAL_MSG: [GENERAL_ONLY_REF, SPECIFIC_REF], SPECIFIC_MSG: [SPECIFIC_REF]}
       
    ##### General function for getting data points:
    def Q_implicature_simulation_datapoint(specific_cost, dcost=1.0, alpha=2.0):
        # Values to obtain:
        is_max = False
        listener_val = None
        speaker_val = None
        # Set-up:
        lexica = Lexica(baselexicon=BASELEXICON, costs={GENERAL_MSG: 0.0, SPECIFIC_MSG: specific_cost}, join_closure=True, nullsem=True, nullcost=5.0, disjunction_cost=dcost)
        ref_probs = np.repeat(1.0/len(lexica.states), len(lexica.states))
        lexprior = np.repeat(1.0/len(lexica.lexica2matrices()), len(lexica.lexica2matrices()))
        # Run the model:
        mod = Pragmod(lexica=lexica.lexica2matrices(), messages=lexica.messages, states=lexica.states, costs=lexica.cost_vector(), lexprior=lexprior, prior=ref_probs, alpha=alpha)
        langs = mod.run_expertise_model(n=3, display=False, digits=2)
        # Get the values we need:
        speaker = mod.speaker_lexical_marginalization(langs[-2])
        listener = mod.listener_lexical_marginalization(langs[-3])
        general_msg_index = lexica.messages.index(GENERAL_MSG)
        general_only_state = lexica.states.index(GENERAL_ONLY_REF)
        disj_state_index = lexica.states.index(DISJ_REF)
        disj_msg_index = lexica.messages.index(DISJ_MSG)
        speaker_val = speaker[disj_state_index, disj_msg_index]
        listener_val = listener[general_msg_index, general_only_state]
        # Determine whether max, with a bit of rounding to avoid spurious mismatch diagnosis:
        maxspkval = np.max(speaker[disj_state_index])
        is_max = np.round(speaker_val, 10) == np.round(maxspkval, 10)
        # Return values:
        return (listener_val, speaker_val, is_max)

    ##### Plot creation:
    matplotlib.rc('font', family='serif', serif='times') # Not sure why this has to be set to get the legend font to change.
    # Values to vary:
    specific_costs = [0.0,1.0,2.0,3.0,4.0]
    disjunction_costs = np.arange(0.0, 5.0, 1)
    alphas = np.array([1.0, 2.0, 3.0, 4.0])    
    # Panels:
    variable_lookup = {r'C(\textit{or})': disjunction_costs, r'\alpha': alphas}
    variable_filename_suffixes = ['alphas', 'or']
    ylims = {r'C(\textit{or})':  [-0.05, 0.45], r'\alpha': [0.15, 0.75]}    
    for variable_name, suffix in zip(variable_lookup, variable_filename_suffixes):
        # Figure set-up:
        fig, ax = plt.subplots(nrows=1, ncols=1)
        fig.set_figheight(7)
        fig.set_figwidth(8)        
        variables = variable_lookup[variable_name]
        ann_index = 0
        ann_adj = 1.01
        ha = 'left'
        va = 'top'
        if variable_name == r'\alpha':
            ha = 'right'
            va = 'center'
            ann_index = -1
            ann_adj = 0.98
        for color, variable in zip(COLORS, variables):
            dcost = variable
            alpha = 2.0            
            if variable_name == r'\alpha':                
                dcost = 1.0
                alpha = variable
            vals = []
            for cost in specific_costs:
                vals.append(Q_implicature_simulation_datapoint(cost, dcost=dcost, alpha=alpha))
            listener_vals, speaker_vals, _ = zip(*vals)                
            max_booleans = [(i, j) for i, j, is_max in vals if is_max]            
            # Plotting (multiple lines with max-value annotations)
            ax.plot(listener_vals, speaker_vals, color=color, linewidth=2)
            if max_booleans:
                maxx, maxy = zip(*max_booleans)
                ax.plot(maxx, maxy, linestyle=':', linewidth=6, color=color)
            ax.annotate(r'$%s = %s$' % (variable_name, variable), xy=(listener_vals[ann_index]*ann_adj, speaker_vals[ann_index]), fontsize=16, ha=ha, va=va, color=color)
        # Axes:
        ax.set_xlabel(r'$L_1(%s \mid \textit{%s})$' % (GENERAL_ONLY_REF, GENERAL_MSG), fontsize=18)
        ax.set_ylabel(r'$S_2(\textit{%s} \mid %s)$' % (DISJ_MSG.replace(' v ', r' or '), DISJ_REF.replace(' v ', r' \vee ')), fontsize=18)        
        ax.set_xlim([0.2, 1.05])   
        ax.set_ylim([0.0, 1.05])
        # Save the panel:
        plt.setp(ax.get_xticklabels(), fontsize=16)
        plt.setp(ax.get_yticklabels(), fontsize=16)
        plt.savefig("%s-%s.pdf" % (output_filename, suffix), bbox_inches='tight')
    
######################################################################

def I_implicature_simulation(output_filename="fig/I-implicature-simulation", dcost=1.0, alpha=3.0):
    ##### General set-up:
    # Messages:
    SUPERKIND_MSG = r'general'
    COMMON_MSG = r'unmarked\_specific'
    UNCOMMON_MSG = r'marked\_specific'
    DISJ_MSG = "%s%s%s" % (SUPERKIND_MSG, DISJUNCTION_SIGN, UNCOMMON_MSG)
    # Referents:
    COMMON_REF = r'r_{\textsc{COMMON}}'
    UNCOMMON_REF = r'r_{\textsc{UNCOMMON}}'
    DISJ_REF = "%s%s%s" % (COMMON_REF, DISJUNCTION_SIGN, UNCOMMON_REF)
    # Common structures:
    BASELEXICON = {SUPERKIND_MSG: [UNCOMMON_REF, COMMON_REF], COMMON_MSG: [COMMON_REF], UNCOMMON_MSG: [UNCOMMON_REF]}
    LEXICAL_COSTS = {SUPERKIND_MSG: 0.0, COMMON_MSG: 0.0, UNCOMMON_MSG: 0.0}
   
    ##### General function for getting data points:
    def I_implicature_simulation_datapoint(common_ref_prob, dcost=1.0, alpha=2.0):
        # Values to obtain:
        is_max = False
        listener_val = None
        speaker_val = None
        # Set-up:
        lexica = Lexica(baselexicon=BASELEXICON, costs=LEXICAL_COSTS, join_closure=True, nullsem=True, nullcost=5.0, disjunction_cost=dcost)
        ref_probs = np.array([common_ref_prob, (1.0-common_ref_prob)/2.0, (1.0-common_ref_prob)/2.0])
        lexprior = np.repeat(1.0/len(lexica.lexica2matrices()), len(lexica.lexica2matrices()))
        # Run the model:
        mod = Pragmod(lexica=lexica.lexica2matrices(), messages=lexica.messages, states=lexica.states, costs=lexica.cost_vector(), lexprior=lexprior, prior=ref_probs, alpha=alpha)
        langs = mod.run_expertise_model(n=3, display=False, digits=2)
        # Get the values we need:
        speaker = mod.speaker_lexical_marginalization(langs[-2])
        listener = mod.listener_lexical_marginalization(langs[-3])
        superkind_term_index = mod.messages.index(SUPERKIND_MSG)
        common_state_index = mod.states.index(COMMON_REF)
        disj_term_index = mod.messages.index(DISJ_MSG)
        disj_state_index = mod.states.index(DISJ_REF)
        # Fill in listener_val and speaker_val:
        listener_val = listener[superkind_term_index, common_state_index]
        speaker_val = speaker[disj_state_index, disj_term_index]
        # Determine whether max, with a bit of rounding to avoid spurious mismatch diagnosis:
        maxspkval = np.max(speaker[disj_state_index])
        is_max = np.round(speaker_val, 10) == np.round(maxspkval, 10)
        # Return values:
        return (listener_val, speaker_val, is_max)

    ##### Plot creation:
    matplotlib.rc('font', family='serif', serif='times') # Not sure why this has to be set to get the legend font to change.
    # Values to vary:
    common_ref_probs = np.arange(1.0/3.0, 1.0/1.0, 0.01)
    disjunction_costs = np.arange(1.0, 5.0, 1)
    alphas = np.array([1.06, 2.0, 3.0, 4.0])    
    # Panels:
    variable_lookup = {r'C(\textit{or})': disjunction_costs, r'\alpha': alphas}
    variable_filename_suffixes = ['alphas', 'or']
    ylims = {r'C(\textit{or})':  [-0.05, 0.45], r'\alpha': [0.15, 0.75]}
    for variable_name, suffix in zip(variable_lookup, variable_filename_suffixes):
        # Figure set-up:
        fig, ax = plt.subplots(nrows=1, ncols=1)
        fig.set_figheight(7)
        fig.set_figwidth(8)
        variables = variable_lookup[variable_name]
        ha = 'left'
        va = 'top'
        if variable_name == r'\alpha':
            ha = 'right'
            va = 'top'
        for color, variable in zip(COLORS, variables):
            dcost = variable
            alpha = 2.0            
            if variable_name == r'\alpha':                
                dcost = 1.0
                alpha = variable
            vals = []
            for ref_prob in common_ref_probs:
                vals.append(I_implicature_simulation_datapoint(ref_prob, dcost=dcost, alpha=alpha))
            listener_vals, speaker_vals, _ = zip(*vals)                
            max_booleans = [(i, j) for i, j, is_max in vals if is_max]            
            # Plotting (multiple lines with max-value annotations)
            ax.plot(listener_vals, speaker_vals, color=color, linewidth=2)
            if max_booleans:
                maxx, maxy = zip(*max_booleans)
                ax.plot(maxx, maxy, linestyle=':', linewidth=6, color=color)
            # Annotation:
            if variable_name == r'\alpha' and variable == variables[-1]: # Avoid label overlap for alpha=3 and alpha=4.
                va = 'bottom'
            ax.annotate(r'$%s = %s$' % (variable_name, variable), xy=(listener_vals[0]*0.98, speaker_vals[0]), fontsize=16, ha=ha, va=va, color=color)
            ax.set_ylim(ylims[variable_name])
        # Axes:
        ax.set_xlabel(r'$L_1(%s \mid \textit{%s})$' % (COMMON_REF, SUPERKIND_MSG), fontsize=18)
        ax.set_ylabel(r'$S_2(\textit{%s} \mid %s)$' % (DISJ_MSG.replace(' v ', r' or '), DISJ_REF.replace(' v ', r' \vee ')), fontsize=18)
        ax.set_xlim([0.0,1.0])
        # Save the panel:
        plt.setp(ax.get_xticklabels(), fontsize=16)
        plt.setp(ax.get_yticklabels(), fontsize=16)
        plt.savefig("%s-%s.pdf" % (output_filename, suffix), bbox_inches='tight')
        

Q_implicature_simulation()            
#I_implicature_simulation()

