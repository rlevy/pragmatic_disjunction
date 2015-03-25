#/usr/bin/env python

######################################################################
# Examples from 
#
# Potts, Christopher and Roget Levy. 2015. Negotiating lexical
# uncertainty and expertise with disjunction. Proceedings of the 41st
# Annual Meeting of the Berkeley Linguistics Society.
#
# ---Christopher Potts
#
######################################################################

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
from utils import display_matrix


plt.style.use('ggplot')
matplotlib.rc('text', usetex=True)
matplotlib.rc('font', family='serif', serif='times')
matplotlib.rcParams['text.latex.preamble'] = [r"\usepackage{amsmath}", r'\boldmath']
matplotlib.rcParams['xtick.major.pad']='1'
matplotlib.rcParams['ytick.major.pad']='0'
COLORS = ['#1B9E77', '#D95F02', '#7570B3', '#E7298A', '#66A61E', '#E6AB02', '#A6761D', '#666666'] + matplotlib.colors.cnames.values()


######################################################################
##### Illustrative examples
    
w1 = '1'; w2 = '2'; w3 = '3'
p = 'p'; q = 'q'; porq = 'p v q'; pandq = 'p & q'

def simple_disjunction():    
    lexicon = np.array([[1.0, 1.0, 0.0],
                        [1.0, 0.0, 1.0],
                        [1.0, 0.0, 0.0],
                        [1.0, 1.0, 1.0],
                        [1.0, 1.0, 1.0]])
    mod = Pragmod(
        lexica=None,
        messages=[p, q, pandq, porq, NULL_MSG],
        meanings=[w1, w2, w3],
        costs=np.array([0.0, 0.0, 1.0, 1.0, 5.0]),
        prior=np.repeat(1.0/4.0, 3),
        lexprior=None,
        temperature=1.0,
        alpha=1.0,
        beta=0.0)
    mod.run_base_model(lexicon, n=2, display=True, digits=2)

def basic_scalar():
    lexica = [np.array([[1.0,1.0], [1.0, 0.0], [1.0, 1.0]]),
              np.array([[1.0,0.0], [1.0, 0.0], [1.0, 1.0]]),
              np.array([[0.0,1.0], [1.0, 0.0], [1.0, 1.0]])]    
    mod = Pragmod(lexica=lexica,
                  messages=['cheap', 'free', NULL_MSG],
                  meanings=[w1, w2],
                  costs=np.array([0.0, 0.0, 5.0]),
                  prior=np.repeat(1.0/2.0, 2),
                  lexprior=np.repeat(1.0/3.0, 3),
                  temperature=1.0,
                  alpha=1.0,
                  beta=2.0)
    for lex in lexica:
        print "=" * 70
        print mod.lex2str(lex)
        mod.run_base_model(lex, n=2, display=True, digits=2)      
    mod.run_expertise_model(n=3, display=True, digits=2)
          
def compositional_disjunction():
    lexica = Lexica(
        baselexicon={p: [w1, w2], q: [w1, w3], pandq:[w1]},
        costs={p:0.0, q:0.0, pandq:1.0},
        join_closure=True,
        nullsem=True,
        nullcost=5.0,
        disjunction_cost=1.0)
    lexica.display()
    mod = Pragmod(lexica=lexica.lexica2matrices(),
                  messages=lexica.messages,
                  meanings=lexica.states,
                  costs=lexica.cost_vector(),
                  prior=np.repeat(1.0/len(lexica.states), len(lexica.states)),
                  lexprior=np.repeat(1.0/len(lexica), len(lexica)),
                  temperature=1.0,
                  alpha=1.0,
                  beta=1.0)
    mod.run_expertise_model(n=2, display=True, digits=2)

######################################################################
##### Hurfordian and disjunction examples

def generic_disjunction_example(
        alpha=1.0,
        beta=1.0,
        disjunction_cost=1.0,
        n=2,
        fulldisplay=False,
        unknown_word=None):
    """Common code for our two illustrative examples, which
    differ only in the above keyword parameters. Increase n to see
    greater depths of recursion. use fulldisplay=True to see more
    details."""    
    # Use the lexicon generation convenience function to
    # generate all the join-closure lexica and calculate
    # the necessary message costs:
    lexica = Lexica(
        baselexicon={'A': ['1'], 'B': ['2'], 'X':['1', '2']}, 
        costs={'A':0.0, 'B':0.0, 'X':0.0},
        join_closure=True,
        nullsem=True,
        nullcost=5.0,
        disjunction_cost=disjunction_cost,
        unknown_word=unknown_word)
    # Lexical matrices:
    lexmats = lexica.lexica2matrices()         
    # Pragmatic models for the above lexical space.
    mod = Pragmod(
        lexica=lexmats,
        messages=lexica.messages,
        meanings=lexica.states,
        costs=lexica.cost_vector(),
        prior=np.repeat(1.0/len(lexica.states), len(lexica.states)),
        lexprior=np.repeat(1.0/len(lexmats), len(lexmats)),
        temperature=1.0,
        alpha=alpha,
        beta=beta)
    if fulldisplay:
        lexica.display()
        # Run the base model on the individual lexica so we can show those lower steps:
        for lex in lexmats:
            print "=" * 70
            print mod.lex2str(lex)
            mod.run_base_model(lex, n=2, display=True, digits=2)         
    ## Run the anxious experts model - fulldisplay=True for a fuller picture:
    langs = mod.run_expertise_model(n=n, display=fulldisplay, digits=2)
    # Look at the specific table we care about:
    msg_index = mod.messages.index('A v X')
    final_listener = langs[-1]
    mod.display_joint_listener(final_listener[msg_index], digits=2)
    return langs
        
def hurfordian_example(n=2, fulldisplay=False):
    generic_disjunction_example(alpha=2.0, beta=1.0, disjunction_cost=1.0, n=n, fulldisplay=fulldisplay)
        
def definitional_example(n=2, fulldisplay=False):
    generic_disjunction_example(alpha=5.0, beta=7.0, disjunction_cost=0.01, n=n, fulldisplay=fulldisplay)

def focal_definitional_example(n=2, fulldisplay=False):
    generic_disjunction_example(alpha=5.0, beta=7.0, disjunction_cost=0.01, n=n, fulldisplay=fulldisplay, unknown_word='X')  

######################################################################
#####

def definitional_speaker_alpha_beta_plot(output_filename='fig/definitional-speaker-alpha-beta.pdf', alpha=3.0, dcost=0.01, betas=np.arange(0.0, 11.0, 1.0)):
    KNOWN_MSG = 'A'
    UNKNOWN_MSG = 'X'
    DISJ_MSG = KNOWN_MSG + DISJUNCTION_SIGN + UNKNOWN_MSG
    DEF_STATE = r'w_{1}'
    DEF_LEXICON = 0        
    BASELEXICON = {'A': [r'w_{1}'], 'B': [r'w_{2}'], 'C': [r'w_{3}'], 'D': [r'w_{4}'], 'X': [r'w_{1}', r'w_{2}', r'w_{3}', r'w_{4}']}
    LEXICAL_COSTS = {'A':0.0, 'B':0.0, 'C':0.0, 'D':0.0, 'X':0.0}
    #BASELEXICON={'A': [r'w_{1}'], 'B': [r'w_{2}'], 'X':[r'w_{1}', r'w_{2}']} 
    #LEXICAL_COSTS={'A':0.0, 'B':0.0, 'X':0.0}
    speaker_vals = []
    lexica = Lexica(baselexicon=BASELEXICON, costs=LEXICAL_COSTS, join_closure=True, nullsem=True, nullcost=5.0, disjunction_cost=dcost)
    DEF_LEXICON = [i for i, lex in enumerate(lexica.lexica2matrices()) if lex[lexica.messages.index(KNOWN_MSG)].all() == lex[lexica.messages.index(UNKNOWN_MSG)].all()][0]
    display_matrix(lexica.lexica2matrices()[DEF_LEXICON], rnames=lexica.messages, cnames=lexica.states)
    for beta in betas:        
        ref_prior = np.repeat(1.0/len(lexica.states), len(lexica.states))
        lexprior = np.repeat(1.0/len(lexica.lexica2matrices()), len(lexica.lexica2matrices()))
        #lexprior[-1] = 0.5
        mod = Pragmod(lexica=lexica.lexica2matrices(), messages=lexica.messages, states=lexica.states, costs=lexica.cost_vector(), lexprior=lexprior, prior=ref_prior, alpha=alpha, beta=beta)
        langs = mod.run_expertise_model(n=3, display=False, digits=2)
        # Get the values we need:
        speaker = langs[-2][DEF_LEXICON]
        disj_msg_index = mod.messages.index(DISJ_MSG)
        def_state_index = mod.states.index(DEF_STATE)
        # Fill in listener_val and speaker_val:
        speaker_val = speaker[def_state_index, disj_msg_index]        
        # Determine whether max, with a bit of rounding to avoid spurious mismatch diagnosis:
        maxspkval = np.max(speaker[def_state_index])
        is_max = np.round(speaker_val, 10) == np.round(maxspkval, 10)
        speaker_vals.append([speaker_val, is_max])
    # Figure set-up:
    fig, ax = plt.subplots(nrows=1, ncols=1)
    fig.set_figheight(7)
    fig.set_figwidth(8)
    vals, maxes = zip(*speaker_vals)
    max_booleans = [(betas[i], vals[i]) for i, is_max in enumerate(maxes) if is_max]         
    ax.plot(betas, vals, linewidth=2, color=COLORS[1])
    if max_booleans:
        maxx, maxy = zip(*max_booleans)
        ax.plot(maxx, maxy, linestyle='', markersize=6, marker='o', color=COLORS[1])
    ax.tick_params(axis='both', which='both', bottom='off', left='off', top='off', right='off', labelbottom='on', labelsize=12)
    ax.set_xlabel(r'$\beta$', fontsize=18)
    ax.set_ylabel(r'$S_2(\textit{%s} \mid %s)$' % (DISJ_MSG.replace(' v ', r' or '), DEF_STATE), fontsize=18)        
    ax.set_ylim([0, 1.05])   
    ax.set_xlim([0.0, 10.5])
    # Save the figure:
    plt.savefig(output_filename, bbox_inches='tight')


def definitional_speaker_lexprior_plot(output_filename='fig/definitional-speaker-lexprior.pdf', alpha=2.0, beta=3.0, dcost=0.01):
    KNOWN_MSG = 'A'
    UNKNOWN_MSG = 'X'
    DISJ_MSG = KNOWN_MSG + DISJUNCTION_SIGN + UNKNOWN_MSG
    DEF_STATE = r'w_{1}'
    DEF_LEXICON = 0        
    BASELEXICON = {'A': [r'w_{1}'], 'B': [r'w_{2}'], 'C': [r'w_{3}'], 'D': [r'w_{4}'], 'X': [r'w_{1}', r'w_{2}', r'w_{3}', r'w_{4}']}
    LEXICAL_COSTS = {'A':0.0, 'B':0.0, 'C':0.0, 'D':0.0, 'X':0.0}
    BASELEXICON={'A': [r'w_{1}'], 'B': [r'w_{2}'], 'X':[r'w_{1}', r'w_{2}']} 
    LEXICAL_COSTS={'A':0.0, 'B':0.0, 'X':0.0}
    speaker_vals = []
    lexica = Lexica(baselexicon=BASELEXICON, costs=LEXICAL_COSTS, join_closure=True, nullsem=True, nullcost=5.0, disjunction_cost=dcost)
    DEF_LEXICON = [i for i, lex in enumerate(lexica.lexica2matrices()) if lex[lexica.messages.index(KNOWN_MSG)].all() == lex[lexica.messages.index(UNKNOWN_MSG)].all()][0]
    display_matrix(lexica.lexica2matrices()[DEF_LEXICON], rnames=lexica.messages, cnames=lexica.states)
    unknown_lex_probs = np.arange(0.0, 1.0, 0.1)
    for prob in unknown_lex_probs:
        lexprior = np.repeat(1.0/(1.0-prob), len(lexica.lexica2matrices()))
        lexprior[-1] = prob               
        ref_prior = np.repeat(1.0/len(lexica.states), len(lexica.states))
        mod = Pragmod(lexica=lexica.lexica2matrices(), messages=lexica.messages, states=lexica.states, costs=lexica.cost_vector(), lexprior=lexprior, prior=ref_prior, alpha=alpha, beta=beta)
        langs = mod.run_expertise_model(n=3, display=False, digits=2)
        # Get the values we need:
        speaker = langs[-2][DEF_LEXICON]
        disj_msg_index = mod.messages.index(DISJ_MSG)
        def_state_index = mod.states.index(DEF_STATE)
        # Fill in listener_val and speaker_val:
        speaker_val = speaker[def_state_index, disj_msg_index]        
        # Determine whether max, with a bit of rounding to avoid spurious mismatch diagnosis:
        maxspkval = np.max(speaker[def_state_index])
        is_max = np.round(speaker_val, 10) == np.round(maxspkval, 10)
        speaker_vals.append([speaker_val, is_max])
    # Figure set-up:
    fig, ax = plt.subplots(nrows=1, ncols=1)
    fig.set_figheight(7)
    fig.set_figwidth(8)
    vals, maxes = zip(*speaker_vals)
    ax.plot(unknown_lex_probs, vals, linewidth=2, color=COLORS[1])
    max_booleans = [(unknown_lex_probs[i], vals[i]) for i, is_max in enumerate(maxes) if is_max]             
    if max_booleans:
        maxx, maxy = zip(*max_booleans)
        ax.plot(maxx, maxy, linestyle='', markersize=6, marker='o', color=COLORS[1])
    ax.tick_params(axis='both', which='both', bottom='off', left='off', top='off', right='off', labelbottom='on', labelsize=12)
    ax.set_xlabel(r'Prior expectation for $\mathcal{L}(X) = W$', fontsize=18)
    ax.set_ylabel(r'$S_2(\textit{%s} \mid %s)$' % (DISJ_MSG.replace(' v ', r' or '), DEF_STATE), fontsize=18)        
    ax.set_ylim([0, 1.05])   
    ax.set_xlim([0.0, 1.05])
    # Save the figure:
    plt.savefig(output_filename, bbox_inches='tight')    
    
######################################################################
##### Parameter exploration

class ListenerParameterExperiment:
    def __init__(self,
            baselexicon={'A': ['1'], 'B': ['2'], 'C': ['3'], 'D': ['4'], 'X': ['1', '2', '3', '4']},
            lexical_costs={'A':0.0, 'B':0.0, 'C':0.0, 'D':0.0, 'X':0.0},
            unknown_word=None,
            dcosts=np.arange(0.0, 0.21, 0.01),
            alphas=np.arange(0.0, 15.0, 1),
            betas=np.arange(0.0, 15.0, 1),
            depths=[10],
            results_filename=None,
            results=None,
            plot_filename=None,
            logx=True,
            xlim=(-3.0, 3.0)):
        # Parameter exploration parameters:
        self.results = results                   # Optionally read in existing results for plotting.
        self.results_filename = results_filename # Output filename for the pickled results.
        self.baselexicon = baselexicon           # The basic lexicon to explore.
        self.lexical_costs = lexical_costs       # Costs for the basic lexicon.
        self.unknown_word = unknown_word         # If present, then it is presumed by Lexica to be constrained to an atomic meaning.
        self.dcosts = dcosts                     # Vector of disjunction costs to explore.
        self.alphas = alphas                     # Vector of alpahs to explore.
        self.betas = betas                       # Vector of betas to explore.
        self.depths = depths                     # Vector of integer depths to explore.
        # Plotting parameters:
        self.plot_filename = plot_filename       # Optional output filename for the plot; if None, then plt.show().
        self.logx = logx                         # Should the beta/alpha be put on the log scale (default: True)
        self.xlim = xlim                         # The current default is good for the current settings.
        # Parameters not exposed because assumed
        # fixed; these are here mainly to avoid
        # mistakes and facilitate future work:
        self.msg = 'A v X'
        self.left_disjunct = 'A'
        self.right_disjunct = 'X'
        self.hurford_state = '1 v 2'
        self.definitional_state = '1'
        self.hurford_cls = 'H'
        self.definitional_cls = 'D'
        self.other_cls = 'O'

    def run(self):
        """The main method: parameter exploration if it's not already done, and then the plot"""
        # If the results are not already computed, get them:
        if not self.results:
            self.results = self.explore_listener_parameters()
            pickle.dump(self.results, file(self.results_filename, 'w'), 2)
        # Classify the parameters into Hurfordian and definitional
        hc = self._get_cls_params(self.hurford_cls, self.hurford_state)
        defin = self._get_cls_params(self.definitional_cls, self.definitional_state)
        # Create the plot:
        self.alpha_beta_cost_scatterplot(hc, defin)
    
    def explore_listener_parameters(self):
        """Explore a large parameter space, classifying the parameter vectors
        based on the max listener <world,lex> inference given self.msg""" 
        results = defaultdict(list)
        for dcost, alpha, beta, depth in product(self.dcosts, self.alphas, self.betas, self.depths):
            params = {'alpha': alpha, 'beta': beta, 'depth': depth, 'disjunction_cost': dcost}
            lexica = Lexica(
                baselexicon=self.baselexicon, 
                costs=self.lexical_costs,
                join_closure=True,
                nullsem=True,
                nullcost=5.0,
                disjunction_cost=dcost,
                unknown_word=self.unknown_word)
            lexmats = lexica.lexica2matrices()
            mod = Pragmod(
                lexica=lexmats,
                messages=lexica.messages,
                meanings=lexica.states,
                costs=lexica.cost_vector(),
                prior=np.repeat(1.0/len(lexica.states), len(lexica.states)),
                lexprior=np.repeat(1.0/len(lexmats), len(lexmats)),
                temperature=1.0,
                alpha=alpha,
                beta=beta)
            # Run the model:
            langs = mod.run_expertise_model(n=depth, display=False)
            # Get the listener's joint probability table for this message:
            msg_index = mod.messages.index(self.msg)
            prob_table = langs[-1][msg_index]        
            sorted_probs = sorted(prob_table.flatten())
            max_pair = None
            max_prob = sorted_probs[-1]
            # No ties allowed!
            if max_prob != sorted_probs[-2]:
                for i, j in product(range(prob_table.shape[0]), range(prob_table.shape[1])):
                    if prob_table[i, j] == max_prob:
                        max_pair = (i, mod.meanings[j])
            # Add the target probability:
            params['prob'] = max_prob
            # Print to show progress:
            print max_pair, params
            # Store this dictionary of results -- parameters plus the predicted probability
            # max_pair is a lexicon index and state name.
            results[max_pair].append(params)
        return results
    
    def _classify_lexica(self):
        classified = []
        lexica = Lexica(baselexicon=self.baselexicon, join_closure=True, nullsem=True)
        for i, lex in enumerate(lexica.lexica2matrices()):
            a_index = lexica.messages.index(self.left_disjunct)
            a_sem = set([s for j, s in enumerate(lexica.states) if lex[a_index][j] > 0.0])
            x_index = lexica.messages.index(self.right_disjunct)
            x_sem = set([s for j, s in enumerate(lexica.states) if lex[x_index][j] > 0.0])
            cls = self.other_cls
            if len(a_sem & x_sem) == 0:
                cls = self.hurford_cls            
            elif a_sem == x_sem:
                cls = self.definitional_cls
            classified.append((i, cls))
        return classified                        

    def _get_cls_params(self, cls, state):
        cls_lex = []
        for lexindex, c in self._classify_lexica():
            if c == cls:
                cls_lex += self.results[(lexindex, state)]
        return cls_lex

    def alpha_beta_cost_scatterplot(self, hc, defin):
        """Create the scatterplot with beta/alpha on the x-axis and disjunction costs on the y-axis"""
        # High-level plotting styles:
        plt.style.use('ggplot')
        matplotlib.rc('text', usetex=True)
        matplotlib.rc('font', family='serif', serif='cm10')
        matplotlib.rcParams['text.latex.preamble'] = [r"\usepackage{amsmath}", r'\boldmath']
        # Set-up:
        transform = np.log if self.logx else (lambda x : x)    
        fig, ax = plt.subplots(nrows=1, ncols=1)
        fig.set_figheight(10)
        fig.set_figwidth(16)
        labsize = 24
        markersize = 15
        ticksize = 18
        colors = ['#1B9E77', '#D95F02', '#7570B3']
        # Hurfordian:
        h_ba = [r['beta']/r['alpha'] for r in hc]
        h_g = [r['disjunction_cost'] for r in hc]
        # Definitional:
        d_ba = [r['beta']/r['alpha'] for r in defin]
        d_g = [r['disjunction_cost'] for r in defin]        
        # Both (if any):
        both = [(x,y) for x, y in zip(d_ba, d_g) if (x,y) in zip(h_ba, h_g)]
        b_ba = []; b_g = []
        if both:
            b_ba, b_g = zip(*both)
        # Remove overlap:
        h_ba, h_g = zip(*[(x,y) for x,y in zip(h_ba, h_g) if (x,y) not in both])
        d_ba, d_g = zip(*[(x,y) for x,y in zip(d_ba, d_g) if (x,y) not in both])                        
        # Plotting:
        for i, vals in enumerate(((h_ba, h_g, 'Hurfordian'), (d_ba, d_g, 'Definitional'), (b_ba, b_g, 'Both'))):
            x, y, label = vals
            ax.plot(self.jitter(transform(x)), self.jitter(y), linestyle="", marker=".", label=label, markersize=markersize, color=colors[i])
        # Labeling:
        xlab = r"\beta/\alpha"    
        if self.logx:
            xlab = r"\log(%s)" % xlab
        ax.set_xlabel(r'$%s$' % xlab, fontsize=labsize, color='black')
        ax.set_ylabel(r'$C(or)$', fontsize=labsize, color='black')
        ax.legend(loc='upper left', bbox_to_anchor=(0,1.1), ncol=3, fontsize=labsize)
        # Ticks and bounds:
        if self.xlim:
            x1,x2,y1,y2 = ax.axis()
            x1, x2 = self.xlim
            ax.axis((x1, x2, y1, y2))
        plt.setp(ax.get_xticklabels(), fontsize=ticksize)
        plt.setp(ax.get_yticklabels(), fontsize=ticksize)
        if self.plot_filename:
            plt.savefig(self.plot_filename, bbox_inches='tight')
        else:
            plt.show()

    def jitter(self, x):
        """Jitter while respecting the data bounds"""
        mu = 0.0
        sd = 0.001
        j = x + np.random.normal(mu, sd, len(x))
        j = np.maximum(j, np.min(x))
        j = np.minimum(j, np.max(x))
        return j

######################################################################
        
if __name__ == '__main__':

    ## Some simple warm-up examples:
    # simple_disjunction()
    # basic_scalar()
    # compositional_disjunction()

    ## From the poster:
    #hurfordian_example(n=2, fulldisplay=False)
    #definitional_example(n=3, fulldisplay=False)
    #focal_definitional_example(n=2, fulldisplay=True)
    
    ## Parameter exploration with a large lexicon; this takes a long time to run!
    # ListenerParameterExperiment(
    #     results_filename='paramexplore-lex5.pickle',
    #     # If the results are already computed:
    #     # results=pickle.load(file("paramexplore-lex5.pickle")),
    #     plot_filename='paramexplore-lex5.pdf').run()

    ## Parameter exploration as above but constraining the unknown word X to an atomic meaning:
    # ListenerParameterExperiment(
    #     results_filename='paramexplore-lex5-focal.pickle',
    #     # If the results are already computed:
    #     # results=pickle.load(file("paramexplore-lex5-focal.pickle")),
    #     plot_filename='paramexplore-lex5-focal.pdf',
    #     unknown_word='X').run()

    definitional_speaker_lexprior_plot()
