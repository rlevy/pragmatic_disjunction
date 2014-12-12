import numpy as np
import cPickle as pickle
import matplotlib
import matplotlib.pyplot as plt
from lsa import *

def listener_runs(baselexicon, unknown_word='X'):
    lexical_costs = {let:0.0 for let in baselexicon.keys()}
    results = listener_explore_hyperparameters(baselexicon=baselexicon,
                                               lexical_costs=lexical_costs,
                                               msg='A v X',
                                               temps=[1.0],
                                               unknown_word=unknown_word,
                                               dcosts=np.arange(0.0, 0.21, 0.01),
                                               alphas=np.arange(0.0, 15.0, 1),
                                               betas=np.arange(0.0, 15.0, 1),
                                               depths=[10])
    output_filename = 'paramexplore-defin-lex%s.pickle' % len(baselexicon)
    if unknown_word:
        output_filename = output_filename.replace('defin-', 'defin-X-')
    pickle.dump(results, file(output_filename, 'w'), 2)


def speaker_definitional_hyperparameter_runs(baselexicon):
    lexical_costs = {let:0.0 for let in baselexicon.keys()}

    results = speaker_explore_hyperparameters(baselexicon=baselexicon,
                                          lexical_costs={'A':0.0, 'B':0.0, 'X':0.0},
                                          state='1',
                                          lexicon=0,
                                          temps=[1.0],
                                          dcosts=np.arange(0.0, 0.21, 0.01),
                                          alphas=np.arange(0.0, 15.0, 1),
                                          betas=np.arange(0.0, 15.0, 1),
                                          depths=[10])
    pickle.dump(results, file('paramexplore-defin-speaker-lex%s.pickle' % len(baselexicon), 'w'), 2)


def param_scatterplots(vals):
    """Plot the pairwise relationships between our parameters for some reading (<Lex, state> pair) of interest.
    vals should be a list of parameter value dictionaries, as in the values stored in the dict output of
    speaker_definitional_hyperparameter_runs
    """
    fig, axarray = plt.subplots(nrows=1, ncols=3)
    fig.set_figheight(8)
    fig.set_figwidth(30)
    params = ('alpha', 'beta', 'disjunction_cost')
    ax_index = 0    
    for i in range(len(params)-1):
        p1 = params[i]
        for j in range((i+1), len(params)):
            p2 = params[j]
            if p1 != p2:
                _twoparam_scatterplot(axarray[ax_index], vals, p1, p2) 
                ax_index +=1
            
def _twoparam_scatterplot(ax, vals, p1, p2):
    """Support for param_scatterplots"""
    vals1 = [r[p1] for r in vals]
    vals2 = [r[p2] for r in vals]
    # Add these invisibly to get the plot set up:
    ax.plot(vals1, vals2, linestyle="", marker=".", color='white')
    x1,x2,y1,y2 = ax.axis()
    ax.axis((0.0, x2, 0.0, y2))
    if p1 in ('alpha', 'beta') and p2 in ('alpha', 'beta'):
        amax = max([x2, y2])
        ax.axis((0.0, amax, 0.0, amax))
        diagvals = np.arange(0.0, amax+1.0, 1.0)
        ax.plot(diagvals, diagvals, linestyle='-', color='red')
    # Add these afterwords to ensure that they are not obscured by the diagonal line:
    ax.plot(vals1, vals2, linestyle="", marker=".")
    ax.set_xlabel(p1)
    ax.set_ylabel(p2)

def alpha_beta_gamma_scatterplot(val_lists, xmin=-1.0, xmax=1.0):
    fig, axarray = plt.subplots(nrows=1, ncols=len(val_lists))
    fig.set_figheight(10)
    fig.set_figwidth(len(val_lists)*10)
    lex_index = 3
    for i, df in enumerate(val_lists):
        ax = axarray[i]
        alpha_beta = [np.log(r['beta']/r['alpha']) for r in df]
        disj = [r['disjunction_cost'] for r in df]
        ax.plot(alpha_beta, disj, linestyle="", marker=".")
        x1,x2,y1,y2 = ax.axis()
        ax.axis((xmin, xmax, 0.0, y2))
        ax.set_xlabel(r'$\log(\beta/\alpha)$')
        ax.set_ylabel('disjunction_cost')   
        axarray[i].set_title("Lexicon size = %s" % lex_index)
        lex_index += 1 


def alpha_beta_gamma_scatterplot(param_pairs, logx=True, title=True, output_filename=None):
    transform = np.log if logx else (lambda x : x)    
    fig, axarray = plt.subplots(nrows=len(param_pairs), ncols=1)
    fig.set_figheight(len(param_pairs)*10)
    fig.set_figwidth(16)
    lex_index = 3

    labsize = 24    
    
    for i, params in enumerate(param_pairs):
        hc, defin = params
        try:
            ax = axarray[i]
        except:
            ax = axarray
        # Hurfordian:
        h_ba = [r['beta']/r['alpha'] for r in hc]
        h_g = [r['disjunction_cost'] for r in hc]
        # Definitional:
        d_ba = [r['beta']/r['alpha'] for r in defin]
        d_g = [r['disjunction_cost'] for r in defin]        
        # Both (if any):
        both = [(x,y) for x, y in zip(d_ba, d_g) if (x,y) in zip(h_ba, h_g)]
        b_ba, b_g = zip(*both)
        # Remove overlap:
        h_ba, h_g = zip(*[(x,y) for x,y in zip(h_ba, h_g) if (x,y) not in both])
        d_ba, d_g = zip(*[(x,y) for x,y in zip(d_ba, d_g) if (x,y) not in both])                        
        # Plotting:
        for i, vals in enumerate(((h_ba, h_g, 'Hurfordian'), (d_ba, d_g, 'Definitional'), (b_ba, b_g, 'Both'))):
            x, y, label = vals
            ax.plot(jitter(transform(x)), jitter(y), linestyle="", marker=".", label=label, markersize=15, color=colors[i])
        # In case we want to specify the axis limits:
        x1,x2,y1,y2 = ax.axis()
        ax.axis((x1, x2, y1, y2))
        # Labeling:
        if logx:
            ax.set_xlabel(r'$\log(\beta/\alpha)$', fontsize=labsize)
        else:
            ax.set_xlabel(r'$\beta/\alpha$', fontsize=labsize)
        ax.set_ylabel(r'$C(or)$', fontsize=labsize)
        if title:
            ax.set_title("Lexicon size: %s" % lex_index) ; lex_index += 1 
        #ax.legend(bbox_to_anchor=(1.0, 1.0), fontsize=16)
        ax.legend(loc='upper left', bbox_to_anchor=(0,1.1), ncol=3, fontsize=labsize)
        plt.setp(ax.get_xticklabels(), fontsize=18)
        plt.setp(ax.get_yticklabels(), fontsize=18)
        if output_filename:
            plt.savefig(output_filename, bbox_inches='tight')
        else:
            plt.show()
        
def jitter(x):
    """Jitter while respecting the data bounds"""
    mu = 0.0
    sd = 0.001
    j = x + np.random.normal(mu, sd, len(x))
    j = np.maximum(j, np.min(x))
    j = np.minimum(j, np.max(x))
    return j
    

if __name__ == '__main__':

    pass

    # speaker_definitional_hyperparameter_runs({'A':0.0, 'B':0.0, 'X':0.0})
    # speaker_definitional_hyperparameter_runs({'A': ['1'], 'B': ['2'], 'C': ['3'], 'X': ['1', '2', '3']})
    # speaker_definitional_hyperparameter_runs({'A': ['1'], 'B': ['2'], 'C': ['3'], 'D': ['4'], 'X': ['1', '2', '3', '4']})

    #listener_runs({'A': ['1'], 'B':['2'], 'X': ['1', '2']}, unknown_word='X')
    #listener_runs({'A': ['1'], 'B': ['2'], 'C': ['3'], 'X': ['1', '2', '3']},  unknown_word='X')
    #listener_runs({'A': ['1'], 'B': ['2'], 'C': ['3'], 'D': ['4'], 'X': ['1', '2', '3', '4']},  unknown_word='X')
