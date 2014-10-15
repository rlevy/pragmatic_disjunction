import os
import numpy as np
from pragmods import Pragmod, display_matrix
from lexica import Lexica
import copy
import csv
from itertools import product
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.font_manager as font_manager


class Experiment:
    def __init__(self,
                 n=3,
                 disjunction_cost=0.0,             
                 lexical_costs={'A':0.0, 'B':0.0, 'X':0.0},
                 temperature=1.0,
                 alpha=1.0,
                 beta=1.0,
                 baselexicon={'A': ['1'], 'B': ['2'], 'X':['1', '2']},
                 prior=None,
                 lexprior = None):
        self.n = n
        self.disjunction_cost = disjunction_cost
        self.lexical_costs = lexical_costs
        self.temperature = temperature
        self.alpha = alpha
        self.beta = beta        
        self.baselexicon = baselexicon
        self.prior = prior
        self.lexprior = lexprior

    def build(self):
        lex = Lexica(baselexicon=self.baselexicon,
                     join_closure=True,
                     disjunction_cost=self.disjunction_cost,
                     nullsem=True,
                     costs=copy.copy(self.lexical_costs))
        
        self.lexica = lex.lexica2matrices()
        self.states = lex.states
        self.messages = lex.messages

        if self.prior == None:
            self.prior = np.repeat(1.0/len(self.states), len(self.states))

        if self.lexprior == None:
            self.lexprior = np.repeat(1.0/len(self.lexica), len(self.lexica))
            
        self.model = Pragmod(lexica=self.lexica,
                             messages=self.messages,
                             meanings=self.states,
                             costs=lex.cost_vector(),
                             prior=self.prior,
                             lexprior=self.lexprior,
                             temperature=self.temperature,
                             alpha=self.alpha,
                             beta=self.beta)

        self.langs = self.model.run_expertise_model(n=self.n, display=False)

    def listener_inference(self, msg='A v X'):
        final_lis_lang = self.langs[-1]
        msg_index = self.messages.index(msg)
        prob_table = []
        for lex_index in range(len(self.lexica)):
            row = np.array([final_lis_lang[lex_index][msg_index][j] for j in range(len(self.states))])
            prob_table.append(row)
        return np.array(prob_table)

    def display_listener_inference(self, msg='A v X'):
        print "--------------------------------------------------"
        print self.params2str()
        prob_table = self.listener_inference(msg=msg)        
        print "".join([x.rjust(10) for x in [""] + self.states])
        for lex_index in range(len(self.lexica)):
            label = "Lex%s" % lex_index
            print label.rjust(10), "".join([str(round(x, 3)).rjust(10) for x in prob_table[lex_index]])

    def params2str(self, joiner='; ', exclude=[]):
        vals = []
        params = {'n': r'$n$',
                  'temperature': r'$\lambda$',
                  'disjunction_cost': r'cost($\vee$)',
                  'lexical_costs': r'costs',
                  'alpha': r'$\alpha$',
                  'beta': r'$\beta$'}
        for x in sorted(params):
            if x not in exclude:
                if x == 'lexical_costs':
                    for p, c in sorted(self.lexical_costs.items()):
                        vals.append('cost(%s): %s' % (p, c))
                else:
                    vals.append("%s: %s" % (params[x], getattr(self, x)))
        return joiner.join(vals)

    def lex2str(self, lexicon_or_lexicon_index):
        lexicon = lexicon_or_lexicon_index
        if isinstance(lexicon, int):
            lexicon = self.lexica[lexicon_or_lexicon_index]            
        def state_sorter(x):
            return sorted(x, cmp=(lambda x, y: cmp(len(x), len(y))))
        entries = []
        for p_index, p in enumerate(sorted(self.baselexicon.keys())):
            sem = [s for i, s in enumerate(self.states) if lexicon[p_index][i] > 0.0]            
            entry = p + "={" + ",".join(state_sorter(sem)) + "}"
            entries.append(entry)
        return "; ".join(entries)

    def plot_listener_inference_depth_values(self, msg='A v X', target_state='1 v 2', n_values=np.arange(1, 5, 1), legend_loc='upper right', output_filename=None):
        self.plot_listener_inference_parameter_space(msg=msg, target_state=target_state, parameter_name='n', parameter_text='Depth', parameter_values=n_values, legend_loc=legend_loc, output_filename=output_filename)

    def plot_listener_inference_beta_values(self, msg='A v X', target_state='1 v 2', beta_values=np.arange(0.01, 5.0, 0.01), legend_loc='upper right', output_filename=None): 
        self.plot_listener_inference_parameter_space(msg=msg, target_state=target_state, parameter_name='beta', parameter_text=r"$\beta$", parameter_values=beta_values, legend_loc=legend_loc, output_filename=output_filename)

    def plot_listener_inference_alpha_values(self, msg='A v X', target_state='1 v 2', alpha_values=np.arange(0.01, 5.0, 0.01), legend_loc='upper right', output_filename=None):
        self.plot_listener_inference_parameter_space(msg=msg, target_state=target_state, parameter_name='alpha', parameter_text=r"$\alpha$", parameter_values=alpha_values, legend_loc=legend_loc, output_filename=output_filename)

    def plot_listener_inference_disjunction_costs(self, msg='A v X', target_state='1 v 2', disjunction_cost_values=np.arange(0.0, 5.0, 0.01), legend_loc='upper right', output_filename=None): 
        self.plot_listener_inference_parameter_space(msg=msg, target_state=target_state, parameter_name='disjunction_cost', parameter_values=disjunction_cost_values, legend_loc=legend_loc, output_filename=output_filename)

    def plot_listener_inference_lambda_values(self, msg='A v X', target_state='1 v 2', lambda_values=np.arange(0.01, 5.0, 0.01), legend_loc='upper right', output_filename=None): 
        self.plot_listener_inference_parameter_space(msg=msg, target_state=target_state, parameter_name='temperature', parameter_text=r"$\lambda$", parameter_values=lambda_values, legend_loc=legend_loc, output_filename=output_filename)
       

    def plot_listener_inference_parameter_space(self,
                                                msg='A v X',
                                                target_state='1 v 2',
                                                parameter_name='disjunction_cost',
                                                parameter_text=None,
                                                parameter_values=np.arange(0.0, 5.0, 0.01),
                                                legend_loc='upper right',
                                                output_filename=None):
        if parameter_text == None: parameter_text = parameter_name        
        setup = {'family': 'sans-serif', 'sans-serif':['Helvetica'], 'weight':'normal', 'size':18}
        title_size = 18
        axis_label_size = 18
        matplotlib.rc('font', **setup) 
        matplotlib.rc('xtick', labelsize=14)
        matplotlib.rc('ytick', labelsize=14)
        matplotlib.rcParams.update({'font.size': 12})
        fig = plt.figure(figsize=(13, 9))
        # Computation
        param_val_pairs = []
        original = getattr(self, parameter_name)        
        probs = defaultdict(list)
        for paramval in parameter_values:
            setattr(self, parameter_name, paramval)
            self.build()
            target_state_index = self.states.index(target_state)
            self.display_listener_inference(msg=msg)             
            prob_table = self.listener_inference(msg=msg)
            for lex_index in range(len(self.lexica)):
                prob = prob_table[lex_index][target_state_index]
                maxval = False
                if np.max(prob_table) == prob:
                    maxval = True
                probs[lex_index].append((paramval, prob, maxval))        
        # The first set of colors is good for colorbind people:
        colors = ['#1B9E77', '#D95F02', '#7570B3', '#E7298A', '#66A61E', '#E6AB02', '#A6761D', '#666666'] + matplotlib.colors.cnames.values()
        for lex_index in range(len(self.lexica)):
            paramvals, vals, maxval_markers = zip(*probs[lex_index])
            lex_rep = self.lex2str(self.lexica[lex_index])
            plt.plot(paramvals, vals, marker="", linestyle="-", label=lex_rep, color=colors[lex_index], linewidth=3)
            # Dots mark max-values in the joint table --- best inferences for the listener:
            dots = [(paramval, val) for paramval, val, marker in probs[lex_index] if marker]
            if dots:           
                dotsx, dotsy = zip(*dots)
                plt.plot(dotsx, dotsy, marker="o", linestyle="", color=colors[lex_index],  linewidth=3)    
        plt.title("Listener hears '%s'\n\n%s" % (msg, self.params2str(exclude=[parameter_name])), fontsize=title_size)
        plt.xlabel(parameter_text, fontsize=axis_label_size)
        plt.ylabel(r"Listener probability for $\langle$Lex, %s$\rangle$" % target_state, fontsize=axis_label_size)
        plt.legend(loc=legend_loc)
        x1,x2,y1,y2 = plt.axis()
        plt.axis((x1, x2, 0.0, 1.0))
        setattr(self, parameter_name, original)
        if output_filename:
            plt.savefig(output_filename)
        else:
            plt.show()

def explore_hyperparameters(baselexicon={'A': ['1'], 'B': ['2'], 'X':['1', '2']},
                            lexical_costs={'A':0.0, 'B':0.0, 'X':0.0},
                            msg='A v X'):    
    temps = np.arange(0.0, 3.2, 0.2)
    dcosts = np.arange(0.0, 3.2, 0.2)
    alphas = np.arange(0.0, 3.2, 0.2)
    betas = np.arange(0.0, 3.2, 0.2)
    depths = [3]
    results = defaultdict(list)
    for temp, dcost, alpha, beta, depth in product(temps, dcosts, alphas, betas, depths):
        params = {'lambda': temp, 'alpha': alpha, 'beta': beta, 'depth': depth, 'disjunction_cost': dcost}
        experiment = Experiment(baselexicon=baselexicon, lexical_costs=lexical_costs, n=depth, temperature=temp, alpha=alpha, beta=beta, disjunction_cost=dcost)
        experiment.build()
        prob_table = experiment.listener_inference(msg=msg)
        sorted_probs = sorted(prob_table.flatten())
        max_pair = None
        max_prob = sorted_probs[-1]
        # No ties allowed!
        if max_prob != sorted_probs[-2]:
            for i, j in product(range(prob_table.shape[0]), range(prob_table.shape[1])):
                if prob_table[i, j] == max_prob:
                    max_pair = (i, experiment.states[j])
        #if max_pair == (0, '1'):
        print max_pair, params
        results[max_pair].append(params)
    return results


if __name__ == '__main__':
    
    hurford = Experiment(temperature=1.0, disjunction_cost=0.01, beta=1.0, alpha=1.0)
    hurford.build()
    hurford.display_listener_inference(msg='A v X')
       
