import os
import numpy as np
from pragmods import Pragmod, display_matrix
from lexica import Lexica, DISJUNCTION_SIGN
import copy
import csv
from itertools import product
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.font_manager as font_manager
import matplotlib.patches as mpatches

######################################################################
##### PLOT PARAMETERS

title_size = 18
axis_label_size = 18
setup = {'family': 'sans-serif', 'weight':'normal', 'size':18}
matplotlib.rc('font', **setup) 
matplotlib.rc('xtick', labelsize=14)
matplotlib.rc('ytick', labelsize=14)
matplotlib.rcParams.update({'font.size': 12})

# The first set of colors is good for colorbind people:
colors = ['#1B9E77', '#D95F02', '#7570B3', '#E7298A', '#66A61E', '#E6AB02', '#A6761D', '#666666'] + matplotlib.colors.cnames.values()

fig_width = 18
fig_height = 12

######################################################################

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
                 lexprior=None,
                 null_cost=5.0,
                 unknown_word=None):
        self.n = n
        self.disjunction_cost = disjunction_cost
        self.lexical_costs = lexical_costs
        self.temperature = temperature
        self.alpha = alpha
        self.beta = beta        
        self.baselexicon = baselexicon
        self.prior = prior
        self.lexprior = lexprior
        self.null_cost = null_cost
        self.unknown_word = unknown_word

    def build(self):
        lex = Lexica(baselexicon=self.baselexicon,
                     join_closure=True,
                     disjunction_cost=self.disjunction_cost,
                     nullsem=True,
                     null_cost=self.null_cost,
                     costs=copy.copy(self.lexical_costs),
                     unknown_word=self.unknown_word)
        
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

    ######################################################################
    ##### LISTENER PERSPECTIVE

    def listener_inference(self, msg='A v X'):
        final_lis_lang = self.langs[-1]
        msg_index = self.messages.index(msg)
        prob_table = []
        for lex_index in range(len(self.lexica)):
            row = np.array([final_lis_lang[lex_index][msg_index][j] for j in range(len(self.states))])
            prob_table.append(row)
        return np.array(prob_table)

    def plot_listener_inference(self, msg='A v X', width=0.2, initial_color_index=0):
        fig, ax = plt.subplots(1,1)
        fig.set_figheight(len(self.states)*len(self.messages)*0.75)
        fig.set_figwidth(14)
        innerlabels = self.states
        outerlabels = ["Lex%s" % (i+1) for i in range(len(self.lexica))]
        mat = self.listener_inference(msg=msg)
        m, n = mat.shape
        barsetwidth = width*n
        ind = np.arange(0.0, (barsetwidth+width)*m, barsetwidth+width)
        ind = ind[::-1]        
        for j in range(n-1, -1, -1):
            xpos = ind+(width*j)
            vals = mat[:, j]        
            ax.barh(xpos, vals, width, color=colors[initial_color_index+j], label=innerlabels[j])
            for i in range(m):
                ax.text(0.01, xpos[i]+(width/2.0), innerlabels[j], rotation='horizontal', ha='left', va='center', fontsize=18)
        ax.set_yticks(ind+barsetwidth/2.0)
        ax.set_yticklabels(outerlabels, fontsize=24)
        ax.set_ylim([min(xpos), max(ind+barsetwidth+width)])             
        
    def show_max_lex_state_values(self, joint_prob_table, precision=10):
        max_prob = np.round(np.max(joint_prob_table), precision)
        for i, j in product(range(len(self.lexica)), range(len(self.states))):
            if np.round(joint_prob_table[i, j], precision) == max_prob:
                print "<Lex%s, %s>: %s" % (i, self.states[j], joint_prob_table[i, j])

    def display_listener_inference(self, msg='A v X', digits=3):
        colwidth = max([len(x) for x in self.messages + self.states] + [digits]) + 4
        print "--------------------------------------------------"
        print self.params2str()
        prob_table = self.listener_inference(msg=msg)        
        headervals = [""] + self.states + ['sum(lex)']
        print self.rowformatter(headervals, colwidth=colwidth)
        for lex_index in range(len(self.lexica)):
            rowvals = ["Lex%s" % lex_index] + [round(x, digits) for x in prob_table[lex_index]] + [np.round(np.sum(prob_table[lex_index]), digits)]
            print self.rowformatter(rowvals, colwidth=colwidth)
        print self.rowformatter(['sum(state)'] + list(np.round(np.sum(prob_table, axis=0), digits)), colwidth=colwidth)

    def plot_listener_inference_depth_values(self, msg='A v X', target_state='1 v 2', n_values=np.arange(1, 5, 1), legend_loc='upper right', output_filename=None, progress_report=True):
        self.plot_listener_inference_parameter_space(msg=msg, target_state=target_state, parameter_name='n', parameter_text='Depth', parameter_values=n_values, legend_loc=legend_loc, output_filename=output_filename, progress_report=progress_report)

    def plot_listener_inference_beta_values(self, msg='A v X', target_state='1 v 2', beta_values=np.arange(0.01, 5.0, 0.01), legend_loc='upper right', output_filename=None, progress_report=True): 
        self.plot_listener_inference_parameter_space(msg=msg, target_state=target_state, parameter_name='beta', parameter_text=r"$\beta$", parameter_values=beta_values, legend_loc=legend_loc, output_filename=output_filename, progress_report=progress_report)

    def plot_listener_inference_alpha_values(self, msg='A v X', target_state='1 v 2', alpha_values=np.arange(0.01, 5.0, 0.01), legend_loc='upper right', output_filename=None, progress_report=True):
        self.plot_listener_inference_parameter_space(msg=msg, target_state=target_state, parameter_name='alpha', parameter_text=r"$\alpha$", parameter_values=alpha_values, legend_loc=legend_loc, output_filename=output_filename, progress_report=progress_report)

    def plot_listener_inference_disjunction_costs(self, msg='A v X', target_state='1 v 2', disjunction_cost_values=np.arange(0.0, 5.0, 0.01), legend_loc='upper right', output_filename=None, progress_report=True): 
        self.plot_listener_inference_parameter_space(msg=msg, target_state=target_state, parameter_name='disjunction_cost', parameter_values=disjunction_cost_values, legend_loc=legend_loc, output_filename=output_filename, progress_report=progress_report)

    def plot_listener_inference_lambda_values(self, msg='A v X', target_state='1 v 2', lambda_values=np.arange(0.01, 5.0, 0.01), legend_loc='upper right', output_filename=None, progress_report=True): 
        self.plot_listener_inference_parameter_space(msg=msg, target_state=target_state, parameter_name='temperature', parameter_text=r"$\lambda$", parameter_values=lambda_values, legend_loc=legend_loc, output_filename=output_filename, progress_report=progress_report)
       
    def plot_listener_inference_parameter_space(self, msg='A v X', target_state='1 v 2', parameter_name='disjunction_cost',  parameter_text=None, parameter_values=np.arange(0.0, 5.0, 0.01), legend_loc='upper right',  output_filename=None, progress_report=True):    
        probs = defaultdict(list)
        # Store the original to respect the problem:
        original = getattr(self, parameter_name)
        # Calculate and organize the values:
        for paramval in parameter_values:           
            setattr(self, parameter_name, paramval)
            self.build()
            target_state_index = self.states.index(target_state)
            if progress_report:
                self.display_listener_inference(msg=msg)             
            prob_table = self.listener_inference(msg=msg)
            for lex_index in range(len(self.lexica)):
                prob = prob_table[lex_index][target_state_index]
                sorted_probs = sorted(prob_table.flatten())
                maxval = False
                # Add a True max flag iff this prob is the max and the max is unique:
                if sorted_probs[-1] == prob and sorted_probs[-1] != sorted_probs[-2]:
                    maxval = True
                probs[lex_index].append((paramval, prob, maxval))
        # Restore the original:
        setattr(self, parameter_name, original)
        # Plot:               
        if parameter_text == None: parameter_text = parameter_names
        fig = plt.figure(figsize=(fig_width, fig_height))     
        for lex_index in range(len(self.lexica)):
            paramvals, vals, maxval_markers = zip(*probs[lex_index])
            lex_rep = self.lex2str(self.lexica[lex_index])
            plt.plot(paramvals, vals, marker="", linestyle="-", label=lex_rep, color=colors[lex_index], markersize=0, linewidth=3)
        for lex_index in range(len(self.lexica)):
            paramvals, vals, maxval_markers = zip(*probs[lex_index]) 
            # Dots mark max-values in the joint table --- best inferences for the listener:
            dots = [(paramval, val) for paramval, val, max_marker in probs[lex_index] if max_marker]
            if dots:           
                dotsx, dotsy = zip(*dots)
                plt.plot(dotsx, dotsy, marker="o", linestyle="", color=colors[lex_index], markersize=8)    
        plt.title("Listener hears '%s'\n\n%s" % (msg, self.params2str(exclude=[parameter_name])), fontsize=title_size)
        plt.xlabel(parameter_text, fontsize=axis_label_size)
        plt.ylabel(r"Listener probability for $\langle$Lex, %s$\rangle$" % target_state, fontsize=axis_label_size)
        plt.legend(loc=legend_loc)
        plt.text(0.01, 0.95, 'dots mark max values', fontsize=14)
        x1,x2,y1,y2 = plt.axis()
        plt.axis((x1, x2, 0.0, 1.0))
        if output_filename:
            plt.savefig(output_filename)
        else:
            plt.show()
                
    ######################################################################
    ##### SPEAKER PERSPECTIVE
            
    def speaker_behavior(self, state="1"):
        final_spk_index = self.langs[-2]
        state_index = self.states.index(state)
        prob_table = []
        for msg_index in range(len(self.messages)):
            row = np.array([final_spk_index[msg_index][state_index][j] for j in range(len(self.lexica))])
            prob_table.append(row)
        return np.array(prob_table)

    def show_max_message_values(self, spk_prob_table, precision=10):
        for j in range(len(self.lexica)):
            probs = np.round(spk_prob_table[ : , j], precision)
            maxprob = np.max(probs)
            indices = [i for i, val in enumerate(probs) if val == maxprob]
            msgs = [self.messages[i] for i in indices]
            print "Lex%s %s: {%s} \t prob = %s" % (j, self.lex2str(self.lexica[j]), ", ".join(msgs), maxprob)
    
    def display_speaker_behavior(self, state='1', digits=3):
        colwidth = max([len(x) for x in self.messages] + [digits]) + 4
        print "--------------------------------------------------"
        print self.params2str()
        prob_table = self.speaker_behavior(state=state)
        headervals = [""] + ["Lex%s" % i for i in range(len(self.lexica))]
        print self.rowformatter(headervals, colwidth=colwidth)
        for msg_index, msg in enumerate(self.messages):
            rowvals = [msg] + [round(x, digits) for x in prob_table[msg_index]]
            print self.rowformatter(rowvals, colwidth=colwidth)
            
    def plot_speaker_behavior_disjunction_costs(self, state='1 v 2', lexicon=0, disjunction_cost_values=np.arange(0.0, 5.0, 0.05), legend_loc='upper right', output_filename=None, progress_report=True): 
        self.plot_speaker_behavior_parameter_space(state=state, lexicon=lexicon, parameter_name='disjunction_cost', parameter_values=disjunction_cost_values, legend_loc=legend_loc, output_filename=output_filename, progress_report=progress_report)

    def plot_speaker_behavior_beta_values(self, state='1 v 2', lexicon=0, beta_values=np.arange(0.01, 5.0, 0.05), legend_loc='upper right', output_filename=None, progress_report=True): 
        self.plot_speaker_behavior_parameter_space(state=state, lexicon=lexicon, parameter_name='beta', parameter_text=r"$\beta$", parameter_values=beta_values, legend_loc=legend_loc, output_filename=output_filename, progress_report=progress_report)

    def plot_speaker_behavior_alpha_values(self,  state='1 v 2', lexicon=0, alpha_values=np.arange(0.01, 5.0, 0.05), legend_loc='upper right', output_filename=None, progress_report=True):
        self.plot_speaker_behavior_parameter_space(state=state, lexicon=lexicon, parameter_name='alpha', parameter_text=r"$\alpha$", parameter_values=alpha_values, legend_loc=legend_loc, output_filename=output_filename, progress_report=progress_report)
        
    def plot_speaker_behavior_parameter_space(self, state='1 v 2', lexicon=0, parameter_name='disjunction_cost', parameter_text=None, parameter_values=np.arange(0.0, 5.0, 0.05), legend_loc='upper right',  output_filename=None, progress_report=True):    
        probs = defaultdict(list)
        # Store the original to respect the problem:
        original = getattr(self, parameter_name)
        # Calculate and organize the values:
        for paramval in parameter_values:           
            setattr(self, parameter_name, paramval)
            self.build()
            if progress_report:
                self.display_speaker_behavior(state=state)             
            prob_table = self.speaker_behavior(state=state)
            for msg_index, msg in enumerate(self.messages):
                prob = prob_table[msg_index, lexicon]
                probs[msg_index].append((paramval, prob))
        # Restore the original parameter setting:
        setattr(self, parameter_name, original)
        # Plotting:              
        if parameter_text == None: parameter_text = parameter_name
        fig = plt.figure(figsize=(fig_width, fig_height))   
        for msg_index, msg in enumerate(self.messages):
            paramvals, vals = zip(*probs[msg_index])
            plt.plot(paramvals, vals, marker="", linestyle="-", label=msg, color=colors[msg_index], markersize=0, linewidth=3)
        # Annotations:
        lex_rep = self.lex2str(self.lexica[lexicon])
        plt.title("Speaker observes <%s, Lexicon: %s> \n\n%s" % (state, lex_rep, self.params2str(exclude=[parameter_name])), fontsize=title_size)
        plt.xlabel(parameter_text, fontsize=axis_label_size)
        plt.ylabel(r"Speaker probability for producing message", fontsize=axis_label_size)
        plt.legend(loc=legend_loc)
        x1,x2,y1,y2 = plt.axis()
        plt.axis((x1, x2, -0.01, 1.01))
        if output_filename:
            plt.savefig(output_filename)
        else:
            plt.show()
        

    ######################################################################
    ##### PRINTING

    def rowformatter(self, row, colwidth=12):        
        return "".join([str(x).rjust(colwidth) for x in row])

    def params2str(self, joiner='; ', exclude=[]):
        vals = []
        params = {'n': r'$n$',
                  'temperature': r'$\lambda$',
                  'disjunction_cost': r'cost($\vee$)',
                  'lexical_costs': r'costs',
                  'null_cost': r'cost$(\emptyset)$',
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
            sem = [s for i, s in enumerate(self.states) if lexicon[p_index][i] > 0.0 and not DISJUNCTION_SIGN in s]
            entry = p + "={" + ",".join(state_sorter(sem)) + "}"
            entries.append(entry)
        return "; ".join(entries)


######################################################################
            
def listener_explore_hyperparameters(baselexicon={'A': ['1'], 'B': ['2'], 'X':['1', '2']},
                                     lexical_costs={'A':0.0, 'B':0.0, 'X':0.0},
                                     msg='A v X',
                                     temps=[1.0],
                                     dcosts=np.arange(0.0, 0.21, 0.01),
                                     alphas=np.arange(0.0, 15.0, 1),
                                     betas=np.arange(0.0, 15.0, 1),
                                     depths=[10]):
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
        params['prob'] = max_prob
        print max_pair, params
        results[max_pair].append(params)
    return results

def speaker_explore_hyperparameters(baselexicon={'A': ['1'], 'B': ['2'], 'X':['1', '2']},
                                    lexical_costs={'A':0.0, 'B':0.0, 'X':0.0},
                                    state='1',
                                    lexicon=0,
                                    temps=[1.0],
                                    dcosts=np.arange(0.0, 0.21, 0.01),
                                    alphas=np.arange(0.0, 15.0, 1),
                                    betas=np.arange(0.0, 15.0, 1),
                                    depths=[10]):
    results = defaultdict(list)
    for temp, dcost, alpha, beta, depth in product(temps, dcosts, alphas, betas, depths):
        params = {'lambda': temp, 'alpha': alpha, 'beta': beta, 'depth': depth, 'disjunction_cost': dcost}
        experiment = Experiment(baselexicon=baselexicon, lexical_costs=lexical_costs, n=depth, temperature=temp, alpha=alpha, beta=beta, disjunction_cost=dcost)
        experiment.build()
        prob_table = experiment.speaker_behavior(state=state)
        probs = prob_table[ : , lexicon]
        sorted_probs = sorted(probs)
        max_msg = None
        max_prob = sorted_probs[-1]
        # No ties allowed!
        if max_prob != sorted_probs[-2]:
            for i in range(len(probs)):
                if probs[i] == max_prob:
                    max_msg = experiment.messages[i]
        params['prob'] = max_prob
        print max_msg, params
        results[max_msg].append(params)
    return results

######################################################################

if __name__ == '__main__':
   
    hurford = Experiment(n=2, temperature=1.0, disjunction_cost=0.01, beta=1.0, alpha=1.0, null_cost=5.0)
    hurford.build()
    hurford.display_listener_inference(msg='A v X')
    hurford.plot_listener_inference()
    
    print "======================================================================"    

    hurford = Experiment(n=2, temperature=1.0, disjunction_cost=1.0, beta=1.0, alpha=1.0, null_cost=5.0,
                         baselexicon={'A': ['1'], 'B': ['2'], 'C': ['3'], 'D': ['4'], 'X': ['1', '2', '3', '4']},
                         lexical_costs={'A':0.0, 'B':0.0, 'C':0.0, 'D':0.0, 'X':0.0})
    hurford.build()
    print hurford.params2str()
    prob_table = hurford.listener_inference(msg='A v X')
    hurford.show_max_lex_state_values(prob_table)

    print "======================================================================"

    hurford = Experiment(n=2, temperature=2.0, disjunction_cost=0.01, beta=1.0, alpha=1.0, null_cost=5.0)
    hurford.build()
    hurford.display_listener_inference(msg='A v X')
    
    print "======================================================================"

    hurford = Experiment(n=2, temperature=2.4, disjunction_cost=0.00, beta=1.9, alpha=0.6, null_cost=5.0,
                         baselexicon={'A': ['1'], 'B': ['2'], 'C': ['3'], 'D': ['4'], 'X': ['1', '2', '3', '4']},
                         lexical_costs={'A':0.0, 'B':0.0, 'C':0.0, 'D':0.0, 'X':0.0})
    hurford.build()
    print hurford.params2str()
    prob_table = hurford.listener_inference(msg='A v X')
    hurford.show_max_lex_state_values(prob_table)

    print "======================================================================"

    hurford = Experiment(n=10, temperature=2.0, disjunction_cost=0.1, beta=1.0, alpha=1.0, null_cost=5.0,
                         baselexicon={'A': ['1'], 'B': ['2'], 'C': ['3'], 'D': ['4'], 'X': ['1', '2', '3', '4']},
                         lexical_costs={'A':0.0, 'B':0.0, 'C':0.0, 'D':0.0, 'X':0.0},
                         unknown_word='X')
    hurford.build()
    langs = hurford.langs
    msg_index = hurford.messages.index('A v X')
    lis_index = 2
    header = [""] + ["Lex%s" % i for i in range(len(hurford.lexica))]
    print "".join([str(x).rjust(8) for x in header])
    for i in range(2, len(langs), 2):
        lang = langs[i]
        row = list(np.round(np.sum(lang, axis=2)[:, msg_index], 3))
        row = ["L%s" % lis_index] + row
        print "".join([str(x).rjust(8) for x in row])
        lis_index += 1  
        
