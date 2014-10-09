import numpy as np
from pragmods import Pragmod, display_matrix
from lexica import Lexica
import copy
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib

class HurfordExperiment:
    def __init__(self,
                 n=3,
                 disjunction_cost=0.0,             
                 lexical_costs={'A':0.0, 'B':0.0, 'C':0.0},
                 temperature=1.0,
                 alpha=1.0,
                 beta=1.0,
                 baselexicon={'A': ['1'], 'B': ['2'], 'C':['1', '2']}):
        self.n = n
        self.disjunction_cost = disjunction_cost
        self.lexical_costs = lexical_costs
        self.temperature = temperature
        self.alpha = alpha
        self.beta = beta        
        self.baselexicon = baselexicon

    def build(self):
        lex = Lexica(baselexicon=self.baselexicon,
                     join_closure=True,
                     disjunction_cost=self.disjunction_cost,
                     nullsem=True,
                     costs=copy.copy(self.lexical_costs))
        
        self.lexica = lex.lexica2matrices()
        self.states = lex.states
        self.messages = lex.messages

        self.model = Pragmod(lexica=self.lexica,
                             messages=self.messages,
                             meanings=self.states,
                             costs=lex.cost_vector(),
                             prior=np.repeat(1.0/len(self.states), len(self.states)),
                             lexprior=np.repeat(1.0/len(lex), len(lex)),
                             temperature=self.temperature,
                             alpha=self.alpha,
                             beta=self.beta)

        self.langs = self.model.run_expertise_model(n=self.n, display=False)

    def listener_inference(self, msg='A v C'):
        final_lis_lang = self.langs[-1]
        msg_index = self.messages.index(msg)
        prob_table = []
        for lex_index in range(len(self.lexica)):
            row = np.array([final_lis_lang[lex_index][msg_index][j] for j in range(len(self.states))])
            prob_table.append(row)
        return np.array(prob_table)

    def display_listener_inference(self, msg='A v C'):
        print "--------------------------------------------------"
        print self.params2str()
        prob_table = self.listener_inference(msg=msg)        
        print "".join([x.rjust(10) for x in [""] + self.states])
        for lex_index in range(len(self.lexica)):
            label = "Lex%s" % lex_index
            print label.rjust(10), "".join([str(round(x, 3)).rjust(10) for x in prob_table[lex_index]])

    def params2str(self, joiner='; ', exclude=[]):
        vals = []
        for x in ('n', 'temperature', 'disjunction_cost', 'lexical_costs', 'alpha', 'beta'):
            if x not in exclude:
                vals.append("%s: %s" % (x, getattr(self, x)))
        return joiner.join(vals)

    def lex2str(self, lexicon):
        entries = []
        for p_index, p in enumerate(sorted(self.baselexicon.keys())):
            sem = [s for i, s in enumerate(self.states) if lexicon[p_index][i] > 0.0]
            entry = p + "={" + ",".join(sorted(sem)) + "}"
            entries.append(entry)
        return "; ".join(entries)

    def plot_listener_inference_beta_values(self,
                                            msg='A v C',
                                            target_state='1 v 2',                                                 
                                            beta_values=np.arange(0.01, 5.0, 0.01),
                                            legend_loc='upper right'):
        self.plot_listener_inference_parameter_space(msg=msg,
                                                     target_state=target_state,
                                                     parameter_name='beta',
                                                     parameter_values=beta_values,
                                                     legend_loc=legend_loc)

    def plot_listener_inference_alpha_values(self,
                                            msg='A v C',
                                            target_state='1 v 2',                                                 
                                            alpha_values=np.arange(0.01, 5.0, 0.01),
                                            legend_loc='upper right'):
        self.plot_listener_inference_parameter_space(msg=msg,
                                                     target_state=target_state,
                                                     parameter_name='alpha',
                                                     parameter_values=alpha_values,
                                                     legend_loc=legend_loc)

    def plot_listener_inference_disjunction_costs(self,
                                                  msg='A v C',
                                                  target_state='1 v 2',                                                 
                                                  disjunction_cost_values=np.arange(0.0, 5.0, 0.01),
                                                  legend_loc='upper right'):
        self.plot_listener_inference_parameter_space(msg=msg,
                                                     target_state=target_state,
                                                     parameter_name='disjunction_cost',
                                                     parameter_values=disjunction_cost_values,
                                                     legend_loc=legend_loc)

    def plot_listener_inference_lambda_values(self,
                                              msg='A v C',
                                              target_state='1 v 2',
                                              lambda_values=np.arange(0.01, 5.0, 0.01),
                                              legend_loc='upper right'):
        self.plot_listener_inference_parameter_space(msg=msg,
                                                     target_state=target_state,
                                                     parameter_name='temperature',
                                                     parameter_values=lambda_values,
                                                     legend_loc=legend_loc)
       

    def plot_listener_inference_parameter_space(self,
                                                msg='A v C',
                                                target_state='1 v 2',
                                                parameter_name='disjunction_cost',
                                                parameter_values=np.arange(0.0, 5.0, 0.01),
                                                legend_loc='upper right'):
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
        fig = plt.figure(figsize=(11, 7))
        # The first set of colors is good for colorbind people:
        colors = ['#1B9E77', '#D95F02', '#7570B3', '#E7298A', '#66A61E', '#E6AB02', '#A6761D', '#666666'] + matplotlib.colors.cnames.values()
        for lex_index in range(len(self.lexica)):
            paramvals, vals, maxval_markers = zip(*probs[lex_index])
            lex_rep = self.lex2str(self.lexica[lex_index])
            plt.plot(paramvals, vals, marker="", linestyle="-", label=lex_rep, color=colors[lex_index], linewidth=3)
            dots = [(paramval, val) for paramval, val, marker in probs[lex_index] if marker]
            if dots:           
                dotsx, dotsy = zip(*dots)
                plt.plot(dotsx, dotsy, marker="o", linestyle="", color=colors[lex_index],  linewidth=3)    
        plt.title("Listener hears '%s'\n%s" % (msg, self.params2str(exclude=[parameter_name])))
        plt.xlabel(parameter_name)
        plt.ylabel("Listener probability for <Lex, %s>" % target_state)
        plt.legend(loc=legend_loc)
        x1,x2,y1,y2 = plt.axis()
        plt.axis((x1, x2, 0.0, 1.0))
        setattr(self, parameter_name, original)
        plt.show()

                    
if __name__ == '__main__':    

   hurford = HurfordExperiment(temperature=2.0, disjunction_cost=2.0, beta=2.0)
   #hurford.build()
   #hurford.display_listener_inference()
   # hurford.plot_listener_inference_disjunction_costs()
   # hurford.plot_listener_inference_disjunction_costs(msg='A v C', target_state='1', legend_loc='lower right')
   # hurford.plot_listener_inference_beta_values(msg='A v C', target_state='1')


   hurford = HurfordExperiment(n=3,
                               temperature=2.0,
                               beta=1.0,
                               disjunction_cost=1.0,
                               baselexicon={'A': ['1'], 'B': ['2'], 'C': ['3'], 'D': ['4'], 'X': ['1', '2', '3', '4']},
                               lexical_costs={'A':0.0, 'B':0.0, 'C':0.0, 'D':0.0, 'X':0.0})

   #hurford.build()
   #hurford.display_listener_inference(msg='A v X')
   #hurford.plot_listener_inference_disjunction_costs(msg='A v X', target_state='1', legend_loc='right')
   #print display_matrix(hurford.lexica[0], rnames=hurford.messages, cnames=hurford.states)
   #hurford.plot_listener_inference_lambda_values(msg='A v X', target_state='1', legend_loc='right')
   hurford.plot_listener_inference_beta_values(msg='A v X', target_state='1', legend_loc='right')
   #hurford.plot_listener_inference_alpha_values(msg='A v X', target_state='1 v 2', legend_loc='right')
    
