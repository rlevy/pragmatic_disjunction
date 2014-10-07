import numpy as np
from pragmods import Pragmod, display_matrix
from lexica import Lexica
import copy
from collections import defaultdict
import matplotlib.pyplot as plt


class HurfordExperiment:
    def __init__(self,
                 n=3,
                 disjunction_cost=0.0,             
                 lexical_costs={'A':0.0, 'B':0.0, 'C':0.0},
                 temperature=1.0):
        self.n = n
        self.disjunction_cost = disjunction_cost
        self.lexical_costs = lexical_costs
        self.temperature = temperature
        self.baselexicon = {'A': ['1'], 'B': ['2'], 'C':['1', '2']}

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
                             temperature=self.temperature)

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
        for x in ('n', 'temperature', 'disjunction_cost', 'lexical_costs'):
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

    def plot_listener_inference_disjunction_costs(self,
                                                  msg='A v C',
                                                  target_state='1 v 2',                                                 
                                                  disjunction_cost_values=np.arange(0.0, 5.0, 0.01),
                                                  legend_loc='upper right'):
        cost_val_pairs = []
        original_dcost = self.disjunction_cost        
        probs = defaultdict(list)
        for dcost in disjunction_cost_values:                        
            self.disjunction_cost = dcost
            self.build()
            target_state_index = self.states.index(target_state)
            self.display_listener_inference(msg=msg)             
            prob_table = self.listener_inference(msg=msg)
            for lex_index in range(len(self.lexica)):
                probs[lex_index].append((dcost, prob_table[lex_index][target_state_index]))
        fig = plt.figure(figsize=(9, 5))
        for lex_index in range(len(self.lexica)):
            dcosts, vals = zip(*probs[lex_index])
            lex_rep = self.lex2str(self.lexica[lex_index])
            plt.plot(dcosts,
                     vals,
                     marker="",
                     linestyle="-",
                     label=lex_rep)
        plt.title("Listener hears '%s'\n%s" % (msg, self.params2str(exclude=['disjunction_cost'])))
        plt.xlabel("Disjunction cost")
        plt.ylabel("Listener probability for <Lex, %s>" % target_state)
        plt.legend(loc=legend_loc)
        self.disjunction_cost = original_dcost
        plt.show()
                    
if __name__ == '__main__':    

   hurford = HurfordExperiment(temperature=1.5)
   #hurford.build()
   #hurford.display_listener_inference()
   hurford.plot_listener_inference_disjunction_costs()
   hurford.plot_listener_inference_disjunction_costs(msg='A v C', target_state='1', legend_loc='lower right')
