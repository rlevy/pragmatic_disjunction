import numpy as np
from pragmods import Pragmod, display_matrix
from lexica import Lexica
        
def hurford(n=2, disjunction_cost=0.0):
    
    baselexicon = {'A': ['1'], 'B': ['2'], 'C':['1', '2']}

    lexica = Lexica(baselexicon=baselexicon, join_closure=True, disjunction_cost=disjunction_cost, nullsem=True, costs={'A':0.0, 'B':0.0, 'C':0.0})
    mats = lexica.lexica2matrices()
    
    mod = Pragmod(lexica=mats,
                  messages=lexica.messages,
                  meanings=lexica.states,
                  costs=lexica.cost_vector(),
                  prior=np.repeat(1.0/len(lexica.states), len(lexica.states)),
                  lexprior=np.repeat(1.0/len(lexica), len(lexica)),
                  temperature=1.0)

    for i, mat in enumerate(mats):                        
        display_matrix(mat, rnames=lexica.messages, cnames=lexica.states)
        print i
    
    langs = mod.run_expertise_model(n=n, display=True)
    #langs = mod.run_uncertainty_model(n=n, display=True)
   
    print "Listener joint inferences given message 'A v C':\n"
    print "".join([x.rjust(10) for x in [""] + lexica.states])
    msg_index = lexica.messages.index('A v C')
    final_lis_lang = langs[-1]
    for lex_index in range(len(mats)):
        row = [final_lis_lang[lex_index][msg_index][j] for j in range(len(lexica.states))]
        label = "Lex%s" % lex_index
        print label.rjust(10), "".join([str(round(x, 3)).rjust(10) for x in row])

    target_lexicon = 2
    target_state = 2
    return final_lis_lang[target_lexicon][msg_index][target_state]
        
def plot_hurford_targets():
    import xy_plot
    cost_val_pairs = []
    for cost in np.arange(0.0, 5.0, 0.1):
        cost_val_pairs.append((cost, hurford(disjunction_cost=cost)))
    costs, vals = zip(*cost_val_pairs)
    xy_plot.lineplot(costs, vals, marker='o', xlab="Disjunction cost", ylab="Lis3 probability for (Lex1, 1 v 2)", title="Listener hears 'A v C'")
    
                    
if __name__ == '__main__':    

   
    hurford(n=3, disjunction_cost=0.0)
    #plot_hurford_targets()

