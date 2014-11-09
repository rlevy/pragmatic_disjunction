import numpy as np
from pragmods import Pragmod, display_matrix
from lexica import Lexica

def scalars(n=3):
    """Scalar example without and with disjunction; compare with Bergen et al.'s figure 5 and figure 9"""
    baselexicon = {'some': [r'$w_{\exists\neg\forall}$', r'$w_{\forall}$'], 'all': [r'$w_{\forall}$']}
    basic_lexica = Lexica(baselexicon=baselexicon)
    lexica = Lexica(baselexicon=baselexicon, join_closure=True, disjunction_cost=0.1)
    lexica.display()    
    mod = Pragmod(lexica=lexica.lexica2matrices(),
                  messages=lexica.messages,
                  meanings=lexica.states,
                  costs=lexica.cost_vector(),
                  prior=np.repeat(1.0/len(lexica.states), len(lexica.states)),
                  lexprior=np.repeat(1.0/len(lexica), len(lexica)),
                  temperature=1.0)
    mod.plot_expertise_listener(output_filename='../paper/fig/scalar-expertise-listener-marginalized.pdf', n=n)
    mod.plot_expertise_speaker(output_filename='../paper/fig/scalar-expertise-speaker.pdf', n=n)
    mod.plot_expertise_speaker(output_filename='../paper/fig/scalar-expertise-speaker-lexsum.pdf', n=n, lexsum=True)


def manner(n=3):
    """Settings for Bergen et al.'s figure 6. Seems to reproduce the effects they report."""
    lexica = Lexica(baselexicon={'SHORT': [r'$w_{RARE}$', r'$w_{FREQ}$'], r'long': [r'$w_{RARE}$', r'$w_{FREQ}$']},
                    costs={'SHORT':1.0, r'long':2.0},
                    null_cost=5.0,
                    join_closure=False,
                    disjunction_cost=0.1)
    lexica.display()   
    mod = Pragmod(lexica=lexica.lexica2matrices(),
                  messages=lexica.messages,
                  meanings=lexica.states,
                  costs=lexica.cost_vector(),
                  prior=np.array([2.0/3.0, 1.0/3.0]),
                  lexprior=np.repeat(1.0/len(lexica), len(lexica)),
                  temperature=1.0,
                  alpha=3.0)
    mod.plot_expertise_listener(output_filename='../paper/fig/manner-expertise-listener-marginalized.pdf', n=n)
    mod.plot_expertise_speaker(output_filename='../paper/fig/manner-expertise-speaker.pdf', n=n)
    mod.plot_expertise_speaker(output_filename='../paper/fig/manner-expertise-speaker-lexsum.pdf', n=n, lexsum=True)

def disjunction(n=3):
    """Settings for Bergen et al.'s figure 6. Seems to reproduce the effects they report."""
   # bl = {'p': [r'$w_1$', r'$w_2$'], 'q':[r'$w_2$', r'$w_3$']}
    bl = {'p': ['1', '2'], 'q':['2', '3']}
    lexica = Lexica(baselexicon=bl,
                    atomic_states=['1', '2', '3'],
                    disjunction_cost=1.0,
                    conjunction_cost=0.0,
                    null_cost=5.0,                    
                    join_closure=True,
                    meet_closure=True,
                    block_trivial_messages=True,
                    block_ineffability=False)
    
    lexica.display()
    for key, val in lexica.lexica[8].items():
        print key, len(val), val
    mod = Pragmod(lexica=lexica.lexica2matrices(),
                  messages=lexica.messages,
                  meanings=lexica.states,
                  costs=lexica.cost_vector(),
                  prior=np.repeat(1.0/len(lexica.states), len(lexica.states)),
                  lexprior=np.repeat(1.0/len(lexica), len(lexica)),
                  temperature=1.0,
                  alpha=1.0)
    mod.plot_expertise_listener(output_filename='../paper/fig/scalardisj-expertise-listener-marginalized.pdf', n=n)
    mod.plot_expertise_speaker(output_filename='../paper/fig/scalardisj-expertise-speaker.pdf', n=n)
    mod.plot_expertise_speaker(output_filename='../paper/fig/scalardisj-expertise-speaker-lexsum.pdf', n=n, lexsum=True)


        
                                      
if __name__ == '__main__':    

    #scalars(n=3)
    #manner(n=3)
    disjunction(n=2)

