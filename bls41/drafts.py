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
