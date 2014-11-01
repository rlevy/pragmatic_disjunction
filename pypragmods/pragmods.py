#/usr/bin/env python

import copy
import numpy as np

import matplotlib.pyplot as plt

def rownorm(mat):
    """Row normalization of a matrix"""
    return np.divide(mat.T, np.sum(mat, axis=1)).T
    
def colnorm(mat):
    """Column normalization of a matrix"""    
    return np.divide(mat, np.sum(mat, axis=0))

def safelog(vals):
    with np.errstate(divide='ignore'):
        return np.log(vals)

def display_matrix(mat, display=True, rnames=None, cnames=None, title='', digits=4):
    """Utility function for displaying strategies to standard output.
    The display parameter saves a lot of conditionals in the important code"""
    if display:
        mat = np.round(mat, digits)
        cwidth = 2 + max([len(x) for x in rnames+cnames] + [digits+2])
        # Divider bar of the appropriate width:
        print "-" * (cwidth * (max(len(cnames), len(rnames)) + 1))
        print title
        # Matrix with even-width columns wide enough for the data:
        print ''.rjust(cwidth) + "".join([str(s).rjust(cwidth) for s in cnames])        
        for i in range(mat.shape[0]):  
            print str(rnames[i]).rjust(cwidth) + "".join(str(x).rjust(cwidth) for x in mat[i, :])
    
######################################################################
        
class Pragmod:    
    def __init__(self, 
                 lexica=None, 
                 messages=None, 
                 meanings=None, 
                 costs=None, 
                 prior=None, 
                 lexprior=None, 
                 temperature=1.0,
                 alpha=1.0,
                 beta=1.0):
        self.lexica = lexica                # list of np.arrays of dimension m x n
        self.messages = messages            # list or tuple of strings of length m
        self.meanings = meanings            # list or tuple of strings of length n
        self.costs = costs                  # np.array of length m
        self.prior = prior                  # np.array of length n
        self.lexprior = lexprior            # np.array of length len(self.lexica)
        self.temperature = temperature      # usually \lambda, but lambda is a Python builtin; should be > 0.0
        self.alpha = alpha                  # speaker value for the world state
        self.beta = beta                    # speaker value for the lexicon

    ##################################################################
    ##### Iteration models

    def run_base_model(self, lex, n=2, display=True):
        """Basic model with a specified messages x meanings matrix of truth values lex"""
        return self.run(n=n, display=display, initial_listener=self.l0(lex), start_level=0)
    
    def run_uncertainty_model(self, n=2, display=True):
        """The lexical uncertainty model of Bergen et al. 2012, 2014"""
        return self.run(n=n, display=display, initial_listener=self.UncertaintyListener(), start_level=1)    

    def run_anxiety_model(self, n=2, display=True):
        """One-shot version of the social anxiety model of Smith et al. 2013"""        
        return self.run(n=n, display=display, initial_listener=self.UncertaintyAnxietyListener(marginalize=True), start_level=1)

    def run(self, lex=None, n=2, display=True, initial_listener=None, start_level=0):
        """Generic iterator. n is the depth of iteration. initial_listener is one of the
        listener methods, applied to a lexical argument in the case of the base model.
        display=True prints all matrices to standard output. start_level controls which 
        listener number to begin with for displaying the model."""
        langs = [initial_listener]
        for i in range(1, (n-1)*2, 2):
            langs.append(self.S(langs[i-1]))
            langs.append(self.L(langs[i]))
        if display:
            self.display_iteration(langs, start_level=start_level)        
        return langs  

    def run_expertise_model(self, n=2, display=True):
        langs = [self.UncertaintyAnxietyListener(marginalize=False)]
        for i in range(1, (n-1)*2, 2):
            langs.append(self.ExpertiseSpeaker(langs[i-1]))
            langs.append(self.ExpertiseListener(langs[i]))
        if display:
            self.display_expertise_iteration(langs)      
        return langs  
        
    ##################################################################
    ##### Agents

    def l0(self, lex):
        """Maps the truth-conditional lexicon lex to a probabilistic one incorporating priors."""
        return rownorm(lex * self.prior)  

    def L(self, spk):
        """The general listener differs from l0 only in transposing the incoming speaker matrix."""
        return self.l0(spk.T)

    def S(self, lis):
        """Bayesian speaker incorporating costs."""
        return rownorm(np.exp(self.temperature * ((self.alpha * safelog(lis.T)) - self.costs)))
    
    def s1(self, lex):
        """Convenience function for S(l0(lex))"""
        return self.S(self.l0(lex))

    def l1(self, lex):
        """Convenience function for L(S(l0(lex)))"""
        return self.L(self.s1(lex))        
    
    def UncertaintyListener(self): 
        """The lexical uncertainty listener reasons over the marginal of S(L0(lex)) for all lexicons lex."""
        result = [self.lexprior[i] * self.prior * self.s1(lex).T for i, lex in enumerate(self.lexica)]
        return rownorm(np.sum(result, axis=0))

    def UncertaintyAnxietyListener(self, marginalize=False):
        """Social anxiety listener of Smith et al. 2013."""
        lik = self.lex_lik()
        result = [(self.l1(lex).T * lik[i]).T for i, lex in enumerate(self.lexica)]
        if marginalize:
            result = np.sum(result, axis=0)
        return result
                       
    def lex_lik(self):
        """Creates a lexicon x utterance matrix, normalized columnwise for P(Lex|u)."""
        p = np.array([np.sum(self.s1(lex), axis=0) * self.lexprior[i] for i, lex in enumerate(self.lexica)])
        return colnorm(p)

    def ExpertiseSpeaker(self, listeners):
        """Our expertise speaker"""
        lis = np.sum(listeners, axis=0)
        lexprobs = np.sum(listeners, axis=2)               
        result = []
        # Embedded for-loops for readability/comprehension; could be replaced:
        for u in range(len(self.messages)):
            lex = np.zeros((len(self.meanings), len(self.lexica)))
            for m in range(len(self.meanings)):
                for l in range(len(self.lexica)):
                    lex[m,l] = np.exp(self.temperature * ((self.alpha*safelog(lis[u,m])) + (self.beta*safelog(lexprobs[l][u])) - self.costs[u]))
            result.append(lex)
        return result / np.sum(result, axis=0)

    def ExpertiseListener(self, speakers):
        """Our expertise listener""" 
        result = []
        # Embedded for-loops for readability/comprehension; could be replaced:
        for k in range(len(self.lexica)):                    
            lis = np.zeros((len(self.messages), len(self.meanings)))                        
            for i in range(len(self.messages)):
                for j in range(len(self.meanings)):
                    lis[i,j] = speakers[i][j,k] * self.prior[j] * self.lexprior[k]
            result.append(lis)
        totals = np.sum(result, axis=(0, 2))
        return [(r.T / totals).T for r in result]
        
    ##################################################################
    ##### Display functions

    def display_expertise_iteration(self, langs):
        print "======================================================================"
        [self.display_listener_matrix(l, title="1 - Lex%s" % i) for i, l in enumerate(langs[0])]
        self.display_listener_matrix(np.sum(langs[0], axis=0), title="1 - marginalized")
        level = 2
        for index in range(1, len(langs), 2):
            print "======================================================================"         
            [self.display_lex_matrix(l, title="%s - %s" % (level, self.messages[i])) for i, l in enumerate(langs[index])]
            spk = np.zeros((len(self.messages), len(self.meanings)))
            for j, u in enumerate(langs[index]): 
                spk[j] += np.sum(u, axis=1)           
            self.display_speaker_matrix(rownorm(spk.T), title="%s - marginalized" % level)
            print "======================================================================"
            [self.display_listener_matrix(l, title="%s - Lex%s" % (level, i)) for i, l in enumerate(langs[index+1])]
            self.display_listener_matrix(np.sum(langs[index+1], axis=0), title="%s - marginalized" % level)
            level += 1
            
    def display_iteration(self, langs, start_level=0):
        self.display_listener_matrix(langs[0], title=start_level)        
        start_level += 1
        display_funcs = (self.display_speaker_matrix, self.display_listener_matrix)
        for i, lang in enumerate(langs[1: ]):
            display_funcs[i % 2](lang, title=start_level)
            if i % 2: start_level += 1

    def display_speaker_matrix(self, mat, display=True, title='', digits=4):
        display_matrix(mat, display=display, title='S%s' % title, rnames=self.meanings, cnames=self.messages, digits=digits)

    def display_listener_matrix(self, mat, display=True, title='', digits=4):
        display_matrix(mat, display=display, title='L%s' % title, rnames=self.messages, cnames=self.meanings, digits=digits)

    def display_lex_matrix(self, mat, display=True, title='', digits=4):
        cnames = ['Lex%s' % i for i in range(len(self.lexica))]
        display_matrix(mat, display=display, title='S%s' % title, rnames=self.meanings, cnames=cnames, digits=digits)
    
    def plot_expertise_iteration(self):
        fig, ax = plt.subplots(4, len(self.lexica)+1)
        fig.set_figheight(20)
        fig.set_figwidth(35)
        for lexindex, lex in enumerate(self.lexica):            
            self.plot_listener_matrix(self.l0(lex), ax[0][lexindex], lex=lex)
            self.plot_speaker_matrix(self.s1(lex), ax[1][lexindex], lex=None)
            self.plot_listener_matrix(self.l1(lex), ax[2][lexindex], lex=None)
        langs = self.UncertaintyAnxietyListener(marginalize=False)
        for lexindex, lex in enumerate(langs):
            self.plot_listener_matrix(lex, ax[3][lexindex], lex=None)
        self.plot_listener_matrix( self.UncertaintyAnxietyListener(marginalize=True), ax[3][len(self.lexica)], lex=None)
        print self.ExpertiseSpeaker(langs)
        plt.savefig('/Volumes/CHRIS/Desktop/temp.png')

    def plot_expertise_listener(self, n=3, output_filename=None):
        fig, ax = plt.subplots(1,1)
        fig.set_figheight(6)
        fig.set_figwidth(8)
        langs = self.run_expertise_model(n=n)
        final_listener = langs[-1]
        marginalized = np.sum(final_listener, axis=0)
        self.plot_listener_matrix(marginalized, ax, lex=None)
        if output_filename:
            plt.savefig(output_filename)
        else:
            plt.show()

    def plot_expertise_speaker(self, n=3, output_filename=None, lexsum=False):        
        langs = self.run_expertise_model(n=n)
        final_speaker = langs[-2]
        if lexsum:
            spk = np.zeros((len(self.messages), len(self.meanings)))
            for j, u in enumerate(final_speaker): 
                spk[j] += np.sum(u, axis=1) 
            fig, ax = plt.subplots()
            fig.set_figheight(6)
            fig.set_figwidth(8)
            self.plot_speaker_matrix(rownorm(spk.T), ax, lex=None)
        else:
            final_speaker = final_speaker.T
            fig, ax = plt.subplots(1,len(self.lexica))
            fig.set_figheight(6)
            fig.set_figwidth(8*len(self.lexica))
            for lexindex in range(len(self.lexica)):
                self.plot_speaker_matrix(final_speaker[lexindex], ax[lexindex], lex=self.lexica[lexindex])
        if output_filename:
             plt.savefig(output_filename)
        else:
            plt.show()
        
    def plot_listener_matrix(self, mat, ax, lex=None):
        self.plot_matrix(mat, ax, lex=lex, outerlabels=self.messages, innerlabels=self.meanings)

    def plot_speaker_matrix(self, mat, ax, lex=None):
        self.plot_matrix(mat, ax, lex=lex, outerlabels=self.meanings, innerlabels=self.messages, initial_color_index=len(self.messages)+1)

    def plot_matrix(self, mat, ax, lex=None, outerlabels=None, innerlabels=None, width=0.2, initial_color_index=0):
        from lsa import colors
        from lexica import DISJUNCTION_SIGN
        m, n = mat.shape
        barsetwidth = width*n
        ind = np.arange(0.0, (barsetwidth+width)*m, barsetwidth+width)
        ind = ind[::-1]        
        for j in range(n-1, -1, -1):
            xpos = ind+(width*j)
            vals = mat[:, j]        
            ax.barh(xpos, vals, width, color=colors[initial_color_index+j], label=innerlabels[j])
            for i in range(m):
                ax.text(0.01, xpos[i]+(width/2.0), innerlabels[j], rotation='horizontal', ha='left', va='center')
        ax.set_yticks(ind+barsetwidth/2.0)
        ax.set_yticklabels(outerlabels)
        if lex != None:
            ax.set_title(self.lex2str(lex))
                
    def lex2str(self, lexicon_or_lexicon_index):
        from lexica import DISJUNCTION_SIGN
        lexicon = lexicon_or_lexicon_index
        if isinstance(lexicon, int):
            lexicon = self.lexica[lexicon_or_lexicon_index]            
        def state_sorter(x):
            return sorted(x, cmp=(lambda x, y: cmp(len(x), len(y))))
        entries = []
        for p_index, p in enumerate(self.messages):
            sem = [s for i, s in enumerate(self.meanings) if lexicon[p_index][i] > 0.0 and not DISJUNCTION_SIGN in s]
            entry = p + "={" + ",".join(state_sorter(sem)) + "}"
            entries.append(entry)
        return "; ".join(entries)
