#!/usr/bin/env python

import random
from pragmods import *
from lexica import *
import csv
from collections import defaultdict

def explore_parameter_space(n=25,
                            unknown_word='X',
                            unknown_disjunction='A v X',
                            target_lexicon_index=0):

    report = {
        'iterations':n,
        'disjunction_cost': 0.001,
        'null_cost': 4.0,
        'alpha': random.uniform(0.1, 5.0),
        'beta': random.uniform(0.1, 5.0),
        'temperature': random.uniform(1.0, 5.0),
        'unknown_word_has_atomic_meaning': True
    }

    baselexicon = {
        'A': ['w1'], 
        'B': ['w2'], 
        'C': ['w3'],
        'D': ['w4'],
        'X': ['w1', 'w2', 'w3', 'w4']}

    lexica = Lexica(baselexicon=baselexicon,
                    join_closure=True,
                    disjunction_cost=report['disjunction_cost'],
                    null_cost=report['null_cost'])
        
    mats = lexica.lexica2matrices()

    unk_index = lexica.messages.index(unknown_word)

    unknown_disjunction_index = lexica.messages.index(unknown_disjunction)

    if report['unknown_word_has_atomic_meaning']:        
        mats = [mat for mat in mats if np.sum(mat[unk_index]) == 1]
           
    mod = Pragmod(lexica=mats,
                  messages=lexica.messages,
                  meanings=lexica.states,
                  costs=lexica.cost_vector(),
                  prior=np.repeat(1.0/len(lexica.states), len(lexica.states)),
                  lexprior=np.repeat(1.0/len(mats), len(mats)),
                  temperature=report['temperature'],
                  alpha=report['alpha'],
                  beta=report['beta'])

    langs = mod.run_expertise_model(n=n, display=False)

    final_speaker = langs[-2]

    report['speaker_unknown_disjunct_value'] = np.round(final_speaker[unknown_disjunction_index][target_lexicon_index][0], 4)
    
    final_listener = langs[-1]

    report['listener_unknown_disjunct_value'] = np.round(final_listener[target_lexicon_index][unknown_disjunction_index][0], 4)

    return report


def create_report(output_filename='explore_disjunctions.csv', runs=500):

    writer = csv.writer(file(output_filename, 'w'))

    # One report to get column names; ignore these values for perspicuity:
    report = explore_parameter_space()
    colnames = sorted(report.keys())
    writer.writerow(colnames)

    # For stat report to satisfy curiosity immediately:
    stats = defaultdict(float)
    
    # The actual runs:
    for i in range(runs):
        report = explore_parameter_space()
        row = [report[cname] for cname in colnames]
        writer.writerow(row)
       
        if report['speaker_unknown_disjunct_value'] >= 0.90:
            stats['speaker_result'] += 1
        if report['listener_unknown_disjunct_value'] >= 0.90:
            stats['listener_result'] += 1

    print stats
        
    
create_report()


