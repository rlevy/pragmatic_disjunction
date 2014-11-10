#!/usr/bin/env python

import csv
import zss
import sys
sys.path.append('/home/njsmith/zss')

# z2 = zss.ZSS("/local/scratch/njs/google_books_ngrams_v2/eng-us-all-20120701-2gram.zss")
# z2 is basically all the 2-gram lines from the original text files in sorted order with random access.
# so lines like
#  NGRAM \t YEAR \t TOKENS \t BOOKS
#print repr(z2.iter_from("it was\t").next())


def add_vals(src_filename="countrycapitals.csv", 
             output_filename="countrycapitals-gb2counts.csv", 
             first_year=1960):
    
    z1 = zss.ZSS("/local/scratch/njs/google_books_ngrams_v2/eng-us-all-20120701-1gram.zss")
    z2 = zss.ZSS("/local/scratch/njs/google_books_ngrams_v2/eng-us-all-20120701-2gram.zss")
    z3 = zss.ZSS("/local/scratch/njs/google_books_ngrams_v2/eng-us-all-20120701-3gram.zss")
    z4 = zss.ZSS("/local/scratch/njs/google_books_ngrams_v2/eng-us-all-20120701-4gram.zss")
    z5 = zss.ZSS("/local/scratch/njs/google_books_ngrams_v2/eng-us-all-20120701-4gram.zss")
    # Reader:
    reader = csv.reader(file(src_filename))
    header = reader.next()
    # Writer:
    writer = csv.writer(file(output_filename, 'w'))
    writer.writerow(header + ['CountryFreq', 'CapitalFreq', 'CountryCapitalFreq', 'CapitalCountryFreq'])
    # Iterate through source reader and write to writer:
    for row in reader:
        # Vals to check:
        x = row[0]
        y = row[1]
        countrycapital = '%s or %s' % (x, y)
        capitalcountry = '%s or %s' % (y, x)
        # Look-up dbs:
        x_db = z1
        y_db = z1
        countrycapital_db = z3
        capitalcountry_db = z3
        # Manage potential bigram disjuncts:
        if x.find(' ') > -1 and y.find(' ') > -1:
            x_db = z2
            y_db = z2
            disj_db = z5
        elif x.find(' ') > -1:
            x_db = z2
            disj_db = z4
        elif y.find(' ') > -1:
            y_db = z2
            disj_db = z4
        # Look-ups:
        xfreq = get_ngram(x, x_db, first_year=first_year)
        yfreq = get_ngram(y, y_db, first_year=first_year)                
        countrycapitalfreq = get_ngram(countrycapital, countrycapital_db, first_year=first_year)
        capitalcountryfreq = get_ngram(capitalcountry, capitalcountry_db, first_year=first_year)
        # Write:
        writer.writerow(row + [xfreq, yfreq, countrycapitalfreq, capitalcountryfreq])
                
def get_ngram(target_ngram, db, first_year=1960):
    total = 0
    for line in db.iter_from("%s\t" % target_ngram):
        ngram, year_str, tokens_str, books = line.split("\t")
        if ngram != target_ngram:
            break
        year = int(year_str)
        if year >= first_year:
            total += int(tokens_str)
    return total

if __name__ == '__main__':

    add_vals()
