import csv
import sys
sys.path.append('/home/rlevy/src/python/ucsdcpl/')
from condprob import lookup_ngram

def creator(src_filename='countrycapitals.csv', output_fielname='countrycapitals-g1tcounts.csv'):
    reader = csv.reader(file(src_filename))      
    header = reader.next()
    writer = csv.writer(file(output_fielname, 'w'))  
    writer.writerow(header + ['CountryFreq', 'CapitalFreq', 'CountryCapitalFreq', 'CapitalCountryFreq'])
    for row in reader:
        print row
        country, capital = row
        vals = row + [lookup_ngram([country]),
                      lookup_ngram([capital]),
                      lookup_ngram([country, 'or', capital]),
                      lookup_ngram([capital, 'or', country])]            
        writer.writerow(vals)
    
creator()
