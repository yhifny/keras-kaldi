# -*- coding: utf-8 -*-
import re
from heapq import heapify
import codecs
import sys



def levenshtein(a,b):
    "Calculates the Levenshtein distance between a and b."
    n, m = len(a), len(b)
    if n > m:
        # Make sure n <= m, to use O(min(n,m)) space
        a,b = b,a
        n,m = m,n

    current = list(range(n+1))
    for i in range(1,m+1):
        previous, current = current, [i]+[0]*n
        for j in range(1,n+1):
            add, delete = previous[j]+1, current[j-1]+1
            change = previous[j-1]
            if a[j-1] != b[i-1]:
                change = change + 1
            current[j] = min(add, delete, change)

    return current[n]

def map_timit48_to_timit39(phone_list):
  
    map_dic={
                'vcl':'sil',
                'cl':'sil',
                'pau':'sil',
                'epi':'sil',
                'el':'l',
                'en':'n',
                'zh':'sh',
                'ao':'aa',
                'ix':'ih',
                'ax':'ah'
            }
    output=[]
    for phone in phone_list:
        if phone not in map_dic:
            output.append(phone)
        else:
            output.append(map_dic[phone])
    
    return output    
    
def wer_timit39(original, result):
    r"""
    The WER is defined as the editing/Levenshtein distance on word level
    divided by the amount of words in the original text.
    In case of the original having more words (N) than the result and both
    being totally different (all N words resulting in 1 edit operation each),
    the WER will always be 1 (N / N = 1).
    """
    # The WER ist calculated on word (and NOT on character) level.
    # Therefore we split the strings into words first:
    ref_id   = original.split()[0]
    ref      = map_timit48_to_timit39(original.split()[1:])
    hyp_id   = result.split()[0]
    hyp      = map_timit48_to_timit39(result.split()[1:])
    
    assert(ref_id == hyp_id)
    
    return levenshtein(ref, hyp) / float(len(ref))

def wers_timit39(originals, results):
    count = len(originals)
    try:
        assert count > 0
    except:
        print(originals)
        raise("ERROR assert count>0 - looks like data is missing")
    rates = []
    mean = 0.0
    assert count == len(results)
    for i in range(count):
        rate = wer_timit39(originals[i], results[i])
        mean = mean + rate
        rates.append(rate)
    return rates, mean / float(count)



