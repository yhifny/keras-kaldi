#!/usr/bin/python
# -*- coding: cp1256 -*-
#
#   Usage: GenScliteNum infile.txt
#   Author: Yasser Hifny
#

import sys

        
        
if __name__ == '__main__':    
    if len(sys.argv)!=2:
        print ("Usage: GenScliteNum infile.txt")

index=0

# load data and build dictionary
for line in open(sys.argv[1], 'r') .readlines():
    print (line.strip(), " ( ", index, " )")
    index=index+1


