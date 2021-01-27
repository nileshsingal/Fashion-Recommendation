import numpy as np
import pandas as pd
import pickle
import os
import json



df = pickle.load(open('styles.dat','rb')) 

dftop = pickle.load(open('top50for44k.dat','rb'))

x = int(input('enter row id:'))
#print(dftop.loc[1000].image)
im = df.loc[x].image
topr = dftop.loc[x]
od = {}

od['Input Image'] = im

oid = {}

for i in range(0,6):
    #topr[i]
    oid[i+1]={'image':df.loc[topr[i][0]].image, 'score':topr[i][1]}

od['Output'] = oid

print(json.dumps(od, indent=4))
