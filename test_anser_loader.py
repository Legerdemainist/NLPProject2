# Created by Bleau Moores, Lisa Ewen, Tim Heydrich
# Last Modified: 27/03/2020 by Tim Heydrich

import numpy as np
import pandas as pd

# ======================================================================================================================
# ======================================================================================================================

# Reads in the answers from Senseval2.key or EnglishLS.test.ket or both depending on the passed value
# The data is stored and returned in a pandas dataframe
def get_test_ansers(sens):
    if sens == 2:
        f = open("./data/senseval2/Senseval2.key", "r")
        all = []
        for x in f:
            all.append(x.split())
    elif sens == 3:
        f = open("./data/senseval3/EnglishLS.test.key", "r")
        all = []
        for x in f:
            all.append(x.split())
    elif sens == 23:
        f = open("./data/senseval2/Senseval2.key", "r")
        all = []
        for x in f:
            all.append(x.split())
        f = open("./data/senseval3/EnglishLS.test.key", "r")
        for x in f:
            all.append(x.split())

    targets = []
    senses = []
    for a in all:
        target = a[1]
        i = 2
        sense = a[2]
        if len(sense) == 1:
            if sense[0] == "P" or sense[0] == "U":
                if len(a) < 4:
                    continue
                senses.append(a[3])
                targets.append(target)
            else:
                senses.append(sense)
                targets.append(target)
        else:
            senses.append(sense)
            targets.append(target)

    senses = np.asarray(senses)
    targets = np.asarray(targets)
    #print(senses.shape)
    #senses = np.transpose([senses])
    #targets = np.transpose([targets])
    #print(targets)

    d = pd.DataFrame([senses, targets], index=['Senses', 'Targets']).T
    #print(d.head(20))
    return d

# ======================================================================================================================
# ======================================================================================================================
