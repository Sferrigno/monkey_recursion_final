import string
import copy


def getCareAboutPos(careAbout, r):
    # returns the position of the header
    # corresponding to this type of response
    # which we care about!
    d = {}
    for i in xrange(len(r)):
        if r[i] in careAbout:
            d[i] = r[i]
    return d


# def getResps(ids, r):


def getKidsResponses(file, careAbout):
    data = open(file, "r")
    l = data.readline()
    # resps = dict()
    ids = dict()
    lastStart = 0
    c = 0
    allResps = []
    # l = l.replace("\r")
    r = l.split(",")
    # r = r.split("\r")
    # print r
    for i in r:
        if "\r" in i:
            allResps.append(r[lastStart:c + 1])
            lastStart = c
        c += 1

    #retD = {}
    vals = {}

    for k in xrange(len(allResps)):
        if k == 0:
            ids = getCareAboutPos(careAbout, allResps[k])
        else:
            #relies on subject being in
            #first column!
            sub = allResps[k][0]
            sub = sub.split("\r")[1]
            #print sub
            if sub not in vals:
                vals[sub] = []
            for key in ids.keys():
                val = allResps[k][key]
                if val != "":

                    vals[sub].append(val)
    data.close()

    return vals



def getMonkeyTsimaneResponses(file, careAbout, subset={}):
    data = open(file, "r")
    l = data.readline()
    # resps = dict()
    r = l.split(",")
    if len(subset.keys()) > 0:
        k = subset.keys()[0]
        #print r, k
        ind_sub = int(r.index(k))
    else:
        ind_sub = -1
    ind = int(r.index(careAbout))
    l = data.readline()
    vals = {}
    while l != "":
        r = l.split(",")
        if (ind_sub == -1) or (r[ind_sub] in subset[k]):
            part = r[0]
            cr = r[ind]
            if part not in vals:
                vals[part] = []
            if cr != "":
                vals[part].append(cr)

        l = data.readline()



    return vals


def getCounts(resps, dictMap):
    allP = {}
    for r in resps:
        tmp = ""
        for a in r:
            paren = dictMap[a]
            tmp += paren + ","
        tmp = tuple(tmp[:len(tmp) - 1].split(","))
        if tmp not in allP:
            allP[tmp] = 0

        allP[tmp] += 1
    return allP


def getCountData(file, careAbout, which, 
        subset={}, dictMap = {"A": "[", "B": "]", "C": "(", "D": ")"}):
    if which == "Kids" or which == "Adults":
        careAbout = [careAbout]
        resps = getKidsResponses(file, careAbout)
    else:
        resps = getMonkeyTsimaneResponses(file, careAbout, subset)
    
    allC = []
    for key in resps.keys():

        if len(resps[key]) > 0:
            if "FORWARD" in careAbout[0]:
                allC.append(resps[key][0])
            else:
                cs = getCounts(resps[key], dictMap)
                allC.append(cs)
    # counts = getCounts(resps, dictMap)
    return allC


if __name__ == "__main__":
    import scipy.stats as st

    ##
    """
    file = "stevesdata/RecursionKids.csv"

    count = getCountData(file, careAbout, "Kids")

    file = "stevesdata/RecursionTsimane.csv"

    """

    careAbout = "Order pressed"
    file = "stevesdata/RecursionMonkey.csv"

    count = getCountData(file, careAbout, "Monkey")


    count = getCountData(file, careAbout, "blorb", 
                subset={"Experiment" :
                 ["Experiment 2"]})


   # file = "stevesdata/RecursionAdults.csv"
    #count = getCountData(file, careAbout, "Adults")
    #file = "stevesdata/RecursionKids_MoreSubs.csv"

    #careAbout = "Order pressed"
    #count = getCountData(file, careAbout, "Kids")

    #careAbout= "FORWARDS DIGITS"

    #forward_dig = getCountData(file, careAbout, "Kids")

    for c in count:
        tot = 0.
        p = 0.
        for x in c:
            if x[0] == "(":
                p += c[x]
            tot += c[x]

        print p/tot



    CE =  ["([])", "[()]"]
    n_CE = []
    n_tot = []
    which = {}
    for part in count:
        print part
        n_CE_i = 0.
        tot_i = 0.
        for resp in part:
            n_resp = part[resp]
            if resp not in which:
                which[resp] = 0
            which[resp] += n_resp

            r= "".join(list(resp))

            if r in CE:
                n_CE_i  += n_resp
            tot_i += n_resp
        n_CE.append(n_CE_i)
        n_tot.append(tot_i)

    print n_CE,n_tot, sum(n_CE)/sum(n_tot)
    print sorted(map(lambda tup:tup[0]/float(tup[1]), zip(n_CE, n_tot)))

    for r in which:
        print r, which[r]/sum(n_tot)