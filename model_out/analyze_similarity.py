import numpy as np
import copy

def KLDiv(p,q): 
    p = np.array(p)
    q = np.array(q)
    return np.sum(p*np.log10(p/q))

def cosine_sim(x,y):
    return np.dot(x,y)/np.sqrt(np.dot(x,x)*np.dot(y,y))
def euc_dist(x,y):
    return np.sum(np.subtract(x,y)**2)



def get_data(file):
    f = open(file, 'r')
    l = f.readline()
    spl = l.split(",")
    who_index = spl.index('who')
    id_index = spl.index('id')
    val_index = spl.index('val')
    
    l = f.readline()
    dct = {}
    while l != "":
        spl = l.replace(" ","").split(",")
        who = spl[who_index]
        part = spl[id_index]
        value = spl[val_index]

        if (who,part) not in dct:
            dct[(who,part)] = []
        dct[(who,part)].append(float(value))

        l = f.readline()

    return dct


def sim_vecs(vecs):
    keys = copy.deepcopy(vecs.keys())
    dct = {}
    for v1 in xrange(len(keys)):
        vec1 = keys[v1]

        for v2 in xrange(len(keys)):
            vec2 = keys[v2]
            #sim = cosine_sim(vecs[vec1],vecs[vec2])
            sim=euc_dist(vecs[vec1],vecs[vec2])
            #sim = -(KLDiv(vecs[vec1],vecs[vec2]) + KLDiv(vecs[vec2],vecs[vec1]))

            assert((vec1,vec2) not in dct)
            dct[(vec1,vec2)] = sim

    return dct


def output_sim(sims, f):
    o = "who1,part1,who2,part2,sim\n"
    for sim in sims:
        who1 = sim[0][0]
        part1 = sim[0][1]
        who2 = sim[1][0]
        part2 = sim[1][1]
        s = sims[sim]

        o += "%s,%s,%s,%s,%f\n" % (who1,part1,who2,part2,s)

    out = open(f,"w+")
    out.write(o)
    out.close()

def output_sim_rowcol(sims, f):
    n_col = 0
    for s in sims:
        if int(s[0][1]) > n_col:
            n_col = int(s[0][1])

    n_col += 1

    keys = (sorted(sims.keys(),
         key=lambda tup: (int(tup[0][1])+1) * (n_col) + int(tup[1][1])))
    o = ""


    for key in keys[:n_col]:
        o += " ,%s%s" % (key[1][0], key[1][1])
    #o += "\n"

    for key in keys:
        print key
        if key[1][1] == '0':
            o += "\n"

            o += "%s%s" % (key[0][0], key[0][1])

        o += ",%f" % sims[key]

    
    out = open(f, "w+")

    out.write(o)
    out.close()

if __name__ == "__main__":
    file = "thetas.csv"
    vecs=  get_data(file)
    sims = sim_vecs(vecs)


    output_sim_rowcol(sims, "similarity_matrix.csv")