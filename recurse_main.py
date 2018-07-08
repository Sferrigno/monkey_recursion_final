import pymc3 as pm
from helpers import *
import matplotlib.pyplot as plt
import seaborn as sns
#from pymc3.backends import Text




def model():
    global data
    alpha_prior = 0.1
    beta_prior = 1.
    alpha_init = np.ones((N_GROUPS,1))
    noise_init = np.ones((N_GROUPS,1))*1e-2

    parts_ones = np.ones((TOTAL_PARTS))
    data_ones = np.ones(len(data[0]))

    hds = store_hds_old(paren_lst,filt)
    ns = np.sum(data, axis=1)


    smooth =  np.ones((TOTAL_PARTS,N_ALGS)) * beta_prior

    #bias in choice of starting parenthesis
    start_p = store_start_p(paren_lst, n=TOTAL_PARTS, lst = ["("])
    start_np = 1 - start_p

    #init_beta = np.array([0.1,0.1,0.1,0.1,0.1,0.24,0.1,0.1,0.1,0.1,0.1,0.1])*25
    init_beta = np.ones(N_ALGS) * beta_prior
    with pm.Model() as m:
        alpha = pm.Exponential('alpha', alpha_prior, shape=1) 

        #alpha = tt.as_tensor([10])
        #alpha = pm.Deterministic('alpha', alpha)



        beta = pm.Dirichlet('beta', init_beta,
                            #np.ones(N_ALGS)*beta_prior,
                        # testval=np.ones(N_ALGS),
                            shape=N_ALGS) 



        theta = pm.Dirichlet('theta',  alpha * beta, 
                           shape=(TOTAL_PARTS,N_ALGS)) 


       # noise = pm.Deterministic('noise',
               #      tt.as_tensor(np.ones(TOTAL_PARTS)*0.05))

        #x1 = pm.Exponential('x1', 1, shape=1)
        #x1 = 10
        #x2 = pm.Beta('x2', 1,20, shape=1)
        #nw1 = x2 * x1
        #nw2 = (1-x2) * x1
        nw1 = 1
        nw2 = 9
 


        noise = pm.Beta("noise", nw1,nw2, shape=TOTAL_PARTS, testval=0.1)
        #noise = tt.tile(noise, (1, N_ALGS))

        new_algs = map(lambda x: theta[x].dot(format_algs_theano(hds, noise[x])), np.arange(TOTAL_PARTS))
       
        theta_resp = tt.concatenate([new_algs], axis=0)
        #theta_resp = theta + noise * 0.5

        bias = pm.Beta("bias", 1,1,shape=(TOTAL_PARTS,1))
        #bias = tt.tile(bias, (1,N_ALGS))

        biased_theta_resps = start_p * bias * theta_resp + start_np * (1.-bias) * theta_resp
        sum_norm = biased_theta_resps.sum(axis=1).reshape((TOTAL_PARTS,1))
        biased_theta_resps = biased_theta_resps / sum_norm

        #biased_theta_resps = theta_resp

        pm.Multinomial('resp', n=ns, p = biased_theta_resps, 
               shape=(TOTAL_PARTS, N_RESPS), observed=data)

        #db = Text('trace')
        step = pm.NUTS()

        trace = pm.sample(MCMC_STEPS,step=step,
            tune=BURNIN,target_accept=0.9, thin=MCMC_THIN)
        print_star("Model Finished!")


    if MCMC_CHAINS > 1:
        print pm.gelman_rubin(trace)

    summary = pm.df_summary(trace)

    print summary
    which = 45
    samp =100



    return trace, summary






if __name__ == "__main__":

    #make parentheses lists, 
    #and hypotheses (e.g. OOMC)

    MCMC_STEPS = 200
    MCMC_THIN = 1
    MCMC_CHAINS=1
    BURNIN =20
    who = "kids"

    paren_lst = make_lists()
    hyps = make_lists(prims=["O","M", "C"])
    gen = get_hyps_gen(hyps)

    for g in gen:
        l = list(gen[g].keys())
        x = "".join(l)
        if "*" not in x:
            print g, gen[g]

    filt = filter_hyps(copy.deepcopy(gen),
                         thresh=0.5, rem_dup=True)
    print
    for f in filt:
        print f, filt[f]

    alg_names = [x for x in filt]
    alg_types = get_algs_of_type(filt)
    print alg_names


    ##############################################



    ##extract data from files
    careAbout = "Order pressed"
    

    monkey_data = getCountData("stevesdata/RecursionMonkey.csv", 

                                careAbout, "Monkeys",
                                subset={"Exposure": "2"})

    kids_data = getCountData("stevesdata/RecursionKids.csv", 
                                careAbout, "Kids")


    kid_dig = getCountData("stevesdata/RecursionKids.csv", 
                                "FORWARDS DIGITS", "Kids")
    adults_data = getCountData("stevesdata/RecursionAdults.csv", 
                                careAbout, "Adults")
    tsimane_data = getCountData("stevesdata/RecursionTsimane.csv", 
                                careAbout, "Tsimane")

    if who == "kids":
        data_use = kids_data
    elif who == "monkeys":
        data_use = monkey_data
    elif who == "adults":
        data_use = adults_data
    else:
        data_use = tsimane_data
    #################################################

    data_assignments = lst_format_data(paren_lst, data_use)



    data = data_assignments[0]
    assignments = data_assignments[1]

    alg_0 = get_0_columns(format_algs(paren_lst,filt, sm=0.0))

    dat_0 = get_0_columns(data)
    both_0 = list(alg_0.intersection(dat_0))

    paren_lst = np.delete(np.array(paren_lst), both_0)
    algorithms = format_algs(paren_lst,filt, sm=0.05)

    #data = np.delete(data, both_0, axis=1)


    data_assignments = lst_format_data(paren_lst, data_use)#,
                     #kids_data, 
                    # tsimane_data,adults_data)

    data = data_assignments[0]
    assignments = data_assignments[1]
    ce_paren = [list(paren_lst).index("([])"), list(paren_lst).index("[()]")]
    ce_prob = []
    for d in data:
        ce_prob.append((d[ce_paren[0]] + d[ce_paren[1]])/float(sum(d)))

    #data_assignments = lst_format_data(paren_lst,
                   #  kids_data)


    print data
    #algorithms = np.delete(algorithms, both_0, axis=1)
    #algorithms =  algorithms/algorithms.sum(axis=1)[:,None]


    N_GROUPS = 1 #len(groups)
    TOTAL_SAMPLES = np.sum(data)
    TOTAL_PARTS = len(data)
    N_ALGS = len(algorithms)
    N_RESPS = len(algorithms[0])
    #N_ALGS = len()


    print_star("TOTAL_SAMPLES", TOTAL_SAMPLES)
    print_star("N_GROUPS", N_GROUPS)

    print_star("N_ALGS",N_ALGS)
    print_star("TOTAL_PARTS", TOTAL_PARTS)
    print_star("N_RESPS", N_RESPS)




    trace, model_out = model()


    means= model_out['mean']
    sds = model_out['sd']

    ###################################################




    grouped = group_vars(means, ["alpha", "beta", "theta", "noise"])

    alpha = grouped["alpha"]
    beta = grouped["beta"]
    theta = grouped["theta"]
    noise = grouped["noise"]


    grouped_sds = group_vars(sds, ["alpha", "beta", "theta", "noise"])
    alpha_sd = grouped_sds["alpha"]
    beta_sd = grouped_sds["beta"]
    theta_sd = grouped_sds["theta"]
    noise_sd = grouped_sds["noise"]

    noise_names = noise[0]
    noise_vals = noise[1]
    noise_sds = noise_sd[1]
    alpha_names = alpha[0]
    alpha_vals = alpha[1]
    alpha_sds = alpha_sd[1]
    beta_names = np.array(beta[0]).reshape(N_GROUPS, N_ALGS)
    beta_vals = np.array(beta[1]).reshape(N_GROUPS, N_ALGS)
    beta_sds = np.array(beta_sd[1]).reshape(N_GROUPS, N_ALGS)
    theta_names = np.array(theta[0]).reshape(TOTAL_PARTS, N_ALGS)
    theta_vals = np.array(theta[1]).reshape(TOTAL_PARTS, N_ALGS)
    theta_sds = np.array(theta_sd[1]).reshape(TOTAL_PARTS, N_ALGS)

    out_noise = "model_out/noise_full_%s.csv" % who
    out_alpha = "model_out/alpha_full_%s.csv" % who
    out_theta = "model_out/theta_full_%s.csv" % who
    out_beta = "model_out/beta_full_%s.csv" % who

    add_dig = []
    if who == "kids":
        add_dig = kid_dig

    noise = output_full_alpha_noise(trace, 'noise',  
        name=who,  thin=MCMC_THIN, 
        out=out_noise, added=add_dig)
    
    output_full_beta(trace, 'beta',  group=who,
        names=alg_names, thin=MCMC_THIN, out=out_beta)
    output_full_theta(trace, 'theta', group=who,
        names=alg_names, thin=MCMC_THIN, 
                    out=out_theta, added=noise) 
    output_full_alpha_noise(trace, 'alpha',  
       name=who, thin=MCMC_THIN, out=out_alpha)


    recursive_betas = amount_alg_type(alg_types, beta_vals, 
                                    which_type="Recursive")
    crossing_betas = amount_alg_type(alg_types, beta_vals, 
                                    which_type="Crossing")
    tail_betas = amount_alg_type(alg_types, beta_vals, 
                                    which_type="Tail")

    print_star("Noise", noise_vals)
    print_star("Recursive Betas", recursive_betas)
    print_star("Crossing Betas", crossing_betas)
    print_star("Tail Betas", tail_betas)




    ####################################################

    ind = np.arange(len(beta_names[0]))
    fig, ax = plt.subplots()
    ax.bar(ind, beta_vals[0])
    #ax.set_xticks(np.arange(len(alpha_names)))
    ax.set_xticks(ind)
    ax.set_xticklabels(alg_names)
    fig.savefig("beta.png")

    #####################################################


