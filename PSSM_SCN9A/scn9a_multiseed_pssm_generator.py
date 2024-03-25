

import pandas as pd
import statistics as stats
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import math

def read_and_clean(filename, func_cutoff, nonfunc_cutoff):
    
    # Read in sequence data 
    # data_file = "sirna_screen_data_KL25082020.csv" # *****
    data_file = "All_data_2023-12-05_scaled_cleaned_SCN9A.csv"

    scn9a_data = pd.read_csv(data_file)
    #display(scn9a_data)
    expr_key = "Scaled_%UNT"

    # Rename Columns
    scn9a_data.rename(columns={'45mer - 14+21+10':'45mer'}, inplace=True)

    scn9a_data['Gene Name'] = scn9a_data['Gene']

    # Relevant Columns = Gene Name, 45mer, STDEV, "Scaled_%UNT"

    # Remove sequences with STDEV > stdv_cutoff_pcnt of the expression itself
    stdv_cutoff_pcnt = 0.30 # *******

    n = len(scn9a_data) # number of sequences before removal
    scn9a_data = scn9a_data[stdv_cutoff_pcnt*(scn9a_data[expr_key]) > scn9a_data["STDEV"]]
    print("Removed sequences with stdev > "+str(int(stdv_cutoff_pcnt*100))+"% of the expression itself",
        "("+str(n-len(scn9a_data)),"sequences removed)")
    
    # Removed sequences with Expression % > max_expr_cutoff_pcnt
    max_expr_cutoff_pcnt = 125 # *******

    n = len(scn9a_data) # number of sequences before removal
    scn9a_data = scn9a_data[scn9a_data[expr_key] < max_expr_cutoff_pcnt]
    print("Removed sequences with expression > "+str(int(max_expr_cutoff_pcnt))+"%",
        "("+str(n-len(scn9a_data)),"sequences removed)")

    scn9a_data["label"] = scn9a_data[expr_key].apply(lambda x: isfunctional(x, func_cutoff, nonfunc_cutoff))

    return scn9a_data

def some_stats(scn9a_data, expr_key):

    import numpy as np
    print(len(scn9a_data),"sequences total")
    print("Average Expression (%):",np.mean(scn9a_data[expr_key]))
    print("Min Expression (%):",min(scn9a_data[expr_key]))
    print("Max Expression (%):",max(scn9a_data[expr_key]))
    print("number of genes:",len(list(set(list(scn9a_data["Gene Name"])))))

    # Determine number of sequences per gene
    genes = list(set(list(scn9a_data["Gene Name"])))

    num_seqs_per_gene = []
    for g in genes:
        num_seqs_per_gene.append(len(scn9a_data[scn9a_data["Gene Name"]==g]))
        
    print(int(round(np.mean(num_seqs_per_gene),0)),"sequences per gene on average")

    return None

def plot_sirna_expression(scn9a_data, expr_key):
    
    # sort by expression %
    scn9a_data.sort_values(by=[expr_key],inplace=True)
    scn9a_data.reset_index(drop=True,inplace=True) # reindex 


    # plot data
    ax = scn9a_data.plot(
        y = expr_key,
    #     x = "gene",
        kind='bar',
        yerr=scn9a_data["STDEV"],
        legend=False,
        figsize=(15,6),
        fontsize = 12,
        color="gray"
    )
    plt.title("Raw Data",fontsize = 12)
    ax.xaxis.set_visible(False)# remove x-axis
    # set title and axis labels
    ax.set_ylabel("Target Expression (%)",fontsize = 12)
    # plt.show()
    return None

def isfunctional(x, func_cutoff, nonfunc_cutoff):
    #func_cutoff = 25 # siRNAs with expression %'s LESS than this value will be included in training ******
    #nonfunc_cutoff = 75 # siRNAs with expression %'s GREATER than this value will be included in training ******
    if x<func_cutoff:
        return "functional"
    elif x>nonfunc_cutoff:
        return "nonfunctional"
    else:
        return "mid"

def plot_cutoff(scn9a_data, expr_key, func_cutoff, nonfunc_cutoff):

    func_col = '#ffb805' # color of functional datapoints ****
    nonfunc_col = '#4287f5' # color of nonfunctional datapoints ****
    mid_col = '#cfcfcf' # color of excluded datapoints ****

    #func_cutoff = 25 # siRNAs with expression %'s LESS than this value will be included in training ******
    #nonfunc_cutoff = 75 # siRNAs with expression %'s GREATER than this value will be included in training ******

    # describe data
    print("number of functional: ",(scn9a_data[scn9a_data["label"] == "functional"]).shape[0])
    print("number of nonfunctional: ",(scn9a_data[scn9a_data["label"] == "nonfunctional"]).shape[0])
    print("number of excluded: ",(scn9a_data[scn9a_data["label"] == "mid"]).shape[0])
    print("total: ",scn9a_data.shape[0])
    print("total used in evaluation (# functional + # nonfunctional):",(scn9a_data[scn9a_data["label"] == "functional"]).shape[0]+(scn9a_data[scn9a_data["label"] == "nonfunctional"]).shape[0])


    # Plot expression cutoffs
    
    scn9a_data.sort_values(by=expr_key,inplace=True)
    # color list to color by value 
    colors=scn9a_data[scn9a_data["label"] == "functional"].shape[0]*[func_col]+scn9a_data[scn9a_data["label"] == "mid"].shape[0]*[mid_col]+scn9a_data[scn9a_data["label"] == "nonfunctional"].shape[0]*[nonfunc_col] 
    ax = scn9a_data.plot(y = expr_key,kind='bar', yerr=scn9a_data["STDEV"],
                        legend=False,figsize=(8,6),fontsize = 12,
                        # color by value
                        color= colors 
                        )
    # set title and axis labels
    plt.title("Expression Cutoffs "+
            "\n Functional: <"+str(func_cutoff)+"%"+
            "\n Nonfunctional: >"+str(nonfunc_cutoff)+"%",
            fontsize = 12)
    ax.xaxis.set_visible(False)# remove x-axis
    ax.set_ylabel("Target Expression (%)",fontsize = 12)
    plt.show()

    return None

def train_test_splitting(scn9a_data, random_state):
    
    training_set_size_pcnt = 0.75 # portion of dataset that will be included in training set as a decimal (ex: use 0.75 to represent 75%) ****
    # Introduce a random state for reproducability
    sirna_train_data, sirna_test_data = train_test_split(scn9a_data, test_size=1-training_set_size_pcnt, random_state=random_state)
    sirna_test_data.reset_index(drop=True, inplace=True)
    #print(sirna_test_data)
    return sirna_train_data, sirna_test_data

def plot_training_cutoff(sirna_train_data, expr_key, func_cutoff, nonfunc_cutoff):
    func_col = '#ffb805' # color of functional datapoints ****
    nonfunc_col = '#4287f5' # color of nonfunctional datapoints ****
    mid_col = '#cfcfcf' # color of excluded datapoints ****
    #func_cutoff = 25 # siRNAs with expression %'s LESS than this value will be included in training ******
    #nonfunc_cutoff = 75 # siRNAs with expression %'s GREATER than this value will be included in training ******


    # describe data
    print("Training Set:")
    print("number of functional: ",(sirna_train_data[sirna_train_data["label"] == "functional"]).shape[0])
    print("number of nonfunctional: ",(sirna_train_data[sirna_train_data["label"] == "nonfunctional"]).shape[0])
    print("number of excluded: ",(sirna_train_data[sirna_train_data["label"] == "mid"]).shape[0])
    print("total: ",sirna_train_data.shape[0])
    print("total used in evaluation (# functional + # nonfunctional):",(sirna_train_data[sirna_train_data["label"] == "functional"]).shape[0]+(sirna_train_data[sirna_train_data["label"] == "nonfunctional"]).shape[0])


    # Plot expression cutoffs
    import matplotlib.pyplot as plt
    sirna_train_data = sirna_train_data.sort_values(by=expr_key)
    # color list to color by value 
    colors=sirna_train_data[sirna_train_data["label"] == "functional"].shape[0]*[func_col]+sirna_train_data[sirna_train_data["label"] == "mid"].shape[0]*[mid_col]+sirna_train_data[sirna_train_data["label"] == "nonfunctional"].shape[0]*[nonfunc_col] 
    ax = sirna_train_data.plot(y = expr_key,kind='bar', yerr=sirna_train_data["STDEV"],
                        legend=False,figsize=(8,6),fontsize = 12,
                        # color by value
                        color= colors 
                        )
    # set title and axis labels
    plt.title("Training Set \nExpression Cutoffs "+
            "\n Functional: <"+str(func_cutoff)+"%"+
            "\n Nonfunctional: >"+str(nonfunc_cutoff)+"%",
            fontsize = 12)
    ax.xaxis.set_visible(False)# remove x-axis
    ax.set_ylabel("Target Expression (%)",fontsize = 12)
    plt.show()

    return None



def plot_test_cutoff(sirna_test_data, expr_key, func_cutoff, nonfunc_cutoff):
    func_col = '#ffb805' # color of functional datapoints ****
    nonfunc_col = '#4287f5' # color of nonfunctional datapoints ****
    mid_col = '#cfcfcf' # color of excluded datapoints ****
    #func_cutoff = 25 # siRNAs with expression %'s LESS than this value will be included in training ******
    #nonfunc_cutoff = 75 # siRNAs with expression %'s GREATER than this value will be included in training ******


    # describe data
    print("Test Set:")
    print("number of functional: ",(sirna_test_data[sirna_test_data["label"] == "functional"]).shape[0])
    print("number of nonfunctional: ",(sirna_test_data[sirna_test_data["label"] == "nonfunctional"]).shape[0])
    print("number of excluded: ",(sirna_test_data[sirna_test_data["label"] == "mid"]).shape[0])
    print("total: ",sirna_test_data.shape[0])
    print("total used in evaluation (# functional + # nonfunctional):",(sirna_test_data[sirna_test_data["label"] == "functional"]).shape[0]+(sirna_test_data[sirna_test_data["label"] == "nonfunctional"]).shape[0])


    # Plot expression cutoffs
    import matplotlib.pyplot as plt
    sirna_test_data = sirna_test_data.sort_values(by=expr_key)
    # color list to color by value 
    colors=sirna_test_data[sirna_test_data["label"] == "functional"].shape[0]*[func_col]+sirna_test_data[sirna_test_data["label"] == "mid"].shape[0]*[mid_col]+sirna_test_data[sirna_test_data["label"] == "nonfunctional"].shape[0]*[nonfunc_col] 
    ax = sirna_test_data.plot(y = expr_key,kind='bar', yerr=sirna_test_data["STDEV"],
                        legend=False,figsize=(8,6),fontsize = 12,
                        # color by value
                        color= colors 
                        )
    # set title and axis labels
    plt.title("Test Set \nExpression Cutoffs "+
            "\n Functional: <"+str(func_cutoff)+"%"+
            "\n Nonfunctional: >"+str(nonfunc_cutoff)+"%",
            fontsize = 12)
    ax.xaxis.set_visible(False)# remove x-axis
    ax.set_ylabel("Target Expression (%)",fontsize = 12)
    plt.show()

    return None


def train_weight_matrix(sirna_train_data):

    train_seqs_all = pd.DataFrame([list(x) for x in sirna_train_data["45mer"]]) 
    train_freqs_all = []
    for i in range(0,train_seqs_all.columns[-1]+1):
        x = pd.DataFrame(train_seqs_all[i].value_counts(sort=False)).transpose()
        a = float(x["A"].item())
        u = float(x["U"].item())
        c = float(x["C"].item())
        g = float(x["G"].item())
    #     print(x)
    #     print("a:",a,"c:",c,"g:",g,"u:",u)
        train_freqs_all.append([a/len(train_seqs_all),u/len(train_seqs_all),c/len(train_seqs_all),g/len(train_seqs_all)])
    # convert train_freqs_all from list to data_frame
    train_freqs_all = pd.DataFrame(train_freqs_all,columns=["A","U","C","G"])
    print("All Training Data Frequencies:", train_freqs_all)

    return train_freqs_all

def func_nonfunc_subtracted(sirna_train_data):

    # Get frequencies per position for functional sequences
    train_seqs_func = sirna_train_data[sirna_train_data["label"] == "functional"]["45mer"]

    # split into individual columns
    train_seqs_func = pd.DataFrame([list(x) for x in train_seqs_func])
    train_freqs_func = []
    for i in range(0,train_seqs_func.columns[-1]+1):
        x = pd.DataFrame(train_seqs_func[i].value_counts(sort=False)).transpose()
        if 'A' in x.columns:
            a = float(x["A"].item())
        else:
            a = 0
        if 'U' in x.columns:
            u = float(x["U"].item())
        else:
            u = 0
        if 'C' in x.columns:
            c = float(x["C"].item())
        else:
            c = 0
        if 'G' in x.columns:
            g = float(x["G"].item())
        else:
            g = 0
        #print(x)
        #print("a:",a,"c:",c,"g:",g,"u:",u)
        
        
        train_freqs_func.append([a/len(train_seqs_func),u/len(train_seqs_func),c/len(train_seqs_func),g/len(train_seqs_func)])

    # convert train_freqs_func from list to data_frame
    train_freqs_func = pd.DataFrame(train_freqs_func,columns=["A","U","C","G"])


    # Get frequences per position for nonfuncional sequences
    train_seqs_nonfunc = sirna_train_data[sirna_train_data["label"] == "nonfunctional"]["45mer"] #NOTE: 45mer is actually 45mer 
    # split into individual columns
    train_seqs_nonfunc = pd.DataFrame([list(x) for x in train_seqs_nonfunc])
    train_freqs_nonfunc = []
    for i in range(0,train_seqs_nonfunc.columns[-1]+1):
        x = pd.DataFrame(train_seqs_nonfunc[i].value_counts(sort=False)).transpose()
        if 'A' in x.columns:
            a = float(x["A"].item())
        else:
            a = 0
        if 'U' in x.columns:
            u = float(x["U"].item())
        else:
            u = 0
        if 'C' in x.columns:
            c = float(x["C"].item())
        else:
            c = 0
        if 'G' in x.columns:
            g = float(x["G"].item())
        else:
            g = 0
    #     print(x)
    #     print("a:",a,"c:",c,"g:",g,"u:",u)
        
        train_freqs_nonfunc.append([a/len(train_seqs_nonfunc),u/len(train_seqs_nonfunc),c/len(train_seqs_nonfunc),g/len(train_seqs_nonfunc)])
    # convert train_freqs_nonfunc from list to data_frame
    train_freqs_nonfunc = pd.DataFrame(train_freqs_nonfunc,columns=["A","U","C","G"])

    # Subtract nonfuncional frequencies from funcional frequencies
    freqs_subtracted = train_freqs_func - train_freqs_nonfunc
    print("Functionl and Nonfunctional Frequencies subtracted:")
    print(freqs_subtracted)
    print(len(sirna_train_data[sirna_train_data["label"]=="nonfunctional"]))

    return freqs_subtracted

def gen_rand_seq(sirna_train_data, train_freqs_all, intro_seed):

    iter_num_func = len(sirna_train_data[sirna_train_data["label"]=="functional"])# iteration number (should the number of functional sequences in training dataset) ***
    iter_num_nonfunc = len(sirna_train_data[sirna_train_data["label"]=="nonfunctional"])# iteration number (should the number of nonfunctional sequences in training dataset) ***

    import numpy as np

    # Generate random sequences using the per position base preferences of the entire dataset as the probabilities
    def random_seq_gen(freqs,n): #freqs Dataframe of frequencies of A,U,C,G  
        '''Returns a list of n random sequences generated with the given per position frequency propbabilities (freq)'''
        BASES = ("A","U","C","G")
        l = len(freqs)
        seq_ls = []
        for i in range(0,n):
            seq = ''
            for k in range(0,l):
                # get position frequency for given positon (P)
                P = list(freqs.loc[k])
                # generate random sequence of length l given a per position frequency P
                seq+=''.join(np.random.choice(BASES, p=P)) 
                k+=1
            seq_ls.append(seq)
        
        return seq_ls

    def compute_counts(seq_ls):
        ''' Compute base counts per position and return dataframe of per position base counts'''
        seq_ls_df = pd.DataFrame([list(x) for x in seq_ls])
        total_A = []
        total_U = []
        total_C = []
        total_G = []
        for i in range(len(seq_ls_df.columns)): # loop through columns
            a = (seq_ls_df[i][seq_ls_df[i] == "A"].count())
            u = (seq_ls_df[i][seq_ls_df[i] == "U"].count())
            c = (seq_ls_df[i][seq_ls_df[i] == "C"].count())
            g = (seq_ls_df[i][seq_ls_df[i] == "G"].count())
            # a,c,g,u = seq_ls_df[i].value_counts(sort=False) # does not work if one or more bases is not present at that possition
            total_A.append(a)
            total_U.append(u)
            total_G.append(g)
            total_C.append(c)
        ct_df = pd.DataFrame([total_A,total_U,total_G,total_C]).transpose()
        ct_df.columns = ["A","U","C","G"]
        return ct_df



    def compute_stats(fn,ct_df_ls):
        '''Computes the provided numpy stats function fn (np.mean/np.median/np.std) from a list of per position base count dataframes'''
        stat_df = pd.DataFrame(
            ([fn([df["A"][j] for df in cts_df_ls_func]) for j in range(0,len(ct_df_ls[0]))], # Extract Column A, row j and computes the mean, loops through each dataframe in the list and through each column in each dataframe
            [fn([df["U"][j] for df in cts_df_ls_func]) for j in range(0,len(ct_df_ls[0]))], # Extract Column U, row j and computes the mean, loops through each dataframe in the list and through each column in each dataframe
            [fn([df["C"][j] for df in cts_df_ls_func]) for j in range(0,len(ct_df_ls[0]))], # Extract Column C, row j and computes the mean, loops through each dataframe in the list and through each column in each dataframe
            [fn([df["G"][j] for df in cts_df_ls_func]) for j in range(0,len(ct_df_ls[0]))]) # Extract Column G, row j and computes the mean, loops through each dataframe in the list and through each column in each dataframe
        ).transpose()
        stat_df.columns =["A","U","C","G"]
        return stat_df


    # Generate Random Sequences corresponding to the number of functional sequences
    cts_df_ls_func = [] 

    # set random seed for reproducability
    
    np.random.seed(intro_seed) 
        
    for i in range(0,iter_num_func):
        cts_df_ls_func.append(compute_counts(random_seq_gen(train_freqs_all,iter_num_func)))
    print("Random sequence generation for Functional sequences complete, generated",iter_num_func*iter_num_func,"sequences")

    # compute statistics
    # mean_df_func = compute_stats(np.mean, cts_df_ls_func)
    median_df_func = compute_stats(np.median, cts_df_ls_func)
    std_df_func = compute_stats(np.std, cts_df_ls_func)

    # Generate Random Sequences corresponding to the number of nonfunctional sequences
    cts_df_ls_nonfunc = []
    for i in range(0,iter_num_func):
        cts_df_ls_nonfunc.append(compute_counts(random_seq_gen(train_freqs_all,iter_num_nonfunc)))
    print("Random sequence generation for Nonfunctional sequences complete, generated",iter_num_nonfunc*iter_num_nonfunc,"sequences")

    # compute statistics
    # mean_df_nonfunc = compute_stats(np.mean, cts_df_ls_nonfunc)
    median_df_nonfunc = compute_stats(np.median, cts_df_ls_nonfunc)
    std_df_nonfunc = compute_stats(np.std, cts_df_ls_nonfunc)

    return median_df_func, median_df_nonfunc, std_df_func, std_df_nonfunc, iter_num_nonfunc, iter_num_func


def compute_test_statistic(md_f,md_nf,std_f,std_nf,n_f,n_nf):
    return ((md_nf/n_nf)-(md_f/n_f))/math.sqrt( (((std_nf**2)/(n_nf**2))/n_nf) + (((std_f**2)/(n_f**2))/n_f))

def gen_test_stats(median_df_func, median_df_nonfunc, std_df_func, std_df_nonfunc, iter_num_nonfunc, iter_num_func):

    # generate test statistics (one per base per position)
    a_ls=[]
    u_ls=[]
    c_ls=[]
    g_ls=[]
    # loop through each position
    for i in range(0,len(median_df_func)):
        a_ls.append(compute_test_statistic(median_df_func["A"][i],median_df_nonfunc["A"][i],std_df_func["A"][i],std_df_nonfunc["A"][i],iter_num_func,iter_num_nonfunc))
        u_ls.append(compute_test_statistic(median_df_func["U"][i],median_df_nonfunc["U"][i],std_df_func["U"][i],std_df_nonfunc["U"][i],iter_num_func,iter_num_nonfunc))
        c_ls.append(compute_test_statistic(median_df_func["C"][i],median_df_nonfunc["C"][i],std_df_func["C"][i],std_df_nonfunc["C"][i],iter_num_func,iter_num_nonfunc))
        g_ls.append(compute_test_statistic(median_df_func["G"][i],median_df_nonfunc["G"][i],std_df_func["G"][i],std_df_nonfunc["G"][i],iter_num_func,iter_num_nonfunc))
    test_stats_df = pd.DataFrame((a_ls,u_ls,c_ls,g_ls)).transpose()
    test_stats_df.columns =["A","U","C","G"]

    return test_stats_df

def gen_p_values(test_stats_df, iter_num_func, iter_num_nonfunc):

    # generate p-values from test statistics
    from scipy.stats import t
    dof = iter_num_func+iter_num_nonfunc-2 # degrees of freedom

    def compute_p_value(test_stat,dof):
        return 2*t.cdf(-1*abs(test_stat),dof) # distribution function

    # generate test statistics (one per base per position)
    a_ls=[]
    u_ls=[]
    c_ls=[]
    g_ls=[]
    # loop through each position
    for i in range(0,len(test_stats_df)):
        a_ls.append(compute_p_value(test_stats_df["A"][i],dof))
        u_ls.append(compute_p_value(test_stats_df["U"][i],dof))
        c_ls.append(compute_p_value(test_stats_df["C"][i],dof))
        g_ls.append(compute_p_value(test_stats_df["G"][i],dof))
    p_vals_df = pd.DataFrame((a_ls,u_ls,c_ls,g_ls)).transpose()
    p_vals_df.columns =["A","U","C","G"]

    return p_vals_df

def build_weight_matrix(p_vals_df, freqs_subtracted, random_state, intro_seed):
    
    p_value_cutoff = 1e-5 # ****
    #  Bonferroni correction for multiple comparisons
    p_value_cutoff = p_value_cutoff**4 

    p_value_data = p_vals_df
    freq_sub_data = freqs_subtracted

    # for each element in p_value_data, if < than p_value_cutoff set to 1, else set to 0
    def convert_p_val_cutoff(x):
        if(x<p_value_cutoff):
            return 1
        else:
            return 0

    p_value_data = p_value_data.map(convert_p_val_cutoff)

    # convert freq subtracted values to weights
    def convert_sub_to_freq(x):
        return(int(x*100))

    freq_sub_data = freq_sub_data.map(convert_sub_to_freq)


    # multiply the dataframes to get the final weight matrix

    weight_matrix = pd.DataFrame(freq_sub_data.values*p_value_data.values, columns=freq_sub_data.columns, index=freq_sub_data.index)



    out_weight_matrix = pd.DataFrame(weight_matrix)
    out_weight_matrix["split_seed"] = random_state
    out_weight_matrix["pop_seed"] = intro_seed
    out_weight_matrix["position"] = (out_weight_matrix.index + 1)

    print("Weight Matrix:")
    print(out_weight_matrix)

    # save weight matrix to file
    pssm_outfile=f"pssms/pssm_scn9a_data_2023-12-05_split_seed-{random_state}_pop_seed-{intro_seed}.csv"
    out_weight_matrix.to_csv(pssm_outfile,index=False)
    print("Weight Matrix file saved to:",pssm_outfile)

    return weight_matrix

def assess_performance(expr_key, sirna_test_data, func_cutoff, weight_matrix):

    def score_sequences(df,weight_matrix):
        # add new column to df for scores
        df["score"] = -999 # initialize to score -999
        seqs = list(df["45mer"])

        # convert weight_matrix to list of lists (4 elements per list - 45 elements in outer list)
        weights = [list(x) for x in weight_matrix.to_numpy()]
        
        score_ls = []
        j=0 # keeps track of seqs
        while(j<len(seqs)):
            x=seqs[j]
            seq_ls = list(x)
            score = 0
            i = 0 # keeps track of positions in 45mer

            while i<45:
                b = seq_ls[i]
                pos_score = 0
                if(b == "A"):
                    pos_score=weights[i][0]
                elif(b == "U" or b == "T"):
                    pos_score=weights[i][1]
                elif(b == "C"):
                    pos_score=weights[i][2]
                elif(b == "G"):
                    pos_score=weights[i][3]
                score+=pos_score
                i+=1

            df.loc[j, "score"] = score
                
            j+=1
            score_ls.append([x,score])

        return df

    # Determine actual activity of sequences in test set
    def get_activity(df,func_cutoff):
        # add new column to df for functionality
        df["activity"] = "" # initialize to empty string
        exprs = list(df[expr_key])
        j=0 # keeps track of each siRNA
        while(j<len(df)):
            func = "X"
            x=exprs[j]

            if x < func_cutoff:
                func = "functional"
            else:
                func = "nonfunctional"

            df.loc[j, "activity"] = func
            j+=1

        return df


    # Compute actual activity of the test sequences (based on the chosen functional cutoff %)
    get_activity(sirna_test_data,func_cutoff)
    # Score test set data with new weight matrix
    score_sequences(sirna_test_data,weight_matrix)    

    return sirna_test_data


def main():
    func_cutoff = 30
    nonfunc_cutoff = 70
    expr_key = "Scaled_%UNT"
    filename = "All_data_2023-12-05_scaled_cleaned_SCN9A.csv"
    cleaned = read_and_clean(filename, func_cutoff, nonfunc_cutoff)
    some_stats(cleaned, expr_key)
    #plot_sirna_expression(cleaned, expr_key)
    #plot_cutoff(cleaned, expr_key, func_cutoff, nonfunc_cutoff)
    
    
    # SELECT SEED AND QUANTITY OF RANDOM STATE FOR REPRODUCABILITY:
    seed = 5
    random_states = 3
    for random_state in range(random_states):

        sirna_train_data, sirna_test_data = train_test_splitting(cleaned, random_state=random_state)
        #plot_training_cutoff(sirna_train_data, expr_key, func_cutoff, nonfunc_cutoff)
        #plot_test_cutoff(sirna_test_data, expr_key, func_cutoff, nonfunc_cutoff)
        train_freqs_all = train_weight_matrix(sirna_train_data)
        freqs_subtracted = func_nonfunc_subtracted(sirna_train_data)
        median_df_func, median_df_nonfunc, std_df_func, std_df_nonfunc, iter_num_nonfunc, iter_num_func = gen_rand_seq(sirna_train_data, train_freqs_all, intro_seed=seed)
        test_stats_df = gen_test_stats(median_df_func, median_df_nonfunc, std_df_func, std_df_nonfunc, iter_num_nonfunc, iter_num_func)
        p_vals_df = gen_p_values(test_stats_df, iter_num_func, iter_num_nonfunc)
        weight_matrix = build_weight_matrix(p_vals_df, freqs_subtracted, random_state=random_state, intro_seed=seed)
        scored_sirna_test_data = assess_performance(expr_key, sirna_test_data, func_cutoff, weight_matrix)
        print(scored_sirna_test_data)

if __name__ == "__main__":
    main()
