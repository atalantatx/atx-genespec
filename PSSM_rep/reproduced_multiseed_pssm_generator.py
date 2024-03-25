

import pandas as pd
import statistics as stats
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import math

def read_and_clean(filename, func_cutoff, nonfunc_cutoff):
    
    # Read in sequence data 
    # data_file = "sirna_screen_data_KL25082020.csv" # *****
    data_file = "bdna_data_KL25082020.csv"

    bdna_data = pd.read_csv(data_file)
    #display(bdna_data)
    expr_key = "Average % mRNA Expression Normalized to Untreated (Primary Screen Results)"

    # Rename Columns
    bdna_data.rename(columns={'80mer':'45mer'}, inplace=True)

    gene_names = []
    # Capitalize all Duplex Names
    bdna_data['Duplex Name'] = bdna_data['Duplex Name'].str.upper()

    # Split on underscore to get only gene name
    for duplex in bdna_data['Duplex Name']:
        if duplex.count("_") == 1:
            gene_names.append(duplex.split("_")[0])
        else:
            gene_names.append("_".join(duplex.split("_", 2)[:2]))

    # Sort Gene Name
    bdna_data['Gene Name'] = gene_names
    #print(bdna_data['45mer'])

    # Relevant Columns = Gene Name, 45mer, Std Dev, "Average % mRNA Expression Normalized to Untreated (Primary Screen Results)"

    # Remove sequences with std dev > stdv_cutoff_pcnt of the expression itself
    stdv_cutoff_pcnt = 0.30 # *******

    n = len(bdna_data) # number of sequences before removal
    bdna_data = bdna_data[stdv_cutoff_pcnt*(bdna_data[expr_key]) > bdna_data["Std Dev"]]
    print("Removed sequences with stdev > "+str(int(stdv_cutoff_pcnt*100))+"% of the expression itself",
        "("+str(n-len(bdna_data)),"sequences removed)")
    
    # Removed sequences with Expression % > max_expr_cutoff_pcnt
    max_expr_cutoff_pcnt = 125 # *******

    n = len(bdna_data) # number of sequences before removal
    bdna_data = bdna_data[bdna_data[expr_key] < max_expr_cutoff_pcnt]
    print("Removed sequences with expression > "+str(int(max_expr_cutoff_pcnt))+"%",
        "("+str(n-len(bdna_data)),"sequences removed)")

    bdna_data["label"] = bdna_data[expr_key].apply(lambda x: isfunctional(x, func_cutoff, nonfunc_cutoff))

    #print(bdna_data.loc[bdna_data['Duplex Name'] == "7A_40715"])

    return bdna_data

def some_stats(bdna_data, expr_key):

    import numpy as np
    print(len(bdna_data),"sequences total")
    print("Average Expression (%):",np.mean(bdna_data[expr_key]))
    print("Min Expression (%):",min(bdna_data[expr_key]))
    print("Max Expression (%):",max(bdna_data[expr_key]))
    print("number of genes:",len(list(set(list(bdna_data["Gene Name"])))))

    # Determine number of sequences per gene
    genes = list(set(list(bdna_data["Gene Name"])))

    num_seqs_per_gene = []
    for g in genes:
        num_seqs_per_gene.append(len(bdna_data[bdna_data["Gene Name"]==g]))
        
    print(int(round(np.mean(num_seqs_per_gene),0)),"sequences per gene on average")

    return None

def plot_sirna_expression(bdna_data, expr_key):
    
    # sort by expression %
    bdna_data.sort_values(by=[expr_key],inplace=True)
    bdna_data.reset_index(drop=True,inplace=True) # reindex 


    # plot data
    ax = bdna_data.plot(
        y = expr_key,
    #     x = "gene",
        kind='bar',
        yerr=bdna_data["Std Dev"],
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

def plot_cutoff(bdna_data, expr_key, func_cutoff, nonfunc_cutoff):

    func_col = '#ffb805' # color of functional datapoints ****
    nonfunc_col = '#4287f5' # color of nonfunctional datapoints ****
    mid_col = '#cfcfcf' # color of excluded datapoints ****

    #func_cutoff = 25 # siRNAs with expression %'s LESS than this value will be included in training ******
    #nonfunc_cutoff = 75 # siRNAs with expression %'s GREATER than this value will be included in training ******

    # describe data
    print("number of functional: ",(bdna_data[bdna_data["label"] == "functional"]).shape[0])
    print("number of nonfunctional: ",(bdna_data[bdna_data["label"] == "nonfunctional"]).shape[0])
    print("number of excluded: ",(bdna_data[bdna_data["label"] == "mid"]).shape[0])
    print("total: ",bdna_data.shape[0])
    print("total used in evaluation (# functional + # nonfunctional):",(bdna_data[bdna_data["label"] == "functional"]).shape[0]+(bdna_data[bdna_data["label"] == "nonfunctional"]).shape[0])


    # Plot expression cutoffs
    
    bdna_data.sort_values(by=expr_key,inplace=True)
    # color list to color by value 
    colors=bdna_data[bdna_data["label"] == "functional"].shape[0]*[func_col]+bdna_data[bdna_data["label"] == "mid"].shape[0]*[mid_col]+bdna_data[bdna_data["label"] == "nonfunctional"].shape[0]*[nonfunc_col] 
    ax = bdna_data.plot(y = expr_key,kind='bar', yerr=bdna_data["Std Dev"],
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

def train_test_splitting(bdna_data, random_state):
    
    training_set_size_pcnt = 0.75 # portion of dataset that will be included in training set as a decimal (ex: use 0.75 to represent 75%) ****
    # Introduce a random state for reproducability
    sirna_train_data, sirna_test_data = train_test_split(bdna_data, test_size=1-training_set_size_pcnt, random_state=random_state)
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
    ax = sirna_train_data.plot(y = expr_key,kind='bar', yerr=sirna_train_data["Std Dev"],
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
    ax = sirna_test_data.plot(y = expr_key,kind='bar', yerr=sirna_test_data["Std Dev"],
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
        a = float(x["A"].item())
        u = float(x["U"].item())
        c = float(x["C"].item())
        g = float(x["G"].item())
    #     print(x)
    #     print("a:",a,"c:",c,"g:",g,"u:",u)
        
        
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
        a = float(x["A"].item())
        u = float(x["U"].item())
        c = float(x["C"].item())
        g = float(x["G"].item())
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
    pssm_outfile=f"pssms/pssm_bdna_data_KL25082020_split_seed-{random_state}_pop_seed-{intro_seed}.csv"
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

def compare_to_umass(sirna_test_data, weight_matrix, func_cutoff, nonfunc_cutoff, random_state, intro_seed):

    from sklearn.metrics import precision_recall_curve, auc
    #from sklearn.metrics import plot_precision_recall_curve
    import matplotlib.pyplot as plt

    precision, recall, thresholds = precision_recall_curve( sirna_test_data.activity, 
                                                        sirna_test_data.score,pos_label="functional")

    # Find AUC
    auc_precision_recall = auc(recall, precision)
    print("Area under curve:", auc_precision_recall)

    # Find absolute difference between generated matrix and original
    umass = pd.read_csv("umass_weight_matrix.csv")
    difference = umass - weight_matrix
    abs_base_diff = difference.abs().sum()
    print("Absolute differences of base pairs:\n",abs_base_diff)
    print("Total sum of absolute differences:", abs_base_diff.sum())

    # Plot Curve
    import pylab as pl
    color = "#3498db"
    # label for data (Based on cutoffs used in matrix)
    lab = "< "+str(func_cutoff)+"% | > "+str(nonfunc_cutoff)+"%"
    pl.rcParams["figure.figsize"] = (9,5)
    pl.rcParams.update({'font.size': 12})
    pl.plot(recall,precision, marker='.',linewidth=2, markersize=8,color=color,label = lab)
    pl.ylim([0.0, 1.05])
    pl.xlim([0.0, 1.00])
    pl.xlabel('Sensitivity / Recall')
    pl.ylabel('Positive Predictive Power / Precision')
    pl.ylim([0.0, 1.05])
    pl.xlim([0.0, 1.00])
    pl.title("Algorithm Performance on Test Set w/ Split Seed: "+str(random_state)+" Population Seed: "+str(intro_seed))
    pl.suptitle("Precision Recall Curve")
    pl.legend(title="Cutoffs:",frameon=False)#loc="lower left")

    outfile = f"plots/pr-rc_bdna_data_KL25082020_split_seed-{random_state}_pop_seed-{intro_seed}.png"
    plt.savefig(outfile)

    return None


def main():
    func_cutoff = 25
    nonfunc_cutoff = 75
    expr_key = "Average % mRNA Expression Normalized to Untreated (Primary Screen Results)"
    filename = "bdna_data_KL25082020.csv"
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
        #compare_to_umass(sirna_test_data, weight_matrix, func_cutoff, nonfunc_cutoff, random_state=random_state, intro_seed=seed)

if __name__ == "__main__":
    main()
