#!/usr/bin/env python
# %%
import pandas as pd
import xgboost as xgb
import numpy as np
# import seaborn as sns
import optuna
from optuna.samplers import TPESampler, RandomSampler
import pickle, os, yaml
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, matthews_corrcoef, confusion_matrix
from scipy.stats import pearsonr, linregress
# from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import Lasso, LassoCV, Ridge, RidgeCV,  ElasticNet, ElasticNetCV
from sklearn.preprocessing import MinMaxScaler

with open('scn9a_params.yml', 'r') as file:
    doc = yaml.safe_load(file)

sel_target = doc["sel_target"]
n_trials = doc["n_trials"]
output_path = doc["output_path"] #"../outputs/RandomForestRegressor/FeatureSet4/random_split/"

os.makedirs(output_path, exist_ok=True)

activity_cols = ["Oligo_ID", "Gene Species", "Scaled", "Train-Test-Split", "Screen"]

# %%
feature_columns = [line.strip() for line in open(doc["featset_columns"]).readlines()]
dataset = pd.read_parquet(doc["features"])
dataset["Oligo_ID"] = dataset["Oligo_ID"].str.replace(
    "PRNP-Human|PRNP-Mouse", "PRNP", regex=True
)
dataset["Gene"] = dataset["Gene"].str.replace(
    "PRNP-Human|PRNP-Mouse", "PRNP", regex=True
)
print(sum(dataset.columns.isin(["Gene"])))


scaled_data = pd.read_csv(doc["dataset"])
scaled_data["Scaled"] = scaled_data["scaled"]
merge_dataset = dataset.merge(
    scaled_data, on=["Oligo_ID", "Gene"], how="inner", copy=True
)

print(merge_dataset.shape)

# %%

# random_state = doc["random_state"]
train_frac = doc["train_frac"]

def trend_plot(y_preds, y_test_p, y_active, title, outfile):
    # m, b = np.polyfit(y_preds, y_test, 1)
    if len(np.unique(y_preds)) >1:
        plt.figure()
        plt.rcdefaults()

        scaler = MinMaxScaler()
        # print(y_test)
        y_preds = scaler.fit_transform(y_preds.reshape(-1, 1))[:,0]
        # print(y_preds)

        linear_fit = linregress(y_preds, y_test_p)
        slope = linear_fit.slope
        pvalue = linear_fit.pvalue
        intercept = linear_fit.intercept

        pcc_res = pearsonr(y_preds, y_test_p)
        scale_val = 0.75
        plt.figure(figsize=(16*scale_val, 9*scale_val))
        plt.tight_layout()

        dot_colors = {
            "0":"red",
            "1": "blue",
            "999":"green"
        }

        fig, ax = plt.subplots()
        fig.set_size_inches(16*scale_val, 9*scale_val)
        for g in np.unique(y_active):
            ix = np.where(y_active == g)
            ax.scatter(x=y_preds[ix], y=y_test_p[ix],c=dot_colors[str(g)],label=g)
        ax.legend(loc='center right',bbox_to_anchor=(1, 0.5))
        if max(y_preds) > 1.8:
            plt.xlim(-0.1, 1.1)
            text_x = -190
        else:
            plt.xlim(min(y_preds) * 0.9, max(y_preds) * 1.1)
            text_x = min(y_preds)
            text_y = max(y_test_p)
        textstr = "\n".join(
            [
                "Reg Line: y = {:.2f}x + {:.2f}".format(slope, intercept),
                "P-value: " + str(round(pvalue, 3)),
                "corr: " + str(round(pcc_res.statistic, 3)),
            ]
        )
        props = dict(boxstyle="round", facecolor="wheat", alpha=0.25)

        plt.ylim(-0.1, 1.4)
        plt.xlim(-0.1, 1.2)
        plt.plot(y_preds, intercept + slope * y_preds, color="black")

        # place a text box in upper left in axes coords
        plt.text(
            text_x, text_y, textstr, fontsize=14, verticalalignment="bottom", bbox=props
        )
        
        plt.title(title)
        plt.xlabel("Predictions")
        plt.ylabel("Actual")
        plt.legend(loc='upper left')
        plt.savefig(outfile)
        plt.close()

scores_dataframes = []
test_set_outs = []

for i in range(doc["boostrap_reps"]):
    # Random State
    random_state = np.random.randint(low=1, high=1000,)

    # Gene and non-gene data
    gene_data = pd.DataFrame(merge_dataset.loc[merge_dataset["Gene"] == sel_target])
    non_gene_data = pd.DataFrame(merge_dataset.loc[merge_dataset["Gene"].isin(doc["add_genes"])])
    
    # Active Assignment for gene and non-gene data
    threshold = doc["threshold"]
    gene_data["Active"] = 999
    # print(gene_data["Scaled"].quantile(0.25))
    # print(gene_data["Scaled"].quantile(0.75))
    active_th = gene_data["Scaled"].quantile(threshold)
    inactive_th = gene_data["Scaled"].quantile(1-threshold)
    gene_data.loc[gene_data["Scaled"]<=active_th,"Active"] = 1
    gene_data.loc[gene_data["Scaled"]>=inactive_th,"Active"] = 0
    
    active_th = non_gene_data["Scaled"].quantile(threshold)
    inactive_th = non_gene_data["Scaled"].quantile(1-threshold)
    non_gene_data["Active"] = 999
    non_gene_data.loc[non_gene_data["Scaled"]<=active_th,"Active"] = 1
    non_gene_data.loc[non_gene_data["Scaled"]>=inactive_th,"Active"] = 0
    
    # Initial Train
    gene_train = pd.DataFrame(gene_data.sample(frac=train_frac, random_state=random_state))
    print(gene_train["Active"].value_counts())
    # print(non_gene_data["Active"].value_counts())
    # continue

    # test = pd.read_csv("../dataset/tiling_reg_dataset.csv")
    # optimize = pd.read_csv("../dataset/primary_reg_dataset.csv")
    # train = pd.DataFrame(optimize[train.columns.to_list()])

    # # train = pd.concat([train,optimize],join="inner")
    non_train = pd.DataFrame(gene_data[~gene_data["Oligo_ID"].isin(gene_train["Oligo_ID"])])
    optimize = pd.DataFrame(non_train.sample(frac=0.5, random_state=random_state))
    test = pd.DataFrame(non_train[~non_train["Oligo_ID"].isin(optimize["Oligo_ID"])])

    train = pd.concat([gene_train,non_gene_data])

    # %%
    print("Num of Train", train.shape[0])
    print("Num of Optimize", optimize.shape[0])
    print("Num of Test", test.shape[0])
    print("Num of Test in Train", sum(test["Oligo_ID"].isin(train["Oligo_ID"])))
    print("Num of Test in Optimize", sum(optimize["Oligo_ID"].isin(test["Oligo_ID"])))
    print("Num of Optimize in Train", sum(optimize["Oligo_ID"].isin(train["Oligo_ID"])))


    # %%
    train["Scaled"] = train["Scaled"]

    # %%
    train["Oligo_ID"].str.split("_").str[0].unique()

    # %%
    print(optimize.shape)
    print(train.shape)
    print(test.shape)
    print(test.shape[0] + optimize.shape[0])

    # %%


    # %%
    train[train["Gene"] == sel_target]

    # sum(train["Oligo_ID"].isin(optimize["Oligo_ID"]) )


    # %%
    df = pd.DataFrame()
    df["ID_match"] = test["Oligo_ID"].isin(train["Oligo_ID"]).any()
    df


    # trend_plot(optimize_preds, y_opt, title=study.study_name+' '+str(n_trials)+" trials_training", outfile="../outputs/RandomForestRegressor/plots_scn9a_splits/"+"pt_scatter_optimize_"+study.study_name+"_trials_"+str(n_trials))

    # %%
    test

    available_features = list(set(feature_columns).intersection(train.columns))
    # %%
    X_train        = train.loc[train["Active"]!=800,available_features]
    y_train        = train.loc[train["Active"]!=800,"Active"]
    y_train_scaled = train.loc[train["Active"]!=800,"Scaled"]
    X_opt = optimize[available_features]
    y_opt = optimize["Active"]
    y_opt_scaled = optimize["Scaled"]
    X_test = test[available_features]
    y_test = test["Active"]
    y_test_scaled = test["Scaled"]
    print(X_train.shape)
    print(X_opt.shape)
    print(X_test.shape)
    # n_trials = 5
    # Create the Optuna study
    # study = create_study(direction='minimize', study_name=gene_species)
    model_path = f"{output_path}/model"
    model_filename = f"{model_path}_{sel_target}_pt_xgbr_trail_{n_trials}_featset1_orig.pkl"
    # exit()

    # %%
    plt.hist(y_opt, density=True)
    plt.hist(y_test, density=True, alpha=0.5)


    # Define the objective function
    def objective(trial):
        params = {
            # "n_estimators": trial.suggest_int("n_estimators", 1, 1000),
            # "alpha" : trial.suggest_float("alpha",0,1),
            "l1_ratio" : trial.suggest_float("l1_ratio",0,1),
            "eps" : trial.suggest_float("eps",0,1),
        }

        model = ElasticNetCV(**params,n_jobs=16,max_iter=10000)

        model.fit(X_train, y_train_scaled)

        y_pred = model.predict(X_opt)
        # y_pred_prob = model.predict_proba(X_opt)

        # y_pred = model.predict(X_train)
        # scaler  = MinMaxScaler()
        # y_pred = scaler.fit_transform(pred.reshape(-1, 1))

        # mse = mean_squared_error(y_opt, y_pred)
        mae = mean_absolute_error(y_opt_scaled, y_pred)

        # r2score = r2_score(y_opt, y_pred)
        # r2_score_val = r2_score(y_opt, y_pred)
        # y_pred = model.predict(X_test)
        mse = mean_squared_error(y_opt_scaled, y_pred)
        # scaler = MinMaxScaler()
        # y_pred_transform = scaler.fit_transform(y_pred)
        # print(y_pred_transform)

        # mcc = matthews_corrcoef(y_opt,y_pred)
        # mae = mean_absolute_error(y_opt_scaled, y_pred_prob[:,0])

        # res = pearsonr(y_opt_scaled, y_pred) #Unable to use statistic with xgb
        # statistic = res.statistic
        return mae


    # n_trials=500
    sampler = TPESampler(seed=7)
    # sampler = RandomSampler(seed=5)
    study = optuna.create_study(
        direction="minimize", study_name=sel_target, sampler=sampler
    )
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(objective, n_trials=n_trials, n_jobs=1, show_progress_bar=True,)


    best_params = study.best_params
    print(best_params)
    model = ElasticNetCV(**best_params,n_jobs=16,max_iter=10000)
    model.fit(X_train, y_train_scaled)
    with open(model_filename, "wb") as f:
        pickle.dump(model, f)
    scaler = MinMaxScaler()
    predictions = model.predict(X_test)
    training_preds = model.predict(X_train)
    optimize_preds = model.predict(X_opt)

    # plt.hist(predictions)
    # plt.hist(training_preds)
    # plt.hist(optimize_preds)


    mse = mean_squared_error(y_test_scaled, predictions)
    res = pearsonr(y_test_scaled, predictions)
    statistic = res.statistic
    pvalue = res.pvalue

    training_mse = mean_squared_error(y_train_scaled, training_preds)
    training_res = pearsonr(y_train_scaled, training_preds)
    training_statistic = training_res.statistic
    training_pvalue = training_res.pvalue

    optimize_res = pearsonr(y_opt_scaled, optimize_preds)

    test_df = test.copy()
    test_df["Predictions"] = predictions
    test_df["Model"] = "XGBoostRegressor"
    test_df["Correlation"] = statistic
    test_df["n_trials"] = n_trials
    test_df["Feature_set"] = "Featset1"
    test_df["Param_space"] = "Original"
    test_df["random_state"] = random_state
    test_df["Active"] = y_test
    test_set_out = test_df[
        [
            "Oligo_ID",
            "Scaled",
            "Predictions",
            "Correlation",
            "Model",
            "n_trials",
            "Feature_set",
            "Param_space",
            "random_state",
            "Active"
        ]
    ]
    test_set_outs.append(test_set_out)

    scores_dataframe = pd.DataFrame(
        {
            "Model": "RandomForestRegressor",
            "Feature_set": "Featset1",
            "Param_space": "Original",
            "Gene": sel_target,
            "train_statistic": [training_statistic],
            "test_statistic": [statistic],
            "train_P_value": [training_pvalue],
            "test_P_value": [pvalue],
            "test_Mean_Squared_Error": [mse],
            "train_Mean_Squared_Error": [training_mse],
            "n_trials": n_trials,
            "Total_train": len(y_train),
            "Total_test": len(y_test),
            "random_state": random_state
        }
        | best_params
    )

    scores_dataframes.append(scores_dataframe)

    # Append the best hyperparameters to the scores dataframe
    # scores_dataframe.append(best_params, ignore_index=True)
    # optuna.visualization.matplotlib.plot_optimization_history(study)
    # plt.title("Optimization history - " + study.study_name+' '+str(n_trials)+" trials")
    # plt.savefig("plots/"+study.study_name+"_trials_"+str(n_trials))
    # plt.close()
    # print(predictions)
    # print(y_test.mean())
    # print(y_test.shape)
    plt.tight_layout()
    trend_plot(
        predictions,
        y_test_scaled.array,
        y_test.array,
        title=study.study_name
        + " "
        + str(n_trials)
        + " trials_test"
        + f" random state ({random_state})",
        outfile=output_path
        + "scatter_test_"
        + study.study_name
        + "_trials_"
        + str(n_trials),
    )
    trend_plot(
        training_preds,
        y_train_scaled.array,
        y_train.array,
        title=study.study_name
        + " "
        + str(n_trials)
        + f" trials_training"
        + f" random state ({random_state})",
        outfile=output_path
        + "scatter_training_"
        + study.study_name
        + "_trials_"
        + str(n_trials),
    )
    trend_plot(
        optimize_preds,
        y_opt_scaled.array,
        y_opt.array,
        title=study.study_name
        + " "
        + str(n_trials)
        + " trials_optimizing"
        + f" random state ({random_state})",
        outfile=output_path
        + "scatter_optimize_"
        + study.study_name
        + "_trials_"
        + str(n_trials),
    )

    # return scores_dataframe, test_set_out

    # Iterate over the unique gene species and train and evaluate the model for each gene species

    # scores_dataframe.to_csv('XG_scores_mse.csv', index=False)
    # scores = pd.read_csv('XG_scores_mse.csv')
    # scores_dataframe.to_csv(
    #     f"{output_path}/{sel_target}_split_results_{n_trials}.csv",
    #     index=None,
    # )
    # test_set_out.to_csv(
    #     f"{output_path}/{sel_target}_split_preds_{n_trials}.csv",
    #     index=None,
    # )
    # metrics.to_csv("Num_trials_"+str(n_trials)+'_XGBR_scores_MSE.csv', index=False)

    # %%
    # trend_plot(optimize_preds, y_opt, title=study.study_name+' '+str(n_trials)+" trials_training", outfile="../outputs/RandomForestRegressor/plots_scn9a_splits/"+"pt_scatter_optimize_"+study.study_name+"_trials_"+str(n_trials))
    # plt.hist(training_preds)
    plt.hist(y_test)
    plt.hist(predictions, alpha=0.5)

    # %%
    scores_dataframe.transpose()

    # %%
    fig, ax = plt.subplots()
    plt.tight_layout()
    optuna.visualization.matplotlib.plot_optimization_history(study)
    plt.title(
        "Optimization history - " + study.study_name + " " + str(n_trials) + " trials"
    )
    plt.savefig(
        output_path
        + "optimization_history_"
        + study.study_name
        + "_trials_"
        + str(n_trials)
    )
    # plt.close()


scores_dataframe = pd.concat(scores_dataframes)
test_set_out = pd.concat(test_set_outs)

scores_dataframe.to_csv(
    f"{output_path}/{sel_target}_split_results_{n_trials}.csv",
    index=None,
)
test_set_out.to_csv(
    f"{output_path}/{sel_target}_split_preds_{n_trials}.csv",
    index=None,
)
