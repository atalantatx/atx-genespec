#!/usr/bin/env python
# %%
import pandas as pd
import xgboost as xgb
import numpy as np
import optuna
from optuna.samplers import TPESampler, RandomSampler
import pickle, os, yaml
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr, linregress
# from sklearn.ensemble import RandomForestRegressor, RandomForestRegressor
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

scores_dataframes = []
test_set_outs = []

for i in range(doc["boostrap_reps"]):
    random_state = np.random.randint(low=1, high=1000)
    gene_data = pd.DataFrame(merge_dataset.loc[merge_dataset["Gene"] == sel_target])
    train = pd.DataFrame(gene_data.sample(frac=train_frac, random_state=random_state))

    # test = pd.read_csv("../dataset/tiling_reg_dataset.csv")
    # optimize = pd.read_csv("../dataset/primary_reg_dataset.csv")
    # train = pd.DataFrame(optimize[train.columns.to_list()])

    # # train = pd.concat([train,optimize],join="inner")
    non_train = pd.DataFrame(gene_data[~gene_data["Oligo_ID"].isin(train["Oligo_ID"])])
    optimize = pd.DataFrame(non_train.sample(frac=0.5, random_state=random_state))
    test = pd.DataFrame(non_train[~non_train["Oligo_ID"].isin(optimize["Oligo_ID"])])

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


    # %%
    def trend_plot(y_preds, y_test, title, outfile):
        # m, b = np.polyfit(y_preds, y_test, 1)
        # print(np.unique(y_preds))
        if len(np.unique(y_preds)) >1:
            fig = plt.figure()
            # fig(figsize=(6, 4), dpi=80)
            fig.set_size_inches(6, 4)
            linear_fit = linregress(y_preds, y_test)
            slope = linear_fit.slope
            pvalue = linear_fit.pvalue
            intercept = linear_fit.intercept

            pcc_res = pearsonr(y_preds, y_test)
            plt.figure(figsize=(10, 6))
            ax = plt.scatter(x=y_preds, y=y_test)
            if max(y_preds) > 1.8:
                plt.xlim(-0.1, 1.1)
                text_x = -190
            else:
                plt.xlim(min(y_preds) * 0.9, max(y_preds) * 1.1)
                text_x = min(y_preds)
                text_y = max(y_test)
            textstr = "\n".join(
                [
                    "Reg Line: y = {:.2f}x + {:.2f}".format(slope, intercept),
                    "P-value: " + str(round(pvalue, 3)),
                    "corr: " + str(round(pcc_res.statistic, 3)),
                ]
            )
            props = dict(boxstyle="round", facecolor="wheat", alpha=0.25)

            plt.ylim(-0.1, 1.4)
            plt.plot(y_preds, intercept + slope * y_preds, color="black")

            # place a text box in upper left in axes coords
            plt.text(
                text_x, text_y, textstr, fontsize=14, verticalalignment="bottom", bbox=props
            )
            plt.title(title)
            plt.xlabel("Predictions")
            plt.ylabel("Actual")
            plt.savefig(outfile)
            plt.close()


    # trend_plot(optimize_preds, y_opt, title=study.study_name+' '+str(n_trials)+" trials_training", outfile="../outputs/RandomForestRegressor/plots_scn9a_splits/"+"pt_scatter_optimize_"+study.study_name+"_trials_"+str(n_trials))

    # %%
    test

    # %%
    X_train = train[feature_columns]
    y_train = train["Scaled"]
    print(sum(X_train.columns.isin(activity_cols)))
    X_opt = optimize[feature_columns]
    y_opt = optimize["Scaled"]
    X_test = test[feature_columns]
    y_test = test["Scaled"]
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
            "n_alphas": trial.suggest_int("n_alphas", 1, 100),
            "fit_intercept": trial.suggest_categorical("fit_intercept",[True,False]),
            # "eps" : trial.suggest_float("eps",0,0.1)
        }

        model = LassoCV(**params, n_jobs=16, random_state=random_state,max_iter=10000,selection="random")
        # model = xgb.XGBRegressor(n_estimators=n_estimators, max_depth=max_depth, reg_alpha=reg_alpha, gamma=gamma,
        #     reg_lambda=reg_lambda, colsample_bytree=colsample_bytree, min_child_weight=min_child_weight,
        #     learning_rate=learning_rate, objective='reg:squarederror',subsample=subsample, n_jobs=64, seed=51)

        model.fit(X_train, y_train)

        y_pred = model.predict(X_opt)
        # y_pred = model.predict(X_train)
        # scaler  = MinMaxScaler()
        # y_pred = scaler.fit_transform(pred.reshape(-1, 1))

        # mse = mean_squared_error(y_opt, y_pred)
        mae = mean_absolute_error(y_opt, y_pred)
        r2score = r2_score(y_opt, y_pred)
        # mae = mean_absolute_error(y_train, y_pred)
        # r2_score_val = r2_score(y_opt, y_pred)
        # y_pred = model.predict(X_test)
        # mse = mean_squared_error(y_test, y_pred)
        # print(mse)

        # res = pearsonr(y_test, y_pred) #Unable to use statistic with xgb
        # statistic = res.statistic
        return mae


    # n_trials=500
    # sampler = TPESampler(seed=5)
    sampler = RandomSampler(seed=5)
    study = optuna.create_study(
        direction="minimize", study_name=sel_target, sampler=sampler
    )
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(objective, n_trials=n_trials, n_jobs=1, show_progress_bar=True)


    best_params = study.best_params
    print(best_params)
    model = LassoCV(
        **best_params,
        random_state=random_state,
        n_jobs=16,
    )
    model.fit(X_train, y_train)
    with open(model_filename, "wb") as f:
        pickle.dump(model, f)
    scaler = MinMaxScaler()
    predictions = model.predict(X_test)
    print(model.score(X_test,y_test))
    training_preds = model.predict(X_train)
    optimize_preds = model.predict(X_opt)

    # plt.hist(predictions)
    # plt.hist(training_preds)
    # plt.hist(optimize_preds)


    mse = mean_squared_error(y_test, predictions)
    res = pearsonr(y_test, predictions)
    statistic = res.statistic
    pvalue = res.pvalue

    training_mse = mean_squared_error(y_train, training_preds)
    training_res = pearsonr(y_train, training_preds)
    training_statistic = training_res.statistic
    training_pvalue = training_res.pvalue

    optimize_res = pearsonr(y_opt, optimize_preds)

    test_df = test.copy()
    test_df["Predictions"] = predictions
    test_df["Model"] = "XGBoostRegressor"
    test_df["Correlation"] = statistic
    test_df["n_trials"] = n_trials
    test_df["Feature_set"] = "Featset1"
    test_df["Param_space"] = "Original"
    test_df["random_state"] = random_state
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
            "random_state"
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
        y_test,
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
        y_train,
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
        y_opt,
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
    optuna.visualization.matplotlib.plot_optimization_history(study)
    plt.tight_layout()
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
