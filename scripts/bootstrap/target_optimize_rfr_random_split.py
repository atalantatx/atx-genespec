# %%
import pandas as pd
import xgboost as xgb
import numpy as np
import optuna
from optuna.samplers import TPESampler, RandomSampler
import pickle
import os
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr, linregress
from sklearn.ensemble import RandomForestRegressor, RandomForestRegressor

from sklearn.preprocessing import MinMaxScaler

sel_target = "SCN9A"
n_trials = 10
output_path = "../outputs/RandomForestRegressor/FeatureSet4/random_split/"

activity_cols = ["Oligo_ID", "Gene Species", "Scaled", "Train-Test-Split", "Screen"]

# %%
feature_columns = [line.strip() for line in open("FeatSet4_columns.txt").readlines()]
dataset = pd.read_parquet("disirna.all_features_v2.features_v2.parquet.gzip")
dataset["Oligo_ID"] = dataset["Oligo_ID"].str.replace(
    "PRNP-Human|PRNP-Mouse", "PRNP", regex=True
)
dataset["Gene"] = dataset["Gene"].str.replace(
    "PRNP-Human|PRNP-Mouse", "PRNP", regex=True
)
print(sum(dataset.columns.isin(["Gene"])))


scaled_data = pd.read_csv("All_data_2023-09-22_scaled.csv")
scaled_data["Scaled"] = scaled_data["scaled"]
merge_dataset = dataset.merge(
    scaled_data, on=["Oligo_ID", "Gene"], how="inner", copy=True
)

merge_dataset.shape

# %%

random_state = 40
train_frac = 0.8
# random_state = np.random.randint(low=1, high=1000)
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


# %%
plt.hist(y_opt, density=True)
plt.hist(y_test, density=True, alpha=0.5)


# Define the objective function
def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 150),
        "max_depth": trial.suggest_int("max_depth", 4, 4),
        "min_samples_split": trial.suggest_float("min_samples_split", 0, 1),
        "max_features": trial.suggest_float("max_features", 0.2, 0.5),
        "max_samples": trial.suggest_float("max_samples", 0, 0.5),
        # "min_samples_leaf": trial.suggest_int('min_samples_leaf', 2, 10),
        # "max_features": trial.suggest_categorical('max_features', ['log2', 'sqrt',None]),
        # "max_leaf_nodes": trial.suggest_int('max_leaf_nodes', 10, 100),
        # "bootstrap": trial.suggest_categorical('bootstrap', [True, False]),
    }

    model = RandomForestRegressor(**params, n_jobs=16, random_state=7, bootstrap=True)
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
model = RandomForestRegressor(
    **best_params,
    random_state=51,
    n_jobs=16,
)
model.fit(X_train, y_train)
with open(model_filename, "wb") as f:
    pickle.dump(model, f)
scaler = MinMaxScaler()
predictions = model.predict(X_test)
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
    ]
]

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
    }
    | best_params
)

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
scores_dataframe.to_csv(
    f"{output_path}/{sel_target}_split_results_{n_trials}.csv",
    index=None,
)
test_set_out.to_csv(
    f"{output_path}/{sel_target}_split_preds_{n_trials}.csv",
    index=None,
)
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
