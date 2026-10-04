import marimo

__generated_with = "0.15.0"
app = marimo.App(width="medium")


@app.cell
def _():
    # Configuration to import local modules when running in a notebook
    import os
    import sys

    # Get the project repository path
    ROOT_DIR = os.path.abspath("..")

    # Add to the system path
    sys.path.append(ROOT_DIR)
    return ROOT_DIR, os


@app.cell
def _():
    import pandas as pd
    from itertools import chain
    from lib.utils import replace_values
    return pd, replace_values


@app.cell
def _(ROOT_DIR, os, pd):
    df = pd.read_csv(
        os.path.join(ROOT_DIR, "data/external/results_ranking_approach.csv")
    )
    return (df,)


@app.cell
def _(ROOT_DIR, os):
    DEST_DIR = os.path.join(ROOT_DIR, "data/post_processed/tables_data")
    return


@app.cell
def _():
    return


@app.cell
def _(df):
    df.columns
    return


@app.cell
def _(df):
    df_2 = df.copy()
    # Deselect models
    drop_models = ["openai/llama-4-scout-17b-16e-instruct", "openai/llama3.1-70b-instruct-fp8", "openai/llama-4-maverick-17b-128e-instruct-fp8"]
    df_2 = df_2[~df_2["path_model"].isin(drop_models)]
    # Display selected models
    df_2["path_model"].unique()
    return (df_2,)


@app.cell
def _(df_2):
    df_3 = df_2.copy()
    # Rename column names
    df_3 = df_3.rename(
        columns={
            "num_cves (last)": "num_cves",
            "num_demonstrations (last)": "num_demonstrations",
            "metrics/MAP": "MAP",
            "metrics/Recall@10": "Recall@10",
            "metrics/Recall@5": "Recall@5",
            "metrics/Recall@20": "Recall@20",
        }
    )
    # Convert from float64 to integer
    df_3["num_cves"] = df_3["num_cves"].astype('Int64')
    df_3["num_demonstrations"] = df_3["num_demonstrations"].astype('Int64')
    return (df_3,)


@app.cell
def _():
    # Train and test
    # Demonstrations train
    # Model names both train and test
    # Combined approach both train and test
    # cvss CWE and timestamp, attack descriptions train
    return


@app.cell
def _(df_3):
    df_train = df_3.loc[df_3["data_split"] == "train"]
    df_test = df_3.loc[df_3["data_split"] == "test"]
    return


@app.function
def select_ranking_metrics():
    metric_cols = ["MAP", "Recall@10", "Recall@5"]
    return metric_cols


@app.function
# TODO: Merge with the other Notebook and add to utils
def remove_duplicates(df, display_cols, keep="last"):
    """
    Remove duplicate experiment configuration
    """
    # Sort id values to later keep the last experiment
    df_sorted = df.sort_values(by="Id", key=lambda col: col.str.split('-').str[1].astype(int))
    subset = display_cols.copy()
    if "Id" in display_cols:
        subset.remove("Id")
    # Drop duplicates, keep="last" keeps the last experiment
    df_3 = df_sorted.drop_duplicates(subset=subset, keep=keep)
    return df_3


@app.function
# TODO: Merge with the other Notebook and add to utils
def display_results(
    df,
    display_cols,
    data_split=None,
    mapping_type=None,
    mapping_method=None,
    num_cves=None,
    metrics=None,
    average_options=None,
    handle_duplicates=True,
    keep_duplicate="last",
    strategy="ranking",
):
    if mapping_method:
        df = df.loc[df["mapping_method"] == mapping_method]
    if num_cves:
        df = df.loc[df["num_cves"] == num_cves]
    # Fiter on data split
    if data_split:
        df = df.loc[df["data_split"] == data_split]
    # Filter on mapping type
    if mapping_type:
        df = df.loc[df["mapping_type"] == mapping_type]
    # print(df.columns)
    if strategy == "exact_match":
        pass
        # select_metrics = get_relevant_metric_names(metrics, average_options)
    if strategy == "ranking":
        select_metrics = select_ranking_metrics()
    # print(select_metrics)
    if handle_duplicates:
        df = remove_duplicates(df, display_cols, keep_duplicate)
    display_cols_2 = display_cols.copy()
    display_cols_2.extend(select_metrics)
    df = df.loc[:, display_cols_2]
    return df


@app.cell
def _(mo):
    mo.md(r"""# Two Step approach""")
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md(r"""## Ablation study""")
    return


@app.cell
def _():
    display_columns_1 = [
        "Id",
        "include_attack_descriptions",
        "num_demonstrations",
        "include_cvss",
        "include_cwe",
    ]
    return (display_columns_1,)


@app.cell
def _(mo):
    mo.md(r"""Analysing the ablations with the parameters (columns) above. And 100 CVEs""")
    return


@app.cell
def _(df_3):
    df_3.columns
    return


@app.cell
def _(df_3, display_columns_1):
    exp_res = display_results(
        df_3,
        display_columns_1,
        data_split="train",
        mapping_type="exploitation_technique",
        num_cves=100,
        handle_duplicates=True,
    )
    return (exp_res,)


@app.cell
def _(exp_res):
    exp_res
    return


@app.cell
def _(df_3, display_columns_1):
    prim_res = display_results(
        df_3,
        display_columns_1,
        data_split="train",
        mapping_type="primary_impact",
        num_cves=100,
    )
    return (prim_res,)


@app.cell
def _(df_3, display_columns_1):
    sec_res = display_results(
        df_3,
        display_columns_1,
        data_split="train",
        mapping_type="secondary_impact",
        num_cves=100,
    )
    return (sec_res,)


@app.cell
def _():
    # Ablations ordered. Listed values are true values
    ranked_combinations = [
        ["include_attack_descriptions", "include_cvss", "include_cwe", 235],
        ["include_attack_descriptions", "include_cvss", 235],
        ["include_attack_descriptions", 235],
        [235],
        [30],
        [0]
    ]
    return (ranked_combinations,)


@app.cell
def _(pd):
    def find_matching_row(df, combination):
        """
        Finds the row in the DataFrame that matches all conditions in a given combination.

        Args:
            df: The pandas DataFrame to search.
            combination: A list containing strings (for boolean columns) and an
                         integer (for 'num_demonstrations').

        Returns:
            A DataFrame containing the matching row, or an empty DataFrame if no match is found.
        """
        # Start with a filter that is True for all rows
        final_filter = pd.Series([True] * len(df), index=df.index)

        # Define the boolean columns you care about
        boolean_cols = [
            "include_attack_descriptions",
            "include_cvss",
            "include_cwe",
        ]

        # --- Handle the boolean flags ---
        # For each boolean column, check if its name is in the combination list.
        # The condition is True if the column shoxuld be True AND its name is in the list,
        # or if the column should be False AND its name is NOT in the list.
        for col in boolean_cols:
            final_filter &= df[col] == (col in combination)

        # --- Handle the numeric value ---
        # Find the integer in the combination list and apply that filter.
        for item in combination:
            if isinstance(item, int):
                final_filter &= df["num_demonstrations"] == item
                break  # Assume only one integer per combination

        return df[final_filter]
    return (find_matching_row,)


@app.cell
def _(find_matching_row, pd, ranked_combinations):
    def display_ranked_results(df, combinations):
        df_final = pd.DataFrame()
        for combi in ranked_combinations:
            row = find_matching_row(df, combi)
            df_final = pd.concat([df_final, row])
        return df_final
    return (display_ranked_results,)


@app.cell
def _(display_ranked_results, exp_res, ranked_combinations):
    exp_res_2 = display_ranked_results(exp_res, ranked_combinations)
    exp_res_2
    return (exp_res_2,)


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    Exploitation technique

    * We see that reducing the demonstrations performance drops across all metrics
    * CWE and CVSS used together has a slight negative impacts compared to leaving them out
    * Leaving out attack descriptions improves MAP but reduces Recall@5
    """
    )
    return


@app.cell
def _(display_ranked_results, prim_res, ranked_combinations):
    prim_res_2 = display_ranked_results(prim_res, ranked_combinations)
    prim_res_2
    return (prim_res_2,)


@app.cell
def _(mo):
    mo.md(
        r"""
    Primary impact

    * Recall@10 and Recall@5 if we reduce the number of demonstrations, but less than for the exploitation_technique. MAP stays the same
    * Leaving out both CWE and CVSS has a positive impact accross Recall@10 and Recall@5, marginal reduction in MAP
    * Including attack descriptions has a clear positive impact on both recall metrics, and marginally better on MAP
    """
    )
    return


@app.cell
def _(display_ranked_results, ranked_combinations, sec_res):
    sec_res_2 = display_ranked_results(sec_res, ranked_combinations)
    sec_res_2
    return (sec_res_2,)


@app.cell
def _(mo):
    mo.md(
        r"""
    Secondary impact

    * Reducing the number of demonstrations reduces performance accross all metrics
    * Dropping CVSS has a positive impact
    * Dropping attack descriptions has a negative impact
    """
    )
    return


@app.cell
def _(mo):
    mo.md(r"""### Combining tables for paper""")
    return


@app.function
def process_ablation_features(all_ablations, mapping_type=None):
    # Gather features in a list
    # include_attack_descriptions True -> A
    # include_cvss -> CVSS
    # include_cwe -> CWE
    # Add new column features
    features_col = []
    for row in all_ablations.itertuples():
        feature = ""
        if row.include_attack_descriptions:
            feature += "A"
        if row.include_cvss:
            feature += "CVSS"
        if row.include_cwe:
            feature += "CWE"
        features_col.append(feature)

    all_ablations_2 = all_ablations.copy()
    all_ablations_2["features"] = features_col
    if mapping_type:
        all_ablations_2["mapping_type"] = mapping_type
    print(all_ablations_2.columns)
    all_ablations_2 = all_ablations_2.loc[:, ["mapping_type", "features", "num_demonstrations", "MAP", "Recall@10", "Recall@5"]]
    all_ablations_2 = all_ablations_2.sort_values(by="mapping_type")
    return all_ablations_2


@app.cell
def _(exp_res_2):
    exp_res_7 = process_ablation_features(exp_res_2, mapping_type="exploitation_technique")
    return (exp_res_7,)


@app.cell
def _(exp_res_7):
    exp_res_7
    return


@app.cell
def _(prim_res_2, sec_res_2):
    prim_res_3 = process_ablation_features(prim_res_2, mapping_type="primary_impact")
    sec_res_3 = process_ablation_features(sec_res_2, mapping_type="secondary_impact")
    return prim_res_3, sec_res_3


@app.cell
def _(exp_res_7, pd, prim_res_3, sec_res_3):
    # Combine three dataframes by stacking them
    combined_ablations = pd.concat([exp_res_7, prim_res_3, sec_res_3], axis=0)
    combined_ablations
    return (combined_ablations,)


@app.function
def rename_latex_compatible_csv(df):
    # Rename columns
    df_2 = df.copy()
    df_2 = df_2.rename(columns={"mapping_type": "mapping-type", "num_demonstrations": "num-demonstrations", "Recall@10": "Recall-at-ten", "Recall@5": "Recall-at-five"})
    # Rename rows
    df_2["mapping-type"] = df_2["mapping-type"].replace({"exploitation_technique": "exploitation-technique", "primary_impact": "primary-impact", "secondary_impact": "secondary-impact"})
    return df_2


@app.cell
def _(combined_ablations):
    combined_ablations_2 = rename_latex_compatible_csv(combined_ablations)
    combined_ablations_2
    return


@app.cell
def _():
    # combined_ablations_2.to_csv(os.path.join(DEST_DIR, "ranking_approach_ablations.csv"), float_format='%.2f', index=False)
    return


@app.cell
def _(mo):
    mo.md(r"""## Results on the full data""")
    return


@app.cell
def _(df_3):
    # filter train and test and model type, exp prim and secd
    df_3.head()
    return


@app.cell
def _():
    return


@app.cell
def _():
    display_columns_2 = ["Id", "path_model", "mapping_method"]
    return (display_columns_2,)


@app.cell
def _(df_3):
    df_combined = df_3.loc[df_3["mapping_method"] == "combined_approach"]
    return (df_combined,)


@app.cell
def _():
    # We manually select the combined_approach experiments which matches our criteria for selecting mapping methods to include. 
    # Mapping method should be included if F1 and recall both are greater than 0.2
    # The following experiments satiesfies our criteria
    combined_ids = ["CVET2-1029", "CVET2-1030", "CVET2-1035", "CVET2-1036", "CVET2-1109", "CVET2-1110", "CVET2-1111", "CVET2-1112", "CVET2-1114", "CVET2-1116", "CVET2-1117", "CVET2-1118"]
    return (combined_ids,)


@app.cell
def _(combined_ids, df_combined):
    df_combined_2 = df_combined.loc[df_combined["Id"].isin(combined_ids)]
    df_combined_2
    return (df_combined_2,)


@app.cell
def _(df_3):
    df_4 = df_3.copy()
    df_4 = df_4.loc[df_4["mapping_method"] != "combined_approach"]
    df_4["mapping_method"].unique()
    return (df_4,)


@app.cell
def _(df_4, df_combined_2, pd):
    df_5 = pd.concat([df_4, df_combined_2], axis=0)
    df_5.loc[df_5["mapping_method"] == "combined_approach"]
    return (df_5,)


@app.cell
def _(df_5):
    df_5
    return


@app.cell
def _():
    display_columns_5 = ["Id", "path_model", "mapping_method", "include_cvss", "include_cwe"]
    return (display_columns_5,)


@app.cell
def _(df_5, display_columns_5):
    display_results(
        df=df_5,
        display_cols=display_columns_5,
        data_split="train",
        mapping_type="exploitation_technique",
        mapping_method=None,
        num_cves=236,
        handle_duplicates=False,
    )
    return


@app.cell
def _(df_5, display_columns_2):
    display_results(
        df=df_5,
        display_cols=display_columns_2,
        data_split="test",
        mapping_type="exploitation_technique",
        mapping_method=None,
        num_cves=60,
        handle_duplicates=True,
    )
    return


@app.cell
def _(mo):
    mo.md(r"""### Combing tables for paper""")
    return


@app.function
#TODO Rename to postprocessing?
def latex_processing(df, model=None, sort_column=None, data_split="train"):
    # Final latex processing
    df_3 = df.copy()
    # Remove the beginning of the column name in path_model
    df_3['path_model'] = df_3['path_model'].str.replace('openai/', '')
    # Simplify llama model name
    df_3['path_model'] = df_3['path_model'].str.replace('-instruct-fp8', '')
    # Rename columns
    if data_split == "train":
        df_3 = df_3.rename(columns={"path_model": "model", "MAP": "train MAP", "Recall@10": "train Recall@10", "Recall@5": "train Recall@5"})
    elif data_split == "test":
        df_3 = df_3.rename(columns={"path_model": "model", "MAP": "test MAP", "Recall@10": "test Recall@10", "Recall@5": "test Recall@5"})
    df_3 = df_3.round(3)
    # Select model to keep
    if model:
        df_3 = df_3.loc[df_3["model"] == model]
        # Drop model column
        df_3 = df_3.drop(columns=["model"])
    if sort_column:
        df_3 = df_3.sort_values(by=sort_column)
    return df_3


@app.cell
def _():
    display_columns_4 = ["mapping_type", "path_model", "mapping_method"]
    # DEBUG
    # display_columns_4 = ["Id", "mapping_type", "path_model", "mapping_method"]
    return (display_columns_4,)


@app.cell
def _(df_5, display_columns_4):
    results_train = display_results(
        df_5,
        data_split="train",
        mapping_method=None,
        display_cols=display_columns_4,
        num_cves=236,
    )
    return (results_train,)


@app.cell
def _(results_train):
    results_train
    return


@app.cell
def _(results_train):
    results_latex_processed_train = latex_processing(results_train, sort_column="mapping_type", data_split="train")
    return (results_latex_processed_train,)


@app.cell
def _(results_latex_processed_train):
    results_latex_processed_train
    return


@app.cell
def _(df_5, display_columns_4):
    results_test = display_results(
        df_5,
        data_split="test",
        mapping_method=None,
        display_cols=display_columns_4,
        num_cves=60,
    )
    return (results_test,)


@app.cell
def _(results_test):
    results_latex_processed_test = latex_processing(results_test, sort_column="mapping_type", data_split="test")
    return (results_latex_processed_test,)


@app.cell
def _(results_latex_processed_test, results_latex_processed_train):
    # Merge results_latex_processed_train and results_latex_processed_test
    df_combined_processed = results_latex_processed_train.merge(results_latex_processed_test, how="left")
    return (df_combined_processed,)


@app.cell
def _(pd):
    def process_model_name(df: pd.DataFrame) -> pd.DataFrame:
        """
        Processes model names in a DataFrame based on the mapping method.

        This function creates a new DataFrame and updates the 'model' column
        for 'gpt-4o-mini' entries.
        Args:
            df: The input DataFrame with 'model' and 'mapping_method' columns.

        Returns:
            A new DataFrame with the processed model names.
        """
        # Create a copy of the DataFrame to avoid modifying the original
        new_df = df.copy()

        # --- Efficiently update the 'model' column using boolean masking ---

        # Condition 1: model is 'gpt-4o-mini' and method is 'in_context_learner'
        gpt_condition_basic = (new_df['model'] == 'gpt-4o-mini') & \
                          (new_df['mapping_method'] == 'in_context_learner')

        # Condition 2: model is 'gpt-4o-mini' and method is 'combined_approach'
        gpt_condition_combined = (new_df['model'] == 'gpt-4o-mini') & \
                             (new_df['mapping_method'] == 'combined_approach')

        llama_condition_basic = (new_df['model'] == 'llama3.3-70b') & \
                          (new_df['mapping_method'] == 'in_context_learner')

        llama_condition_combined = (new_df['model'] == 'llama3.3-70b') & \
                          (new_df['mapping_method'] == 'combined_approach')

        # Apply the new names to the copied DataFrame using .loc
        # .loc is used here to ensure the values are set on the DataFrame itself,
        # avoiding any potential SettingWithCopyWarning.
        new_df.loc[gpt_condition_basic, 'model'] = 'gpt-4o-mini-basic'
        new_df.loc[gpt_condition_combined, 'model'] = 'gpt-4o-mini-combined'
        new_df.loc[llama_condition_basic, 'model'] = 'llama3.3-70b-basic'
        new_df.loc[llama_condition_combined, 'model'] = 'llama3.3-70b-combined'

        new_df.drop(columns=['mapping_method'], inplace=True)
        # Sort mapping type, exp, prim, sec
        # Sort on model, gpt-mini-basic, llama-basic, gpt-mini-combined, llama-combined.
        new_df = new_df.sort_values(by=['mapping_type', 'model'], ascending=[True, True])

        return new_df
    return (process_model_name,)


@app.cell
def _(df_combined_processed, process_model_name):
    df_combined_processed_2 = process_model_name(df=df_combined_processed)
    return (df_combined_processed_2,)


@app.cell
def _():
    # sorted_df = df_combined_processed_2.sort_values(by=['mapping_type', 'model'], ascending=[True, True])
    return


@app.cell
def _(df_combined_processed_2):
    df_combined_processed_2
    return


@app.cell
def _(df_combined_processed_2):
    df_combined_processed_3 = df_combined_processed_2.copy()
    return (df_combined_processed_3,)


@app.cell
def _(df_combined_processed_3, replace_values):
    df_combined_processed_4 = replace_values(df_combined_processed_3)
    return (df_combined_processed_4,)


@app.cell
def _(df_combined_processed_4):
    df_combined_processed_4
    return


@app.cell
def _():
    # df_combined_processed_4.to_csv(os.path.join(DEST_DIR, "ranking_approach_models.csv"), float_format='%.2f', index_label=False, index=False)
    return


if __name__ == "__main__":
    app.run()
