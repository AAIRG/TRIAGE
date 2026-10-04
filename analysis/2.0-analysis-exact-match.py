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
    from lib.utils import process_metrics, aggregate_column_values, filter_metric_columns, get_metric_name
    return (
        aggregate_column_values,
        filter_metric_columns,
        get_metric_name,
        process_metrics,
    )


@app.cell
def _():
    import pandas as pd
    return (pd,)


@app.cell
def _(ROOT_DIR, os, pd):
    df = pd.read_csv(
        os.path.join(ROOT_DIR, "data/external/results_exact_match.csv")
    )
    return (df,)


@app.cell
def _(ROOT_DIR, os):
    DEST_DIR = os.path.join(ROOT_DIR, "data/post_processed/tables_data")
    return


@app.cell
def _(mo):
    mo.md(r"""##Aggregate metrics in the data""")
    return


@app.cell
def _():
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    * Separate metrics are reported for each mapping type.
    * Uncategorized metrics e.g precision are actually micro averaged
    """
    )
    return


@app.cell
def _(df):
    df.head()
    return


@app.cell
def _(df):
    df_2 = df.copy()

    # Workaround. Fixing missing mapping type for selected experiments
    # TODO: Fix this in Neptune later
    # Mapping type should be exploitation technique for the following mapping_method and Ids 
    # tactic_technique ID. train: 817, test: 818
    # affected_object ID. train: 494, test: 756
    incomplete_experiments = ["CVET2-817", "CVET2-818", "CVET2-494", "CVET2-756"]
    df_2.loc[df_2["Id"].isin(incomplete_experiments), "mapping_type"] = "exploitation_technique"

    # Deselect models
    drop_models = ["openai/llama-4-scout-17b-16e-instruct", "openai/llama3.1-70b-instruct-fp8", "openai/llama-4-maverick-17b-128e-instruct-fp8"]

    df_2 = df_2[~df_2["path_model"].isin(drop_models)]
    # Display selected models
    df_2["path_model"].unique()
    return (df_2,)


@app.cell
def _(df_2, filter_metric_columns):
    # Test that we filter all micro_precision above
    micro_precision_cols_test = filter_metric_columns(
        df_2, metric="precision", average="micro"
    )
    return (micro_precision_cols_test,)


@app.cell
def _(micro_precision_cols_test):
    # Should be three mapping types, each with precision and micro precision. in total six columns.
    assert len(micro_precision_cols_test) == 6
    return


@app.cell
def _(aggregate_column_values, df_2, micro_precision_cols_test):
    df_test_2 = df_2.copy()
    df_test_2 = aggregate_column_values(
        df_test_2, metric_name="micro_precision", cols=micro_precision_cols_test
    )
    return (df_test_2,)


@app.cell
def _(df_test_2):
    # Check that micro_f1 is a column
    assert "micro_precision" in df_test_2.columns
    return


@app.cell
def _(df_test_2):
    # Explore values
    df_test_2.loc[:3, "micro_precision"]
    return


@app.cell
def _(df_2, process_metrics):
    df_3 = process_metrics(df_2)
    return (df_3,)


@app.cell
def _(df_3):
    # Drop experiments without a mapping type
    df_4 = df_3[df_3["mapping_type"].notna()]
    # Rename column names
    df_4 = df_4.rename(
        columns={
            "num_cves (last)": "num_cves",
            "num_demonstrations (last)": "num_demonstrations",
        }
    )
    # Convert from float64 to integer
    df_4["num_cves"] = df_4["num_cves"].astype('Int64')
    df_4["num_demonstrations"] = df_4["num_demonstrations"].astype('Int64')
    return (df_4,)


@app.cell
def _(df_4):
    df_4
    return


@app.cell
def _():
    # Train_data, each methodology type including two_step approach for the most amount of data
    return


@app.cell
def _(get_metric_name):
    def get_relevant_metric_names(metrics=None, average_options=None):
        # NB: metrics and average_options should be lists.
        if metrics is None:
            metrics = ["f1", "precision", "recall"]
        if average_options is None:
            average_options = ["micro", "macro", "weighted"]
        metric_names = []
        for metric in metrics:
            for average in average_options:
                metric_name = get_metric_name(metric, average)
                metric_names.append(metric_name)
        return metric_names
    return (get_relevant_metric_names,)


@app.cell
def _(get_relevant_metric_names):
    select_metrics = get_relevant_metric_names()
    return


@app.cell
def _(get_relevant_metric_names):
    select_metrics_2 = get_relevant_metric_names(
        metrics=None, average_options=["micro"]
    )
    print(select_metrics_2)
    return


@app.cell
def _(df_4):
    len(df_4)
    return


@app.cell
def _():
    # Define columns to display in the dataframe. These columns are in addition to the metric columns.
    # Clear display_cols
    display_cols = None
    # display_cols = ['path_model', 'mapping_method', 'include_cvss', 'include_cwe', 'include_affected_object','include_timestamp', 'use_parent_techniques', 'use_all_enterprise_techniques']
    # display_cols = ['path_model', 'mapping_method', 'include_cvss', 'include_cwe', 'use_parent_techniques', 'use_all_enterprise_techniques']
    display_cols = ["Id", "path_model", "mapping_method"]
    # feature_dict = {key: False for key in feature_cols}
    return (display_cols,)


@app.function
# Define a custom sorting key
def extract_last_three_digits(id_value):
    return int(id_value.split('-')[1])


@app.function
def remove_duplicates(df, display_cols, keep="last"):
    """
    Remove duplicate experiment configuration
    """
    # Sort id values to later keep the last experiment
    df_sorted = df.sort_values(by='Id', key=lambda x: x.map(extract_last_three_digits))
    subset = display_cols.copy()
    if "Id" in display_cols:
        subset.remove("Id")
    # Drop duplicates, keep="last" keeps the last experiment
    df_2 = df_sorted.drop_duplicates(subset=subset, keep=keep)
    return df_2


@app.cell
def _(get_relevant_metric_names):
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
        keep_duplicate="last"):
        if num_cves:
            df = df.loc[df["num_cves"] == num_cves]
        # Fiter on data split
        if data_split:
            df = df.loc[df["data_split"] == data_split]
        # Filter on mapping type
        if mapping_type:
            df = df.loc[df["mapping_type"] == mapping_type]
        if mapping_method:
            df = df.loc[df["mapping_method"] == mapping_method]
        # print(df.columns)
        select_metrics = get_relevant_metric_names(metrics, average_options)
        # print(select_metrics)
        if handle_duplicates:
            df = remove_duplicates(df, display_cols, keep_duplicate)
        display_cols_2 = display_cols.copy()
        display_cols_2.extend(select_metrics)
        df = df.loc[:, display_cols_2]
        return df
    return (display_results,)


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Tests
    Testing functions and code below
    """
    )
    return


@app.cell
def _(df_4, display_cols, display_results):
    # Test function
    results_1 = display_results(
        df_4,
        data_split="train",
        display_cols=display_cols,
        num_cves=236,
        mapping_type="exploitation_technique",
        average_options=["micro"],
        handle_duplicates=False
    )
    return (results_1,)


@app.cell
def _():
    # remove_duplicates(results_1, display_cols=display_cols)
    return


@app.cell
def _(results_1):
    results_1.head()
    return


@app.cell
def _(display_cols, results_1):
    # Test handling duplicates

    # Sort id values to later keep the last experiment
    results_1_sorted = results_1.sort_values(by="Id") 
    display_cols_2 = display_cols.copy()
    display_cols_2.remove("Id")
    print(display_cols_2)
    # Drop duplicates and keep the last value = last experiment
    results_1_cleaned = results_1_sorted.drop_duplicates(subset=display_cols_2, keep='last')
    return (results_1_cleaned,)


@app.cell
def _(results_1_cleaned):
    results_1_cleaned
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md(r"""## Train Results""")
    return


@app.cell
def _(display_cols):
    display_cols
    return


@app.cell
def _(df_4, display_cols, display_results):
    results_exp_train = display_results(
        df_4,
        data_split="train",
        display_cols=display_cols,
        mapping_type="exploitation_technique",
        num_cves=236,
        average_options=["micro"],
    )
    return (results_exp_train,)


@app.cell
def _(results_exp_train):
    results_exp_train
    return


@app.cell
def _(df_4, display_cols, display_results):
    results_prim_train = display_results(
        df_4,
        data_split="train",
        display_cols=display_cols,
        num_cves=236,
        mapping_type="primary_impact",
        average_options=["micro"],
    )
    return (results_prim_train,)


@app.cell
def _(results_prim_train):
    results_prim_train
    return


@app.cell
def _(df_4, display_cols, display_results):
    results_sec_train = display_results(
        df_4,
        data_split="train",
        display_cols=display_cols,
        num_cves=236,
        mapping_type="secondary_impact",
        average_options=["micro"],
    )
    return (results_sec_train,)


@app.cell
def _(results_sec_train):
    results_sec_train
    return


@app.cell
def _(mo):
    mo.md(r"""## Test Results""")
    return


@app.cell
def _(df_4, display_cols, display_results):
    results_exp_test = display_results(
        df_4,
        data_split="test",
        display_cols=display_cols,
        num_cves=60,
        mapping_type="exploitation_technique",
        average_options=["micro"],
    )
    return (results_exp_test,)


@app.cell
def _(results_exp_test):
    results_exp_test
    return


@app.cell
def _(df_4, display_cols, display_results):
    results_prim_test = display_results(
        df_4,
        data_split="test",
        display_cols=display_cols,
        num_cves=60,
        mapping_type="primary_impact",
        average_options=["micro"],
    )
    return (results_prim_test,)


@app.cell
def _(results_prim_test):
    results_prim_test
    return


@app.cell
def _(df_4, display_cols, display_results):
    results_sec_test = display_results(
        df_4,
        data_split="test",
        display_cols=display_cols,
        num_cves=60,
        mapping_type="secondary_impact",
        average_options=["micro"],
    )
    return (results_sec_test,)


@app.cell
def _(results_sec_test):
    results_sec_test
    return


@app.cell
def _(mo):
    mo.md(r"""## Combining tables for paper""")
    return


@app.function
#TODO Rename to postprocessing?
def latex_processing(df, model=None, sort_column=None, data_split="train"):
    # Final latex processing
    df_2 = df.copy()
    # Remove the beginning of the column name in path_model
    df_2['path_model'] = df_2['path_model'].str.replace('openai/', '')
    # Simplify llama model name
    df_2['path_model'] = df_2['path_model'].str.replace('-instruct-fp8', '')
    # Rename columns
    if data_split == "train":
        df_2 = df_2.rename(columns={"path_model": "model", "micro_f1": "train F1 score", "micro_precision": "train precision", "micro_recall": "train recall"})
    elif data_split == "test":
        df_2 = df_2.rename(columns={"path_model": "model", "micro_f1": "test F1 score", "micro_precision": "test precision", "micro_recall": "test recall"})
    df_3 = df_2.round(3)
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
    display_cols_3 = ["path_model", "mapping_type", "mapping_method"]
    # DEBUG. include Id
    # display_cols_3 = ["Id", "path_model", "mapping_type", "mapping_method"]
    return (display_cols_3,)


@app.cell
def _(df_4, display_cols_3, display_results):
    results_train = display_results(
        df_4,
        data_split="train",
        display_cols=display_cols_3,
        num_cves=236,
        average_options=["micro"],
        handle_duplicates=True
    )
    return (results_train,)


@app.cell
def _(results_train):
    results_train
    return


@app.cell
def _(results_train):
    results_train.columns
    return


@app.cell
def _(results_train):
    results_latex_processed_train = latex_processing(results_train, model="gpt-4o-mini", sort_column="mapping_type")
    return (results_latex_processed_train,)


@app.cell
def _(results_latex_processed_train):
    results_latex_processed_train
    return


@app.cell
def _(df_4, display_cols_3, display_results):
    results_test = display_results(
        df_4,
        data_split="test",
        display_cols=display_cols_3,
        num_cves=60,
        average_options=["micro"],
    )
    results_latex_processed_test = latex_processing(results_test, model="gpt-4o-mini", sort_column="mapping_type", data_split="test")
    return (results_latex_processed_test,)


@app.cell
def _(results_latex_processed_test):
    results_latex_processed_test
    return


@app.cell
def _(results_latex_processed_test, results_latex_processed_train):
    # Merge results_latex_processed_train and results_latex_processed_test
    df_combined = results_latex_processed_train.merge(results_latex_processed_test, how="left")
    return (df_combined,)


@app.cell
def _(df_combined):
    df_combined
    return


@app.cell
def _(df_combined):
    df_combined_method = df_combined.copy()
    df_combined_method.columns = [col.replace('_', '-') for col in df_combined_method.columns]
    df_combined_method = df_combined_method.map(lambda x: x.replace('_', '-') if isinstance(x, str) else x)
    return (df_combined_method,)


@app.cell
def _(df_combined_method):
    df_combined_method
    return


@app.cell
def _():
    # Save table as csv
    # df_combined_method.to_csv(os.path.join(DEST_DIR, "unranked_classification.csv"), float_format='%.2f', index_label=False, index=False)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
