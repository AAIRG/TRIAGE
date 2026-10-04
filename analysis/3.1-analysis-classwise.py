import marimo

__generated_with = "0.14.17"
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
    import json
    import pandas as pd
    return json, pd


@app.cell
def _():
    from lib.utils import (
        compute_classwise_recall_at_k,
        get_full_attack_name,
        postprocess,
    )
    return compute_classwise_recall_at_k, get_full_attack_name, postprocess


@app.cell
def _(ROOT_DIR, os):
    DEST_DIR = os.path.join(ROOT_DIR, "data/post_processed/tables_data")
    return


@app.cell
def _(ROOT_DIR, os, pd):
    df_attack_techniques = pd.read_csv(
        os.path.join(
            ROOT_DIR,
            "data/pre_processed/attack_techniques_base.csv",
        )
    )
    return (df_attack_techniques,)


@app.cell
def _(df_attack_techniques):
    df_attack_techniques
    return


@app.cell
def _(get_full_attack_name):
    def add_attack_name(df_classwise, df_attack_techniques):
        df_classwise_2 = df_classwise.copy()
        attack_names = []
        for row in df_classwise_2.itertuples():
            if row.attack_id == "N/A":
                attack_name = row.attack_id
            else:
                attack_name = get_full_attack_name(
                    df_attack_techniques, attack_id=row.attack_id
                )
            attack_names.append(attack_name)
        df_classwise_2["attack-name"] = attack_names
        return df_classwise_2
    return (add_attack_name,)


@app.cell
def _():
    # Remove 0 values
    # Order columns
    # Run for all mapping types
    # Add frequency
    return


@app.cell
def _():
    #
    return


@app.cell
def _(postprocess):
    def count_predicitons(data):
        # Initialize a dictionary to count labels
        pred_counts = {}
        true_counts = {}

        # Iterate over each item in the data
        for item in data:
            true_labels, predictions = postprocess(item)

            # Assuming each item has a 'predictions' key with a list of labels
            for label in predictions:
                # Count each label
                if label in pred_counts:
                    pred_counts[label] += 1
                else:
                    pred_counts[label] = 1

            for label in true_labels:
                if label in true_counts:
                    true_counts[label] += 1
                else:
                    true_counts[label] = 1

        return true_counts, pred_counts
    return (count_predicitons,)


@app.cell
def _(pd):
    def create_df_countings(counted_true):
        df = pd.DataFrame([counted_true]).T
        df_2 = df.copy()
        df_2 = df_2.fillna(0)
        # Rename columns for clarity
        df_2.columns = ["n-true"]
        # Convert from float64 to integer
        df_2["n-true"] = df_2["n-true"].astype("Int64")
        df_2 = df_2.reset_index(names="attack_id")
        return df_2
    return (create_df_countings,)


@app.cell
def _():
    # Read exp id
    #
    # Count
    # Count positive values and zero values Remove recall-at-ten zero values
    return


@app.cell
def _(
    add_attack_name,
    compute_classwise_recall_at_k,
    count_predicitons,
    create_df_countings,
    pd,
):
    def create_classwise_df(data, df_attack_techniques, k=10):
        cls_rec_at_k = compute_classwise_recall_at_k(data, k=k)
        cls_rec_at_k = {
            k: v
            for k, v in sorted(
                cls_rec_at_k.items(), key=lambda item: item[1], reverse=True
            )
        }
        df_scores = pd.DataFrame(
            list(cls_rec_at_k.items()), columns=["attack_id", "recall-at-ten"]
        )
        df_scores_2 = add_attack_name(df_scores, df_attack_techniques)

        counted_true, counted_predictions = count_predicitons(data)

        df_countings = create_df_countings(counted_true=counted_true)

        df_scores_3 = df_scores_2.merge(df_countings, on="attack_id")

        # Rename attack id to make latex compatible CSV
        df_scores_3 = df_scores_3.rename(columns={"attack_id": "attack-id"})

        # Reorder columns
        df_scores_3 = df_scores_3[
            ["attack-id", "attack-name", "recall-at-ten", "n-true"]
        ]

        return df_scores_3
    return (create_classwise_df,)


@app.function
def merge_train_test(df_train, df_test):
    merged_df = df_train.merge(
        df_test,
        how="outer",
        on=["attack-id", "attack-name"],
        suffixes=("-train", "-test"),
    )
    # Fill empty values
    merged_df_2 = merged_df.copy()
    merged_df_2["recall-at-ten-train"] = merged_df_2[
        "recall-at-ten-train"
    ].fillna(0)
    merged_df_2["recall-at-ten-test"] = merged_df_2[
        "recall-at-ten-test"
    ].fillna(0)

    # Filling empty values in n-true-train and n-true-test with 0
    merged_df_2["n-true-train"] = merged_df_2["n-true-train"].fillna(0)
    merged_df_2["n-true-test"] = merged_df_2["n-true-test"].fillna(0)

    # Matex Latex compatible

    return merged_df_2


@app.function
def filter_positive_recall(df):
    # Filter the dataframe to include rows where at least one of 'recall-at-ten-train' or 'recall-at-ten-test' is positive
    positive_recall_df = df[
        (df["recall-at-ten-train"] > 0) | (df["recall-at-ten-test"] > 0)
    ]
    return positive_recall_df


@app.function
def count_ratio_positive(df, data_split=None):
    # Remove None
    df = df.loc[df["attack-id"] != "N/A"]
    # n_true must be > 0, recall == 0
    if data_split == "train":
        n_true = "n-true-train"
        recall = "recall-at-ten-train"
        count_positive = len(df[(df[n_true] > 0)])
        count_recall_positive = len(df[(df[recall] > 0)])
    elif data_split == "test":
        n_true = "n-true-test"
        recall = "recall-at-ten-test"
        count_positive = len(df[(df[n_true] > 0)])
        count_recall_positive = len(df[(df[recall] > 0)])
    else:
        train_techniques = df.loc[(df["n-true-train"] > 0)]["attack-id"]
        test_techniques = df.loc[(df["n-true-test"] > 0)]["attack-id"]
        # All techniques existing in the ground truth in one of the splits
        unique_techniques = set(train_techniques).union(set(test_techniques))
        n_unique_techniques = len(unique_techniques)
        # Techniques with true positive predictions
        tp_train = df.loc[df["recall-at-ten-train"] > 0]["attack-id"]
        tp_test = df.loc[df["recall-at-ten-test"] > 0]["attack-id"]
        tp = set(tp_train).union(set(tp_test))
        n_tp = len(tp)

    if data_split is None:
            answer = f"{n_tp} of {n_unique_techniques} existing techniques have one or more positive predictions"
    else:
        answer = f"{count_recall_positive} of {count_positive} existing techniques in {data_split} have one or more positive predictions"
    return answer


@app.cell
def _():
    return


@app.cell
def _(mo):
    mo.md(r"""## Exploitation Technique""")
    return


@app.cell
def _(ROOT_DIR, json, os):
    train_exp_path = os.path.join(
        ROOT_DIR, "models/categorized_mapping/experiments/CVET2-1029/history.json"
    )
    with open(train_exp_path, "r") as train_exp_file:
        train_exp_data = json.load(train_exp_file)
    return (train_exp_data,)


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(create_classwise_df, df_attack_techniques, train_exp_data):
    train_exp_classwise_3 = create_classwise_df(
        train_exp_data, df_attack_techniques
    )
    return (train_exp_classwise_3,)


@app.cell
def _():
    # recall-at-ten train, n-true-train, recall-at-ten test, n-true-test
    return


@app.cell
def _(ROOT_DIR, json, os):
    # Test
    test_exp_path = os.path.join(
        ROOT_DIR, "models/categorized_mapping/experiments/CVET2-1030/history.json"
    )
    with open(test_exp_path, "r") as test_exp_file:
        test_exp_data = json.load(test_exp_file)
    return (test_exp_data,)


@app.cell
def _(create_classwise_df, df_attack_techniques, test_exp_data):
    test_exp_classwise = create_classwise_df(test_exp_data, df_attack_techniques)
    return (test_exp_classwise,)


@app.cell
def _():
    # Merge train and test
    # If attack id does not exist add n_true 0, recall-at-ten N/A
    return


@app.cell
def _(test_exp_classwise, train_exp_classwise_3):
    exp_merged_3 = merge_train_test(train_exp_classwise_3, test_exp_classwise)
    exp_merged_3
    return (exp_merged_3,)


@app.cell
def _(exp_merged_3):
    exp_merged_positive = filter_positive_recall(exp_merged_3)
    return


@app.cell
def _():
    # exp_merged_positive.to_csv(os.path.join(DEST_DIR, "exp_classwise.csv"), float_format='%.2f', index_label=False, index=False)
    return


@app.function
def filter_recall_and_examples(df, mapping_type, examples_threshold=1, recall_at_ten_threshold=0, has_none=True):
    """
    For a test dataframe and a mapping type, filter dataframe based threshold recall value and number of in-context examples in train
    Print out the coverage of techniques among the relevant ones. 
    """
    df_filtered = df.copy()
    df_filtered_examples = df_filtered.loc[df["n-true-train"] >= examples_threshold]
    df_filtered_final = df_filtered_examples.loc[df_filtered_examples["recall-at-ten-test"] > 0]
    all_techniques = len(df)
    relevant_techniques = len(df_filtered_examples)
    num_tech_true_pred = len(df_filtered_final)
    if has_none:
        # Ommit the None value if it exist
        all_techniques -= 1
        relevant_techniques -= 1
        num_tech_true_pred -= 1
    print(f"{relevant_techniques} of {all_techniques} {mapping_type} techniques in test have at least {examples_threshold} mapping(s) in train: ")
    print(f"{num_tech_true_pred} techniques have true predictions: ")


@app.cell
def _(mo):
    mo.md(r"""## Primary Impact""")
    return


@app.cell
def _(ROOT_DIR, json, os):
    train_prim_path = os.path.join(
        ROOT_DIR, "models/categorized_mapping/experiments/CVET2-783/history.json"
    )
    with open(train_prim_path, "r") as train_prim_file:
        train_prim_data = json.load(train_prim_file)
    return (train_prim_data,)


@app.cell
def _(create_classwise_df, df_attack_techniques, train_prim_data):
    train_prim_classwise = create_classwise_df(
        train_prim_data, df_attack_techniques
    )
    return (train_prim_classwise,)


@app.cell
def _(ROOT_DIR, json, os):
    # Test
    test_prim_path = os.path.join(
        ROOT_DIR, "models/categorized_mapping/experiments/CVET2-762/history.json"
    )
    with open(test_prim_path, "r") as test_prim_file:
        test_prim_data = json.load(test_prim_file)
    return (test_prim_data,)


@app.cell
def _(create_classwise_df, df_attack_techniques, test_prim_data):
    test_prim_classwise = create_classwise_df(test_prim_data, df_attack_techniques)
    return (test_prim_classwise,)


@app.cell
def _(test_prim_classwise, train_prim_classwise):
    prim_merged = merge_train_test(train_prim_classwise, test_prim_classwise)
    prim_merged
    return (prim_merged,)


@app.cell
def _(prim_merged):
    prim_merged_positive = filter_positive_recall(prim_merged)
    prim_merged_positive
    return


@app.cell
def _():
    # prim_merged_positive.to_csv(os.path.join(DEST_DIR, "prim_classwise.csv"), float_format='%.2f')
    return


@app.cell
def _(mo):
    mo.md(r"""## Secondary Impact""")
    return


@app.cell
def _(ROOT_DIR, json, os):
    # train
    train_sec_path = os.path.join(
        ROOT_DIR, "models/categorized_mapping/experiments/CVET2-782/history.json"
    )
    with open(train_sec_path, "r") as train_sec_file:
        train_sec_data = json.load(train_sec_file)
    return (train_sec_data,)


@app.cell
def _(create_classwise_df, df_attack_techniques, train_sec_data):
    train_sec_classwise = create_classwise_df(train_sec_data, df_attack_techniques)
    return (train_sec_classwise,)


@app.cell
def _(ROOT_DIR, json, os):
    # test
    test_sec_path = os.path.join(
        ROOT_DIR, "models/categorized_mapping/experiments/CVET2-761/history.json"
    )
    with open(test_sec_path, "r") as test_sec_file:
        test_sec_data = json.load(test_sec_file)
    return (test_sec_data,)


@app.cell
def _(create_classwise_df, df_attack_techniques, test_sec_data):
    test_sec_classwise = create_classwise_df(test_sec_data, df_attack_techniques)
    return (test_sec_classwise,)


@app.cell
def _(test_sec_classwise, train_sec_classwise):
    sec_merged = merge_train_test(train_sec_classwise, test_sec_classwise)
    return (sec_merged,)


@app.cell
def _(sec_merged):
    sec_merged
    return


@app.cell
def _(sec_merged):
    sec_merged_positive = filter_positive_recall(sec_merged)
    sec_merged_positive
    return


@app.cell
def _():
    # sec_merged_positive.to_csv(os.path.join(DEST_DIR, "sec_classwise.csv"), float_format='%.2f')
    return


if __name__ == "__main__":
    app.run()
