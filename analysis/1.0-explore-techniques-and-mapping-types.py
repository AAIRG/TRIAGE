import marimo

__generated_with = "0.15.0"
app = marimo.App(width="medium")


@app.cell
def _():
    # Configuration to import local modules when running in a notebook
    import os
    import sys
    # Get the project repository path
    ROOT_DIR = os.path.abspath('..')

    # Add to the system path
    sys.path.append(ROOT_DIR)
    return ROOT_DIR, os


@app.cell
def _():
    import pandas as pd
    from lib.utils import parse_arguments
    import matplotlib.pyplot as plt
    import numpy as np
    return np, parse_arguments, pd, plt


@app.cell
def _():
    from lib.load_data import BaseDataHandler
    return (BaseDataHandler,)


@app.cell
def _(ROOT_DIR, os):
    # Save figure to Overleaf project via dropbox
    DEST_DIR = os.path.join(ROOT_DIR, "reports/figures")
    # Change to graphs/data?
    DEST_DIR_2 = os.path.join(ROOT_DIR, "data/post_processed/tables_data")
    return


@app.cell
def _(parse_arguments):
    args = parse_arguments()
    return (args,)


@app.cell
def _(args):
    args.use_parent_techniques = True
    return


@app.cell
def _(BaseDataHandler, args):
    data_handler = BaseDataHandler(args)
    return (data_handler,)


@app.cell
def _(data_handler):
    cves_train = data_handler.cves_train
    return


@app.cell
def _(data_handler):
    df_attack_labels = data_handler.df_attack_labels
    return (df_attack_labels,)


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md(r"""## Explore techniques""")
    return


@app.cell
def _(plt, technique_value_counts):
    # Create some sample data since none was provided.
    # This creates a pandas Series with 50 techniques and their counts.

    # Original plotting code provided by the user
    plt.figure(figsize=(10, 6))
    bars = plt.bar(
        technique_value_counts.index, technique_value_counts.values, color="teal"
    )
    plt.xlabel("Technique")
    plt.ylabel("Counts")
    #plt.title("Counts of Techniques")
    plt.xticks(rotation=45)
    ax = plt.gca()

    # The user wants to only include every tenth label, including the first.
    # We can achieve this by getting the tick labels and then setting them again,
    # but making a label an empty string if it's not the first, tenth, twentieth, etc.

    # Get the current tick labels
    labels = [item.get_text() for item in ax.get_xticklabels()]

    # Create a new list of labels.
    # The first label (index 0) and every 10th label thereafter will be kept.
    # All other labels are replaced with an empty string.
    new_labels = [label if i % 5 == 0 else "" for i, label in enumerate(labels)]

    # Set the new tick labels on the plot
    ax.set_xticklabels(new_labels)

    # Adjust plot to ensure everything fits without overlapping
    plt.tight_layout()

    NAME = "counts_of_techniques"
    # plt.savefig(os.path.join(DEST_DIR, NAME + ".pdf"), format="pdf")


    plt.show()
    return


@app.cell
def _(df_attack_labels):
    technique_value_counts = df_attack_labels.value_counts(subset="attack_id")
    return (technique_value_counts,)


@app.cell
def _():
    #technique_value_counts.to_csv(os.path.join(ROOT_DIR, DEST_DIR_2, "technique_value_counts.csv"), float_format='%.2f', index_label=False, index=False)
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    We have a highly imbalanced data, with a significant proportion of the techniques occurring only once. 

    Let's look closer on the techniques for each mapping type
    """
    )
    return


@app.cell
def _(mo):
    mo.md(r"""## Techniques per CVE and mapping type""")
    return


@app.cell
def _(df_attack_labels):
    # Total number of attack techniques for each mapping type
    df_attack_labels.value_counts(subset="mapping_type")
    return


@app.cell
def _(df_attack_labels):
    # Most common technique per mapping type
    exp_labels = df_attack_labels.loc[df_attack_labels["mapping_type"] == "exploitation_technique"]
    return (exp_labels,)


@app.cell
def _():
    # exp_labels.value_counts(subset=["attack_id", "attack_name"])[:6]
    return


@app.cell
def _(df_attack_labels):
    prim_labels = df_attack_labels.loc[df_attack_labels["mapping_type"] == "primary_impact"]
    return (prim_labels,)


@app.cell
def _():
    # prim_labels.value_counts(subset=["attack_id", "attack_name"])[:6]
    return


@app.cell
def _(df_attack_labels):
    sec_labels = df_attack_labels.loc[df_attack_labels["mapping_type"] == "secondary_impact"]
    return (sec_labels,)


@app.cell
def _():
    # sec_labels.value_counts(subset=["attack_id", "attack_name"])[:6]
    return


@app.cell
def _(np, pd):
    def count_technique_labels_per_cve(df_attack_labels, labels):
        none_labels = set(df_attack_labels["CVE ID"]) - set(labels["CVE ID"])
        individual_counts = labels.value_counts(subset="CVE ID").values
        # count options
        count_1 = np.count_nonzero(individual_counts == 1)
        count_2 = np.count_nonzero(individual_counts == 2)
        count_3 = np.count_nonzero(individual_counts == 3)
        count_more = np.count_nonzero(individual_counts > 3)
        df = pd.DataFrame({"Techniques per CVE": ["Zero", "One", "Two", "Three", "More than three"],
                           "occurrences": [len(none_labels), count_1, count_2, count_3, count_more]})
        return df
    return (count_technique_labels_per_cve,)


@app.cell
def _(count_technique_labels_per_cve, df_attack_labels, exp_labels):
    exp_counts = count_technique_labels_per_cve(df_attack_labels, exp_labels)
    return (exp_counts,)


@app.cell
def _(count_technique_labels_per_cve, df_attack_labels, prim_labels):
    prim_counts = count_technique_labels_per_cve(df_attack_labels, prim_labels)
    return (prim_counts,)


@app.cell
def _(count_technique_labels_per_cve, df_attack_labels, sec_labels):
    sec_counts = count_technique_labels_per_cve(df_attack_labels, sec_labels)
    return (sec_counts,)


@app.cell
def _():
    # Combine tables, append prim-/exp-/sec- to the Occurrences

    # Final table occurrences on top. Below prim exp sec
    # Make latex, rename techniques per cve to techniques-per-cve
    return


@app.function
def process_latex_table(exp_counts, prim_counts, sec_counts):
    # Rename
    exp_counts = exp_counts.rename(columns = {"occurrences": "exploitation-occurrences"})
    prim_counts = prim_counts.rename(columns = {"occurrences": "primary-occurrences"})
    sec_counts = sec_counts.rename(columns = {"occurrences": "secondary-occurrences"})
    # Combine tables
    df_combined = exp_counts.merge(prim_counts, on="Techniques per CVE")
    df_combined = df_combined.merge(sec_counts, on="Techniques per CVE")
    df_combined = df_combined.rename(columns={"Techniques per CVE": "techniques-per-cve"})
    return df_combined


@app.cell
def _(exp_counts, prim_counts, sec_counts):
    df_combined = process_latex_table(exp_counts, prim_counts, sec_counts)
    df_combined
    return


@app.cell
def _():
    # df_combined.to_csv(os.path.join(DEST_DIR_2, "mapping_types_per_cve.csv"), float_format='%.2f', index_label=False, index=False)
    return


if __name__ == "__main__":
    app.run()
