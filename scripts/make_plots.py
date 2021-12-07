# © Copyright Technical University of Denmark
"""
Make the evaluation plots. 
Reqires the following:
    --bert_crossvalidation_metrics:               output of cross_validate.py
    --signalp5_crossvalidation_metrics:           file in same format as file above, but for another model to compare to
    --bert_viterbi_paths                          output of average_viterbi_decode.py
    --signalp5_viterbi_paths                      file in same format as file above, but for another model to compare to
    --bert_crossvalidation_metrics_randomized     output of cross_validate.py --randomize_kingdoms \
    --signalp5_crossvalidation_metrics_randomized file in same format as file above, but for another model to compare to
    --output_dir                                  where to save the plots
"""
import pandas as pd
import ast
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import argparse
from sklearn.metrics import confusion_matrix


def get_sp_len(token_ids):
    token_ids = np.array(token_ids)
    sp_pos = np.isin(token_ids, [5, 11, 19, 26, 31])
    pos_indices = np.where(sp_pos)[0]

    return pos_indices.max() + 1 if len(pos_indices) > 0 else np.nan


def get_n_len(token_ids):
    token_ids = np.array(token_ids)
    sp_pos = np.isin(token_ids, [3, 9, 16, 23, 31])
    pos_indices = np.where(sp_pos)[0]
    return pos_indices.max() - pos_indices.min() + 1 if len(pos_indices) > 0 else np.nan


def get_h_len(token_ids):
    token_ids = np.array(token_ids)
    sp_pos = np.isin(token_ids, [4, 10, 18, 25])
    pos_indices = np.where(sp_pos)[0]
    return pos_indices.max() - pos_indices.min() + 1 if len(pos_indices) > 0 else np.nan


def get_c_len(token_ids):
    token_ids = np.array(token_ids)
    sp_pos = np.isin(token_ids, [5, 19])
    pos_indices = np.where(sp_pos)[0]
    return pos_indices.max() - pos_indices.min() + 1 if len(pos_indices) > 0 else np.nan


aas = {
    "A": 0,
    "R": 1,
    "N": 2,
    "D": 3,
    "C": 4,
    "Q": 5,
    "E": 6,
    "G": 7,
    "H": 8,
    "I": 9,
    "L": 10,
    "K": 11,
    "M": 12,
    "F": 13,
    "P": 14,
    "S": 15,
    "T": 16,
    "W": 17,
    "Y": 18,
    "V": 19,
}


def count_aas(sequence, token_ids, skip_n_methionine=False):

    sequence = np.array([aas[x] for x in sequence])
    # count aas in n
    sp_pos = np.where(np.isin(token_ids, [3, 9, 16, 23, 31]))[0]
    if len(sp_pos) > 0:
        start = 1 if skip_n_methionine else sp_pos.min()
        end = sp_pos.max() + 1
        region_aas = sequence[start:end]
        aas_n = np.bincount(region_aas, minlength=20)
    else:
        aas_n = np.zeros(20)

    # count aas in h
    sp_pos = np.where(np.isin(token_ids, [4, 10, 18, 25]))[0]
    if len(sp_pos) > 0:
        start = sp_pos.min()
        end = sp_pos.max() + 1
        region_aas = sequence[start:end]
        aas_h = np.bincount(region_aas, minlength=20)
    else:
        aas_h = np.zeros(20)

    # count aas in c
    sp_pos = np.where(np.isin(token_ids, [5, 19]))[0]
    if len(sp_pos) > 0:
        start = sp_pos.min()
        end = sp_pos.max() + 1
        region_aas = sequence[start:end]
        aas_c = np.bincount(region_aas, minlength=20)
    else:
        aas_c = np.zeros(20)

    return aas_n, aas_h, aas_c


def compute_net_charge(amino_acid_counts):
    # pos. aas contribute one pos. charge, neg. aas one negative. just do pos-neg for net charge
    # D, E are negative
    neg_charge = amino_acid_counts[3] + amino_acid_counts[6]
    # R, K, H are positive
    pos_charge = amino_acid_counts[1] + amino_acid_counts[8] + amino_acid_counts[11]

    return pos_charge - neg_charge


KYTE_DOOLITTE = {
    "I": 4.5,
    "V": 4.2,
    "L": 3.8,
    "F": 2.8,
    "C": 2.5,
    "M": 1.9,
    "A": 1.8,
    "G": -0.4,
    "T": -0.7,
    "S": -0.8,
    "W": -0.9,
    "Y": -1.3,
    "P": -1.6,
    "H": -3.2,
    "E": -3.5,
    "Q": -3.5,
    "D": -3.5,
    "N": -3.5,
    "K": -3.9,
    "R": -4.5,
}

# make new dict mapping idx to kd score
idx_to_hydrophobicity = dict(zip(aas.values(), [KYTE_DOOLITTE[x] for x in aas.keys()]))


def compute_hydrophobicity(amino_acid_counts):
    total_score = 0
    for i in range(len(aas)):
        score = amino_acid_counts[i] * idx_to_hydrophobicity[i]
        total_score += score

    return total_score


def make_conf_matrix(
    y_true,
    y_pred,
    percent=False,
    categories=[
        "Other",
        "Sec\nSPI",
        "Sec\nSPII",
        "Tat\nSPI",
        "Tat\nSPII",
        "Sec\nSPIII",
    ],
    label_size=10,
    axlabel_size=10,
    tick_size=10,
    ax=None,
):
    confusion = confusion_matrix(y_true, y_pred)
    if percent:
        confusion_norm = confusion / confusion.sum(axis=1)[:, None]

        group_counts = ["{0:0.0f}".format(value) for value in confusion.flatten()]
        group_percentages = [
            "{0:.1%}".format(value) for value in confusion_norm.flatten()
        ]
        labels = [f"{v1}\n{v2}" for v1, v2 in zip(group_counts, group_percentages)]
        labels = np.asarray(labels).reshape(len(categories), len(categories))

        b = sns.heatmap(
            confusion_norm,
            cmap="Blues",
            annot=labels,
            xticklabels=categories,
            yticklabels=categories,
            fmt="",
            cbar=False,
            annot_kws={"size": label_size},
            ax=ax,
        )

    else:
        b = sns.heatmap(
            confusion,
            cmap="Blues",
            annot=True,
            xticklabels=categories,
            yticklabels=categories,
        )

    return b


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--bert_crossvalidation_metrics", type=str, default=None)
    parser.add_argument("--signalp5_crossvalidation_metrics", type=str, default=None)

    parser.add_argument("--bert_viterbi_paths", type=str, default=None)
    parser.add_argument("--signalp5_viterbi_paths", type=str, default=None)

    parser.add_argument(
        "--bert_crossvalidation_metrics_randomized", type=str, default=None
    )
    parser.add_argument(
        "--signalp5_crossvalidation_metrics_randomized", type=str, default=None
    )

    parser.add_argument("--output_dir", type=str, default="plots")

    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        print(f"made {args.output_dir}")
        os.makedirs(args.output_dir)

    # confusion matrix
    if args.bert_viterbi_paths is not None:
        df = pd.read_csv(args.bert_viterbi_paths)
        df["Path"] = df["Path"].apply(ast.literal_eval)

        type_to_label = {
            "NO_SP": 0,
            "SP": 1,
            "LIPO": 2,
            "TAT": 3,
            "TATLIPO": 4,
            "PILIN": 5,
        }

        df["True label"] = df["Type"].apply(lambda x: type_to_label[x])

        cols = [
            "Eukarya",
            "Archaea",
            "Gram-Positive Bacteria",
            "Gram-Negative Bacteria",
        ]

        fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(20, 5))

        pad = 5  # in points

        kwargs = {"label_size": 12, "axlabel_size": 12, "tick_size": 15}

        ax = axes[0]  # plt.subplot(4,2,1)
        df_plot = df.loc[df["Kingdom"] == "EUKARYA"]
        df_plot.loc[
            df_plot["Pred label"] != 0, "Pred label"
        ] = 1  # fix SPII, TAT to SPI

        make_conf_matrix(
            df_plot["True label"].astype(int),
            df_plot["Pred label"].astype(int),
            percent=True,
            **kwargs,
            ax=ax,
            categories=["Other", "Sec\nSPI"],
        )

        ax = axes[1]
        df_plot = df.loc[df["Kingdom"] == "ARCHAEA"]
        make_conf_matrix(
            df_plot["True label"].astype(int),
            df_plot["Pred label"].astype(int),
            percent=True,
            **kwargs,
            ax=ax,
        )

        ax = axes[2]
        df_plot = df.loc[df["Kingdom"] == "POSITIVE"]
        make_conf_matrix(
            df_plot["True label"].astype(int),
            df_plot["Pred label"].astype(int),
            percent=True,
            **kwargs,
            ax=ax,
        )

        ax = axes[3]
        df_plot = df.loc[df["Kingdom"] == "NEGATIVE"]
        make_conf_matrix(
            df_plot["True label"].astype(int),
            df_plot["Pred label"].astype(int),
            percent=True,
            **kwargs,
            ax=ax,
        )

        for ax, col in zip(axes, cols):
            ax.annotate(
                col,
                xy=(0.5, 1),
                xytext=(0, pad),
                xycoords="axes fraction",
                textcoords="offset points",
                size=15,
                ha="center",
                va="baseline",
            )

        for ax in axes:
            ax.set_ylabel("True label", size=15)
            ax.set_xlabel("Predicted label", size=15)

        for ax in axes.flat:
            ax.set_yticklabels(
                ax.get_yticklabels(), rotation=0, horizontalalignment="right"
            )

        fig.tight_layout()

        plt.savefig(os.path.join(args.output_dir, "confusion_matrices_bert.png"))

    if args.signalp5_viterbi_paths is not None:

        df = pd.read_csv(args.signalp5_viterbi_paths)
        df["Path"] = df["Path"].apply(ast.literal_eval)

        type_to_label = {
            "NO_SP": 0,
            "SP": 1,
            "LIPO": 2,
            "TAT": 3,
            "TATLIPO": 4,
            "PILIN": 5,
        }

        df["True label"] = df["Type"].apply(lambda x: type_to_label[x])

        cols = [
            "Eukarya",
            "Archaea",
            "Gram-Positive Bacteria",
            "Gram-Negative Bacteria",
        ]

        fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(20, 5))
        # plt.setp(axes.flat, xlabel='X-label', ylabel='Y-label')

        pad = 5  # in points

        kwargs = {"label_size": 12, "axlabel_size": 12, "tick_size": 15}

        ax = axes[0]  # plt.subplot(4,2,1)
        df_plot = df.loc[df["Kingdom"] == "EUKARYA"]
        df_plot.loc[
            df_plot["Pred label"] != 0, "Pred label"
        ] = 1  # fix SPII, TAT to SPI

        make_conf_matrix(
            df_plot["True label"].astype(int),
            df_plot["Pred label"].astype(int),
            percent=True,
            **kwargs,
            ax=ax,
            categories=["Other", "Sec\nSPI"],
        )

        ax = axes[1]
        df_plot = df.loc[df["Kingdom"] == "ARCHAEA"]
        make_conf_matrix(
            df_plot["True label"].astype(int),
            df_plot["Pred label"].astype(int),
            percent=True,
            **kwargs,
            ax=ax,
        )

        ax = axes[2]
        df_plot = df.loc[df["Kingdom"] == "POSITIVE"]
        make_conf_matrix(
            df_plot["True label"].astype(int),
            df_plot["Pred label"].astype(int),
            percent=True,
            **kwargs,
            ax=ax,
        )

        ax = axes[3]
        df_plot = df.loc[df["Kingdom"] == "NEGATIVE"]
        make_conf_matrix(
            df_plot["True label"].astype(int),
            df_plot["Pred label"].astype(int),
            percent=True,
            **kwargs,
            ax=ax,
        )

        for ax, col in zip(axes, cols):
            ax.annotate(
                col,
                xy=(0.5, 1),
                xytext=(0, pad),
                xycoords="axes fraction",
                textcoords="offset points",
                size=15,
                ha="center",
                va="baseline",
            )

        for ax in axes:
            ax.set_ylabel("True label", size=15)
            ax.set_xlabel("Predicted label", size=15)

        for ax in axes.flat:
            ax.set_yticklabels(
                ax.get_yticklabels(), rotation=0, horizontalalignment="right"
            )

        fig.tight_layout()

        plt.savefig(os.path.join(args.output_dir, "confusion_matrices_signalp5.png"))

    # crossvalidation metrics
    if (
        args.bert_crossvalidation_metrics is not None
        and args.signalp5_crossvalidation_metrics is not None
    ):
        metrics_1 = pd.read_csv(
            args.signalp5_crossvalidation_metrics, index_col=0
        ).mean(axis=1)
        df = pd.read_csv(args.bert_crossvalidation_metrics, index_col=0)
        metrics_2 = (
            df.loc[df.index.str.contains("mcc|window")].astype(float).mean(axis=1)
        )
        df = pd.DataFrame({"SignalP 5": metrics_1, "Bert": metrics_2})

        # build additional identifer columns from split index
        exp_df = df.reset_index()["index"].str.split("_", expand=True)
        exp_df.columns = ["kingdom", "type", "metric", "no", "window"]
        exp_df.index = df.index

        # put together
        df = df.join(exp_df)

        nice_label_dict = {
            "NO_SP": "Other",
            "SP": "Sec/SPI",
            "LIPO": "Sec/SPII",
            "TAT": "Tat/SPI",
            "TATLIPO": "Tat/SPII",
            "PILIN": "Sec/SPIII",
            None: None,
        }

        df["type"] = df["type"].apply(lambda x: nice_label_dict[x])

        plt.figure(figsize=(12, 8))

        ax = plt.subplot(2, 1, 1)

        df_plot = df.loc[df["metric"] == "mcc1"][
            ["kingdom", "type", "SignalP 5", "Bert"]
        ]  # , 'crossval_std']]
        df_plot = df_plot.set_index(df_plot["kingdom"] + "\n" + df_plot["type"])
        df_plot = df_plot.sort_index()

        df_plot = df_plot.rename({"crossval_mean": "Bert-CRF"}, axis=1)

        df_plot.plot(kind="bar", ax=ax, ylim=(0, 1), rot=0).legend(loc="lower left")
        plt.title("MCC 1")

        ax = plt.subplot(2, 1, 2)
        df_plot = df.loc[df["kingdom"].isin(["ARCHAEA", "POSITIVE", "NEGATIVE"])]
        df_plot = df_plot.loc[df["metric"] == "mcc2"][
            ["kingdom", "type", "SignalP 5", "Bert"]
        ]  # , 'crossval_std']]
        df_plot = df_plot.set_index(df_plot["kingdom"] + "\n" + df_plot["type"])
        df_plot = df_plot.sort_index()

        df_plot = df_plot.rename({"crossval_mean": "Bert-CRF"}, axis=1)

        df_plot.plot(kind="bar", ax=ax, ylim=(0, 1), rot=0).legend(loc="lower left")
        plt.title("MCC 2")

        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, "mcc.png"))

        df_plot = df.loc[df["metric"] == "recall"][
            ["kingdom", "type", "window", "SignalP 5", "Bert"]
        ]  # , 'crossval_std']]
        df_plot = df_plot.set_index(
            df_plot["type"] + "\n ±" + df_plot["window"].astype(int).astype(str)
        )
        df_plot = df_plot.sort_index()

        plt.figure(figsize=(16, 10))
        for idx, kingdom in enumerate(df_plot["kingdom"].unique()):
            ax = plt.subplot(2, 2, idx + 1)
            df_plot.loc[df_plot["kingdom"] == kingdom][["SignalP 5", "Bert"]].plot(
                kind="bar", ax=ax, title=kingdom + " CS recall", rot=90
            )

        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, "recall.png"))

        df_plot = df.loc[df["metric"] == "precision"][
            ["kingdom", "type", "window", "SignalP 5", "Bert"]
        ]  # , 'crossval_std']]
        df_plot = df_plot.set_index(
            df_plot["type"] + "\n ±" + df_plot["window"].astype(int).astype(str)
        )
        df_plot = df_plot.sort_index()

        plt.figure(figsize=(16, 10))
        for idx, kingdom in enumerate(df_plot["kingdom"].unique()):
            ax = plt.subplot(2, 2, idx + 1)
            df_plot.loc[df_plot["kingdom"] == kingdom][["SignalP 5", "Bert"]].plot(
                kind="bar", ax=ax, title=kingdom + " CS precision", rot=90
            )

        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, "precision.png"))

    ####
    #    Region plots
    ####
    # palettepalette name, list, or dict
    # Colors to use for the different levels of the hue variable.
    # Should be something that can be interpreted by color_palette(), or a dictionary mapping hue levels to matplotlib colors.
    if args.bert_viterbi_paths is not None:
        df = pd.read_csv(args.bert_viterbi_paths)
        df["Path"] = df["Path"].apply(ast.literal_eval)

        # set up a color palette for seaborn to be used in all plots that are grouped by kingdom
        # also fix plotting order for all plots
        palette = {
            "EUKARYA": (0.12156862745098039, 0.4666666666666667, 0.7058823529411765),
            "NEGATIVE": (0.17254901960784313, 0.6274509803921569, 0.17254901960784313),
            "POSITIVE": (0.8392156862745098, 0.15294117647058825, 0.1568627450980392),
            "ARCHAEA": (1.0, 0.4980392156862745, 0.054901960784313725),
        }
        hue_order = list(palette.keys())

        # filter wrong predictions. not interesting for region quality
        type_to_label = {
            "NO_SP": 0,
            "SP": 1,
            "LIPO": 2,
            "TAT": 3,
            "TATLIPO": 4,
            "PILIN": 5,
        }
        pred_correctly = (
            df["Type"].apply(lambda x: type_to_label[x]) == df["Pred label"]
        )
        df = df.loc[pred_correctly]
        df = df.reset_index(drop=True)

        # compute all the region characteristics
        df["len_n"] = df["Path"].apply(lambda x: get_n_len(x))
        df["len_h"] = df["Path"].apply(lambda x: get_h_len(x))
        df["len_c"] = df["Path"].apply(lambda x: get_c_len(x))
        df["len_sp"] = df["Path"].apply(lambda x: get_sp_len(x))
        df["frac_n"] = df["len_n"] / df["len_sp"]
        df["frac_h"] = df["len_h"] / df["len_sp"]
        df["frac_c"] = df["len_c"] / df["len_sp"]

        aas_n_list = []
        aas_h_list = []
        aas_c_list = []
        for idx, row in df.iterrows():
            aas_n, aas_h, aas_c = count_aas(row["Sequence"], row["Path"])
            aas_n_list.append(aas_n)
            aas_h_list.append(aas_h)
            aas_c_list.append(aas_c)

        aas_n = np.stack(aas_n_list)
        aas_h = np.stack(aas_h_list)
        aas_c = np.stack(aas_c_list)

        charges_n = np.apply_along_axis(compute_net_charge, 1, aas_n)
        # NOTE need to fix n charge for eukarya and archaea - M is not formylated
        euk_idx = df["Kingdom"].values == "EUKARYA"
        arc_idx = df["Kingdom"].values == "ARCHAEA"
        charges_n[euk_idx] = charges_n[euk_idx] + 1
        charges_n[arc_idx] = charges_n[arc_idx] + 1

        charges_h = np.apply_along_axis(compute_net_charge, 1, aas_h)
        charges_c = np.apply_along_axis(compute_net_charge, 1, aas_c)

        hydrophobicity_n = np.apply_along_axis(compute_hydrophobicity, 1, aas_n)
        hydrophobicity_h = np.apply_along_axis(compute_hydrophobicity, 1, aas_h)
        hydrophobicity_c = np.apply_along_axis(compute_hydrophobicity, 1, aas_c)

        df["charge_n"] = charges_n
        df["charge_h"] = charges_h
        df["charge_c"] = charges_c
        df["hydrophobicity_n"] = hydrophobicity_n
        df["hydrophobicity_h"] = hydrophobicity_h
        df["hydrophobicity_c"] = hydrophobicity_c

        df.to_csv(os.path.join(args.output_dir, "region_characteristics.csv"))

        ## SPI charges
        plot_df = df.loc[df["Type"] == "SP"]
        plt.figure(figsize=(15, 10))
        ax = plt.subplot(2, 3, 1)
        sns.boxplot(
            x="Kingdom",
            y="charge_n",
            data=plot_df,
            palette=palette,
            hue_order=hue_order,
        )
        plt.ylabel("Net charge")
        plt.title("n-region Sec/SPI")

        plt.subplot(2, 3, 2, sharey=ax)
        sns.boxplot(
            x="Kingdom",
            y="charge_h",
            data=plot_df,
            palette=palette,
            hue_order=hue_order,
        )
        plt.ylabel("Net charge")
        plt.title("h-region Sec/SPI")

        plt.subplot(2, 3, 3, sharey=ax)
        sns.boxplot(
            x="Kingdom",
            y="charge_c",
            data=plot_df,
            palette=palette,
            hue_order=hue_order,
        )
        plt.ylabel("Net charge")
        plt.title("c-region Sec/SPI")

        plot_df = df.loc[df["Type"] == "TAT"]

        ax = plt.subplot(2, 3, 4)
        sns.boxplot(
            x="Kingdom",
            y="charge_n",
            data=plot_df,
            palette=palette,
            hue_order=hue_order,
        )
        plt.ylabel("Net charge")
        plt.title("n-region Tat/SPI")

        plt.subplot(2, 3, 5, sharey=ax)
        sns.boxplot(
            x="Kingdom",
            y="charge_h",
            data=plot_df,
            palette=palette,
            hue_order=hue_order,
        )
        plt.ylabel("Net charge")
        plt.title("h-region Tat/SPI")

        plt.subplot(2, 3, 6, sharey=ax)
        sns.boxplot(
            x="Kingdom",
            y="charge_c",
            data=plot_df,
            palette=palette,
            hue_order=hue_order,
        )
        plt.ylabel("Net charge")
        plt.title("c-region Tat/SPI")

        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, "spi_charges.png"))

        ##SPII charges
        plot_df = df.loc[df["Type"] == "LIPO"]
        plt.figure(figsize=(10, 10))
        ax = plt.subplot(2, 2, 1)
        sns.boxplot(
            x="Kingdom",
            y="charge_n",
            data=plot_df,
            palette=palette,
            hue_order=hue_order,
        )
        plt.ylabel("Net charge")
        plt.title("n-region Sec/SPII")

        plt.subplot(2, 2, 2, sharey=ax)
        sns.boxplot(
            x="Kingdom",
            y="charge_h",
            data=plot_df,
            palette=palette,
            hue_order=hue_order,
        )
        plt.ylabel("Net charge")
        plt.title("h-region Sec/SPII")

        plot_df = df.loc[df["Type"] == "TATLIPO"]
        ax = plt.subplot(2, 2, 3)
        sns.boxplot(
            x="Kingdom",
            y="charge_n",
            data=plot_df,
            palette=palette,
            hue_order=hue_order,
        )
        plt.ylabel("Net charge")
        plt.title("n-region Tat/SPII")

        plt.subplot(2, 2, 4, sharey=ax)
        sns.boxplot(
            x="Kingdom",
            y="charge_h",
            data=plot_df,
            palette=palette,
            hue_order=hue_order,
        )
        plt.ylabel("Net charge")
        plt.title("h-region Tat/SPII")

        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, "spii_charges.png"))
        plt.close()

        ## SPIII charges
        plot_df = df.loc[df["Type"] == "PILIN"]
        plt.figure(figsize=(5, 5))
        sns.boxplot(
            x="Kingdom",
            y="charge_n",
            data=plot_df,
            palette=palette,
            hue_order=hue_order,
        )
        plt.ylabel("Net charge")
        plt.title("Sec/SPIII")

        plt.savefig(os.path.join(args.output_dir, "spiii_charges.png"))
        plt.close()

        ## n-region hydrophobicity
        plt.figure(figsize=(20, 5))

        plot_df = df.loc[df["Type"] == "SP"]
        ax = plt.subplot(1, 4, 1)
        plt.title("n-region Sec/SPI")
        sns.boxplot(
            x="Kingdom",
            y="hydrophobicity_n",
            data=plot_df,
            palette=palette,
            hue_order=hue_order,
        )
        plt.ylabel("Hydrophobicity")

        plot_df = df.loc[df["Type"] == "LIPO"]
        plt.subplot(1, 4, 2, sharey=ax)
        plt.title("n-region Sec/SPII")
        sns.boxplot(
            x="Kingdom",
            y="hydrophobicity_n",
            data=plot_df,
            palette=palette,
            hue_order=hue_order,
        )
        plt.ylabel("Hydrophobicity")

        plot_df = df.loc[df["Type"] == "TAT"]
        plt.subplot(1, 4, 3, sharey=ax)
        plt.title("n-region Tat/SPI")
        sns.boxplot(
            x="Kingdom",
            y="hydrophobicity_n",
            data=plot_df,
            palette=palette,
            hue_order=hue_order,
        )
        plt.ylabel("Hydrophobicity")

        plot_df = df.loc[df["Type"] == "TATLIPO"]
        plt.subplot(1, 4, 4, sharey=ax)
        plt.title("n-region Tat/SPII")
        sns.boxplot(
            x="Kingdom",
            y="hydrophobicity_n",
            data=plot_df,
            palette=palette,
            hue_order=hue_order,
        )
        plt.ylabel("Hydrophobicity")

        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, "n_region_hydrophobicity.png"))
        plt.close()

        ## h-region hydrophobicity
        plt.figure(figsize=(20, 5))

        plot_df = df.loc[df["Type"] == "SP"]
        ax = plt.subplot(1, 4, 1)
        plt.title("h-region Sec/SPI")
        sns.boxplot(
            x="Kingdom",
            y="hydrophobicity_h",
            data=plot_df,
            palette=palette,
            hue_order=hue_order,
        )
        plt.ylabel("Hydrophobicity")

        plot_df = df.loc[df["Type"] == "LIPO"]
        plt.subplot(1, 4, 2, sharey=ax)
        plt.title("h-region Sec/SPII")
        sns.boxplot(
            x="Kingdom",
            y="hydrophobicity_h",
            data=plot_df,
            palette=palette,
            hue_order=hue_order,
        )
        plt.ylabel("Hydrophobicity")

        plot_df = df.loc[df["Type"] == "TAT"]
        plt.subplot(1, 4, 3, sharey=ax)
        plt.title("h-region Tat/SPI")
        sns.boxplot(
            x="Kingdom",
            y="hydrophobicity_h",
            data=plot_df,
            palette=palette,
            hue_order=hue_order,
        )
        plt.ylabel("Hydrophobicity")

        plot_df = df.loc[df["Type"] == "TATLIPO"]
        plt.subplot(1, 4, 4, sharey=ax)
        plt.title("h-region Tat/SPII")
        sns.boxplot(
            x="Kingdom",
            y="hydrophobicity_h",
            data=plot_df,
            palette=palette,
            hue_order=hue_order,
        )
        plt.ylabel("Hydrophobicity")

        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, "h_region_hydrophobicity.png"))
        plt.close()

        ## c-region hydrophobicity
        plt.figure(figsize=(10, 5))

        plot_df = df.loc[df["Type"] == "SP"]
        ax = plt.subplot(1, 2, 1)
        plt.title("c-region Sec/SPI")
        sns.boxplot(
            x="Kingdom",
            y="hydrophobicity_c",
            data=plot_df,
            palette=palette,
            hue_order=hue_order,
        )
        plt.ylabel("Hydrophobicity")

        plot_df = df.loc[df["Type"] == "TAT"]
        plt.subplot(1, 2, 2, sharey=ax)
        plt.title("c-region Tat/SPI")
        sns.boxplot(
            x="Kingdom",
            y="hydrophobicity_c",
            data=plot_df,
            palette=palette,
            hue_order=hue_order,
        )
        plt.ylabel("Hydrophobicity")

        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, "c_region_hydrophobicity.png"))
        plt.close()

        ## SPIII hydrophobicity
        plot_df = df.loc[df["Type"] == "PILIN"]
        plt.figure(figsize=(5, 5))
        sns.boxplot(
            x="Kingdom",
            y="hydrophobicity_n",
            data=plot_df,
            palette=palette,
            hue_order=hue_order,
        )
        plt.ylabel("Hydrophobicity")
        plt.title("Sec/SPIII")
        plt.savefig(os.path.join(args.output_dir, "spiii_hydrophobicity.png"))
        plt.close()

        ## amino acid compositions per region, per kingdom, per class
        dist_list = []
        for sp_type in ["SP", "LIPO", "TAT", "TATLIPO"]:
            for kingdom in ["EUKARYA", "ARCHAEA", "POSITIVE", "NEGATIVE"]:

                plot_df = df.loc[(df["Type"] == sp_type) & (df["Kingdom"] == kingdom)]

                aa_dist_n = aas_n[plot_df.index].sum(axis=0)
                aa_dist_n = aa_dist_n / aa_dist_n.sum()
                aa_dist_h = aas_h[plot_df.index].sum(axis=0)
                aa_dist_h = aa_dist_h / aa_dist_h.sum()
                aa_dist_c = aas_c[plot_df.index].sum(axis=0)
                aa_dist_c = aa_dist_c / aa_dist_c.sum()

                record = {"type": sp_type, "kingdom": kingdom, "region": "n"}
                record.update(dict(zip(aas.keys(), aa_dist_n)))
                dist_list.append(record)

                record = {"type": sp_type, "kingdom": kingdom, "region": "h"}
                record.update(dict(zip(aas.keys(), aa_dist_h)))
                dist_list.append(record)

                record = {"type": sp_type, "kingdom": kingdom, "region": "c"}
                record.update(dict(zip(aas.keys(), aa_dist_c)))
                dist_list.append(record)

        # melted df that can be used by seaborn
        aa_dist_df = pd.DataFrame.from_dict(dist_list).melt(
            id_vars=["type", "kingdom", "region"], var_name="aa", value_name="frequency"
        )

        plt.figure(figsize=(20, 16))

        plt.subplot(4, 3, 1)
        plot_data = aa_dist_df.loc[
            (aa_dist_df["type"] == "SP") & (aa_dist_df["region"] == "n")
        ]
        sns.barplot(data=plot_data, x="aa", y="frequency", hue="kingdom")
        plt.title("n-region SPI")
        plt.xlabel("Amino acid")

        plt.subplot(4, 3, 2)
        plot_data = aa_dist_df.loc[
            (aa_dist_df["type"] == "SP") & (aa_dist_df["region"] == "h")
        ]
        sns.barplot(data=plot_data, x="aa", y="frequency", hue="kingdom")
        plt.title("h-region SPI")
        plt.xlabel("Amino acid")

        plt.subplot(4, 3, 3)
        plot_data = aa_dist_df.loc[
            (aa_dist_df["type"] == "SP") & (aa_dist_df["region"] == "c")
        ]
        sns.barplot(data=plot_data, x="aa", y="frequency", hue="kingdom")
        plt.title("c-region SPI")
        plt.xlabel("Amino acid")

        plt.subplot(4, 3, 4)
        plot_data = aa_dist_df.loc[
            (aa_dist_df["type"] == "TAT") & (aa_dist_df["region"] == "n")
        ]
        sns.barplot(data=plot_data, x="aa", y="frequency", hue="kingdom")
        plt.title("n-region Tat")
        plt.xlabel("Amino acid")

        plt.subplot(4, 3, 5)
        plot_data = aa_dist_df.loc[
            (aa_dist_df["type"] == "TAT") & (aa_dist_df["region"] == "h")
        ]
        sns.barplot(data=plot_data, x="aa", y="frequency", hue="kingdom")
        plt.title("h-region Tat")
        plt.xlabel("Amino acid")

        plt.subplot(4, 3, 6)
        plot_data = aa_dist_df.loc[
            (aa_dist_df["type"] == "TAT") & (aa_dist_df["region"] == "c")
        ]
        sns.barplot(data=plot_data, x="aa", y="frequency", hue="kingdom")
        plt.title("c-region Tat")
        plt.xlabel("Amino acid")

        plt.subplot(4, 3, 7)
        plot_data = aa_dist_df.loc[
            (aa_dist_df["type"] == "LIPO") & (aa_dist_df["region"] == "n")
        ]
        sns.barplot(data=plot_data, x="aa", y="frequency", hue="kingdom")
        plt.title("n-region Sec/SPII")
        plt.xlabel("Amino acid")

        plt.subplot(4, 3, 8)
        plot_data = aa_dist_df.loc[
            (aa_dist_df["type"] == "LIPO") & (aa_dist_df["region"] == "h")
        ]
        sns.barplot(data=plot_data, x="aa", y="frequency", hue="kingdom")
        plt.title("h-region Sec/SPII")
        plt.xlabel("Amino acid")

        plt.subplot(4, 3, 10)
        plot_data = aa_dist_df.loc[
            (aa_dist_df["type"] == "TATLIPO") & (aa_dist_df["region"] == "n")
        ]
        sns.barplot(data=plot_data, x="aa", y="frequency", hue="kingdom")
        plt.title("n-region Tat/SPII")
        plt.xlabel("Amino acid")

        plt.subplot(4, 3, 11)
        plot_data = aa_dist_df.loc[
            (aa_dist_df["type"] == "TATLIPO") & (aa_dist_df["region"] == "h")
        ]
        sns.barplot(data=plot_data, x="aa", y="frequency", hue="kingdom")
        plt.title("h-region Tat/SPII")
        plt.xlabel("Amino acid")

        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, "regions_aa_distributions.png"))
        plt.close()

        ## relative region lengths
        plot_df = df.loc[df["Type"] == "SP"]

        plt.figure(figsize=(15, 10))
        plt.subplot(2, 3, 1)
        sns.boxplot(
            x="Kingdom", y="frac_n", data=plot_df, palette=palette, hue_order=hue_order
        )
        plt.ylim(0, 1)
        plt.title("n-region Sec/SPI")
        plt.ylabel("Relative length")

        plt.subplot(2, 3, 2)
        sns.boxplot(
            x="Kingdom", y="frac_h", data=plot_df, palette=palette, hue_order=hue_order
        )
        plt.ylim(0, 1)
        plt.title("h-region Sec/SPI")
        plt.ylabel("Relative length")

        plt.subplot(2, 3, 3)
        sns.boxplot(
            x="Kingdom", y="frac_c", data=plot_df, palette=palette, hue_order=hue_order
        )
        plt.ylim(0, 1)
        plt.title("c-region Sec/SPI")
        plt.ylabel("Relative length")

        plot_df = df.loc[df["Type"] == "TAT"]

        plt.subplot(2, 3, 4)
        sns.boxplot(
            x="Kingdom", y="frac_n", data=plot_df, palette=palette, hue_order=hue_order
        )
        plt.ylim(0, 1)
        plt.title("n-region Tat/SPI")
        plt.ylabel("Relative length")

        plt.subplot(2, 3, 5)
        sns.boxplot(
            x="Kingdom", y="frac_h", data=plot_df, palette=palette, hue_order=hue_order
        )
        plt.ylim(0, 1)
        plt.title("h-region Tat/SPI")
        plt.ylabel("Relative length")

        plt.subplot(2, 3, 6)
        sns.boxplot(
            x="Kingdom", y="frac_c", data=plot_df, palette=palette, hue_order=hue_order
        )
        plt.ylim(0, 1)
        plt.title("c-region Tat/SPI")
        plt.ylabel("Relative length")

        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, "regions_relative_lengths.png"))
        plt.close()

        plot_df = df.loc[df["Type"] == "SP"]

        plt.figure(figsize=(12, 8))
        ax = plt.subplot(2, 3, 1)
        sns.boxplot(
            x="Kingdom", y="len_n", data=plot_df, palette=palette, hue_order=hue_order
        )
        # plt.ylim(0,1)
        plt.title("n-region Sec/SPI")
        plt.ylabel("Length")

        plt.subplot(2, 3, 2, sharey=ax)
        sns.boxplot(
            x="Kingdom", y="len_h", data=plot_df, palette=palette, hue_order=hue_order
        )
        # plt.ylim(0,1)
        plt.title("h-region Sec/SPI")
        plt.ylabel("Length")

        plt.subplot(2, 3, 3, sharey=ax)
        sns.boxplot(
            x="Kingdom", y="len_c", data=plot_df, palette=palette, hue_order=hue_order
        )
        # plt.ylim(0,1)
        plt.title("c-region Sec/SPI")
        plt.ylabel("Length")

        plot_df = df.loc[df["Type"] == "TAT"]

        ax = plt.subplot(2, 3, 4)
        sns.boxplot(
            x="Kingdom", y="len_n", data=plot_df, palette=palette, hue_order=hue_order
        )
        # plt.ylim(0,1)
        plt.title("n-region Tat/SPI")
        plt.ylabel("Length")

        plt.subplot(2, 3, 5, sharey=ax)
        sns.boxplot(
            x="Kingdom", y="len_h", data=plot_df, palette=palette, hue_order=hue_order
        )
        # plt.ylim(0,1)
        plt.title("h-region Tat/SPI")
        plt.ylabel("Length")

        plt.subplot(2, 3, 6, sharey=ax)
        sns.boxplot(
            x="Kingdom", y="len_c", data=plot_df, palette=palette, hue_order=hue_order
        )
        # plt.ylim(0,1)
        plt.title("c-region Tat/SPI")
        plt.ylabel("Length")

        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, "spi_absolute_lengths.png"))
        plt.close()

        plot_df = df.loc[df["Type"] == "LIPO"]

        plt.figure(figsize=(10, 10))
        plt.subplot(2, 2, 1)
        sns.boxplot(
            x="Kingdom", y="frac_n", data=plot_df, palette=palette, hue_order=hue_order
        )
        plt.ylim(0, 1)
        plt.title("n-region Sec/SPII")
        plt.ylabel("Relative length")

        plt.subplot(2, 2, 2)
        sns.boxplot(
            x="Kingdom", y="frac_h", data=plot_df, palette=palette, hue_order=hue_order
        )
        plt.ylim(0, 1)
        plt.title("h-region Sec/SPII")
        plt.ylabel("Relative length")

        plot_df = df.loc[df["Type"] == "TATLIPO"]

        plt.subplot(2, 2, 3)
        sns.boxplot(
            x="Kingdom", y="frac_n", data=plot_df, palette=palette, hue_order=hue_order
        )
        plt.ylim(0, 1)
        plt.title("n-region Tat/SPII")
        plt.ylabel("Relative length")

        plt.subplot(2, 2, 4)
        sns.boxplot(
            x="Kingdom", y="frac_h", data=plot_df, palette=palette, hue_order=hue_order
        )
        plt.ylim(0, 1)
        plt.title("h-region Tat/SPII")
        plt.ylabel("Relative length")

        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, "spii_relative_lengths.png"))
        plt.close()

        plot_df = df.loc[df["Type"] == "LIPO"]

        plt.figure(figsize=(10, 10))
        plt.subplot(2, 2, 1)
        sns.boxplot(
            x="Kingdom", y="len_n", data=plot_df, palette=palette, hue_order=hue_order
        )
        plt.title("n-region Sec/SPII")
        plt.ylabel("Length")

        plt.subplot(2, 2, 2)
        sns.boxplot(
            x="Kingdom", y="len_h", data=plot_df, palette=palette, hue_order=hue_order
        )
        plt.title("h-region Sec/SPII")
        plt.ylabel("Length")

        plot_df = df.loc[df["Type"] == "TATLIPO"]

        plt.subplot(2, 2, 3)
        sns.boxplot(
            x="Kingdom", y="len_n", data=plot_df, palette=palette, hue_order=hue_order
        )
        plt.title("n-region Tat/SPII")
        plt.ylabel("Length")

        plt.subplot(2, 2, 4)
        sns.boxplot(
            x="Kingdom", y="len_h", data=plot_df, palette=palette, hue_order=hue_order
        )
        plt.title("h-region Tat/SPII")
        plt.ylabel("Length")

        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, "spii_absolute_lengths.png"))
        plt.close()

        ## SPIII lengths
        plot_df = df.loc[df["Type"] == "PILIN"]
        plt.figure(figsize=(5, 5))
        sns.boxplot(
            x="Kingdom", y="len_n", data=plot_df, palette=palette, hue_order=hue_order
        )
        plt.ylabel("Length")
        plt.title("Sec/SPIII")

        plt.savefig(os.path.join(args.output_dir, "spiii_absolute_lengths.png"))
        plt.close()

    ####
    #    Kingdom randomization plots
    ####

    if (
        args.bert_crossvalidation_metrics_randomized is not None
        and args.signalp5_crossvalidation_metrics_randomized is not None
    ):
        # make metric columns for both models
        df = pd.read_csv(args.signalp5_crossvalidation_metrics, index_col=0)
        metrics_1 = (
            df.loc[df.index.str.contains("mcc|window")].astype(float).mean(axis=1)
        )

        df = pd.read_csv(args.signalp5_crossvalidation_metrics_randomized, index_col=0)
        metrics_2 = (
            df.loc[df.index.str.contains("mcc|window")].astype(float).mean(axis=1)
        )
        df = pd.DataFrame({"Correct": metrics_1, "Randomized": metrics_2})

        # build additional identifer columns from split index
        exp_df = df.reset_index()["index"].str.split("_", expand=True)
        exp_df.columns = ["kingdom", "type", "metric", "no", "window"]
        exp_df.index = df.index

        # put together
        df = df.join(exp_df)

        nice_label_dict = {
            "NO_SP": "Other",
            "SP": "Sec/SPI",
            "LIPO": "Sec/SPII",
            "TAT": "Tat/SPI",
            "TATLIPO": "Tat/SPII",
            "PILIN": "Sec/SPIII",
            None: None,
        }

        df["type"] = df["type"].apply(lambda x: nice_label_dict[x])

        plt.figure(figsize=(12, 8))

        ax = plt.subplot(2, 1, 1)

        df_plot = df.loc[df["metric"] == "mcc1"][
            ["kingdom", "type", "Correct", "Randomized"]
        ]  # , 'crossval_std']]
        df_plot = df_plot.set_index(df_plot["kingdom"] + "\n" + df_plot["type"])
        df_plot = df_plot.sort_index()

        df_plot.plot(kind="bar", ax=ax, ylim=(0, 1), rot=0).legend(loc="lower left")
        plt.title("SignalP 5.0 MCC 1")

        ax = plt.subplot(2, 1, 2)
        df_plot = df.loc[df["kingdom"].isin(["ARCHAEA", "POSITIVE", "NEGATIVE"])]
        df_plot = df_plot.loc[df["metric"] == "mcc2"][
            ["kingdom", "type", "Correct", "Randomized"]
        ]  # , 'crossval_std']]
        df_plot = df_plot.set_index(df_plot["kingdom"] + "\n" + df_plot["type"])
        df_plot = df_plot.sort_index()

        df_plot.plot(kind="bar", ax=ax, ylim=(0, 1), rot=0).legend(loc="lower left")
        plt.title("SignalP 5.0 MCC 2")

        plt.tight_layout()
        plt.savefig(
            os.path.join(args.output_dir, "signalp5_randomized_kingdoms_mcc.png")
        )

        df_plot = df.loc[df["metric"] == "recall"][
            ["kingdom", "type", "window", "Correct", "Randomized"]
        ]  # , 'crossval_std']]
        df_plot = df_plot.set_index(
            df_plot["type"] + "\n ±" + df_plot["window"].astype(int).astype(str)
        )
        plt.figure(figsize=(16, 10))
        for idx, kingdom in enumerate(df_plot["kingdom"].unique()):
            ax = plt.subplot(2, 2, idx + 1)
            df_plot.loc[df_plot["kingdom"] == kingdom][["Correct", "Randomized"]].plot(
                kind="bar", ax=ax, title=kingdom + " CS recall", rot=90
            )

        plt.tight_layout()
        plt.savefig(
            os.path.join(args.output_dir, "signalp5_randomized_kingdoms_recall.png")
        )

        df_plot = df.loc[df["metric"] == "precision"][
            ["kingdom", "type", "window", "Correct", "Randomized"]
        ]  # , 'crossval_std']]
        df_plot = df_plot.set_index(
            df_plot["type"] + "\n ±" + df_plot["window"].astype(int).astype(str)
        )
        df_plot = df_plot.sort_index()

        plt.figure(figsize=(16, 10))
        for idx, kingdom in enumerate(df_plot["kingdom"].unique()):
            ax = plt.subplot(2, 2, idx + 1)
            df_plot.loc[df_plot["kingdom"] == kingdom][["Correct", "Randomized"]].plot(
                kind="bar", ax=ax, title=kingdom + " CS precision", rot=90
            )

        plt.tight_layout()
        plt.savefig(
            os.path.join(args.output_dir, "signalp5_randomized_kingdoms_precision.png")
        )

        # make metric columns for both models
        df = pd.read_csv(args.bert_crossvalidation_metrics, index_col=0)
        metrics_1 = (
            df.loc[df.index.str.contains("mcc|window")].astype(float).mean(axis=1)
        )

        df = pd.read_csv(args.bert_crossvalidation_metrics_randomized, index_col=0)
        metrics_2 = (
            df.loc[df.index.str.contains("mcc|window")].astype(float).mean(axis=1)
        )
        df = pd.DataFrame({"Correct": metrics_1, "Randomized": metrics_2})

        # build additional identifer columns from split index
        exp_df = df.reset_index()["index"].str.split("_", expand=True)
        exp_df.columns = ["kingdom", "type", "metric", "no", "window"]
        exp_df.index = df.index

        # put together
        df = df.join(exp_df)

        nice_label_dict = {
            "NO_SP": "Other",
            "SP": "Sec/SPI",
            "LIPO": "Sec/SPII",
            "TAT": "Tat/SPI",
            "TATLIPO": "Tat/SPII",
            "PILIN": "Sec/SPIII",
            None: None,
        }

        df["type"] = df["type"].apply(lambda x: nice_label_dict[x])

        plt.figure(figsize=(12, 8))

        ax = plt.subplot(2, 1, 1)

        df_plot = df.loc[df["metric"] == "mcc1"][
            ["kingdom", "type", "Correct", "Randomized"]
        ]  # , 'crossval_std']]
        df_plot = df_plot.set_index(df_plot["kingdom"] + "\n" + df_plot["type"])
        df_plot = df_plot.sort_index()

        # df_plot = df_plot.rename({'crossval_mean': 'Bert-CRF'}, axis =1)

        df_plot.plot(kind="bar", ax=ax, ylim=(0, 1), rot=0).legend(loc="lower left")
        plt.title("Bert MCC 1")

        ax = plt.subplot(2, 1, 2)
        df_plot = df.loc[df["kingdom"].isin(["ARCHAEA", "POSITIVE", "NEGATIVE"])]
        df_plot = df_plot.loc[df["metric"] == "mcc2"][
            ["kingdom", "type", "Correct", "Randomized"]
        ]  # , 'crossval_std']]
        df_plot = df_plot.set_index(df_plot["kingdom"] + "\n" + df_plot["type"])

        # df_plot = df_plot.rename({'crossval_mean': 'Bert-CRF'}, axis =1)

        df_plot.plot(kind="bar", ax=ax, ylim=(0, 1), rot=0).legend(loc="lower left")
        plt.title("Bert MCC 2")

        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, "bert_randomized_kingdoms_mcc.png"))

        df_plot = df.loc[df["metric"] == "recall"][
            ["kingdom", "type", "window", "Correct", "Randomized"]
        ]  # , 'crossval_std']]
        df_plot = df_plot.set_index(
            df_plot["type"] + "\n ±" + df_plot["window"].astype(int).astype(str)
        )
        df_plot = df_plot.sort_index()

        plt.figure(figsize=(16, 10))
        for idx, kingdom in enumerate(df_plot["kingdom"].unique()):
            ax = plt.subplot(2, 2, idx + 1)
            df_plot.loc[df_plot["kingdom"] == kingdom][["Correct", "Randomized"]].plot(
                kind="bar", ax=ax, title=kingdom + " CS recall", rot=90
            )

        plt.tight_layout()
        plt.savefig(
            os.path.join(args.output_dir, "bert_randomized_kingdoms_recall.png")
        )

        df_plot = df.loc[df["metric"] == "precision"][
            ["kingdom", "type", "window", "Correct", "Randomized"]
        ]  # , 'crossval_std']]
        df_plot = df_plot.set_index(
            df_plot["type"] + "\n ±" + df_plot["window"].astype(int).astype(str)
        )
        df_plot = df_plot.sort_index()

        plt.figure(figsize=(16, 10))
        for idx, kingdom in enumerate(df_plot["kingdom"].unique()):
            ax = plt.subplot(2, 2, idx + 1)
            df_plot.loc[df_plot["kingdom"] == kingdom][["Correct", "Randomized"]].plot(
                kind="bar", ax=ax, title=kingdom + " CS precision", rot=90
            )

        plt.tight_layout()
        plt.savefig(
            os.path.join(args.output_dir, "bert_randomized_kingdoms_precision.png")
        )


if __name__ == "__main__":
    main()
