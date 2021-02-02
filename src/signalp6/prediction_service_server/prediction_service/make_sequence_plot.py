"""
Utility functions for final predictor.
Need to make plot for sequence.

We have 37 different states - impractical to display all. 
How to clean up:
 1) ignore labels that are never higher than a set threshold (0.01) - this is effectively 0
 2) merge class regions that are the same (bad idea?)
"""
import numpy as np
import matplotlib.pyplot as plt

# convert label ids to simplified labels for plotting
IDS_TO_PLOT_LABEL = {
    0: "I",
    1: "M",
    2: "O",
    3: "N",  # SP
    4: "H",
    5: "C",
    6: "I",
    7: "M",
    8: "O",
    9: "N",  # LIPO
    10: "H",
    11: "C",
    12: "c",
    13: "I",
    14: "M",
    15: "O",
    16: "N",  # TAT
    17: "R",
    18: "H",
    19: "C",
    20: "I",
    21: "M",
    22: "O",
    23: "N",  # TATLIPO
    24: "R",
    25: "H",
    26: "C",
    27: "c",
    28: "I",
    29: "M",
    30: "O",
    31: "P",  # PILIN
    32: "c",
    33: "H",
    34: "I",
    35: "M",
    36: "O",
}
# colors for letters, depending on region
REGION_PLOT_COLORS = {
    "N": "red",
    "H": "orange",
    "C": "gold",
    "I": "gray",
    "M": "gray",
    "O": "gray",
    "c": "cyan",
    "R": "lime",
    "P": "red",
}
# colors for the probability lines, should somewhat match the letters
intra_color = "lightcoral"  # use some very light colors for these 'regions', we don't care about them really
tm_color = "khaki"
extra_color = "bisque"
PROB_PLOT_COLORS = {
    0: intra_color,
    1: tm_color,
    2: extra_color,
    3: "red",
    4: "orange",
    5: "gold",
    6: intra_color,
    7: tm_color,
    8: extra_color,
    9: "red",
    10: "orange",
    11: "gold",
    12: "cyan",
    13: intra_color,
    14: tm_color,
    15: extra_color,
    16: "red",
    17: "lime",
    18: "orange",
    19: "gold",
    20: intra_color,
    21: tm_color,
    22: extra_color,
    23: "red",  # TATLIPO
    24: "lime",
    25: "orange",
    26: "gold",
    27: "cyan",
    28: intra_color,
    29: tm_color,
    30: extra_color,
    31: "red",  # PILIN
    32: "gold",
    33: "orange",
    34: intra_color,
    35: tm_color,
    36: extra_color,
}
# labels to write to legend for probability channels
IDX_LABEL_MAP = {
    0: "Other I",
    1: "Other M",
    2: "Other O",
    3: "Sec/SPI n",
    4: "Sec/SPI h",
    5: "Sec/SPI c",
    6: "Sec/SPI I",
    7: "Sec/SPI M",
    8: "Sec/SPI O",
    9: "Sec/SPII n",
    10: "Sec/SPII h",
    11: "Sec/SPII c",
    12: "Sec/SPII cys",
    13: "Sec/SPII I",
    14: "Sec/SPII M",
    15: "Sec/SPII O",
    16: "Tat/SPI n",
    17: "Tat/SPI RR",
    18: "Tat/SPI h",
    19: "Tat/SPI c",
    20: "Tat/SPI I",
    21: "Tat/SPI M",
    22: "Tat/SPI O",
    23: "Tat/SPII n",
    24: "Tat/SPII RR",
    25: "Tat/SPII h",
    26: "Tat/SPII c",
    27: "Tat/SPII cys",
    28: "Tat/SPII I",
    29: "Tat/SPII M",
    30: "Tat/SPII O",
    31: "Sec/SPIII P",  # PILIN
    32: "Sec/SPIII cons.",
    33: "Sec/SPIII h",
    34: "Sec/SPIII I",
    35: "Sec/SPIII M",
    36: "Sec/SPIII O",
}


def sequence_plot(
    marginal_probs,
    viterbi_path,
    amino_acid_sequence,
    hide_threshold=0.01,
    figsize=(16, 6),
    title=None,
):

    amino_acid_sequence = amino_acid_sequence[:70]
    seq_length = len(amino_acid_sequence)
    pos_labels_to_plot = [IDS_TO_PLOT_LABEL[x] for x in viterbi_path]
    joined_ticks = [
        x + "\n" + y for x, y in zip(pos_labels_to_plot, amino_acid_sequence)
    ]
    tick_colors = [REGION_PLOT_COLORS[x] for x in pos_labels_to_plot]

    # import ipdb; ipdb.set_trace()
    fig = plt.figure(figsize=figsize)

    # plt.gca().xaxis.grid(True) #vertical gridlines
    for x in np.arange(
        1, seq_length + 1
    ):  # gridlines are controlled by ticks, but want a gridline at each pos, not tick
        plt.axvline(x, c="whitesmoke")

    # iterate over all channels - use channel idx to assign color/label
    for idx in range(marginal_probs.shape[1]):
        probs = marginal_probs[:, idx]
        # skip low
        if (probs > hide_threshold).any():
            plt.plot(
                np.arange(1, seq_length + 1),
                probs,
                label=IDX_LABEL_MAP[idx],
                c=PROB_PLOT_COLORS[idx],
            )

    # set xticks and apply color to tick labels
    # plt.xticks(np.arange(0,seq_length), joined_ticks)
    # ax = plt.gca()
    # [t.set_color(i) for (i,t) in zip(tick_colors,ax.xaxis.get_ticklabels())]

    # Put AA labels and region label in plot at x positions
    for i, t in enumerate(pos_labels_to_plot):
        plt.text(i + 1, -0.05, t, c=REGION_PLOT_COLORS[t], ha="center", va="center")
    for i, t in enumerate(amino_acid_sequence):
        plt.text(i + 1, -0.1, t, ha="center", va="center")

    # adjust lim to fit pos labels
    plt.ylim((-0.15, 1.05))
    plt.xlim((0, seq_length + 1))

    plt.ylabel("Probability")
    plt.xlabel("Protein sequence")
    if title is not None:
        plt.title(title)

    plt.legend(loc="upper right")
    plt.tight_layout()

    return fig
