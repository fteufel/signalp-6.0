# © Copyright Technical University of Denmark
"""
Utilities to compute additional metrics from webserver outputs.
This is useful for evaluating regions.
The server itself only returns the region positions,
it does not compute any additional properties.

region_features_from_server_output skips all sequences in input that
had no regions predicted (only sequences that are in region_output.gff3)
"""
import os
import pandas as pd
import numpy as np


def read_fasta(file):
    """Read a two-line format fasta file into a pd Dataframe"""
    with open(file, "r") as f:
        lines = f.read().splitlines()
        identifiers = lines[::2]
        sequences = lines[1::2]

        df = pd.DataFrame.from_dict({"ID": identifiers, "Sequence": sequences})
    return df


def make_one_dataframe(regions_gff: str, output_gff: str, processed_fasta: str):
    """
    Takes in three input files produced by the webservice, and merges them
    into one df with the following 11-column format:
    ID	(Start, c-region) (Start, h-region)	(Start, n-region) (Start, twin-arginine motif) (End, c-region) (End, h-region) \
        (End, n-region)	(End, twin-arginine motif)	Sequence	SP type
    """
    # read in everything
    df_regions = pd.read_csv(
        regions_gff,
        sep="\t",
        skiprows=1,
        names=["ID", "Source", "Feature", "Start", "End", ".1", ".2", ".3", ".4"],
    )
    df_regions = df_regions.drop(["Source", ".1", ".2", ".3", ".4"], axis=1)
    df_type = pd.read_csv(
        output_gff,
        sep="\t",
        skiprows=1,
        names=["ID", "Source", "SP type", "Start", "End", ".1", ".2", ".3", "Note"],
    )

    df_seqs = read_fasta(processed_fasta)

    # pivot the df
    df_regions = df_regions.set_index(["ID", "Feature"]).unstack()

    # merge on ID
    df_regions = df_regions.merge(df_seqs, on="ID")
    df_regions = df_regions.merge(df_type[["ID", "SP type", "End", "Note"]], on="ID")

    # make proper SP type. gff3 only specifies 'signal_peptide' and 'lipoprotein_signal_peptide'
    # rest is in Note column.
    for idx, row in df_regions.iterrows():
        if row['Note'] == 'Note=TAT':
            df_regions.loc[idx, 'SP type'] = 'tat_' + df_regions.loc[idx, 'SP type']
        if row['Note'] == 'Note=Pilin':
            df_regions.loc[idx, 'SP type'] ='pilin_' + df_regions.loc[idx, 'SP type']
    df_regions = df_regions.drop('Note',axis=1)

    return df_regions


def compute_net_charge(amino_acid_counts):
    """Given an array of AA counts, compute net charge"""
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
    "U": 2.5, #selenocysteine, assume same as C
    "X": 0, #any 
    "B": -3.5, # D or N
    "Z": -3.5, #Q or E
    "O": -3.9, # Pyrollysine, don't really know, assume same as Lysine
}
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
    "U": 20,
    "X": 21,
    "B": 22,
    "Z": 23,
    "O": 24,
}

# make new dict mapping idx to kd score
idx_to_hydrophobicity = dict(zip(aas.values(), [KYTE_DOOLITTE[x] for x in aas.keys()]))


def compute_hydrophobicity(amino_acid_counts):
    """Given an array of AA counts, compute KD hydrophobicity"""
    total_score = 0
    for i in range(len(aas)):
        score = amino_acid_counts[i] * idx_to_hydrophobicity[i]
        total_score += score

    return total_score


def compute_region_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Takes a dataframe created by make_one_dataframe
    and returns a dataframe containing features of the regions
    """
    # setup df to collect all the results
    df_results = df[["ID", "SP type"]].copy()

    # simple region length features
    # always check whether a region is really there. Otherwise 
    # fails on lipo/no-sp/pilin-only dataframes.
    if ('Start', 'n-region') in df.columns and ('End', 'n-region') in df.columns:
        df_results.loc[:, "len_n"] = df["End", "n-region"] - df["Start", "n-region"]
    else:
        df_results.loc[:, "len_n"] = np.nan
        
    if ('Start', 'h-region') in df.columns and ('End', 'h-region') in df.columns:
        df_results.loc[:, "len_h"] = df["End", "h-region"] - df["Start", "h-region"]
    else:
        df_results.loc[:, "len_h"] = np.nan
        
    if ('Start', 'c-region') in df.columns and ('End', 'c-region') in df.columns:
        df_results.loc[:, "len_c"] = df["End", "c-region"] - df["Start", "c-region"]
    else:
        df_results.loc[:, "len_c"] = np.nan
    
    df_results.loc[:, "len_sp"] = df["End"]

    df_results.loc[:, "rel_len_n"] = df_results["len_n"] / df_results["len_sp"]
    df_results.loc[:, "rel_len_h"] = df_results["len_h"] / df_results["len_sp"]
    df_results.loc[:, "rel_len_c"] = df_results["len_c"] / df_results["len_sp"]

    # replace nas with -1, so that we can apply same function to each row irrespective
    # of whether a region is really there. Mask out spurious results afterwards.
    df_filled = df.fillna(-1)

    # biochemical features
    for idx, row in df_filled.iterrows():
        # cut substrings
        # we do this in arrays, so we can use bincount for frequencies
        sequence = np.array([aas[x] for x in row["Sequence"]])
        # NOTE we need to -1 the start idx (are in +1 format). For the end, need to keep the +1
        if row['SP type'] in ['signal_peptide', 'lipoprotein_signal_peptide', 'tat_signal_peptide', 'tat_lipoprotein_signal_peptide']:
            n_seq = sequence[
                int(row["Start", "n-region"]) - 1 : int(row["End", "n-region"])
            ]
            h_seq = sequence[
                int(row["Start", "h-region"]) - 1 : int(row["End", "h-region"])
            ]
        else:
            n_seq = []
            h_seq = []
        
        if row['SP type'] in ['lipoprotein_signal_peptide', 'tat_lipoprotein_signal_peptide']:
            c_seq = []
        else:
            c_seq = sequence[
                int(row["Start", "c-region"]) - 1 : int(row["End", "c-region"])
            ]


        # count the AAs
        minlength = len(aas)
        aas_n = np.bincount(n_seq, minlength=minlength)
        aas_h = np.bincount(h_seq, minlength=minlength)
        aas_c = np.bincount(c_seq, minlength=minlength)
        aas_all = aas_n + aas_h + aas_c

        # based on the AA counts, compute features
        df_results.loc[idx, "hyd_n"] = compute_hydrophobicity(aas_n)
        df_results.loc[idx, "hyd_h"] = compute_hydrophobicity(aas_h)
        df_results.loc[idx, "hyd_c"] = compute_hydrophobicity(aas_c)
        df_results.loc[idx, "hyd_sp"] = compute_hydrophobicity(aas_all)

        df_results.loc[idx, "chr_n"] = compute_net_charge(aas_n)
        df_results.loc[idx, "chr_h"] = compute_net_charge(aas_h)
        df_results.loc[idx, "chr_c"] = compute_net_charge(aas_c)
        df_results.loc[idx, "chr_sp"] = compute_net_charge(aas_all)

    # mask out based on global label.
    df_results.loc[df_results['SP type']=='lipoprotein_signal_peptide', ['len_c', 'rel_len_c', 'hyd_c', 'chr_c']] = np.nan
    df_results.loc[df_results['SP type']=='tat_lipoprotein_signal_peptide', ['len_c', 'rel_len_c', 'hyd_c', 'chr_c']] = np.nan

    return df_results


def region_features_from_server_output(server_output_path: str):
    """Takes an unzipped SignalP6 webserver output and makes a df containing
    region features
    """
    regions_gff = os.path.join(server_output_path, "region_output.gff3")
    output_gff = os.path.join(server_output_path, "output.gff3")
    processed_fasta = os.path.join(server_output_path, "processed_entries.fasta")
    df = make_one_dataframe(regions_gff, output_gff, processed_fasta)

    df_out = compute_region_features(df)

    return df_out
