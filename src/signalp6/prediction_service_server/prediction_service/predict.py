"""
Main prediction script.
Used by webserver, and can also be used from CLI.
Due to reliance on TorchScript, this can be effectively run witout
any dependencies - all we need is the compiled model and the vocabulary as json.

TODO see whats better - no dependency (save some mb code) or have the original 
code where the model is, should somebody come looking.

The script works on a fasta file.

Args:
    --fastafile 
    --output_path
    --format
    --organism

Example :
    python3 predict.py --fastafile path/to/file.fasta --output output/dir/ --organism Archaea


results are collected in results_summary dict to be dumped as json.
 -general job info in "INFO"
 -individual outputs in "SEQUENCES
"""

import torch
import json
import argparse
import os
import numpy as np
from datetime import datetime
from make_sequence_plot import sequence_plot

GLOBAL_LABEL_DICT = {0: "NO_SP", 1: "SP", 2: "LIPO", 3: "TAT", 4: "TATLIPO", 5: "PILIN"}
TOKENIZER_VOCAB = {
    "[PAD]": 0,
    "[UNK]": 1,
    "[CLS]": 2,
    "[SEP]": 3,
    "[MASK]": 4,
    "L": 5,
    "A": 6,
    "G": 7,
    "V": 8,
    "E": 9,
    "S": 10,
    "I": 11,
    "K": 12,
    "R": 13,
    "D": 14,
    "T": 15,
    "P": 16,
    "N": 17,
    "Q": 18,
    "F": 19,
    "Y": 20,
    "M": 21,
    "H": 21,
    "C": 22,
    "W": 23,
    "X": 24,
    "U": 25,
    "B": 26,
    "Z": 27,
    "O": 28,
    "Eukarya": 29,
    "Archaea": 30,
    "Positive": 31,
    "Negative": 32,
}


def tokenize_sequence(amino_acids: str, kingdom: str):
    """Hard-coded version of the HF tokenizer.
    Possible because AA vocabs are small, removes dependency on transformers.
    Less overhead because has no state."""
    tokenized = [TOKENIZER_VOCAB[x] for x in amino_acids]
    tokenized = (
        [TOKENIZER_VOCAB["[CLS]"], TOKENIZER_VOCAB[kingdom]]
        + tokenized
        + [TOKENIZER_VOCAB["[SEP]"]]
    )

    return tokenized


def model_inputs_from_fasta(fastafile: str, kingdom: str):
    """Parse a fasta file to input id + mask tensors.
    Pad all seqs to full length (73, 70+special tokens).
    traced model requires that."""

    with open(fastafile, "r") as f:
        lines = f.read().splitlines()
        identifiers = lines[::2]
        sequences = lines[1::2]

    # truncate
    input_ids = [x[:70] for x in sequences]
    input_ids = [tokenize_sequence(x, kingdom) for x in input_ids]
    input_ids = [x + [0] * (73 - len(x)) for x in input_ids]

    input_ids = np.vstack(input_ids)
    input_mask = (input_ids > 0) * 1

    identifiers = [x.lstrip(">") for x in identifiers]

    return (
        identifiers,
        sequences,
        torch.LongTensor(input_ids),
        torch.LongTensor(input_mask),
    )


def get_cleavage_sites(tagged_seqs: np.ndarray, cs_tokens=[5, 11, 19, 26, 31]):
    """Convert sequences of tokens to the indices of the cleavage site.
    Inputs:
        tagged_seqs: (batch_size, seq_len) integer array of position-wise labels , with "C" tags for cleavage sites
        cs_tokens = label tokens that indicate a cleavage site
    Returns:
        cs_sites: (batch_size) integer array of position that is a CS. -1 if no SP present in sequence.
    """

    def get_last_sp_idx(x: np.ndarray) -> int:
        """Func1d to get the last index that is tagged as SP. use with np.apply_along_axis. """
        sp_idx = np.where(np.isin(x, cs_tokens))[0]
        if len(sp_idx) > 0:
            max_idx = sp_idx.max() + 1
        else:
            max_idx = -1
        return max_idx

    cs_sites = np.apply_along_axis(get_last_sp_idx, 1, tagged_seqs)
    return cs_sites


def predict(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    input_mask: torch.Tensor,
    batch_size: int = 50,
):
    """Cut batches from tensors and process batch-wise"""
    # process as batches
    all_global_probs = []
    all_marginal_probs = []
    all_viterbi_paths = []
    b_start = 0
    b_end = batch_size

    while b_start < len(input_ids):
        with torch.no_grad():
            ids = input_ids[b_start:b_end, :]
            mask = input_mask[b_start:b_end, :]
            global_probs, marginal_probs, viterbi_paths = model(ids, mask)

            all_global_probs.append(global_probs.numpy())
            all_marginal_probs.append(marginal_probs.numpy())
            all_viterbi_paths.append(viterbi_paths.numpy())

        b_start = b_start + batch_size
        b_end = b_end + batch_size

    all_global_probs = np.concatenate(all_global_probs)
    all_marginal_probs = np.concatenate(all_marginal_probs)
    all_viterbi_paths = np.concatenate(all_viterbi_paths)

    return all_global_probs, all_marginal_probs, all_viterbi_paths


def make_output_table(
    identifiers, global_probs, cleavage_sites, kingdom, output_file_path
):
    """Make the .tsv tabular output for all sequences"""
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

    # get the global preds, post-processing for Eukarya
    if kingdom == "Eukarya":
        p_no = global_probs[:, 0, None]  # keep a dummy dim to put array back together
        p_sp = global_probs[:, 1:, None].sum(axis=1)
        global_probs = np.hstack([p_no, p_sp])
        colnames = ["ID", "Prediction", "OTHER", "SP(Sec/SPI)", "CS Position"]
    else:
        colnames = [
            "ID",
            "Prediction",
            "OTHER",
            "SP(Sec/SPI)",
            "TAT(Tat/SPI)",
            "LIPO(Sec/SPII)",
            "TATLIPO(Sec/SPII)",
            "PILIN(Sec/SPIII)",
            "CS Position",
        ]

    pred_label_id = np.argmax(global_probs, axis=1)

    with open(output_file_path, "w") as f:
        # write the headers
        f.write(
            f"# SignalP-6.0\tOrganism: {kingdom.lower().capitalize()}\tTimestamp: {timestamp}\n"
        )
        f.write("# " + "\t".join(colnames) + "\n")

        for idx, identifier in enumerate(identifiers):
            # format everything for writing
            prediction = GLOBAL_LABEL_DICT[pred_label_id[idx]]
            probs = global_probs[idx]
            probs = [f"{p:.6f}" for p in probs]
            cs = str(cleavage_sites[idx]) if cleavage_sites[idx] != -1 else ""
            writelist = [identifier] + [prediction] + probs + [cs]
            f.write("\t".join(writelist) + "\n")


def make_plots(
    identifiers: list,
    sequences: list,
    global_probs: np.ndarray,
    marginal_probs: np.ndarray,
    viterbi_paths: np.ndarray,
    output_dir,
):

    eps_name_list = []
    png_name_list = []
    txt_name_list = []
    for idx, identifier in enumerate(identifiers):

        fig = sequence_plot(marginal_probs[idx], viterbi_paths[idx], sequences[idx])
        eps_fname = "output_%s_plot.eps" % (identifiers[idx])
        png_fname = "output_%s_plot.png" % (identifiers[idx])
        eps_path = os.path.join(output_dir, eps_fname)
        png_path = os.path.join(output_dir, png_fname)

        fig.savefig(png_path, format="png")
        fig.savefig(eps_path, format="eps")

        eps_name_list.append(eps_path)
        png_name_list.append(png_path)
        txt_name_list.append("")

    return eps_name_list, png_name_list, txt_name_list


def add_sequences_to_summary(
    summary: dict,
    identifiers: list,
    global_probs: np.ndarray,
    cleavage_sites: np.ndarray,
    kingdom: str,
    eps_save_paths,
    png_save_paths,
    txt_save_paths,
):
    """Adds each predicted sequence to the JSON output. Also makes plots and adds their
    save paths to the JSON."""
    if kingdom == "Eukarya":
        p_no = global_probs[:, 0, None]
        p_sp = global_probs[:, 1:, None].sum(axis=1)
        global_probs = np.hstack([p_no, p_sp])
        prednames = ["Signal Peptide (Sec/SPI)", "Other"]
    else:
        prednames = [
            "Other",
            "Signal Peptide (Sec/SPI)",
            "TAT signal peptide (Tat/SPI)",
            "Lipoprotein signal peptide (Sec/SPII)",
            "TAT Lipoprotein signal peptide (Sec/SPII)",
            "Pilin-like signal peptide (Sec/SPIII)",
        ]

    pred_label_id = np.argmax(global_probs, axis=1)

    for idx, identifier in enumerate(identifiers):
        seq_dict = {
            "Name": identifier,
            "Plot_eps": eps_save_paths[idx],
            "Plot_png": png_save_paths[idx],
            "Plot_txt": txt_save_paths[idx],
            "Prediction": prednames[pred_label_id[idx]],
            "Likelihood": [round(x.item(), 4) for x in global_probs[idx]],
            "Protein_types": prednames,
        }
        if cleavage_sites[idx] == -1:
            seq_dict["CS_pos"] = ""
        else:
            seq_dict[
                "CS_pos"
            ] = f"Cleavage site between pos. {cleavage_sites[idx]} and {cleavage_sites[idx]+1}"

        summary["SEQUENCES"][identifier] = seq_dict

    return summary


def main():
    parser = argparse.ArgumentParser("SignalP 6.0")
    parser.add_argument("--fastafile", type=str, help="fasta file to predict")
    parser.add_argument("--output_dir", type=str, help="path to save the outputs in")
    # choices-nargs setup to limit options + have default value
    parser.add_argument(
        "--format",
        type=str,
        default="short",
        const="short",
        nargs="?",
        choices=["short", "long"],
    )
    parser.add_argument(
        "--organism",
        type=str,
        default="Eukarya",
        const="Eukarya",
        nargs="?",
        choices=["Eukarya", "Archaea", "Positive", "Negative"],
    )

    args = parser.parse_args()
    # some default values for input parameters, remove before release
    file_path = None
    shortput = False
    write_path = None

    assert os.path.isfile(
        args.fastafile
    ), f"Specified file path {args.fastafile} is not a existing file."

    if not os.path.exists(args.output_dir):
        try:
            os.makedirs(args.output_dir)
        except:
            raise ValueError(f"{args.output_dir} does not exist and cannot be created.")

    identifiers, aa_sequences, input_ids, input_mask = model_inputs_from_fasta(
        args.fastafile, args.organism
    )

    # remove/replace special symbols
    # identifiers = format_identifiers(identifiers)

    # this is where we will collect the results
    result_summary = {
        "INFO": {"failedjobs": 0, "size": len(identifiers)},
        "CSV_FILE": "",
        "MATURE_FILE": "",
        "GFF_FILE": "",
        "ZIP_FILE": "",
        "SEQUENCES": {},
        "FORMAT": args.format,
    }

    # check the parser output whether there's something wrong
    # If the parser gives us no data, write error file to output dir
    if len(identifiers) < 1:
        with open(os.path.join(write_path, "errors.json"), "w+") as outf:
            outf.write(
                "{ parse_error: %r}" % ("Could not parse input!" + " Check data file.")
            )
        result_summary["INFO"]["failedjobs"] = 1

        # TODO
        # write_error_file()
        return None

    # load model and run data
    model = torch.jit.load("../checkpoints/ensemble_model_signalp6.pt")
    global_probs, marginal_probs, viterbi_paths = predict(model, input_ids, input_mask)
    cleavage_sites = get_cleavage_sites(viterbi_paths)

    # write tabular output
    tsv_output_file = os.path.join(args.output_dir, "prediction_results.tsv")
    make_output_table(
        identifiers, global_probs, cleavage_sites, args.organism, tsv_output_file
    )
    result_summary["CSV_FILE"] = tsv_output_file

    # write plots - only if long output format
    eps_save_paths, png_save_paths, txt_save_paths = (
        [""] * len(identifiers),
        [""] * len(identifiers),
        [""] * len(identifiers),
    )
    eps_save_paths, png_save_paths, txt_save_paths = make_plots(
        identifiers,
        aa_sequences,
        global_probs,
        marginal_probs,
        viterbi_paths,
        args.output_dir,
    )
    add_sequences_to_summary(
        result_summary,
        identifiers,
        global_probs,
        cleavage_sites,
        args.organism,
        eps_save_paths,
        png_save_paths,
        txt_save_paths,
    )

    with open(os.path.join(args.output_dir, "output.json"), "w+") as outf:
        json.dump(result_summary, outf, indent=2)


if __name__ == "__main__":
    main()
