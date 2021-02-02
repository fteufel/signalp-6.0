#!/usr/bin/env python
# coding: utf-8
"""
This module executes a prediction on samples from a fasta file,
the path of which should be provided as a command line argument.

The desired output folder should also be provided as an argument.

Optionally, you can either use -s/--short to skip creation of 
sequence figures or -t/--format long/short.


Args:
    -f,--file <file_path> 
    -o,--output <write_path>
    -s,--short 
    -t,--format <format>

Example: 
python predict.py -f ../../data/full_sequences/gpi_test_animal.txt -o ../../tmp/
"""
import sys
from utility import *
from datetime import datetime
import scipy.stats as st
import matplotlib.cm as cm
from singlepointernet_model import SinglePointerNet
import pdb
from os.path import join as path_join
from os.path import isfile, isdir
import json
from os import mkdir


def predict(
    models: pd.core.frame.DataFrame, test_in: torch.Tensor, test_lengths: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """models:
    test_in: pre padded and integer encoded samples to evaluate.
    test_lengths: The length of the input sequences unpadded.

    Note, the order of the two input vectors are assumed to be the
    same. That is to say, the same index is expected to refer to the
    same sample.

    Performs an ensemble prediction using the models in the models
    dataframe. The log probabilities are added together and softmaxed.

    Returns the predicted positions, the likelihood of the predicted
    positions and the log probabilities of each model separated.

    """
    batch_size = 1000
    max_size = test_in.shape[0]
    models_count = len(models)

    log_probs = torch.Tensor(models_count, max_size, 101)
    ensemble_out = torch.Tensor(max_size, 101)
    prediction = torch.LongTensor(max_size)

    with torch.no_grad():
        for dexes in get_batch(max_size, batch_size):
            test_batch_size = dexes[1] - dexes[0]

            for i, model in models.iterrows():
                # model is the i'th row in models
                # After: outputs[:i+1] contain the log probability results of
                #        models.loc[:i,'model']
                m = model["model"]  # m is a SinglePointerNet pytorch model

                # Set the state of the model to evaluation mode
                m.eval()

                test_in_batch = test_in[dexes[0] : dexes[1]]
                test_len_batch = test_lengths[dexes[0] : dexes[1]]
                hiddens = m.init_hidden(test_batch_size)
                cells = m.init_hidden(test_batch_size)

                output = m(test_in_batch, hiddens, cells, test_len_batch)

                log_probs[i, dexes[0] : dexes[1]] = output

        # Ensemble Probability
        ensemble_out = torch.softmax(torch.mean(log_probs, dim=0), dim=1)

        # Ensemble Position Prediction
        prediction = torch.argmax(ensemble_out, dim=1).view(-1)

    return prediction, ensemble_out, log_probs


def make_prediction_graphs(
    test_data: pd.core.frame.DataFrame,
    write_path: str,
    probabilities: torch.Tensor,
    all_outputs: torch.Tensor,
    result_summary: dict,
    shortput: bool,
):
    """Creates prediction graphs for the position predictions in the
    DataFrame test_data and saves them to the directory write_path.

    test_data:      The DataFrame should at least contain the following:
                    'sequence':   The original sequences.
                    'seq_length': The length of the input sequences
                                  before padding. That is, after left
                                  truncation and after addition of
                                  terminal symbol.
                    'name':       The sequences identifier.
    write_path:     The directory to write to.
    probabilities:  The probability distribution for each sample in
                    test_data (The order should be the same)
    all_outputs:    The log probability distribution output of each
                    ensemble member.
    result_summary: Stores metadata, including filepaths for result
                    display. Should contain at least:
                    'SEQUENCES':  Empty dict
    shortput:       if True then only text output is generated.
    """

    # scale for interval calculations (for prediction graphs)
    # schale = np.nan_to_num(st.sem(torch.softmax(all_outputs,
    #                                             dim=2),
    #                               axis=0,
    #                               nan_policy='omit'))
    # regbase_intervals = list(st.t.interval(0.95,
    #                                        all_outputs.shape[1]-1,
    #                                        loc=probabilities,
    #                                        scale=schale))
    # regbase_intervals[0] = np.nan_to_num(regbase_intervals[0])
    # regbase_intervals[1] = np.nan_to_num(regbase_intervals[1])

    bluest = cm.Blues(np.linspace(0.9, 0.9, 1))[0]
    blues = cm.Blues(np.linspace(0.55, 0.3, 5))

    html_path = write_path
    if "html" in html_path:
        spl = html_path.split("/")
        while "html" not in spl[0]:
            spl.pop(0)
        html_path = "/" + "/".join(spl[1:])

    for ind, row in test_data.iterrows():
        # row is the ind'th row in test_data
        # After: ind prediction graphs and files have been produced.
        # if not short output, otherwise only files.

        this_data = test_data.iloc[ind]
        s = this_data["sequence"]
        s += "*"
        maxlen = this_data["seq_length"]
        o = get_numpy(probabilities[ind])[:maxlen]
        # o_lower = regbase_intervals[0][ind,:maxlen]
        # o_upper = regbase_intervals[1][ind,:maxlen]

        # reductiveo = 1 - np.cumsum(o)
        # additiveo = 1 - np.flip(np.cumsum(np.flip(o)))

        eps_fname = "output_%s_plot.eps" % (this_data["name"])
        png_fname = "output_%s_plot.png" % (this_data["name"])
        if not shortput:
            fig = plt.figure(figsize=(16, 8), frameon=False)
            for xc in range(len(o) - 2):
                plt.axvline(x=xc, color="k", linestyle="--", alpha=0.1)
            plt.axvline(x=maxlen - 2, color="k", linestyle="-", alpha=0.4)
            plt.axvline(x=maxlen - 1, color="r", linestyle="--", alpha=0.1)
            plt.axhline(0.0, color="k", linestyle="--", alpha=0.1)
            # plt.axhline(0.5, color='k', linestyle='--',alpha=0.1)
            plt.axhline(1, color="k", linestyle="--", alpha=0.1)

            plt.plot(o, "-", color=bluest, label="Ensemble prediction")

            plt.legend(loc="upper left")
            plt.ylim(-0.05, 1.05)
            plt.ylabel("Probability")
            plt.xlabel("Protein Sequence")
            plt.title("NetGPI, $\omega$-site prediction: %s" % (this_data["name"]))
            plt.xticks(range(len(o)), s[-maxlen:])

            # Write this sequence's position prediction graph
            png_path = path_join(write_path, png_fname)
            eps_path = path_join(write_path, eps_fname)

            plt.savefig(
                png_path,
                format="png",
                # transparent=True
            )
            plt.savefig(
                eps_path,
                format="eps",
                # transparent=True
            )

            plt.close(fig)
        tab_fname = "output_%s_pred.txt" % (this_data["name"])
        tab_path = path_join(write_path, tab_fname)
        # Write this sequence's position prediction file
        with open(tab_path, "w+") as inf:
            inf.write("# Name=%s\n" % (this_data["name"]))
            inf.write("# pos\taa\tlikelihood\n")
            for pos, aacid, os in zip(range(1, maxlen + 1), s[-maxlen:], o):
                inf.write("%i\t%s\t%.6f\n" % (pos, aacid, os))
        amino_acid = (
            this_data["sequence"][len(this_data["sequence"]) - this_data["result"]]
            if this_data["result"] != 0
            else "*"
        )
        result_summary["SEQUENCES"][this_data["name"]] = {
            "pred_text": "%s residue omega-site predicted at position %i."
            % (amino_acid, len(this_data["sequence"]) + 1 - this_data["result"]),
            "likelihood": [
                this_data["sentinelhood"]
                if this_data["result"] == 0
                else this_data["omegahood"]
            ],
            "Amino-acid": amino_acid,
            "Name": this_data["name"],
            "Plot_eps": path_join(html_path, eps_fname),
            "Plot_png": path_join(html_path, png_fname),
            "Plot_txt": path_join(html_path, tab_fname),
            "Prediction": "GPI-Anchored",
            "likelihood_texts": [
                "Not GPI-Anchored"
                if this_data["result"] == 0
                else "Omega-site, pos:%i, resdue:%s"
                % (len(this_data["sequence"]) + 1 - this_data["result"], amino_acid)
            ],
        }
        if this_data["result"] == 0:
            result_summary["SEQUENCES"][this_data["name"]][
                "Prediction"
            ] = "Not GPI-Anchored"
            result_summary["SEQUENCES"][this_data["name"]]["pred_text"] = ""
    return result_summary


def make_result_files(
    test_data: pd.core.frame.DataFrame, write_path: str, result_summary: dict
):
    """Creates output files from the predictions provided in test_data.

    test_data:      The DataFrame should at least contain the following:
                    'sequence':   The original sequences.
                    'seq_length': The length of the input sequences before
                                  padding. That is, after left truncation and
                                  after addition of terminal symbol.
                    'name':       The sequences identifier.
                    'result':     The predicted positions distance from the
                               sequences end.
                    'likelihood': The probability at the predicted position.
    write_path:     The directory to write to.
    result_summary: Dict containing meta-data for result display.
    """
    html_path = write_path
    if "html" in html_path:
        spl = html_path.split("/")
        while "html" not in spl[0]:
            spl.pop(0)
        html_path = "/" + "/".join(spl[1:])

    # Write results file
    csv_file = path_join(write_path, "output_protein_type.txt")
    mature_file = path_join(write_path, "output_mature.fasta")
    gff_file = path_join(write_path, "output.gff3")
    with open(csv_file, "w+") as outf:
        outf.write(
            "# NetGPI 1.1\tTimestamp: %s\n"
            % (datetime.strftime(datetime.now(), "%Y%m%d%H%M%S"))
        )
        outf.write(
            "# ID\tSeq-length\tPred. GPI-Anchored\tOmega-site pos."
            + "\tLikelihood\tAmino-acid\n"
        )
        for i, row in test_data.iterrows():
            if row["result"] > 0:
                outf.write(
                    "%s\t%i\t%s\t%i\t%.3f\t%s\n"
                    % (
                        row["name"],
                        len(row["sequence"]),
                        "GPI-Anchored",
                        len(row["sequence"]) + 1 - row["result"],
                        row["omegahood"],
                        row["sequence"][len(row["sequence"]) - row["result"]],
                    )
                )
            else:
                outf.write(
                    "%s\t%i\t%s\t-\t%.3f\t%s\n"
                    % (
                        row["name"],
                        len(row["sequence"]),
                        "Not GPI-Anchored",
                        row["sentinelhood"],
                        "*",
                    )
                )

    # List of evaluated proteins
    with open(mature_file, "w+") as outf:
        for i, row in test_data.iterrows():
            if row["result"] > 0:
                outf.write(">%s\n" % (row["name"]))
                outf.write("%s\n" % (row["sequence"]))

    # Write results file gff3 format
    with open(gff_file, "w+") as outf:
        outf.write("## gff-version 3\n")
        for i, row in test_data.iterrows():
            if row["result"] > 0:
                outf.write(
                    "%s\tNetGPI-1.1\tLipidation\t%i\t%i\t%.3f\t.\t.\t.\tNote=GPI-anchor\n"
                    % (
                        row["name"],
                        len(row["sequence"]) + 1 - row["result"],
                        len(row["sequence"]) + 1 - row["result"],
                        row["omegahood"],
                    )
                )

    result_summary["CSV_FILE"] = path_join(html_path, "output_protein_type.txt")
    result_summary["MATURE_FILE"] = path_join(html_path, "output_mature.fasta")
    result_summary["GFF_FILE"] = path_join(html_path, "output.gff3")
    return result_summary


def main():
    # some default values for input parameters, remove before release
    file_path = None
    shortput = False
    write_path = None
    if len(sys.argv) > 1:
        args = (x for x in sys.argv[1:])
        for arg in args:
            parts = arg
            if ":" in arg:
                parts = arg.split(":")
            elif "=" in arg:
                parts = arg.split("=")
            elif arg in ("-f", "--file"):
                file_path = next(args, None)
            elif arg in ("-s", "--short"):
                shortput = True
            elif arg in ("-o", "--output"):
                write_path = next(args, None)
            elif arg in ("-t", "--format"):
                format_str = next(args, None)
                if format_str == "short":
                    shortput = True

            if len(parts) == 2:
                if parts[0] in ("-f", "--file"):
                    file_path = parts[1]
                elif parts[0] in ("-o", "--output"):
                    write_path = parts[1]
                elif parts[0] in ("-t", "--format"):
                    format_str = parts[1]
                    if format_str == "short":
                        shortput = True

    if type(file_path) != str:
        raise ValueError("File path (-f/--file) not provided, improper or nonexistent.")
    elif not isfile(file_path):
        raise ValueError("File path (-f/--file) improper or nonexistent.")

    if type(write_path) != str:
        raise ValueError("Write path (-o/--output) not provided")
    elif not isdir(write_path):
        try:
            mkdir(write_path)
        except Exception:
            raise ValueError(
                """Write path (-o/--output) improper, nonexistent or
                    no write permission."""
            )

    # Read data, parser should only return applicable samples.
    test_data = fetch_data_as_frame(fetching=file_path, file_type="fasta")
    # pdb.set_trace()
    result_summary = {
        "INFO": {"failedjobs": 0, "size": len(test_data)},
        "CSV_FILE": "",
        "MATURE_FILE": "",
        "GFF_FILE": "",
        "ZIP_FILE": "",
        "SEQUENCES": {},
        "FORMAT": "short" if shortput else "long",
    }

    # If the parser gives us no data, write error file to output dir
    if len(test_data) < 1:
        with open(path_join(write_path, "errors.json"), "w+") as outf:
            outf.write(
                "{ parse_error: %r}" % ("Could not parse input!" + " Check data file.")
            )
        result_summary["INFO"]["failedjobs"] = 1

    else:

        test_in = get_variable(torch.LongTensor(test_data["enc_input"]))
        test_lengths = get_variable(torch.LongTensor(test_data["seq_length"]))

        # Read model ensemble
        ensembles = []
        for tp in range(5):
            for vp in range(5):
                if tp == vp:
                    continue
                part_ensemble_data = torch.load(
                    "../../picklejar/gi300_model_ensemble_tp_%d_vp_%d.pt" % (tp, vp),
                    map_location=DEVICE,
                )
                model = SinglePointerNet(
                    part_ensemble_data["params"]["lstm-hunits"],
                    part_ensemble_data["params"]["lstm-layers"],
                    part_ensemble_data["params"]["lstm-dropout"],
                    part_ensemble_data["params"]["attention-dim"],
                    True,
                    part_ensemble_data["params"]["embedding-dim"],
                    22,
                )
                del part_ensemble_data["model"]["state_dict"]["kingd_embeddings.weight"]
                model.load_state_dict(part_ensemble_data["model"]["state_dict"])

                ensembles.append({"model": model, "test-part": tp, "val-part": vp})
        model_ensemble = pd.DataFrame(ensembles)

        # Make prediction
        predictions, probabilities, all_outputs = predict(
            model_ensemble, test_in, test_lengths
        )
        # Class index is 0 indexed !
        test_data["pred"] = get_numpy(predictions) + 1
        # Prediction is based on left truncated sequences, i.e. there is a
        # predesignated maximum length (distance) from the sequences end.
        # Here we get the predicted position's distance from the sequences end.
        test_data["result"] = (test_data["seq_length"]) - test_data["pred"]
        # Awkward confusion incoming! 'output' is here 'renamed' to target
        test_data["target"] = test_data["output"]
        # 'output' is now the ensembles output, which are probability dist.
        test_data["output"] = [np.array(x) for x in get_numpy(probabilities)]
        # The ensembles highest probability, that is, the probability at
        # the predicted position.
        for ind, row in test_data.iterrows():
            test_data.loc[ind, "omegahood"] = np.max(
                get_numpy(probabilities)[ind, : row["seq_length"] - 1]
            )
            test_data.loc[ind, "sentinelhood"] = get_numpy(probabilities)[
                ind, row["seq_length"] - 1
            ]

        result_summary = make_prediction_graphs(
            test_data, write_path, probabilities, all_outputs, result_summary, shortput
        )

    result_summary = make_result_files(test_data, write_path, result_summary)

    with open(path_join(write_path, "output.json"), "w+") as outf:
        json.dump(result_summary, outf, indent=2)


if __name__ == "__main__":
    main()
