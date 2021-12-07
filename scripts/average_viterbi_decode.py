# © Copyright Technical University of Denmark
"""
Averaging the viterbi decoding.

How we do it:
The viterbi decoding itself cannot be averaged, it is discrete.
We average the emissions of multiple models, and we average the
weights of the CRF for all models. Then we run the viterbi decoding
to get one single path, that we consider the average of the models.

"""
import torch
import os
import sys

sys.path.append("/zhome/1d/8/153438/experiments/master-thesis/")
from typing import List, Tuple
import argparse

from signalp6.models import BertSequenceTaggingCRF, ProteinBertTokenizer
from signalp6.training_utils import RegionCRFDataset
from tqdm import tqdm
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_averaged_emissions(
    model_checkpoint_list: List[str], dataloader
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run sequences through all models in list.
    Return only the emissions (Bert+linear projection) and average those.
    Also returns input masks, as CRF needs them when decoding.
    """

    emission_list = []
    global_probs_list = []
    for checkpoint in tqdm(model_checkpoint_list):

        # Load model and get emissions
        model = BertSequenceTaggingCRF.from_pretrained(checkpoint)
        model.to(device)
        model.eval()

        emissions_batched = []
        masks_batched = []
        probs_batched = []
        for i, batch in enumerate(dataloader):

            (
                data,
                targets,
                input_mask,
                global_targets,
                weights,
                kingdoms,
                cleavage_sites,
            ) = batch
            data = data.to(device)
            input_mask = input_mask.to(device)

            with torch.no_grad():
                global_probs, pos_probs, pos_preds, emissions, input_mask = model(
                    data, input_mask=input_mask, return_emissions=True
                )
                emissions_batched.append(emissions)
                masks_batched.append(input_mask.cpu())
                probs_batched.append(global_probs.cpu())

        # covert to CPU tensors after forward pass. Save memory.
        model_emissions = torch.cat(emissions_batched).detach().cpu()
        emission_list.append(model_emissions)
        masks = torch.cat(
            masks_batched
        )  # gathering mask in inner loop is sufficent, same every time.
        model_probs = torch.cat(probs_batched).detach().cpu()
        global_probs_list.append(model_probs)

    # Average the emissions

    emissions = torch.stack(emission_list)
    probs = torch.stack(global_probs_list)
    return emissions.mean(dim=0), masks, probs.mean(dim=0)


def run_averaged_crf(
    model_checkpoint_list: List[str], emissions: torch.Tensor, input_mask: torch.tensor
):
    """Average weights of a list of model checkpoints,
    then run viterbi decoding on provided emissions.
    Running this on CPU should be fine, viterbi decoding
    does not use a lot of multiplications. Only one for each timestep.
    """

    # Gather the CRF weights
    start_transitions = []
    transitions = []
    end_transitions = []

    for checkpoint in model_checkpoint_list:
        model = BertSequenceTaggingCRF.from_pretrained(checkpoint)
        start_transitions.append(model.crf.start_transitions.data)
        transitions.append(model.crf.transitions.data)
        end_transitions.append(model.crf.end_transitions.data)

    # Average and set model weights
    start_transitions = torch.stack(start_transitions).mean(dim=0)
    transitions = torch.stack(transitions).mean(dim=0)
    end_transitions = torch.stack(end_transitions).mean(dim=0)

    model.crf.start_transitions.data = start_transitions
    model.crf.transitions.data = transitions
    model.crf.end_transitions.data = end_transitions

    # Get viterbi decodings
    with torch.no_grad():
        viterbi_paths = model.crf.decode(emissions=emissions, mask=input_mask.byte())

    return viterbi_paths


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        type=str,
        default="../data/signal_peptides/signalp_updated_data/signalp_6_train_set.fasta",
    )
    parser.add_argument(
        "--model_base_path",
        type=str,
        default="/work3/felteu/tagging_checkpoints/signalp_6",
    )
    parser.add_argument("--output_file", type=str, default="viterbi_paths.csv")
    parser.add_argument(
        "--n_partitions",
        type=int,
        default=5,
        help="Number of partitions, for loading the checkpoints and datasets.",
    )

    args = parser.parse_args()

    tokenizer = ProteinBertTokenizer.from_pretrained(
        "/zhome/1d/8/153438/experiments/master-thesis/resources/vocab_with_kingdom",
        do_lower_case=False,
    )

    # Collect results + header info to build output df
    results_list = []

    partitions = list(range(args.n_partitions))
    for partition in partitions:
        # Load data
        dataset = RegionCRFDataset(
            args.data,
            tokenizer=tokenizer,
            partition_id=[partition],
            add_special_tokens=True,
        )
        dl = torch.utils.data.DataLoader(
            dataset, collate_fn=dataset.collate_fn, batch_size=100
        )

        # Put together list of checkpoints
        checkpoints = [
            os.path.join(args.model_base_path, f"test_{partition}_val_{x}")
            for x in set(partitions).difference({partition})
        ]

        # Run
        emissions, input_mask, probs = get_averaged_emissions(checkpoints, dl)
        viterbi_paths = run_averaged_crf(checkpoints, emissions, input_mask)

        # Gather results + info
        for i in range(len(dataset)):
            entrydict = {}
            entry, kingdom, typ, _ = dataset.identifiers[i].split("|")
            entrydict["Entry"] = entry
            entrydict["Kingdom"] = kingdom
            entrydict["Type"] = typ
            entrydict["Path"] = viterbi_paths[i]
            entrydict["Sequence"] = dataset.sequences[i]
            entrydict["True path"] = dataset.labels[i]
            entrydict["Pred label"] = probs[i].argmax().detach().numpy()

            results_list.append(entrydict)

    # Put back together and save
    df = pd.DataFrame.from_dict(results_list)

    df.to_csv(args.output_file)


if __name__ == "__main__":
    main()
