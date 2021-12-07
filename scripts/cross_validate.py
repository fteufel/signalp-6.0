# © Copyright Technical University of Denmark
"""
Compute cross-validated metrics. Run each model
on its test set, compute all metrics and save as .csv
We save all metrics per model in the .csv, mean/sd
calculation we do later in a notebook.
"""
import torch
import os
import argparse

from signalp6.models import BertSequenceTaggingCRF, ProteinBertTokenizer
from signalp6.training_utils import RegionCRFDataset
from signalp6.utils import get_metrics_multistate
from tqdm import tqdm
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
    parser.add_argument("--randomize_kingdoms", action="store_true")
    parser.add_argument("--output_file", type=str, default="crossval_metrics.csv")
    parser.add_argument(
        "--no_multistate",
        action="store_true",
        help="Model to evaluate is no multistate model, use different labels to find CS",
    )
    parser.add_argument(
        "--n_partitions",
        type=int,
        default=5,
        help="Number of partitions, for loading the checkpoints and datasets.",
    )
    parser.add_argument(
        "--use_pvd",
        action="store_true",
        help="Replace viterbi decoding with posterior-viterbi decoding",
    )
    args = parser.parse_args()

    tokenizer = ProteinBertTokenizer.from_pretrained(
        "/zhome/1d/8/153438/experiments/master-thesis/resources/vocab_with_kingdom",
        do_lower_case=False,
    )

    # Collect results + header info to build output df
    results_list = []

    partitions = list(range(args.n_partitions))

    # for result collection
    metrics_list = []
    checkpoint_list = []
    for partition in tqdm(partitions):
        # Load data
        dataset = RegionCRFDataset(
            args.data,
            tokenizer=tokenizer,
            partition_id=[partition],
            add_special_tokens=True,
        )

        if args.randomize_kingdoms:
            import random

            kingdoms = list(set(dataset.kingdom_ids))
            random_kingdoms = random.choices(kingdoms, k=len(dataset.kingdom_ids))
            dataset.kingdom_ids = random_kingdoms
            print("randomized kingdom IDs")

        dl = torch.utils.data.DataLoader(
            dataset, collate_fn=dataset.collate_fn, batch_size=50
        )

        # Put together list of checkpoints
        checkpoints = [
            os.path.join(args.model_base_path, f"test_{partition}_val_{x}")
            for x in set(partitions).difference({partition})
        ]

        # Run

        for checkpoint in checkpoints:

            model = BertSequenceTaggingCRF.from_pretrained(checkpoint)
            setattr(model, "use_pvd", args.use_pvd)  # ad hoc fix to use pvd

            metrics = get_metrics_multistate(
                model,
                dl,
                sp_tokens=[3, 7, 11, 15, 19] if args.no_multistate else None,
                compute_region_metrics=False,
            )
            metrics_list.append(metrics)  # save metrics
            # checkpoint_list.append(f'test_{partition}_val_{x}') #save name of checkpoint

    df = pd.DataFrame.from_dict(metrics_list)
    # df.index = checkpoint_list

    df.T.to_csv(args.output_file)


if __name__ == "__main__":
    main()
