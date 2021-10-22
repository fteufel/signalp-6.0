# SignalP 6.0

Signal peptide prediction model based on a [Bert protein language model encoder](https://github.com/agemagician/ProtTrans) and a conditional random field (CRF) decoder.

This is the development codebase. If you are looking for the prediction service, go to https://services.healthtech.dtu.dk/service.php?SignalP-6.0.

The installation instructions for the installable SignalP 6.0 prediction tool can be found [here](https://github.com/fteufel/signalp-6.0/blob/main/installation_instructions.md).

Install in editable mode with `pip install -e ./` to experiment.

## Data

The training dataset as well as the full dataset before homology partitioning are in `data`. The directory additionally contains the extended vocabulary of the ProtTrans `BertTokenizer` used.

## Training

You can find the training script in `scripts/train_model.py`. The pytorch model is in  `src/signalp6/models`. Please refer to the training script source for the meaning of all parameters.

A basic training command looks like this:
```
python3 scripts/train_model.py --data data/train_set.fasta --test_partition 0 --validation_partition 1 --output_dir testruns --experiment_name testrun1 --remove_top_layers 1 --kingdom_as_token --sp_region_labels --region_regularization_alpha 0.5 --constrain_crf
```


## Other things in package
- `training_utils` contains parts that were used to fit the model, e.g. dataloading and regularization.
- `utils` contains other utilities, such as functions to calculate metrics or region statistics.


