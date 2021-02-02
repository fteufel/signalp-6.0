# signalp-6.0
Signal peptide prediction model based on a [Bert protein language model encoder](https://github.com/agemagician/ProtTrans) and a conditional random field (CRF) decoder.

Install in editable mode with `pip install -e ./` to experiment.

## Data

The training dataset as well as the full dataset before homology partitioning are in `data`. The directory additionally contains the extended vocabulary of the ProtTrans `BertTokenizer` used.

## Training

You can find the training script in `scripts/train_model.py`. The pytorch model is in  `src/signalp6/models`. Please refer to the training script source for the meaning of all parameters.

A basic training command with region tagging, region regularization and layer pruning enabled looks like this:
```
python3 scripts/train_model.py --data data/train_set.fasta --test_partition 0 --validation_partition 1 --output_dir testruns --experiment_name testrun1 --remove_top_layers 1 --kingdom_as_token --sp_region_labels --region_regularization_alpha 0.5 --constrain_crf
```

## Deployment
A version optimized for deployment is in `model_for_deployment`. This consumes all cross-validation checkpoints and creates a single model, that takes care of ensembling the checkpoints internally. Naturally, this multiplies the memory demand of the model by the number of checkpoints. Run with `scripts/torchscript_export_model.py`.


## Other things in package
- `training_utils` contains parts that were used to fit the model, e.g. dataloading and regularization.
- `utils` contains other parts, such as functions to calculate metrics or region statistics.


