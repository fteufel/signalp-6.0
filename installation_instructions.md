# SignalP 6.0 signal peptide prediction tool

These instructions refer to the Python package available for download at https://services.healthtech.dtu.dk/service.php?SignalP-6.0.

## Installation
SignalP 6.0 is distributed as a Python package.

1. Unpack the downloaded `tar.gz` file.
2. \[Optional\]: Create a python environment to install the package in.
3. Open the directory containing the downloaded package, and install it by executing the following command.
```
pip install signalp-6-package/
```

4. Copy the model files to the location at which the signalp module got installed. The model weight files are large, so this might take a while.
```
SIGNALP_DIR=$(python3 -c "import signalp; import os; print(os.path.dirname(signalp.__file__))" )
cp -r signalp-6-package/models/* $SIGNALP_DIR/model_weights/
```
5. The installer created a command `signalp6` on your system that is available within the python environment in which you ran step 2.
6. \[Optional\]: SignalP 6.0 supports different run modes (see [Usage](#Usage)). Your download only included the one you picked. To make multiple run modes available in the same installation, you need to download and install additional model weights. Find instructions for that [below](#installing-additional-modes).
7. \[Optional\]: By default SignalP 6.0 runs on CPU. If you have a GPU available, you can convert your installation to use it. Instructions for that are [below](#converting-to-gpu)

The package was tested on Ubuntu 18.04.2 LTS under WSL2, running Python 3.6. As it is a Python package, it should run on all operating systems that are supported by Pytorch. It typically installs in 1-5 minutes, depending on disk write speed. The dependencies can be found in `requirements.txt`.

## Usage

### How to predict

A command takes the following form 
```
signalp6 --fastafile /path/to/input.fasta --organism other --output_dir path/to/be/saved --format txt --mode fast
```

To get help for the available options, run `signalp6 -h`.  

#### Required

- `--fastafile`, `-ff`, specifies the fasta file with the sequences to be predicted.
To prevent invalid file paths, non-alphanumeric characters in fasta headers are replaced with "`_`" for saving the individual sequence output files.

- `--output_dir`, `-od`, speicifies the directory in which to save the outputs. If it does not exist, it will be created. Note that repeated calls with the same `--output_dir` will overwrite previous prediction results.


#### Options 
- `--organism`, `-org`, is either `other` or `eukarya`. Specifying `eukarya` triggers post-processing of the SP predictions to prevent spurious results (only predicts type Sec/SPI).   
Defaults to `other`.

- `--format`, `-fmt`, can take the values `txt`, `png`, `eps`, `all`, `none`. It defines what output files are created for individual sequences. `txt` produces a tabular `.gff` file with the per-position predictions for each sequence. `png`, `eps`, `all` additionally produce probability plots in the requested format. `none` only writes the summary prediction files. For larger prediction jobs, plotting will slow down the processing speed significantly.  
Defaults to `txt`.

- `--mode`, `-m`, is either `fast`, `slow` or `slow-sequential`. Default is `fast`, which uses a smaller model that approximates the performance of the full model, requiring a fraction of the resources and being significantly faster. `slow` runs the full model in parallel, which requires more than 14GB of RAM to be available. `slow-sequential` runs the full model sequentially, taking the same amount of RAM as `fast` but being 6 times slower. If the specified model is not installed, SignalP will abort with an error.   
Defaults to `fast`.

#### Performance optimization

- `--bsize`, `-bs` is the integer batch size used for prediction. When running on GPU, this should be adjusted to maximize usage of the available memory. On CPU, the choice usually has only a limited effect on performance. Defaults to `10`.

- `--write_procs`, `-wp` is the integer number of parallel processes launched for writing output files. Using multiple processes significantly speeds up writing the outputs for prediction jobs with many sequences. However, due to the way multiprocessing works in Python, this leads to increased memory usage. By setting to `1`, no additional processes are started. Defaults to the number of available CPUs with `8` processes maximum.

### Output interpretation

The installed version produces the same outputs as the SignalP 6.0 webserver. For interpretation instructions, please consult   https://services.healthtech.dtu.dk/service.php?SignalP-6.0/.  

**The following files are always created:**

#### `prediction_results.txt`
A tab delimited file with one line per prediction. 
Columns:  
- ID: the sequence ID parsed from the fasta input.
- Prediction: The predicted type. One of [`OTHER` (No SP), `SP` (Sec/SPI), `LIPO` (Sec/SPII), `TAT` (Tat/SPI), `TATLIPO` (Tat/SPII), `PILIN` (Sec/SPIII)].
- One column for each possible type with the model's probability.
- CS Position: The cleavage site. The sequence positions between which the SPase cleaves and its predicted probability.

#### `processed_entries.fasta`
The sequences that were predicted in one-line fasta format, as they were parsed by the program.

#### `output.gff3`
The start and end positions of all predicted signal peptides in GFF3 format.

#### `region_output.gff3`
The start and end positions of all predicted signal peptide regions in GFF3 format.

#### `output.json`
The prediction results in JSON format, together with details on the run parameters and paths to the generated output files. Useful for integrating SignalP 6.0 in pipelines. 


**These single-sequence files are optional:**

#### `SEQUENCE_NAME.txt`
Contains the predicted label and probabilities at each sequence position in tabular format.

#### `SEQUENCE_NAME.png` and `SEQUENCE_NAME.eps`
A plot of the predicted labels and probabilities of the sequence.


## Installing additional modes

1. Download the package for the mode you would like to add to your installation.
2. Unpack the downloaded package. The model is stored at `signalp-6-package/signalp/model_weights/`.
3. Find the target destination of your current SignalP 6.0 installation. When installed with pip in a virtual environment, this should typically be something like `/PATH/TO/ENV/ENVIRONMENT_NAME/lib/python3.6/site-packages/signalp/model_weights`. A quick way to find out the target destination is to run SignalP 6.0 with a mode that is not installed and look at the resulting `ValueError`.
4. Copy the content of the downloaded package into your installation. `cp -r signalp-6-package/signalp/model_weights/ /PATH/TO/ENV/ENVIRONMENT_NAME/lib/python3.6/site-packages/signalp/model_weights`. The additional mode should now be available and run without throwing an error.

## Converting to GPU
The model weights that come with the installation by default run on your CPU. If you have a GPU available, you can convert your installation to use the GPU instead. This is done via a command that is available after installation:
```
signalp6_convert_models gpu # makes all installed models run on GPU
signalp6_convert_models cpu # reverts all models back to CPU
```
These conversion calls run a bash script that modifies the model files. Conversion will take a while as it involves repacking large zipped archives. 
If the system does not have a GPU, the conversion call will fail. If you try to run GPU-coverted weights on CPU, signalp6 will fail.


## Bugs and Questions
For technical issues (Bugs, missing functionality, poor documentation ...) contact felix.teufel@gmail.com or open an issue at https://github.com/fteufel/signalp-6.0.  
For scientific questions, contact henni@dtu.dk.

-------------------------
## Updates

### 6.0e
- Automatically resolve rare cases where the predicted regions do not match the predicted type.

### 6.0d
- Add `none` as a format for single-sequence output files.
- Option `--write_procs` to disable multiprocessing for writing output files.
- Improved resolving of cases with missing CS.

### 6.0c
- Fix region output when using the `eukarya` mode.
- Handle Sec/SPII and Tat/SPII SPs correctly when resolving cases with no CS reported.

### 6.0b
- Fix issue where some predicted SPs did not have a CS reported. SignalP 6.0 now automatically resolves such cases. Can be disabled with `--skip-resolve`.
- Updated CLI to better match SignalP 5.0:
    - Accept argument aliases `-fasta`, `-batch`, `-format`, `-org`.
    - Accept `euk` as alias for `eukarya`.
- Added short aliases for CLI arguments.
- Improvements to README.md
- Support running models on GPU (thanks to Milan Milenovic, https://github.com/milenovic)
    - model weights can be converted to and from GPU using `signalp6_covert_models`
    - `signalp6` automatically infers whether to run on GPU when loading the model weight files.
- No longer replace whitespace in fasta headers (except when writing files, still replace with `_` in filenames).

### 6.0a
- Initial preprint release.

## Legacy hosts

Download the model weights you need at 
 - slow, memory-hungry (parallel) model at https://mega.nz/file/TFUmSZxT#sQWPIlHtkWLJ1otfqxx1zeERUUu4D4YM3yvWefTXnUU (or at DTU)
 - fast model at https://mega.nz/file/uZ0DVQ7b#oDDerv_8mJmj1ypC_CX8OaWBxfYe4dXDvHGVgOiwCks (or at DTU)
 - slow, memory-saving (sequential) model for now only hosted at DTU.
