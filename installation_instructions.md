# SignalP 6.0 signal peptide prediction tool


## Installation
SignalP 6.0 is distributed as a python package. It is too large to host on Github, please download it at https://services.healthtech.dtu.dk/service.php?SignalP-6.0

1. Unpack the downloaded `tar.gz` file.
2. \[Optional\]: Create a python environment to install the package in.
3. Open the directory containing the downloaded package, and install it with `pip install signalp-6-package`
4. The installer creates a command `signalp6` on your system that is available within the python environment in which you ran step 2.
5. \[Optional\]: SignalP 6.0 supports different run modes (see [Usage](#Usage)). Your installer only included the one you picked. To make multiple run modes available in the same installation, you need to download and install additional model weights. Find instructions for that [below](#installing-additional-modes).

The package was tested on Ubuntu 18.04.2 LTS under WSL2, running Python 3.6. As it is a Python package, it should run on all operating systems that are supported by Pytorch. It 
typically installs in 1-5 minutes, depending on disk write speed. The dependencies can be found in `requirements.txt`

## Usage

### How to predict

A command takes the following form 
```
signalp6 --fastafile /path/to/input.fasta --organism other --output_dir path/to/be/saved --format txt --mode fast
```

- `fastafile` specifies the fasta file with the sequences to be predicted. A demo fasta file can be found [here](https://github.com/fteufel/signalp-6.0/blob/main/data/example_seqs.fasta).
To prevent invalid file paths, non-alphanumeric characters in fasta headers are replaced with "`_`" for saving the individual sequence output files.

- `organism` is either `other` or `Eukarya`. Specifying `Eukarya` triggers post-processing of the SP predictions to prevent spurious results (only predicts type Sec/SPI).

- `format` can take the values `txt`, `png`, `eps`, `all`. It defines what output files are created for individual sequences. `txt` produces a tabular `.gff` file with the per-position predictions for each sequence. `png`, `eps`, `all` additionally produce probability plots in the requested format. For larger prediction jobs, plotting will slow down the processing speed significantly.

- `mode` is either `fast`, `slow` or `slow-sequential`. Default is `fast`, which uses a smaller model that approximates the performance of the full model, requiring a fraction of the resources and being significantly faster. `slow` runs the full model in parallel, which requires more than 14GB of RAM to be available. `slow-sequential` runs the full model sequentially, taking the same amount of RAM as `fast` but being 6 times slower. If the specified model is not installed, SignalP will abort with an error.

The demo fasta file should be finished in less than one minute. Each `signalp6` command initially needs to load the model into memory, so it is advisable 
to submit multiple sequences in one fasta file, rather than calling `signalp6` multiple times.

### Output interpretation

The installed version produces the same outputs as the SignalP 6.0 webserver. For instructions, please consult https://services.healthtech.dtu.dk/service.php?SignalP-6.0/

## Installing additional modes

1. Download the package for the mode you would like to add to your installation.
2. Unpack the downloaded package. The model is stored at `signalp-6-package/signalp/model_weights/`.
3. Find the target destination of your current SignalP 6.0 installation. When installed with pip in a virtual environment, this should typically be something like `/PATH/TO/ENV/ENVIRONMENT_NAME/lib/python3.6/site-packages/signalp/model_weights`. A quick way to find out the target destination is to run SignalP 6.0 with a mode that is not installed and look at the resulting `ValueError`.
4. Copy the content of the downloaded package into your installation. `cp -r signalp-6-package/signalp/model_weights/ /PATH/TO/ENV/ENVIRONMENT_NAME/lib/python3.6/site-packages/signalp/model_weights`. The additional mode should now be available and run without throwing an error.
