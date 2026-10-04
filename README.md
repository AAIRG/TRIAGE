# Replication Package for _"A Systematic Approach to Predict the Impact of Cybersecurity Vulnerabilities Using LLMs"_

This repository contains the replication package for the paper "A Systematic Approach to Predict 
the Impact of Cybersecurity Vulnerabilities Using LLMs", by Anders Mølmen Høst, Pierre Lison, 
and Leon Moonen, accepted for publication in the 24th IEEE International Conference on Trust, 
Security and Privacy in Computing and Communications (TrustCom).

The replication package contains the scripts and data to run the experiments and analyze the results related 
to our paper. A preprint of the paper is avaiable on arXiv at <https://arxiv.org/abs/2508.18439>.


## Organization
- `data/` contains the following subfolders:
  - `external/`: Exported experiment data tracked with Neptune.
  - `input/`: Input data for experiments.
  - `post_processed/tables_data/`: CSV files to produce tables in the paper.
  - `pre_processed/`: Pre-processed data from MITRE ATT&CK, NVD, CWE, and 
    CVE Mapping Methodology (CMM). Labeled data and associated data splits.
  - Individual files in these folders are described in data/README.md.
  individual files in these folders are described in data/README.md
- `lib/` contains the overall approach including the experiment set up and analysis. 
  - `load_data.py`: Handles the data necessary for experimentation or analysis. 
  - `utils.py`: Utility functions used throughout the project. Includes command line arguments.
  - `smet_ranking.py`: Utility functions for `run_smet_ranking.py`.
  - `uncategorized_triage.py`: Utility functions for `run_uncategorized_triage.py`.
- `analysis/` directory contains marimo notebooks to create the tables found 
  in the paper. 
- `lib/core/`: Implements the main components of TRIAGE (Methodology Mappers, 
  In-Context Learner, and combining the final predictions).
- `run_scripts/`: Scripts for running individual mapping methods.
    - Methodology Mappers: 
      - `run_exp_affected_object.py`: Affected object types.
      - `run_functionality.py`: Functionality.
      - `run_job_exploit_method.py`: Exploitation technique.
      - `run_job_tactic_technique_method.py`: Tactic.
      - `run_vulnerability_type.py`; Vulnerability type.
    - `run_combine_approach.py`: The combined approach is the last step in TRIAGE combing 
      intermediate predictions from Methodology Mappers with the In-Context Learner. 
    - `run_in_context_learner.py`: The In-Context Learner.
    - `run_methodology_combine.py`: Combining the methodology mappers. 
    - `run_smet_ranking.py`: Compute ranking metrics for SMET predictions
    - `run_uncategorized_triage.py`: Merging TRIAGE predictions across mapping types to enable 
      comparison with SMET predictions
    - `make_train_test_split.py`: Split data into train and test sets
- `plots/`: Scripts to make figures in the paper
  for experiment configuration. 
- `models/`: Stores the output from experiment runs. The performance metrics of these 
  experiments are stored in `data/external/`, and further explored in `analysis/`.
  - `categorized_mapping/experiments/`: Experiments using the categorized approach with 
    mapping types (exploitation technique, primary impact, secondary impact). Each experiment has its own experiment directory which is 
    a subdirectory named with the experiment ID, for example: `CVET2-1030`. The experiment 
    directories contains the following files:
    - In-Context Learner experiments
      - `context.txt`: The instructed role of the LLM and the first part of the prompt.
      - `question_history.txt`: The prompting log with the assigned instructions and in-context 
        learning examples. Due to the fact that most information is static, only the first 
        example (CVE) is stored.
      - `history.json` : The predictions and model reasoning for each example.
    - Methodology Mappers 
      - Predictions in a CSV file
    - Combination approach
      - JSON file with predictions from both Methodology Mappers and In-Context Learner in addition to the final predictions.
    - `tmp_output/`: Directory to store temporary output for each individual method. This 
      directory comes in handy to easy access intermediate predictions when running the full pipeline with the `combined_approach`.
    - `uncategorized_mapping/SMET_output/`: Output from SMET
    - `uncategorized_mapping/experiments/`: Final processed experiment data from SMET and merge_sort
- `SMET/`: External code from SMET
- `figures/`: Figures in the paper.
- `shell_scripts/`: Shell scripts to run multiple experiments
- Root Directory
  - `config.py`, `constants.py`, `settings.py`: Project configuration files.
  - `prompt_templates.pdf`: Prompt templates from the paper
  - `requirements.txt`: Python package dependencies.

## Prerequisites
### Installation
The main project (`lib/`) requires Python 3.12. To set up the environment, follow these steps:

1. Create a virtual environment.
2. Install the necessary packages using pip from the `requirements.txt` file.

```bash
python -m venv env
source env/bin/activate  # On Windows use `env\Scripts\activate`
pip install -r requirements.txt
```
Running `SMET` experiments requires Python 3.9. To run these experiments switch python version and 
create a second environment from `SMET/requirements.txt`.

### Configuring APIs
- Create the file `config.ini` to configure APIs
- To track experiments with `Neptune`:
  - Follow the setup as provided by `Neptune`: https://docs.neptune.ai/setup
  - Copy/paste project path and api token and make associated variables in `config.ini`
- Configure Lambda and OpenAI by copy/pasting api token and organization ID. 
- Your `config.ini` file should be structured as follows:

```
[neptune]
project_1 = [insert_value]
project_2 = [insert_value]
api_token = [insert_value]

[OPENAI]
token = [insert_value]
org_id = [insert_value]

[LAMBDA]
api_token = [insert_value]
```


## Usage

### Tables and Figures

- To analyze experiment data and create tables, go to `analysis/` and 
  run: `marimo edit [notebook_name]`

- Create the figures by running the scripts in `plots/`

### Run Experiments

To run the final best experiment configuration:

Select options as follows: 
- `--data_split`: Select "train" or "test".
- `--neptune_mode`: Enable tracking with "async" or select "debug"
- `--num_cves`: Select the number of CVEs. If not selected, the full data split (either train or test)
  is used. 

For example: 
```bash
sh shell_scripts/full_pipeline.sh --data_split "test" --neptune_mode "debug" --num_cves 2
```

To run an individual method, select a python script from `run_scripts/` and run.
For example:
```
python -m run_scripts/run_in_context_learner.py --data_split "train" --neptune_mode "debug" 
--num_cves 20
```

All predictions will be stored in `models/tmp_output/`. 
In addition, when --neptune_mode is "async", output predictions will be stored in: 
`models/categorized_mapping/experiments/[EXPERIMENT_ID]/`.

Additional arguments for individual methods can be set and these are documented in `utils.py`. 

#### Experiment with SMET

To run SMET experiments, follow the instructions in `SMET/README.md`.

## Acknowledgement

This work has been financially supported by the Research Council of
Norway through the secureIT project (RCN contract \#288787).  
The empirical evaluation made use of the Experimental Infrastructure 
for Exploration of Exascale Computing (eX3), financially supported 
by the Research Council of Norway under contract \#270053.