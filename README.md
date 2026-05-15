# How to Run Retention Training / Scoring

This file will provide detailed instructions on how to generate new retention models as well as how to run a weekly scoring batch job. The following links may be useful prior to attempting to run these processes:

1. How to utilize the DOIT build system - https://github.allstate.com/D3-Lifetime-Value/classic-specialty-ltv/blob/main/classic_spl_ltv/dodo/dodo_retention.py
2. More detailed information about what our preprocessing and featurizing scripts do - https://confluence.allstate.com/display/EL/V7+Retention+-+Data
3. If you have any questions about retention feel free to reach out to <!-- TODO: update contacts -->

## Retention Training Process

If you plan to re-train the LTV Retention models, the first thing you should do is create a new branch off of the current main branch (currently v9.1). This will ensure that you do not accidentally overwrite any models or encoders that are being used in the weekly scoring process. The command to create a new branch is `git checkout -b <new-branch> <existing-branch>`.

The commands to rerun the retention training process are as follows...

1. `'./doit.sh' prod <END_DATE> 'classic_spl_ltv/dodo/dodo_retention.py' generate_ret_training` - This task runs the `/eltv_pipeline/jobs/retention/common_pipeline_preprocess.py` file which reads in data from disparate sources and joins it together to create the base training dataset.

2. `'./doit.sh' prod <END_DATE> 'classic_spl_ltv/dodo/dodo_retention.py' generate_ret_training` - This task runs the `/classic_spl_ltv/jobs/retention/spl_pipeline_featurize.py` script which builds the features we require for training such as the autoregressive and moving average features, RUFF features, and more. After this step, the training dataset is complete and can be used to train the Auto, Hoc, and Other retention models.

3. `'./doit.sh' prod <END_DATE> 'classic_spl_ltv/dodo/dodo_retention.py' score_auto_short_term` - This task will train and score the short-term auto retention model. This command will kick off both the model training and model scoring processes. First, the `/classic_spl_ltv/jobs/retention/common_train.py` will kick off which splits the data into train / test/ validation sets and then trains the model. Next, the `/classic_spl_ltv/jobs/retention/common_score.py` will kick off which reads in the split data and uses the model output to score each policy. For training runs, the `/classic_spl_ltv/jobs/retention/pipeline/eval_models_template.ipynb` will run which automates much of the model review process. Note that the next 5 commands run the same process described in this step.

4. `'./doit.sh' prod <END_DATE> 'classic_spl_ltv/dodo/dodo_retention.py' score_auto_long_term`
5. `'./doit.sh' prod <END_DATE> 'classic_spl_ltv/dodo/dodo_retention.py' score_hoc_short_term`
6. `'./doit.sh' prod <END_DATE> 'classic_spl_ltv/dodo/dodo_retention.py' score_hoc_long_term`
7. `'./doit.sh' prod <END_DATE> 'classic_spl_ltv/dodo/dodo_retention.py' score_other_short_term`
8. `'./doit.sh' prod <END_DATE> 'classic_spl_ltv/dodo/dodo_retention.py' score_other_long_term`

## Retention Scoring Process

Below are the steps to run the retention scoring batch job. It is rare that these commands will need to be used given that they are encapsulated in the weekly batch job. This section is useful if changes are made to the scoring pipeline and testing needs to be done independent of the other LTV tasks.

The commands to run the retention scoring process are...

1. `'./doit.sh' prod <END_DATE> 'dodo.py' generate_ret_scoring` - This command kicks off the `/eltv_pipeline/jobs/retention/common_pipeline_preprocess.py` with the "--scoring" flag applied. Similarly to the training version, this step joins together all necessary datasets to score LTV Retention.

2. `'./doit.sh' prod <END_DATE> 'dodo.py' generate_ret_score_features` - This command kicks off the `/classic_spl_ltv/jobs/retention/spl_pipeline_featurize.py` with the "--scoring" flag applied. Again, similarly to the training version, this step builds the remaining features needed to score each policy. Once this step is complete, we can start scoring.

3. `'./doit.sh' prod <END_DATE> 'dodo.py' retention_scoring` - This task will handle the entire scoring process. It will first run a "clean" command that will empty out the scored retention data from the target directories for each combination of Auto / Hoc / Other and Short / Long term models. This ensures that we do not release multiple scores per policy. Next, for each combination of model line (Auto, Hoc, Other) and model type (Short, Long) the `/classic_spl_ltv/jobs/retention/common_score.py` will be kicked off, which is where each line is iteratively scored and the results are written to their target output path.

## Troubleshooting

### Environment

See [Weekly Run - Troubleshooting - Environment](<!-- TODO: update link -->) for some environment troubleshooting tips.
