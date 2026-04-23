"""Doit scripts to launch retention training jobs. This is for SPL only.

The training data process script creates a single result used for all models. This is done in https://github.allstate.com/D3-Lifetime-Value/eltv-policy-data-pipeline.
The training data creation script creates both a single result and a dataset for each model.
The model training scripts run for `other - short` only.
"""

from typing import List, Optional

import ltv_helpers.doit_help as doh
import ltv_helpers.logging_help as logh
from classic_spl_ltv.config.paths import paths as p

DOIT_CONFIG = {"reporter": doh.MyReporter}
end_date = doh.get_and_set_end_date()
logh.add_file_log(__file__)

tf = doh.TaskFactory(end_date)


def generate_featurized_data(
    run_type: str,
    outputs: List[str],
    inputs: List[str],
    executor_tier: str = "large-spark",
    num_executors: int = 8,
    max_executors: Optional[int] = None,
):
    assert run_type in ["scoring", "training"], f"{run_type=} not valid value"

    task = tf.generate_python_task(
        py_file_path="/mnt/code/classic_spl_ltv/jobs/retention/spl_pipeline_featurize.py",
        py_args=f"--{run_type}",
        outputs=outputs,
        inputs=inputs,
        is_spark=True,
        list_txt="Generate training or scoring data for retention models",
        executor_tier=executor_tier,
        num_executors=num_executors,
        max_executors=max_executors,
    )

    return task


task_generate_ret_train_features = generate_featurized_data(
    run_type="training",
    outputs=[p.ret_ltv_training],
    inputs=[p.retention_preprocess_train],
    executor_tier="large-spark-clone", # high memory large spark
    num_executors=12,
    max_executors=None,
)


def train_models(
    model_type: str,
    model_line: str,
    run_type: str,
    outputs: List[str],
    inputs: List[str],
):
    """Build a task for training models"""
    task = tf.generate_python_task(
        py_file_path="/mnt/code/classic_spl_ltv/jobs/retention/common_train.py",
        py_args=f"--model_type={model_type} --model_line={model_line} --run_type={run_type}",
        outputs=outputs,
        inputs=inputs,
        is_spark=False,
        list_txt="Train new other retention models",
        high_mem_hardware=True,
    )

    return task


def score_models(
    model_type: str,
    model_line: str,
    outputs: List[str],
    inputs: List[str],
):
    """Build a task for scoring on data with models."""
    task = tf.generate_python_task(
        py_file_path="/mnt/code/classic_spl_ltv/jobs/retention/common_score.py",
        py_args=f"evaluate --model_type={model_type} --model_line={model_line}",
        outputs=outputs,
        inputs=inputs,
        is_spark=False,
        list_txt="Score the new retention model.",
        high_mem_hardware=True,
    )

    return task


def rerun_eval_notebooks(
    outputs: List[str],
    inputs: List[str],
):
    """Build a task for rerunning evaluation notebooks on pre-existing scoring data."""
    task = tf.generate_python_task(
        py_file_path="/mnt/code/classic_spl_ltv/jobs/retention/rerun_eval_notebooks.py",
        outputs=outputs,
        inputs=inputs,
        is_spark=False,
        list_txt="Rerun the model eval notebooks.",
        high_mem_hardware=True,
    )

    return task


# Below are all the individual scripts.
task_train_other_short_term = train_models(
    model_type="short",
    model_line="other",
    run_type="training",
    outputs=[
        p.ret_other_short_lgb,
        p.ret_other_short_enc,
        p.ret_other_short_training_set,
        p.ret_other_short_validation_set,
        p.ret_other_short_test_set,
        p.ret_other_short_labels,
    ],
    inputs=[p.ret_ltv_training],
)


task_score_other_short_term = score_models(
    model_type="short",
    model_line="other",
    outputs=[
        p.ret_other_short_training_set_scored,
        p.ret_other_short_validation_set_scored,
        p.ret_other_short_test_set_scored,
    ],
    inputs=[
        p.ret_other_short_lgb,
        p.ret_other_short_enc,
        p.ret_other_short_training_set,
        p.ret_other_short_validation_set,
        p.ret_other_short_test_set,
        p.ret_other_short_labels,
    ],
)


task_rerun_eval_notebooks = rerun_eval_notebooks(
    outputs=[
        p.ret_other_short_val_ipynb,
    ],
    inputs=[
        p.ret_other_short_training_set_scored,
        p.ret_other_short_validation_set_scored,
        p.ret_other_short_test_set_scored,
    ],
)
