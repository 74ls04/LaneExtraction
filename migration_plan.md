# TensorFlow 1.x to 2.x Migration Plan

This document outlines the steps to migrate the existing TensorFlow 1.x codebase to TensorFlow 2.x, aiming for a minimal rewrite approach as guided by the `MIGRATION.md` document.

## 1. Environment Setup with `uv`

First, we will set up a virtual environment and install the necessary dependencies using `uv`.

- **Initialize Environment:** Run `uv init` in the project's root directory to create a `pyproject.toml` file for managing dependencies.
- **Install Dependencies:** Install the required libraries using `uv pip install`. This will include `tensorflow`, `tf-slim` (to replace `tf.contrib`), `numpy`, `opencv-python`, and `Pillow`.

```bash
# Step 1: Initialize uv (if pyproject.toml doesn't exist)
uv init

# Step 2: Install dependencies
uv pip install tensorflow tf-slim numpy opencv-python Pillow
```

## 2. Automated Code Conversion

We will use the `tf_upgrade_v2` utility to automatically convert the bulk of the TF1.x API calls to the `tf.compat.v1` module. This allows the code to run on a TF2 installation while preserving the original logic.

- **Run Script:** Execute the `tf_upgrade_v2` script on the `code` directory.
- **Output:** The converted code will be placed in a new `code_v2` directory to keep the original source intact.
- **Report:** A `report.txt` file will be generated, highlighting any conversions that require manual review or cannot be automated.

```bash
tf_upgrade_v2 --intree code --outtree code_v2 --reportfile report.txt
```

## 3. Manual `tf.contrib` Replacement

The `tf.contrib` module has been deprecated and is not handled by the automated script. We will manually replace its usage.

- **Identify Usage:** The primary usage is `tf.contrib.layers` within the `cnnmodels` directory (e.g., `resnet.py`).
- **Replace with `tf-slim`:** As recommended by the migration guide, we will replace `tf.contrib.layers` with the equivalent functions from the `tf_slim` library, which we installed in Step 1. This ensures that the model architecture remains consistent.

## 4. Training Framework Update

We will make minimal necessary changes to the training framework to ensure it functions correctly with the migrated code.

- **Update Optimizer:** In `code_v2/framework/model.py`, replace the `tf.train.AdamOptimizer` with its Keras equivalent, `tf.keras.optimizers.Adam`.
- **Update Arguments:** The argument names for the optimizer may need to be updated (e.g., `beta1` to `beta_1`).

## 5. Validation

After the automated and manual changes, we will validate the migration to ensure the model's behavior has not changed.

- **Review Report:** Carefully review the `report.txt` file for any warnings or errors that need to be addressed.
- **Run Training:** Execute the main training script (e.g., `code_v2/laneAndDirectionExtraction/train.py`).
- **Compare Outputs:** Compare key metrics (loss, accuracy) and generated validation images against a run from the original codebase to ensure numerical and functional consistency.
