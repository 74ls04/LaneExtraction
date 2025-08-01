TF1.x -> TF2 migration overview

TensorFlow 2 is fundamentally different from TF1.x in several ways. You can still run unmodified TF1.x code (except for contrib) against TF2 binary installations like so:

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

However, this is not running TF2 behaviors and APIs, and may not work as expected with code written for TF2. If you are not running with TF2 behaviors active, you are effectively running TF1.x on top of a TF2 installation. Read the TF1 vs TF2 behaviors guide for more details on how TF2 is different from TF1.x.

This guide provides an overview of the process to migrate your TF1.x code to TF2. This enables you to take advantage of new and future feature improvements and also make your code simpler, more performant, and easier to maintain.

If you are using tf.keras's high level APIs and training exclusively with model.fit, your code should more or less be fully compatible with TF2 except for the following caveats:

    TF2 has new default learning rates for Keras optimizers.
    TF2 may have changed the "name" that metrics are logged to.

TF2 migration process

Before migrating, learn about the behavior and API differences between TF1.x and TF2 by reading the guide.

    Run the automated script to convert some of your TF1.x API usage to tf.compat.v1.
    Remove old tf.contrib symbols (check TF Addons and TF-Slim).
    Make your TF1.x model forward passes run in TF2 with eager execution enabled.
    Upgrade your TF1.x code for training loops and saving/loading models to TF2 equivalents.
    (Optional) Migrate your TF2-compatible tf.compat.v1 APIs to idiomatic TF2 APIs.

The following sections expand upon the steps outlined above.
Run the symbol conversion script

This executes an initial pass at rewriting your code symbols to run against TF 2.x binaries, but won't make your code idiomatic to TF 2.x nor will it automatically make your code compatible with TF2 behaviors.

Your code will most likely still make use of tf.compat.v1 endpoints to access placeholders, sessions, collections, and other TF1.x-style functionality.

Read the guide to find out more about the best practices for using the symbol conversion script.
Remove usage of tf.contrib

The tf.contrib module has been sunsetted and several of its submodules have been integrated into the core TF2 API. The other submodules are now spun-off into other projects like TF IO and TF Addons.

A large amount of older TF1.x code uses the Slim library, which was packaged with TF1.x as tf.contrib.layers. When migrating your Slim code to TF2, switch your Slim API usages to point to the tf-slim pip package. Then, read the model mapping guide to learn how to convert Slim code.

Alternatively, if you use Slim pre-trained models you may consider trying out Keras's pre-traimed models from tf.keras.applications or TF Hub's TF2 SavedModels exported from the original Slim code.
Make TF1.x model forward passes run with TF2 behaviors enabled
Track variables and losses

TF2 does not support global collections.

Eager execution in TF2 does not support tf.Graph collection-based APIs. This affects how you construct and track variables.

For new TF2 code you would use tf.Variable instead of v1.get_variable and use Python objects to collect and track variables instead of tf.compat.v1.variable_scope. Typically this would be one of:

    tf.keras.layers.Layer
    tf.keras.Model
    tf.Module

Aggregate lists of variables (like tf.Graph.get_collection(tf.GraphKeys.VARIABLES)) with the .variables and .trainable_variables attributes of the Layer, Module, or Model objects.

The Layer and Model classes implement several other properties that remove the need for global collections. Their .losses property can be a replacement for using the tf.GraphKeys.LOSSES collection.

Read the model mapping guide to find out more about using the TF2 code modeling shims to embed your existing get_variable and variable_scope based code inside of Layers, Models, and Modules. This will let you the execute forward passes with eager execution enabled without major rewrites.
Adapting to other behavior changes

If the model mapping guide on its own is insufficient to get your model forward pass running other behavior changes that may be more details, see the guide on TF1.x vs TF2 behaviors to learn about the other behavior changes and how you can adapt to them. Also check out the making new Layers and Models via subclassing guide for details.
Validating your results

See the model validation guide for easy tools and guidance around how you can (numerically) validate that your model is behaving correctly when eager execution is enabled. You may find this especially useful when paired with the model mapping guide.
Upgrade training, evaluation, and import/export code

TF1.x training loops built with v1.Session-style tf.estimator.Estimators and other collections-based approaches are not compatible with the new behaviors of TF2. It is important you migrate all your TF1.x training code as combining it with TF2 code can cause unexpected behaviors.

You can choose from among several strategies to do this.

The highest-level approach is to use tf.keras. The high level functions in Keras manage a lot of the low-level details that might be easy to miss if you write your own training loop. For example, they automatically collect the regularization losses, and set the training=True argument when calling the model.

Refer to the Estimator migration guide to learn how you can migrate tf.estimator.Estimators code to use vanilla and custom tf.keras training loops.

Custom training loops give you finer control over your model such as tracking the weights of individual layers. Read the guide on building training loops from scratch to learn how to use tf.GradientTape to retrieve model weights and use them to update the model.
Convert TF1.x optimizers to Keras optimizers

The optimizers in tf.compat.v1.train, such as the Adam optimizer and the gradient descent optimizer, have equivalents in tf.keras.optimizers.

The table below summarizes how you can convert these legacy optimizers to their Keras equivalents. You can directly replace the TF1.x version with the TF2 version unless additional steps (such as updating the default learning rate) are required.

Note that converting your optimizers may make old checkpoints incompatible.
TF1.x 	TF2 	Additional steps
`tf.v1.train.GradientDescentOptimizer` 	tf.keras.optimizers.SGD 	None
`tf.v1.train.MomentumOptimizer` 	tf.keras.optimizers.SGD 	Include the `momentum` argument
`tf.v1.train.AdamOptimizer` 	tf.keras.optimizers.Adam 	Rename `beta1` and `beta2` arguments to `beta_1` and `beta_2`
`tf.v1.train.RMSPropOptimizer` 	tf.keras.optimizers.RMSprop 	Rename the `decay` argument to `rho`
`tf.v1.train.AdadeltaOptimizer` 	tf.keras.optimizers.Adadelta 	None
`tf.v1.train.AdagradOptimizer` 	tf.keras.optimizers.Adagrad 	None
`tf.v1.train.FtrlOptimizer` 	tf.keras.optimizers.Ftrl 	Remove the `accum_name` and `linear_name` arguments
`tf.contrib.AdamaxOptimizer` 	tf.keras.optimizers.Adamax 	Rename the `beta1`, and `beta2` arguments to `beta_1` and `beta_2`
`tf.contrib.Nadam` 	tf.keras.optimizers.Nadam 	Rename the `beta1`, and `beta2` arguments to `beta_1` and `beta_2`
Note: In TF2, all epsilons (numerical stability constants) now default to 1e-7 instead of 1e-8. This difference is negligible in most use cases.
Upgrade data input pipelines

There are many ways to feed data to a tf.keras model. They will accept Python generators and Numpy arrays as input.

The recommended way to feed data to a model is to use the tf.data package, which contains a collection of high performance classes for manipulating data. The datasets belonging to tf.data are efficient, expressive, and integrate well with TF2.

They can be passed directly to the tf.keras.Model.fit method.

model.fit(dataset, epochs=5)

They can be iterated over directly standard Python:

for example_batch, label_batch in dataset:
    break

If you are still using tf.queue, these are now only supported as data-structures, not as input pipelines.

You should also migrate all feature preprocessing code that usestf.feature_columns. Read the migration guide for more details.
Saving and loading models

TF2 uses object-based checkpoints. Read the checkpoint migration guide to learn more about migrating off name-based TF1.x checkpoints. Also read the checkpoints guide in the core TensorFlow docs.

There are no significant compatibility concerns for saved models. Read the SavedModel guide for more information about migrating SavedModels in TF1.x to TF2. In general,

    TF1.x saved_models work in TF2.
    TF2 saved_models work in TF1.x if all the ops are supported.

Also refer to the GraphDef section in the SavedModel migration guide for more information on working with Graph.pb and Graph.pbtxt objects.
(Optional) Migrate off tf.compat.v1 symbols

The tf.compat.v1 module contains the complete TF1.x API, with its original semantics.

Even after following the steps above and ending up with code that is fully compatible with all TF2 behaviors, it is likely there may be many mentions of compat.v1 apis that happen to be compatible with TF2. You should avoid using these legacy compat.v1 apis for any new code that you write, though they will continue working for your already-written code.

However, you may choose to migrate the existing usages to non-legacy TF2 APIs. The docstrings of individual compat.v1 symbols will often explain how to migrate them to non-legacy TF2 APIs. Additionally, the model mapping guide's section on incremental migration to idiomatic TF2 APIs may help with this as well.
Resources and further reading
As mentioned previously, it is a good practice to migrate all your TF1.x code to TF2. Read the guides in the Migrate to TF2 section of the TensorFlow guide to learn more.


TensorFlow 2.x includes many API changes from TF 1.x and the tf.compat.v1 APIs, such as reordering arguments, renaming symbols, and changing default values for parameters. Manually performing all of these modifications would be tedious and prone to error. To streamline the changes, and to make your transition to TF 2.x as seamless as possible, the TensorFlow team has created the tf_upgrade_v2 utility to help transition legacy code to the new API.
Note: tf_upgrade_v2 is installed automatically for TensorFlow 1.13 and later (including all TF 2.x builds).

Typical usage is like this:

tf_upgrade_v2 \
  --intree my_project/ \
  --outtree my_project_v2/ \
  --reportfile report.txt

It will accelerate your upgrade process by converting existing TensorFlow 1.x Python scripts to TensorFlow 2.x.

The conversion script automates many mechanical API transformations, though many APIs cannot be automatically migrated. It is also not able to fully make your code compatible with TF2 behaviors and APIs. So, it is only a part of your migration journey.
Compatibility modules

Certain API symbols can not be upgraded simply by using a string replacement. Those that cannot be automatically upgraded will be mapped to their locations in the compat.v1 module. This module replaces TF 1.x symbols like tf.foo with the equivalent tf.compat.v1.foo reference. If you are already using compat.v1 APIs by importing TF via import tensorflow.compat.v1 as tf, the tf_upgrade_v2 script will attempt to convert these usages to the non-compat APIs where possible. Note that while some compat.v1 APIs are compatible with TF2.x behaviors, many are not. Therefore, it's recommended to manually proofread replacements and migrate them to new APIs in the tf.* namespace instead of tf.compat.v1 namespace as quickly as possible.

Because of TensorFlow 2.x module deprecations (for example, tf.flags and tf.contrib), some changes can not be worked around by switching to compat.v1. Upgrading this code may require using an additional library (for example, absl.flags) or switching to a package in tensorflow/addons.
Recommended upgrade process

The rest of this guide demonstrates how to use the symbol-rewriting script. While the script is easy to use, it is strongly recommended that you use the script as part of the following process:

    Unit Test: Ensure that the code you’re upgrading has a unit test suite with reasonable coverage. This is Python code, so the language won’t protect you from many classes of mistakes. Also ensure that any dependency you have has already been upgraded to be compatible with TensorFlow 2.x.

    Install TensorFlow 1.15: Upgrade your TensorFlow to the latest TensorFlow 1.x version, at least 1.15. This includes the final TensorFlow 2.0 API in tf.compat.v2.

    Test With 1.15: Ensure your unit tests pass at this point. You’ll be running them repeatedly as you upgrade so starting from green is important.

    Run the upgrade script: Run tf_upgrade_v2 on your entire source tree, tests included. This will upgrade your code to a format where it only uses symbols available in TensorFlow 2.0. Deprecated symbols will be accessed with tf.compat.v1. These will eventually require manual attention, but not immediately.

    Run the converted tests with TensorFlow 1.15: Your code should still run fine in TensorFlow 1.15. Run your unit tests again. Any error in your tests here means there’s a bug in the upgrade script. Please let us know.

    Check the upgrade report for warnings and errors: The script writes a report file that explains any conversions you should double-check, or any manual action you need to take. For example: Any remaining instances of contrib will require manual action to remove. Please consult the RFC for more instructions.

    Install TensorFlow 2.x: At this point it should be safe to switch to TensorFlow 2.x binaries, even if you are running with legacy behaviors

    Test with v1.disable_v2_behavior: Re-running your tests with a v1.disable_v2_behavior() in the tests' main function should give the same results as running under 1.15.

    Enable V2 Behavior: Now that your tests work using the TF2 binaries, you can now begin migrating your code to avoiding tf.estimators and only using supported TF2 behaviors (with no TF2 behavior disabling). See the Migration guides for details.

Using the symbol-rewriting tf_upgrade_v2 script
Setup

Before getting started ensure that TensorFlow 2.x is installed.

import tensorflow as tf

print(tf.__version__)

2024-08-15 01:55:33.141808: E external/local_xla/xla/stream_executor/cuda/cuda_fft.cc:485] Unable to register cuFFT factory: Attempting to register factory for plugin cuFFT when one has already been registered
2024-08-15 01:55:33.163444: E external/local_xla/xla/stream_executor/cuda/cuda_dnn.cc:8454] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered
2024-08-15 01:55:33.169957: E external/local_xla/xla/stream_executor/cuda/cuda_blas.cc:1452] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered
2.17.0

Clone the tensorflow/models git repository so you have some code to test on:

git clone --branch r1.13.0 --depth 1 https://github.com/tensorflow/models

Cloning into 'models'...
remote: Enumerating objects: 2927, done.
remote: Counting objects: 100% (2927/2927), done.
remote: Compressing objects: 100% (2428/2428), done.
remote: Total 2927 (delta 503), reused 2114 (delta 424), pack-reused 0 (from 0)
Receiving objects: 100% (2927/2927), 369.04 MiB | 57.07 MiB/s, done.
Resolving deltas: 100% (503/503), done.
Updating files: 100% (2768/2768), done.

Read the help

The script should be installed with TensorFlow. Here is the builtin help:

tf_upgrade_v2 -h

2024-08-15 01:55:47.051713: E external/local_xla/xla/stream_executor/cuda/cuda_fft.cc:485] Unable to register cuFFT factory: Attempting to register factory for plugin cuFFT when one has already been registered
2024-08-15 01:55:47.070985: E external/local_xla/xla/stream_executor/cuda/cuda_dnn.cc:8454] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered
2024-08-15 01:55:47.076855: E external/local_xla/xla/stream_executor/cuda/cuda_blas.cc:1452] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered
usage: tf_upgrade_v2 [-h] [--infile INPUT_FILE] [--outfile OUTPUT_FILE]
                     [--intree INPUT_TREE] [--outtree OUTPUT_TREE]
                     [--copyotherfiles COPY_OTHER_FILES] [--inplace]
                     [--no_import_rename] [--no_upgrade_compat_v1_import]
                     [--reportfile REPORT_FILENAME] [--mode {DEFAULT,SAFETY}]
                     [--print_all]

Convert a TensorFlow Python file from 1.x to 2.0

Simple usage:
  tf_upgrade_v2.py --infile foo.py --outfile bar.py
  tf_upgrade_v2.py --infile foo.ipynb --outfile bar.ipynb
  tf_upgrade_v2.py --intree ~/code/old --outtree ~/code/new

optional arguments:
  -h, --help            show this help message and exit
  --infile INPUT_FILE   If converting a single file, the name of the file to
                        convert
  --outfile OUTPUT_FILE
                        If converting a single file, the output filename.
  --intree INPUT_TREE   If converting a whole tree of files, the directory to
                        read from (relative or absolute).
  --outtree OUTPUT_TREE
                        If converting a whole tree of files, the output
                        directory (relative or absolute).
  --copyotherfiles COPY_OTHER_FILES
                        If converting a whole tree of files, whether to copy
                        the other files.
  --inplace             If converting a set of files, whether to allow the
                        conversion to be performed on the input files.
  --no_import_rename    Not to rename import to compat.v2 explicitly.
  --no_upgrade_compat_v1_import
                        If specified, don't upgrade explicit imports of
                        `tensorflow.compat.v1 as tf` to the v2 APIs.
                        Otherwise, explicit imports of the form
                        `tensorflow.compat.v1 as tf` will be upgraded.
  --reportfile REPORT_FILENAME
                        The name of the file where the report log is
                        stored.(default: report.txt)
  --mode {DEFAULT,SAFETY}
                        Upgrade script mode. Supported modes: DEFAULT: Perform
                        only straightforward conversions to upgrade to 2.0. In
                        more difficult cases, switch to use compat.v1. SAFETY:
                        Keep 1.* code intact and import compat.v1 module.
  --print_all           Print full log to stdout instead of just printing
                        errors

Example TF1 code

Here is a simple TensorFlow 1.0 script:

head -n 65 models/samples/cookbook/regression/custom_regression.py | tail -n 10

# Calculate loss using mean squared error
  average_loss = tf.losses.mean_squared_error(labels, predictions)

  # Pre-made estimators use the total_loss instead of the average,
  # so report total_loss for compatibility.
  batch_size = tf.shape(labels)[0]
  total_loss = tf.to_float(batch_size) * average_loss

  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = params.get("optimizer", tf.train.AdamOptimizer)

With TensorFlow 2.x installed it does not run:

(cd models/samples/cookbook/regression && python custom_regression.py)

2024-08-15 01:55:50.002184: E external/local_xla/xla/stream_executor/cuda/cuda_fft.cc:485] Unable to register cuFFT factory: Attempting to register factory for plugin cuFFT when one has already been registered
2024-08-15 01:55:50.021138: E external/local_xla/xla/stream_executor/cuda/cuda_dnn.cc:8454] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered
2024-08-15 01:55:50.026917: E external/local_xla/xla/stream_executor/cuda/cuda_blas.cc:1452] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered
Traceback (most recent call last):
  File "/tmpfs/src/temp/site/en/guide/migrate/models/samples/cookbook/regression/custom_regression.py", line 162, in <module>
    tf.logging.set_verbosity(tf.logging.INFO)
AttributeError: module 'tensorflow' has no attribute 'logging'

Single file

The script can be run on a single Python file:

!tf_upgrade_v2 \
  --infile models/samples/cookbook/regression/custom_regression.py \
  --outfile /tmp/custom_regression_v2.py

2024-08-15 01:55:52.838925: E external/local_xla/xla/stream_executor/cuda/cuda_fft.cc:485] Unable to register cuFFT factory: Attempting to register factory for plugin cuFFT when one has already been registered
2024-08-15 01:55:52.859243: E external/local_xla/xla/stream_executor/cuda/cuda_dnn.cc:8454] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered
2024-08-15 01:55:52.865018: E external/local_xla/xla/stream_executor/cuda/cuda_blas.cc:1452] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered
INFO line 38:8: Renamed 'tf.feature_column.input_layer' to 'tf.compat.v1.feature_column.input_layer'
INFO line 57:17: tf.losses.mean_squared_error requires manual check. tf.losses have been replaced with object oriented versions in TF 2.0 and after. The loss function calls have been converted to compat.v1 for backward compatibility. Please update these calls to the TF 2.0 versions.
INFO line 57:17: Renamed 'tf.losses.mean_squared_error' to 'tf.compat.v1.losses.mean_squared_error'
INFO line 62:15: Changed tf.to_float call to tf.cast(..., dtype=tf.float32).
INFO line 65:40: Renamed 'tf.train.AdamOptimizer' to 'tf.compat.v1.train.AdamOptimizer'
INFO line 68:39: Renamed 'tf.train.get_global_step' to 'tf.compat.v1.train.get_global_step'
INFO line 83:9: tf.metrics.root_mean_squared_error requires manual check. tf.metrics have been replaced with object oriented versions in TF 2.0 and after. The metric function calls have been converted to compat.v1 for backward compatibility. Please update these calls to the TF 2.0 versions.
INFO line 83:9: Renamed 'tf.metrics.root_mean_squared_error' to 'tf.compat.v1.metrics.root_mean_squared_error'
INFO line 142:23: Renamed 'tf.train.AdamOptimizer' to 'tf.compat.v1.train.AdamOptimizer'
INFO line 162:2: Renamed 'tf.logging.set_verbosity' to 'tf.compat.v1.logging.set_verbosity'
INFO line 162:27: Renamed 'tf.logging.INFO' to 'tf.compat.v1.logging.INFO'
INFO line 163:2: Renamed 'tf.app.run' to 'tf.compat.v1.app.run'
TensorFlow 2.0 Upgrade Script
-----------------------------
Converted 1 files
Detected 0 issues that require attention
--------------------------------------------------------------------------------


Make sure to read the detailed log 'report.txt'

The script will print errors if it can not find a fix for the code.
Directory tree

Typical projects, including this simple example, will use much more than one file. Typically want to update an entire package, so the script can also be run on a directory tree:

# update the .py files and copy all the other files to the outtree
!tf_upgrade_v2 \
    --intree models/samples/cookbook/regression/ \
    --outtree regression_v2/ \
    --reportfile tree_report.txt

2024-08-15 01:55:55.699073: E external/local_xla/xla/stream_executor/cuda/cuda_fft.cc:485] Unable to register cuFFT factory: Attempting to register factory for plugin cuFFT when one has already been registered
2024-08-15 01:55:55.718072: E external/local_xla/xla/stream_executor/cuda/cuda_dnn.cc:8454] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered
2024-08-15 01:55:55.723647: E external/local_xla/xla/stream_executor/cuda/cuda_blas.cc:1452] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered
INFO line 40:7: Renamed 'tf.test.mock' to 'tf.compat.v1.test.mock'
WARNING line 125:15: Changing dataset.make_one_shot_iterator() to tf.compat.v1.data.make_one_shot_iterator(dataset). Please check this transformation.

INFO line 96:2: Renamed 'tf.logging.set_verbosity' to 'tf.compat.v1.logging.set_verbosity'
INFO line 96:27: Renamed 'tf.logging.INFO' to 'tf.compat.v1.logging.INFO'
INFO line 97:2: Renamed 'tf.app.run' to 'tf.compat.v1.app.run'
INFO line 101:2: Renamed 'tf.logging.set_verbosity' to 'tf.compat.v1.logging.set_verbosity'
INFO line 101:27: Renamed 'tf.logging.INFO' to 'tf.compat.v1.logging.INFO'
INFO line 102:2: Renamed 'tf.app.run' to 'tf.compat.v1.app.run'
INFO line 105:2: Renamed 'tf.logging.set_verbosity' to 'tf.compat.v1.logging.set_verbosity'
INFO line 105:27: Renamed 'tf.logging.INFO' to 'tf.compat.v1.logging.INFO'
INFO line 106:2: Renamed 'tf.app.run' to 'tf.compat.v1.app.run'
INFO line 38:8: Renamed 'tf.feature_column.input_layer' to 'tf.compat.v1.feature_column.input_layer'
INFO line 57:17: tf.losses.mean_squared_error requires manual check. tf.losses have been replaced with object oriented versions in TF 2.0 and after. The loss function calls have been converted to compat.v1 for backward compatibility. Please update these calls to the TF 2.0 versions.
INFO line 57:17: Renamed 'tf.losses.mean_squared_error' to 'tf.compat.v1.losses.mean_squared_error'
INFO line 62:15: Changed tf.to_float call to tf.cast(..., dtype=tf.float32).
INFO line 65:40: Renamed 'tf.train.AdamOptimizer' to 'tf.compat.v1.train.AdamOptimizer'
INFO line 68:39: Renamed 'tf.train.get_global_step' to 'tf.compat.v1.train.get_global_step'
INFO line 83:9: tf.metrics.root_mean_squared_error requires manual check. tf.metrics have been replaced with object oriented versions in TF 2.0 and after. The metric function calls have been converted to compat.v1 for backward compatibility. Please update these calls to the TF 2.0 versions.
INFO line 83:9: Renamed 'tf.metrics.root_mean_squared_error' to 'tf.compat.v1.metrics.root_mean_squared_error'
INFO line 142:23: Renamed 'tf.train.AdamOptimizer' to 'tf.compat.v1.train.AdamOptimizer'
INFO line 162:2: Renamed 'tf.logging.set_verbosity' to 'tf.compat.v1.logging.set_verbosity'
INFO line 162:27: Renamed 'tf.logging.INFO' to 'tf.compat.v1.logging.INFO'
INFO line 163:2: Renamed 'tf.app.run' to 'tf.compat.v1.app.run'
TensorFlow 2.0 Upgrade Script
-----------------------------
Converted 7 files
Detected 1 issues that require attention
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
File: models/samples/cookbook/regression/automobile_data.py
--------------------------------------------------------------------------------
models/samples/cookbook/regression/automobile_data.py:125:15: WARNING: Changing dataset.make_one_shot_iterator() to tf.compat.v1.data.make_one_shot_iterator(dataset). Please check this transformation.



Make sure to read the detailed log 'tree_report.txt'

Note the one warning about the dataset.make_one_shot_iterator function.

Now the script works in with TensorFlow 2.x:

Note that because the tf.compat.v1 module is included in TF 1.15, the converted script will also run in TensorFlow 1.15.

(cd regression_v2 && python custom_regression.py 2>&1) | tail

tf.compat.v1.app.run(main=main)
  File "/tmpfs/src/tf_docs_env/lib/python3.9/site-packages/tensorflow/python/platform/app.py", line 36, in run
    _run(main=main, argv=argv, flags_parser=_parse_flags_tolerate_undef)
  File "/tmpfs/src/tf_docs_env/lib/python3.9/site-packages/absl/app.py", line 308, in run
    _run_main(main, args)
  File "/tmpfs/src/tf_docs_env/lib/python3.9/site-packages/absl/app.py", line 254, in _run_main
    sys.exit(main(argv))
  File "/tmpfs/src/temp/site/en/guide/migrate/regression_v2/custom_regression.py", line 137, in main
    model = tf.estimator.Estimator(
AttributeError: module 'tensorflow' has no attribute 'estimator'

Detailed report

The script also reports a list of detailed changes. In this example it found one possibly unsafe transformation and included a warning at the top of the file:

head -n 20 tree_report.txt

TensorFlow 2.0 Upgrade Script
-----------------------------
Converted 7 files
Detected 1 issues that require attention
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
File: models/samples/cookbook/regression/automobile_data.py
--------------------------------------------------------------------------------
models/samples/cookbook/regression/automobile_data.py:125:15: WARNING: Changing dataset.make_one_shot_iterator() to tf.compat.v1.data.make_one_shot_iterator(dataset). Please check this transformation.

================================================================================
Detailed log follows:

================================================================================
================================================================================
Input tree: 'models/samples/cookbook/regression/'
================================================================================
--------------------------------------------------------------------------------
Processing file 'models/samples/cookbook/regression/regression_test.py'
 outputting to 'regression_v2/regression_test.py'

Note again the one warning about the Dataset.make_one_shot_iterator function.

In other cases the output will explain the reasoning for non-trivial changes:

%%writefile dropout.py
import tensorflow as tf

d = tf.nn.dropout(tf.range(10), 0.2)
z = tf.zeros_like(d, optimize=False)

Writing dropout.py

!tf_upgrade_v2 \
  --infile dropout.py \
  --outfile dropout_v2.py \
  --reportfile dropout_report.txt > /dev/null

2024-08-15 01:56:02.740029: E external/local_xla/xla/stream_executor/cuda/cuda_fft.cc:485] Unable to register cuFFT factory: Attempting to register factory for plugin cuFFT when one has already been registered
2024-08-15 01:56:02.758969: E external/local_xla/xla/stream_executor/cuda/cuda_dnn.cc:8454] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered
2024-08-15 01:56:02.764604: E external/local_xla/xla/stream_executor/cuda/cuda_blas.cc:1452] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered

cat dropout_report.txt

TensorFlow 2.0 Upgrade Script
-----------------------------
Converted 1 files
Detected 0 issues that require attention
--------------------------------------------------------------------------------
================================================================================
Detailed log follows:

================================================================================
--------------------------------------------------------------------------------
Processing file 'dropout.py'
 outputting to 'dropout_v2.py'
--------------------------------------------------------------------------------

3:4: INFO: Changing keep_prob arg of tf.nn.dropout to rate, and recomputing value.

4:4: INFO: Renaming tf.zeros_like to tf.compat.v1.zeros_like because argument optimize is present. tf.zeros_like no longer takes an optimize argument, and behaves as if optimize=True. This call site specifies something other than optimize=True, so it was converted to compat.v1.
--------------------------------------------------------------------------------

Here is the modified file contents, note how the script adds argument names to deal with moved and renamed arguments:

cat dropout_v2.py

import tensorflow as tf

d = tf.nn.dropout(tf.range(10), rate=1 - (0.2))
z = tf.compat.v1.zeros_like(d, optimize=False)

A larger project might contain a few errors. For example convert the deeplab model:

!tf_upgrade_v2 \
    --intree models/research/deeplab \
    --outtree deeplab_v2 \
    --reportfile deeplab_report.txt > /dev/null

2024-08-15 01:56:05.822940: E external/local_xla/xla/stream_executor/cuda/cuda_fft.cc:485] Unable to register cuFFT factory: Attempting to register factory for plugin cuFFT when one has already been registered
2024-08-15 01:56:05.841928: E external/local_xla/xla/stream_executor/cuda/cuda_dnn.cc:8454] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered
2024-08-15 01:56:05.847711: E external/local_xla/xla/stream_executor/cuda/cuda_blas.cc:1452] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered

It produced the output files:

ls deeplab_v2

README.md   datasets        input_preprocess.py        train.py
__init__.py deeplab_demo.ipynb  local_test.sh          utils
common.py   eval.py         local_test_mobilenetv2.sh  vis.py
common_test.py  export_model.py     model.py
core        g3doc           model_test.py

But there were errors. The report will help you pin-point what you need to fix before this will run. Here are the first three errors:

cat deeplab_report.txt | grep -i models/research/deeplab | grep -i error | head -n 3

models/research/deeplab/vis.py:31:7: ERROR: Using member tf.contrib.slim in deprecated module tf.contrib. tf.contrib.slim cannot be converted automatically. tf.contrib will not be distributed with TensorFlow 2.0, please consider an alternative in non-contrib TensorFlow, a community-maintained repository such as tensorflow/addons, or fork the required code.
models/research/deeplab/eval.py:28:7: ERROR: Using member tf.contrib.slim in deprecated module tf.contrib. tf.contrib.slim cannot be converted automatically. tf.contrib will not be distributed with TensorFlow 2.0, please consider an alternative in non-contrib TensorFlow, a community-maintained repository such as tensorflow/addons, or fork the required code.
models/research/deeplab/eval.py:146:8: ERROR: Using member tf.contrib.metrics.aggregate_metric_map in deprecated module tf.contrib. tf.contrib.metrics.aggregate_metric_map cannot be converted automatically. tf.contrib will not be distributed with TensorFlow 2.0, please consider an alternative in non-contrib TensorFlow, a community-maintained repository such as tensorflow/addons, or fork the required code.

"Safety" mode

The conversion script also has a less invasive SAFETY mode that simply changes the imports to use the tensorflow.compat.v1 module:

cat dropout.py

import tensorflow as tf

d = tf.nn.dropout(tf.range(10), 0.2)
z = tf.zeros_like(d, optimize=False)

tf_upgrade_v2 --mode SAFETY --infile dropout.py --outfile dropout_v2_safe.py > /dev/null

2024-08-15 01:56:10.510235: E external/local_xla/xla/stream_executor/cuda/cuda_fft.cc:485] Unable to register cuFFT factory: Attempting to register factory for plugin cuFFT when one has already been registered
2024-08-15 01:56:10.529243: E external/local_xla/xla/stream_executor/cuda/cuda_dnn.cc:8454] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered
2024-08-15 01:56:10.534850: E external/local_xla/xla/stream_executor/cuda/cuda_blas.cc:1452] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered

cat dropout_v2_safe.py

import tensorflow.compat.v1 as tf

d = tf.nn.dropout(tf.range(10), 0.2)
z = tf.zeros_like(d, optimize=False)

As you can see this doesn't upgrade your code, but does allow TensorFlow 1 code to run against TensorFlow 2 binaries. Note that this does not mean your code is running supported TF 2.x behaviors!
Caveats

    Do not update parts of your code manually before running this script. In particular, functions that have had reordered arguments like tf.math.argmax or tf.batch_to_space cause the script to incorrectly add keyword arguments that mismap your existing code.

    The script assumes that tensorflow is imported using import tensorflow as tf, or import tensorflow.compat.v1 as tf.

    This script does not reorder arguments. Instead, the script adds keyword arguments to functions that have their arguments reordered.

    Check out tf2up.ml for a convenient tool to upgrade Jupyter notebooks and Python files in a GitHub repository.

This guide provides an overview and examples of a modeling code shim that you can employ to use your existing TF1.x models in TF2 workflows such as eager execution, tf.function, and distribution strategies with minimal changes to your modeling code.
Scope of usage

The shim described in this guide is designed for TF1.x models that rely on:

    tf.compat.v1.get_variable and tf.compat.v1.variable_scope to control variable creation and reuse, and
    Graph-collection based APIs such as tf.compat.v1.global_variables(), tf.compat.v1.trainable_variables, tf.compat.v1.losses.get_regularization_losses(), and tf.compat.v1.get_collection() to keep track of weights and regularization losses

This includes most models built on top of tf.compat.v1.layer, tf.contrib.layers APIs, and TensorFlow-Slim.

The shim is NOT necessary for the following TF1.x models:

    Stand-alone Keras models that already track all of their trainable weights and regularization losses via model.trainable_weights and model.losses respectively.
    tf.Modules that already track all of their trainable weights via module.trainable_variables, and only create weights if they have not already been created.

These models are likely to work in TF2 with eager execution and tf.functions out-of-the-box.
Setup

Import TensorFlow and other dependencies.

pip uninstall -y -q tensorflow

# Install tf-nightly as the DeterministicRandomTestTool is available only in
# Tensorflow 2.8

pip install -q tf-nightly

import tensorflow as tf
import tensorflow.compat.v1 as v1
import sys
import numpy as np

from contextlib import contextmanager

The track_tf1_style_variables decorator

The key shim described in this guide is tf.compat.v1.keras.utils.track_tf1_style_variables, a decorator that you can use within methods belonging to tf.keras.layers.Layer and tf.Module to track TF1.x-style weights and capture regularization losses.

Decorating a tf.keras.layers.Layer's or tf.Module's call methods with tf.compat.v1.keras.utils.track_tf1_style_variables allows variable creation and reuse via tf.compat.v1.get_variable (and by extension tf.compat.v1.layers) to work correctly inside of the decorated method rather than always creating a new variable on each call. It will also cause the layer or module to implicitly track any weights created or accessed via get_variable inside the decorated method.

In addition to tracking the weights themselves under the standard layer.variable/module.variable/etc. properties, if the method belongs to a tf.keras.layers.Layer, then any regularization losses specified via the get_variable or tf.compat.v1.layers regularizer arguments will get tracked by the layer under the standard layer.losses property.

This tracking mechanism enables using large classes of TF1.x-style model-forward-pass code inside of Keras layers or tf.Modules in TF2 even with TF2 behaviors enabled.
Usage examples

The usage examples below demonstrate the modeling shims used to decorate tf.keras.layers.Layer methods, but except where they are specifically interacting with Keras features they are applicable when decorating tf.Module methods as well.
Layer built with tf.compat.v1.get_variable

Imagine you have a layer implemented directly on top of tf.compat.v1.get_variable as follows:

def dense(self, inputs, units):
  out = inputs
  with tf.compat.v1.variable_scope("dense"):
    # The weights are created with a `regularizer`,
    kernel = tf.compat.v1.get_variable(
        shape=[out.shape[-1], units],
        regularizer=tf.keras.regularizers.L2(),
        initializer=tf.compat.v1.initializers.glorot_normal,
        name="kernel")
    bias = tf.compat.v1.get_variable(
        shape=[units,],
        initializer=tf.compat.v1.initializers.zeros,
        name="bias")
    out = tf.linalg.matmul(out, kernel)
    out = tf.compat.v1.nn.bias_add(out, bias)
  return out

Use the shim to turn it into a layer and call it on inputs.

class DenseLayer(tf.keras.layers.Layer):

  def __init__(self, units, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.units = units

  @tf.compat.v1.keras.utils.track_tf1_style_variables
  def call(self, inputs):
    out = inputs
    with tf.compat.v1.variable_scope("dense"):
      # The weights are created with a `regularizer`,
      # so the layer should track their regularization losses
      kernel = tf.compat.v1.get_variable(
          shape=[out.shape[-1], self.units],
          regularizer=tf.keras.regularizers.L2(),
          initializer=tf.compat.v1.initializers.glorot_normal,
          name="kernel")
      bias = tf.compat.v1.get_variable(
          shape=[self.units,],
          initializer=tf.compat.v1.initializers.zeros,
          name="bias")
      out = tf.linalg.matmul(out, kernel)
      out = tf.compat.v1.nn.bias_add(out, bias)
    return out

layer = DenseLayer(10)
x = tf.random.normal(shape=(8, 20))
layer(x)

Access the tracked variables and the captured regularization losses like a standard Keras layer.

layer.trainable_variables
layer.losses

To see that the weights get reused each time you call the layer, set all the weights to zero and call the layer again.

print("Resetting variables to zero:", [var.name for var in layer.trainable_variables])

for var in layer.trainable_variables:
  var.assign(var * 0.0)

# Note: layer.losses is not a live view and
# will get reset only at each layer call
print("layer.losses:", layer.losses)
print("calling layer again.")
out = layer(x)
print("layer.losses: ", layer.losses)
out

You can use the converted layer directly in Keras functional model construction as well.

inputs = tf.keras.Input(shape=(20))
outputs = DenseLayer(10)(inputs)
model = tf.keras.Model(inputs=inputs, outputs=outputs)

x = tf.random.normal(shape=(8, 20))
model(x)

# Access the model variables and regularization losses
model.weights
model.losses

Model built with tf.compat.v1.layers

Imagine you have a layer or model implemented directly on top of tf.compat.v1.layers as follows:

def model(self, inputs, units):
  with tf.compat.v1.variable_scope('model'):
    out = tf.compat.v1.layers.conv2d(
        inputs, 3, 3,
        kernel_regularizer="l2")
    out = tf.compat.v1.layers.flatten(out)
    out = tf.compat.v1.layers.dense(
        out, units,
        kernel_regularizer="l2")
    return out

Use the shim to turn it into a layer and call it on inputs.

class CompatV1LayerModel(tf.keras.layers.Layer):

  def __init__(self, units, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.units = units

  @tf.compat.v1.keras.utils.track_tf1_style_variables
  def call(self, inputs):
    with tf.compat.v1.variable_scope('model'):
      out = tf.compat.v1.layers.conv2d(
          inputs, 3, 3,
          kernel_regularizer="l2")
      out = tf.compat.v1.layers.flatten(out)
      out = tf.compat.v1.layers.dense(
          out, self.units,
          kernel_regularizer="l2")
      return out

layer = CompatV1LayerModel(10)
x = tf.random.normal(shape=(8, 5, 5, 5))
layer(x)

Warning: For safety reasons, make sure to put all tf.compat.v1.layers inside of a non-empty-string variable_scope. This is because tf.compat.v1.layers with auto-generated names will always auto-increment the name outside of any variable scope. This means the requested variable names will mismatch each time you call the layer/module. So, rather than reusing the already-made weights it will create a new set of variables every call.

Access the tracked variables and captured regularization losses like a standard Keras layer.

layer.trainable_variables
layer.losses

To see that the weights get reused each time you call the layer, set all the weights to zero and call the layer again.

print("Resetting variables to zero:", [var.name for var in layer.trainable_variables])

for var in layer.trainable_variables:
  var.assign(var * 0.0)

out = layer(x)
print("layer.losses: ", layer.losses)
out

You can use the converted layer directly in Keras functional model construction as well.

inputs = tf.keras.Input(shape=(5, 5, 5))
outputs = CompatV1LayerModel(10)(inputs)
model = tf.keras.Model(inputs=inputs, outputs=outputs)

x = tf.random.normal(shape=(8, 5, 5, 5))
model(x)

# Access the model variables and regularization losses
model.weights
model.losses

Capture batch normalization updates and model training args

In TF1.x, you perform batch normalization like this:

  x_norm = tf.compat.v1.layers.batch_normalization(x, training=training)

  # ...

  update_ops = tf.compat.v1.get_collection(tf.GraphKeys.UPDATE_OPS)
  train_op = optimizer.minimize(loss)
  train_op = tf.group([train_op, update_ops])

Note that:

    The batch normalization moving average updates are tracked by get_collection which was called separately from the layer
    tf.compat.v1.layers.batch_normalization requires a training argument (generally called is_training when using TF-Slim batch normalization layers)

In TF2, due to eager execution and automatic control dependencies, the batch normalization moving average updates will be executed right away. There is no need to separately collect them from the updates collection and add them as explicit control dependencies.

Additionally, if you give your tf.keras.layers.Layer's forward pass method a training argument, Keras will be able to pass the current training phase and any nested layers to it just like it does for any other layer. See the API docs for tf.keras.Model for more information on how Keras handles the training argument.

If you are decorating tf.Module methods, you need to make sure to manually pass all training arguments as needed. However, the batch normalization moving average updates will still be applied automatically with no need for explicit control dependencies.

The following code snippets demonstrate how to embed batch normalization layers in the shim and how using it in a Keras model works (applicable to tf.keras.layers.Layer).

class CompatV1BatchNorm(tf.keras.layers.Layer):

  @tf.compat.v1.keras.utils.track_tf1_style_variables
  def call(self, inputs, training=None):
    print("Forward pass called with `training` =", training)
    with v1.variable_scope('batch_norm_layer'):
      return v1.layers.batch_normalization(x, training=training)

print("Constructing model")
inputs = tf.keras.Input(shape=(5, 5, 5))
outputs = CompatV1BatchNorm()(inputs)
model = tf.keras.Model(inputs=inputs, outputs=outputs)

print("Calling model in inference mode")
x = tf.random.normal(shape=(8, 5, 5, 5))
model(x, training=False)

print("Moving average variables before training: ",
      {var.name: var.read_value() for var in model.non_trainable_variables})

# Notice that when running TF2 and eager execution, the batchnorm layer directly
# updates the moving averages while training without needing any extra control
# dependencies
print("calling model in training mode")
model(x, training=True)

print("Moving average variables after training: ",
      {var.name: var.read_value() for var in model.non_trainable_variables})

Variable-scope based variable reuse

Any variable creations in the forward pass based on get_variable will maintain the same variable naming and reuse semantics that variable scopes have in TF1.x. This is true as long as you have at least one non-empty outer scope for any tf.compat.v1.layers with auto-generated names, as mentioned above.
Note: Naming and reuse will be scoped to within a single layer/module instance. Calls to get_variable inside one shim-decorated layer or module will not be able to refer to variables created inside of layers or modules. You can get around this by using Python references to other variables directly if need be, rather than accessing variables via get_variable.
Eager execution & tf.function

As seen above, decorated methods for tf.keras.layers.Layer and tf.Module run inside of eager execution and are also compatible with tf.function. This means you can use pdb and other interactive tools to step through your forward pass as it is running.
Warning: Although it is perfectly safe to call your shim-decorated layer/module methods from inside of a tf.function, it is not safe to put tf.functions inside of your shim-decorated methods if those tf.functions contain get_variable calls. Entering a tf.function resets variable_scopes, which means the TF1.x-style variable-scope-based variable reuse that the shim mimics will break down in this setting.
Distribution strategies

Calls to get_variable inside of @track_tf1_style_variables-decorated layer or module methods use standard tf.Variable variable creations under the hood. This means you can use them with the various distribution strategies available with tf.distribute such as MirroredStrategy and TPUStrategy.
Nesting tf.Variables, tf.Modules, tf.keras.layers & tf.keras.models in decorated calls

Decorating your layer call in tf.compat.v1.keras.utils.track_tf1_style_variables will only add automatic implicit tracking of variables created (and reused) via tf.compat.v1.get_variable. It will not capture weights directly created by tf.Variable calls, such as those used by typical Keras layers and most tf.Modules. This section describes how to handle these nested cases.
(Pre-existing usages) tf.keras.layers and tf.keras.models

For pre-existing usages of nested Keras layers and models, use tf.compat.v1.keras.utils.get_or_create_layer. This is only recommended for easing migration of existing TF1.x nested Keras usages; new code should use explicit attribute setting as described below for tf.Variables and tf.Modules.

To use tf.compat.v1.keras.utils.get_or_create_layer, wrap the code that constructs your nested model into a method, and pass it in to the method. Example:

class NestedModel(tf.keras.Model):

  def __init__(self, units, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.units = units

  def build_model(self):
    inp = tf.keras.Input(shape=(5, 5))
    dense_layer = tf.keras.layers.Dense(
        10, name="dense", kernel_regularizer="l2",
        kernel_initializer=tf.compat.v1.ones_initializer())
    model = tf.keras.Model(inputs=inp, outputs=dense_layer(inp))
    return model

  @tf.compat.v1.keras.utils.track_tf1_style_variables
  def call(self, inputs):
    # Get or create a nested model without assigning it as an explicit property
    model = tf.compat.v1.keras.utils.get_or_create_layer(
        "dense_model", self.build_model)
    return model(inputs)

layer = NestedModel(10)
layer(tf.ones(shape=(5,5)))

This method ensures that these nested layers are correctly reused and tracked by tensorflow. Note that the @track_tf1_style_variables decorator is still required on the appropriate method. The model builder method passed into get_or_create_layer (in this case, self.build_model), should take no arguments.

Weights are tracked:

assert len(layer.weights) == 2
weights = {x.name: x for x in layer.variables}

assert set(weights.keys()) == {"dense/bias:0", "dense/kernel:0"}

layer.weights

And regularization loss as well:

tf.add_n(layer.losses)

Incremental migration: tf.Variables and tf.Modules

If you need to embed tf.Variable calls or tf.Modules in your decorated methods (for example, if you are following the incremental migration to non-legacy TF2 APIs described later in this guide), you still need to explicitly track these, with the following requirements:

    Explicitly make sure that the variable/module/layer is only created once
    Explicitly attach them as instance attributes just as you would when defining a typical module or layer
    Explicitly reuse the already-created object in follow-on calls

This ensures that weights are not created new each call and are correctly reused. Additionally, this also ensures that existing weights and regularization losses get tracked.

Here is an example of how this could look:

class NestedLayer(tf.keras.layers.Layer):

  def __init__(self, units, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.units = units

  @tf.compat.v1.keras.utils.track_tf1_style_variables
  def __call__(self, inputs):
    out = inputs
    with tf.compat.v1.variable_scope("inner_dense"):
      # The weights are created with a `regularizer`,
      # so the layer should track their regularization losses
      kernel = tf.compat.v1.get_variable(
          shape=[out.shape[-1], self.units],
          regularizer=tf.keras.regularizers.L2(),
          initializer=tf.compat.v1.initializers.glorot_normal,
          name="kernel")
      bias = tf.compat.v1.get_variable(
          shape=[self.units,],
          initializer=tf.compat.v1.initializers.zeros,
          name="bias")
      out = tf.linalg.matmul(out, kernel)
      out = tf.compat.v1.nn.bias_add(out, bias)
    return out

class WrappedDenseLayer(tf.keras.layers.Layer):

  def __init__(self, units, **kwargs):
    super().__init__(**kwargs)
    self.units = units
    # Only create the nested tf.variable/module/layer/model
    # once, and then reuse it each time!
    self._dense_layer = NestedLayer(self.units)

  @tf.compat.v1.keras.utils.track_tf1_style_variables
  def call(self, inputs):
    with tf.compat.v1.variable_scope('outer'):
      outputs = tf.compat.v1.layers.dense(inputs, 3)
      outputs = tf.compat.v1.layers.dense(inputs, 4)
      return self._dense_layer(outputs)

layer = WrappedDenseLayer(10)

layer(tf.ones(shape=(5, 5)))

Note that explicit tracking of the nested module is needed even though it is decorated with the track_tf1_style_variables decorator. This is because each module/layer with decorated methods has its own variable store associated with it.

The weights are correctly tracked:

assert len(layer.weights) == 6
weights = {x.name: x for x in layer.variables}

assert set(weights.keys()) == {"outer/inner_dense/bias:0",
                               "outer/inner_dense/kernel:0",
                               "outer/dense/bias:0",
                               "outer/dense/kernel:0",
                               "outer/dense_1/bias:0",
                               "outer/dense_1/kernel:0"}

layer.trainable_weights

As well as regularization loss:

layer.losses

Note that if the NestedLayer were a non-Keras tf.Module instead, variables would still be tracked but regularization losses would not be automatically tracked, so you would have to explicitly track them separately.
Guidance on variable names

Explicit tf.Variable calls and Keras layers use a different layer name / variable name autogeneration mechanism than you may be used to from the combination of get_variable and variable_scopes. Although the shim will make your variable names match for variables created by get_variable even when going from TF1.x graphs to TF2 eager execution & tf.function, it cannot guarantee the same for the variable names generated for tf.Variable calls and Keras layers that you embed within your method decorators. It is even possible for multiple variables to share the same name in TF2 eager execution and tf.function.

You should take special care with this when following the sections on validating correctness and mapping TF1.x checkpoints later on in this guide.
Using tf.compat.v1.make_template in the decorated method

It is highly recommended you directly use tf.compat.v1.keras.utils.track_tf1_style_variables instead of using tf.compat.v1.make_template, as it is a thinner layer on top of TF2.

Follow the guidance in this section for prior TF1.x code that was already relying on tf.compat.v1.make_template.

Because tf.compat.v1.make_template wraps code that uses get_variable, the track_tf1_style_variables decorator allows you to use these templates in layer calls and successfully track the weights and regularization losses.

However, do make sure to call make_template only once and then reuse the same template in each layer call. Otherwise, a new template will be created each time you call the layer along with a new set of variables.

For example,

class CompatV1TemplateScaleByY(tf.keras.layers.Layer):

  def __init__(self, **kwargs):
    super().__init__(**kwargs)
    def my_op(x, scalar_name):
      var1 = tf.compat.v1.get_variable(scalar_name,
                            shape=[],
                            regularizer=tf.compat.v1.keras.regularizers.L2(),
                            initializer=tf.compat.v1.constant_initializer(1.5))
      return x * var1
    self.scale_by_y = tf.compat.v1.make_template('scale_by_y', my_op, scalar_name='y')

  @tf.compat.v1.keras.utils.track_tf1_style_variables
  def call(self, inputs):
    with tf.compat.v1.variable_scope('layer'):
      # Using a scope ensures the `scale_by_y` name will not be incremented
      # for each instantiation of the layer.
      return self.scale_by_y(inputs)

layer = CompatV1TemplateScaleByY()

out = layer(tf.ones(shape=(2, 3)))
print("weights:", layer.weights)
print("regularization loss:", layer.losses)
print("output:", out)

Warning: Avoid sharing the same make_template-created template across multiple layer instances as it may break the variable and regularization loss tracking mechanisms of the shim decorator. Additionally, if you plan to use the same make_template name inside of multiple layer instances then you should nest the created template's usage inside of a variable_scope. If not, the generated name for the template's variable_scope will increment with each new instance of the layer. This could alter the weight names in unexpected ways.
Incremental migration to Native TF2

As mentioned earlier, track_tf1_style_variables allows you to mix TF2-style object-oriented tf.Variable/tf.keras.layers.Layer/tf.Module usage with legacy tf.compat.v1.get_variable/tf.compat.v1.layers-style usage inside of the same decorated module/layer.

This means that after you have made your TF1.x model fully-TF2-compatible, you can write all new model components with native (non-tf.compat.v1) TF2 APIs and have them interoperate with your older code.

However, if you continue to modify your older model components, you may also choose to incrementally switch your legacy-style tf.compat.v1 usage over to the purely-native object-oriented APIs that are recommended for newly written TF2 code.

tf.compat.v1.get_variable usage can be replaced with either self.add_weight calls if you are decorating a Keras layer/model, or with tf.Variable calls if you are decorating Keras objects or tf.Modules.

Both functional-style and object-oriented tf.compat.v1.layers can generally be replaced with the equivalent tf.keras.layers layer with no argument changes required.

You may also consider chunks parts of your model or common patterns into individual layers/modules during your incremental move to purely-native APIs, which may themselves use track_tf1_style_variables.
A note on Slim and contrib.layers

A large amount of older TF 1.x code uses the Slim library, which was packaged with TF 1.x as tf.contrib.layers. Converting code using Slim to native TF 2 is more involved than converting v1.layers. In fact, it may make sense to convert your Slim code to v1.layers first, then convert to Keras. Below is some general guidance for converting Slim code.

    Ensure all arguments are explicit. Remove arg_scopes if possible. If you still need to use them, split normalizer_fn and activation_fn into their own layers.
    Separable conv layers map to one or more different Keras layers (depthwise, pointwise, and separable Keras layers).
    Slim and v1.layers have different argument names and default values.
    Note that some arguments have different scales.

Migration to Native TF2 ignoring checkpoint compatibility

The following code sample demonstrates an incremental move of a model to purely-native APIs without considering checkpoint compatibility.

class CompatModel(tf.keras.layers.Layer):

  def __init__(self, units, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.units = units

  @tf.compat.v1.keras.utils.track_tf1_style_variables
  def call(self, inputs, training=None):
    with tf.compat.v1.variable_scope('model'):
      out = tf.compat.v1.layers.conv2d(
          inputs, 3, 3,
          kernel_regularizer="l2")
      out = tf.compat.v1.layers.flatten(out)
      out = tf.compat.v1.layers.dropout(out, training=training)
      out = tf.compat.v1.layers.dense(
          out, self.units,
          kernel_regularizer="l2")
      return out

Next, replace the compat.v1 APIs with their native object-oriented equivalents in a piecewise manner. Start by switching the convolution layer to a Keras object created in the layer constructor.

class PartiallyMigratedModel(tf.keras.layers.Layer):

  def __init__(self, units, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.units = units
    self.conv_layer = tf.keras.layers.Conv2D(
      3, 3,
      kernel_regularizer="l2")

  @tf.compat.v1.keras.utils.track_tf1_style_variables
  def call(self, inputs, training=None):
    with tf.compat.v1.variable_scope('model'):
      out = self.conv_layer(inputs)
      out = tf.compat.v1.layers.flatten(out)
      out = tf.compat.v1.layers.dropout(out, training=training)
      out = tf.compat.v1.layers.dense(
          out, self.units,
          kernel_regularizer="l2")
      return out

Use the v1.keras.utils.DeterministicRandomTestTool class to verify that this incremental change leaves the model with the same behavior as before.

random_tool = v1.keras.utils.DeterministicRandomTestTool(mode='num_random_ops')
with random_tool.scope():
  tf.keras.utils.set_random_seed(42)
  layer = CompatModel(10)

  inputs = tf.random.normal(shape=(10, 5, 5, 5))
  original_output = layer(inputs)

  # Grab the regularization loss as well
  original_regularization_loss = tf.math.add_n(layer.losses)

print(original_regularization_loss)

random_tool = v1.keras.utils.DeterministicRandomTestTool(mode='num_random_ops')
with random_tool.scope():
  tf.keras.utils.set_random_seed(42)
  layer = PartiallyMigratedModel(10)

  inputs = tf.random.normal(shape=(10, 5, 5, 5))
  migrated_output = layer(inputs)

  # Grab the regularization loss as well
  migrated_regularization_loss = tf.math.add_n(layer.losses)

print(migrated_regularization_loss)

# Verify that the regularization loss and output both match
np.testing.assert_allclose(original_regularization_loss.numpy(), migrated_regularization_loss.numpy())
np.testing.assert_allclose(original_output.numpy(), migrated_output.numpy())

You have now replaced all of the individual compat.v1.layers with native Keras layers.

class NearlyFullyNativeModel(tf.keras.layers.Layer):

  def __init__(self, units, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.units = units
    self.conv_layer = tf.keras.layers.Conv2D(
      3, 3,
      kernel_regularizer="l2")
    self.flatten_layer = tf.keras.layers.Flatten()
    self.dense_layer = tf.keras.layers.Dense(
      self.units,
      kernel_regularizer="l2")

  @tf.compat.v1.keras.utils.track_tf1_style_variables
  def call(self, inputs):
    with tf.compat.v1.variable_scope('model'):
      out = self.conv_layer(inputs)
      out = self.flatten_layer(out)
      out = self.dense_layer(out)
      return out

random_tool = v1.keras.utils.DeterministicRandomTestTool(mode='num_random_ops')
with random_tool.scope():
  tf.keras.utils.set_random_seed(42)
  layer = NearlyFullyNativeModel(10)

  inputs = tf.random.normal(shape=(10, 5, 5, 5))
  migrated_output = layer(inputs)

  # Grab the regularization loss as well
  migrated_regularization_loss = tf.math.add_n(layer.losses)

print(migrated_regularization_loss)

# Verify that the regularization loss and output both match
np.testing.assert_allclose(original_regularization_loss.numpy(), migrated_regularization_loss.numpy())
np.testing.assert_allclose(original_output.numpy(), migrated_output.numpy())

Finally, remove both any remaining (no-longer-needed) variable_scope usage and the track_tf1_style_variables decorator itself.

You are now left with a version of the model that uses entirely native APIs.

class FullyNativeModel(tf.keras.layers.Layer):

  def __init__(self, units, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.units = units
    self.conv_layer = tf.keras.layers.Conv2D(
      3, 3,
      kernel_regularizer="l2")
    self.flatten_layer = tf.keras.layers.Flatten()
    self.dense_layer = tf.keras.layers.Dense(
      self.units,
      kernel_regularizer="l2")

  def call(self, inputs):
    out = self.conv_layer(inputs)
    out = self.flatten_layer(out)
    out = self.dense_layer(out)
    return out

random_tool = v1.keras.utils.DeterministicRandomTestTool(mode='num_random_ops')
with random_tool.scope():
  tf.keras.utils.set_random_seed(42)
  layer = FullyNativeModel(10)

  inputs = tf.random.normal(shape=(10, 5, 5, 5))
  migrated_output = layer(inputs)

  # Grab the regularization loss as well
  migrated_regularization_loss = tf.math.add_n(layer.losses)

print(migrated_regularization_loss)

# Verify that the regularization loss and output both match
np.testing.assert_allclose(original_regularization_loss.numpy(), migrated_regularization_loss.numpy())
np.testing.assert_allclose(original_output.numpy(), migrated_output.numpy())

Maintaining checkpoint compatibility during migration to Native TF2

The above migration process to native TF2 APIs changed both the variable names (as Keras APIs produce very different weight names), and the object-oriented paths that point to different weights in the model. The impact of these changes is that they will have broken both any existing TF1-style name-based checkpoints or TF2-style object-oriented checkpoints.

However, in some cases, you might be able to take your original name-based checkpoint and find a mapping of the variables to their new names with approaches like the one detailed in the Reusing TF1.x checkpoints guide.

Some tips to making this feasible are as follows:

    Variables still all have a name argument you can set.
    Keras models also take a name argument as which they set as the prefix for their variables.
    The v1.name_scope function can be used to set variable name prefixes. This is very different from tf.variable_scope. It only affects names, and doesn't track variables and reuse.

With the above pointers in mind, the following code samples demonstrate a workflow you can adapt to your code to incrementally update part of a model while simultaneously updating checkpoints.
Note: Due to the complexity of variable naming with Keras layers, this is not guaranteed to work for all use cases.

    Begin by switching functional-style tf.compat.v1.layers to their object-oriented versions.

class FunctionalStyleCompatModel(tf.keras.layers.Layer):

  @tf.compat.v1.keras.utils.track_tf1_style_variables
  def call(self, inputs, training=None):
    with tf.compat.v1.variable_scope('model'):
      out = tf.compat.v1.layers.conv2d(
          inputs, 3, 3,
          kernel_regularizer="l2")
      out = tf.compat.v1.layers.conv2d(
          out, 4, 4,
          kernel_regularizer="l2")
      out = tf.compat.v1.layers.conv2d(
          out, 5, 5,
          kernel_regularizer="l2")
      return out

layer = FunctionalStyleCompatModel()
layer(tf.ones(shape=(10, 10, 10, 10)))
[v.name for v in layer.weights]

    Next, assign the compat.v1.layer objects and any variables created by compat.v1.get_variable as properties of the tf.keras.layers.Layer/tf.Module object whose method is decorated with track_tf1_style_variables (note that any object-oriented TF2 style checkpoints will now save out both a path by variable name and the new object-oriented path).

class OOStyleCompatModel(tf.keras.layers.Layer):

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.conv_1 = tf.compat.v1.layers.Conv2D(
          3, 3,
          kernel_regularizer="l2")
    self.conv_2 = tf.compat.v1.layers.Conv2D(
          4, 4,
          kernel_regularizer="l2")

  @tf.compat.v1.keras.utils.track_tf1_style_variables
  def call(self, inputs, training=None):
    with tf.compat.v1.variable_scope('model'):
      out = self.conv_1(inputs)
      out = self.conv_2(out)
      out = tf.compat.v1.layers.conv2d(
          out, 5, 5,
          kernel_regularizer="l2")
      return out

layer = OOStyleCompatModel()
layer(tf.ones(shape=(10, 10, 10, 10)))
[v.name for v in layer.weights]

    Resave a loaded checkpoint at this point to save out paths both by the variable name (for compat.v1.layers), or by the object-oriented object graph.

weights = {v.name: v for v in layer.weights}
assert weights['model/conv2d/kernel:0'] is layer.conv_1.kernel
assert weights['model/conv2d_1/bias:0'] is layer.conv_2.bias

    You can now swap out the object-oriented compat.v1.layers for native Keras layers while still being able to load the recently-saved checkpoint. Ensure that you preserve variable names for the remaining compat.v1.layers by still recording the auto-generated variable_scopes of the replaced layers. These switched layers/variables will now only use the object attribute path to the variables in the checkpoint instead of the variable name path.

In general, you can replace usage of compat.v1.get_variable in variables attached to properties by:

    Switching them to using tf.Variable, OR
    Updating them by using tf.keras.layers.Layer.add_weight. Note that if you are not switching all layers in one go this may change auto-generated layer/variable naming for the remaining compat.v1.layers that are missing a name argument. If that is the case, you must keep the variable names for remaining compat.v1.layers the same by manually opening and closing a variable_scope corresponding to the removed compat.v1.layer's generated scope name. Otherwise the paths from existing checkpoints may conflict and checkpoint loading will behave incorrectly.

def record_scope(scope_name):
  """Record a variable_scope to make sure future ones get incremented."""
  with tf.compat.v1.variable_scope(scope_name):
    pass

class PartiallyNativeKerasLayersModel(tf.keras.layers.Layer):

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.conv_1 = tf.keras.layers.Conv2D(
          3, 3,
          kernel_regularizer="l2")
    self.conv_2 = tf.keras.layers.Conv2D(
          4, 4,
          kernel_regularizer="l2")

  @tf.compat.v1.keras.utils.track_tf1_style_variables
  def call(self, inputs, training=None):
    with tf.compat.v1.variable_scope('model'):
      out = self.conv_1(inputs)
      record_scope('conv2d') # Only needed if follow-on compat.v1.layers do not pass a `name` arg
      out = self.conv_2(out)
      record_scope('conv2d_1') # Only needed if follow-on compat.v1.layers do not pass a `name` arg
      out = tf.compat.v1.layers.conv2d(
          out, 5, 5,
          kernel_regularizer="l2")
      return out

layer = PartiallyNativeKerasLayersModel()
layer(tf.ones(shape=(10, 10, 10, 10)))
[v.name for v in layer.weights]

Saving a checkpoint out at this step after constructing the variables will make it contain only the currently-available object paths.

Ensure you record the scopes of the removed compat.v1.layers to preserve the auto-generated weight names for the remaining compat.v1.layers.

weights = set(v.name for v in layer.weights)
assert 'model/conv2d_2/kernel:0' in weights
assert 'model/conv2d_2/bias:0' in weights

    Repeat the above steps until you have replaced all the compat.v1.layers and compat.v1.get_variables in your model with fully-native equivalents.

class FullyNativeKerasLayersModel(tf.keras.layers.Layer):

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.conv_1 = tf.keras.layers.Conv2D(
          3, 3,
          kernel_regularizer="l2")
    self.conv_2 = tf.keras.layers.Conv2D(
          4, 4,
          kernel_regularizer="l2")
    self.conv_3 = tf.keras.layers.Conv2D(
          5, 5,
          kernel_regularizer="l2")


  def call(self, inputs, training=None):
    with tf.compat.v1.variable_scope('model'):
      out = self.conv_1(inputs)
      out = self.conv_2(out)
      out = self.conv_3(out)
      return out

layer = FullyNativeKerasLayersModel()
layer(tf.ones(shape=(10, 10, 10, 10)))
[v.name for v in layer.weights]

Remember to test to make sure the newly updated checkpoint still behaves as you expect. Apply the techniques described in the validate numerical correctness guide at every incremental step of this process to ensure your migrated code runs correctly.
Handling TF1.x to TF2 behavior changes not covered by the modeling shims

The modeling shims described in this guide can make sure that variables, layers, and regularization losses created with get_variable, tf.compat.v1.layers, and variable_scope semantics continue to work as before when using eager execution and tf.function, without having to rely on collections.

This does not cover all TF1.x-specific semantics that your model forward passes may be relying on. In some cases, the shims might be insufficient to get your model forward pass running in TF2 on their own. Read the TF1.x vs TF2 behaviors guide to learn more about the behavioral differences between TF1.x and TF2