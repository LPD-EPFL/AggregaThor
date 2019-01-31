# AggregaThor

Authors: Georgios Damaskinos, Arsany Guirguis, Sébastien Rouault

This is the framework introduced in [AggregaThor: Byzantine Machine Learning via Robust Gradient Aggregation (SysML19')](https://www.sysml.cc/papers.html),
co-authored by El Mahdi El Mhamdi, who provided the algorithms design and formal guarantees, and Rachid Guerraoui, who supervised the project.

## Structure
* `deploy.py` *To deploy a TF cluster over SSH (no need for NFS).*
```
usage: deploy.py [-h] --cluster CLUSTER [--deploy] [--id ID]
                 [--nice [NICE [NICE ...]]] [--omit] [--MPI]
		 [--runner RUNNER] [--UDP NUM_UDP]

optional arguments:
  -h, --help            show this help message and exit
  --cluster CLUSTER     Full cluster specification, JSON format: {"<jobname>": ["hostname:port", ...], ...}, or special value(s): 'G5k'
  --deploy              Whether this instance must deploy the whole cluster through SSH
  --id ID               This node's role, format: <job>:<id>
  --nice [NICE [NICE ...]]
                        Make this process nice, or list of job(s) which tasks must maximize their respective niceness level
  --omit                Do not start the node's server instance, can be used only with '--id' and '--deploy'
  --MPI			Use MPI instead of gRPC as the underlying communication
  --runner RUNNER	Runner arguments (see below). Should be passed in case of using MPI (see Distributed deployment with MPI/UDP section below)
  --UDP NUM_UDP		Number of workers that will use UDP connections
```
* `runner.py` *To perform training + evaluation of an experiment + aggregator on a cluster.*
```
usage: runner.py [-h] [--client CLIENT] [--server SERVER]
                 [--ps-job-name PS_JOB_NAME] [--ev-job-name EV_JOB_NAME]
                 [--wk-job-name WK_JOB_NAME] --experiment EXPERIMENT
                 [--experiment-args [EXPERIMENT_ARGS [EXPERIMENT_ARGS ...]]]
                 --aggregator AGGREGATOR
                 [--aggregator-args [AGGREGATOR_ARGS [AGGREGATOR_ARGS ...]]]
                 [--optimizer OPTIMIZER]
                 [--optimizer-args [OPTIMIZER_ARGS [OPTIMIZER_ARGS ...]]]
                 [--learning-rate LEARNING_RATE]
                 [--learning-rate-args [LEARNING_RATE_ARGS [LEARNING_RATE_ARGS ...]]]
                 [--l1-regularize L1_REGULARIZE]
                 [--l2-regularize L2_REGULARIZE] --nb-workers NB_WORKERS
                 [--nb-decl-byz-workers NB_DECL_BYZ_WORKERS]
                 [--nb-real-byz-workers NB_REAL_BYZ_WORKERS] [--attack ATTACK]
                 [--attack-args [ATTACK_ARGS [ATTACK_ARGS ...]]]
                 [--max-steps MAX_STEPS] [--checkpoint-dir CHECKPOINT_DIR]
                 [--checkpoint-delta CHECKPOINT_DELTA]
                 [--checkpoint-period CHECKPOINT_PERIOD]
                 [--summary-dir SUMMARY_DIR] [--summary-delta SUMMARY_DELTA]
                 [--summary-period SUMMARY_PERIOD]
                 [--evaluation-file EVALUATION_FILE]
                 [--evaluation-delta EVALUATION_DELTA]
                 [--evaluation-period EVALUATION_PERIOD] [--use-gpu]
                 [--reuse-gpu] [--use-tpu] [--reuse-tpu] [--no-wait] [--trace]
                 [--stdout-to STDOUT_TO] [--stderr-to STDERR_TO] [--MPI]

Start/continue a distributed training session.

optional arguments:
  -h, --help            show this help message and exit
  --client CLIENT       Trusted node URL in the cluster (usually the parameter server) to connect to as a client; one and only one of '--server' and '--client' must be specified
  --server SERVER       Full JSON cluster specification, on which to act as the only parameter server, or special value(s): 'G5k'; one and only one of '--server' and '--client' must be specified
  --ps-job-name PS_JOB_NAME
                        Parameter server job name
  --ev-job-name EV_JOB_NAME
                        Evaluation job name (may be the parameter server job name)
  --wk-job-name WK_JOB_NAME
                        Worker job name
  --experiment EXPERIMENT
                        Experiment to run on the cluster
  --experiment-args [EXPERIMENT_ARGS [EXPERIMENT_ARGS ...]]
                        Additional arguments to pass to the underlying experiment
  --aggregator AGGREGATOR
                        Gradient aggregation rule to use
  --aggregator-args [AGGREGATOR_ARGS [AGGREGATOR_ARGS ...]]
                        Additional arguments to pass to the underlying GAR
  --optimizer OPTIMIZER
                        Optimizer to use
  --optimizer-args [OPTIMIZER_ARGS [OPTIMIZER_ARGS ...]]
                        Additional arguments to pass to the underlying optimizer
  --learning-rate LEARNING_RATE
                        Type of learning rate decay to use
  --learning-rate-args [LEARNING_RATE_ARGS [LEARNING_RATE_ARGS ...]]
                        Additional arguments to pass to the underlying learning rate
  --l1-regularize L1_REGULARIZE
                        l1 regularization strength to use, non-positive for none, non-positive by default
  --l2-regularize L2_REGULARIZE
                        l2 regularization strength to use, non-positive for none, non-positive by default
  --nb-workers NB_WORKERS
                        Total number of workers
  --nb-decl-byz-workers NB_DECL_BYZ_WORKERS
                        Number of declared Byzantine workers (i.e. value of 'f')
  --nb-real-byz-workers NB_REAL_BYZ_WORKERS
                        Number of real Byzantine workers
  --attack ATTACK       Attack to use (ignored if --nb-real-byz-workers is 0)
  --attack-args [ATTACK_ARGS [ATTACK_ARGS ...]]
                        Additional arguments to pass to the underlying attack (ignored if --nb-real-byz-workers is 0)
  --max-steps MAX_STEPS
                        Number of additional steps to perform before stopping the training, non-positive for no limit
  --checkpoint-dir CHECKPOINT_DIR
                        Checkpoint directory to use, will be created if inexistent
  --checkpoint-delta CHECKPOINT_DELTA
                        Save checkpoint after the given step delta, negative for unused
  --checkpoint-period CHECKPOINT_PERIOD
                        Save checkpoint at least every given period (in s), negative for unused
  --summary-dir SUMMARY_DIR
                        Summary directory to use, '-' for none, defaults to '--checkpoint-dir'
  --summary-delta SUMMARY_DELTA
                        Save summaries after the given step delta, negative for unused
  --summary-period SUMMARY_PERIOD
                        Save summaries at least every given period (in s), negative for unused
  --evaluation-file EVALUATION_FILE
                        File in which to write the accuracy evaluations (format: wall time (in s)<tab>global step<tab>name:value<tab>...), '-' for none, defaults to '<checkpoint dir>/eval'
  --evaluation-delta EVALUATION_DELTA
                        Evaluate the model after the given step delta, negative for unused
  --evaluation-period EVALUATION_PERIOD
                        Evaluate the model at least every given period (in s), negative for unused
  --use-gpu             Use target GPU devices if available
  --reuse-gpu           Allow target GPU devices to be used by several entities, implies '--use-gpu'
  --use-tpu             Use target TPU devices if available
  --reuse-tpu           Allow target TPU devices to be used by several entities, implies '--use-tpu'
  --no-wait             Do not wait for a signal before exiting when acting as a server
  --trace               Print a (performance) debugging message for every important step of the graph execution
  --stdout-to STDOUT_TO
                        Redirect the standard output to the given file (overwritten if exists), '-' for none, '-' by default
  --stderr-to STDERR_TO
                        Redirect the standard error output to the given file (overwritten if exists), '-' for none, '-' by default
  --MPI                 Use MPI instead of gRPC as the underlying communication

```
* `experiments` *Directory containing each supported experiment (1 experiment = 1 model + 1 dataset).*
* `aggregators` *Directory containing each supported GAR.*

## Local deployment
1. (optional) [Apply the TF patches](tf_patches/)

2. Run different experiments on the local machine.

   `python3 runner.py --server '{"local": ["127.0.0.1:7000"]}' --ps-job-name local --wk-job-name local --ev-job-name local --experiment mnist --learning-rate-args initial-rate:0.05 --aggregator average --nb-workers 4 --reuse-gpu --max-step 10000 --evaluation-period -1 --checkpoint-period -1 --summary-period -1 --evaluation-delta 100 --checkpoint-delta -1 --summary-delta -1 --no-wait`

## Distributed deployment
1. (optional) [Apply the TF patches](tf_patches/)

2. Deploy once a cluster with `deploy.py`.

   `user@`**_`any`_**`:$ python3 deploy.py --cluster '{"ps": ["cpu_ps:7000"], "workers": ["cpu_worker:7000", "gpu_worker:7000"]}' --deploy --id ps:0 --omit`

   * adapt the provided cluster string to you own cluster.
   The format is: {"_job name_": \["_hostname_:_port_", ...\], ...}

3. Run different experiments on the deployed cluster (no need to call `deploy.py` for each use of `runner.py`).

   `user@`**`cpu_ps`**`:$ python3 runner.py --server '{"ps": ["cpu_ps:7000"], "workers": ["cpu_worker:7000", "gpu_worker:7000"]}' --ev-job-name ps --experiment mnist --learning-rate-args initial-rate:0.05 --aggregator average --nb-workers 4 --use-gpu --max-step 10000 --evaluation-period -1 --checkpoint-period -1 --summary-period -1 --evaluation-delta 100 --checkpoint-delta -1 --summary-delta -1 --no-wait`

   * specify the same cluster as in the `deploy.py` command.
   * IMPORTANT: run this command on the machine named **cpu\_ps** in this example (see Pitfalls/Need for a *process-local* server).

   Please note that only one, trusted parameter server is supported.

## Distributed deployment with MPI/UDP
1. Apply the TF patches (tf_patches/)

2. Run `deploy.py` and pass the arguments of `runner.py` to it with the `RUNNER` argument. This should run from the machine named **cpu\_ps**

   `user@`**`cpu_ps`**`:$ python3 deploy.py --cluster '{"ps": ["cpu_ps:7000"], "workers": ["cpu_worker:7000", "gpu-worker:7000"]}' --deploy --id ps:0 --omit --MPI --UDP 1 --runner "--server '{\"ps\": [\"cpu_ps:7000\"], \"workers\": [\"cpu_worker:7000\", \"gpu_worker:7000\"]}' --ev-job-name ps --experiment mnist --learning-rate-args initial-rate:0.05 --aggregator average --nb-workers 2 --max-step 10000 --evaluation-period -1 --checkpoint-period -1 --summary-period -1 --evaluation-delta 100 --checkpoint-delta -1 --summary-delta -1 --no-wait --MPI"`

## Pitfalls
* Bug/incomplete documentation in distributed TF (r1.10).

   The example given above triggers that bug: you'll realize with `watch nvidia-smi` on *gpu_worker* that the GPUs are actually not in use.
   This induces a huge performance hit, and running onto a single-machine cluster on *gpu_worker* with no `deploy.py` and `--server '{"all": ["gpu_worker:7000"]}' --ps-job-name all --wk-job-name all --ev-job-name all --reuse-gpu` for `runner.py` would increase throughput by **two orders of magnitude**.

   So keep in mind that, depending from where you connect to the cluster, GPUs might be enable or not.
   This breaks the abstraction of a unified cluster that TF seems to be bound to provide.
   This may be because kernel resolution is *wrongly* handled by the server creating the session (that might be CPU-only), and not by the servers carrying the operations out (that could support GPUs).

* Need for a *process-local* server.

   Some operations cannot be serialized, like custom ones, `tf.data.Dataset.from_generator` and `tf.py_func`.
   These can only be carried out by a `tf.train.Server` which is in the same process as the dynamically loaded native library or Python bytecode to execute.
   The `--omit` option of `deploy.py` and `--server` option of `runner.py` are to allow `runner.py` to instantiate the cluster's PS server, and then to use these unserializable operations on the PS *only*.

## Experiments using slim
Google's slim models and datasets are imported by `experiments/slims.py`.

This module requires that there exists (symlinks to) directories named:
* `experiments/slim`, that contains a local copy of [models/research/slim](https://github.com/tensorflow/models/tree/master/research/slim).
* `experiments/datasets`, that contains directories (named `cifar10`, `imagenet`, etc), each containing the associated dataset in a *TFRecord* format (that slim would successfully be able to load and preprocess, see [preparing the datasets](https://github.com/tensorflow/models/tree/master/research/slim#Data)).
