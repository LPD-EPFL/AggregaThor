This directory explain our changes in the core distributed runtime of TensorFlow. This includes the following changes:
1. Modifying the MPI implementation to support UDP.
2. Supporting secure gRPC.
3. Prohibit all except PS jobs to modify the graph.

To apply these modifications, one need to build TensorFlow from source as indicated by this page: https://www.tensorflow.org/install/source.
Then, copy "kernels" folder to "tensorflow/tensorflow/contrib/mpi_collectives/" and run "apply_patches.sh"
This is tested with r1.6 branch.

Here are all dependencies you would need (in addition to those mentioned in the link above) to make this work. We are writing in brackets the version we use (if any):
- Openmpi (v 3.0.0)
- Python (v 3.5)
- Bazel (v 0.9.0)
- libsodium (https://libsodium.gitbook.io/doc/installation)

##MPI-related Info:
This section provides some useful information about how to use our MPI extension (supporting UDP) and what are the avialble options.

## DISCLAIMER:
This is the UDP extension implementation of the TensorFlow communication layer. We base our implementation on the MPI implementation publicly available on https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/mpi

## How to compile and use MPI-enabled TensorFlow

1. Follow the regular TF compilation instructions. During configure step, if you want MPI support, answer yes to this question:

    ```Do you wish to build TensorFlow with MPI support [y/N]```

2. To turn on the MPI connection, add the protocol "grpc+mpi" in the server definition:

    ```server = tf.train.Server(cluster, job_name="local", task_index=0, protocol='grpc+mpi') # default protocol is 'grpc'```

## Runtime options

The following environment variables can be set to modify the behavior at runtime:

**MPI_DISABLED=[0,1]**

This environment variable allows you to disable the MPI path before launch (e.g. for performance or correctness testing).

**MPI_OPTIMAL_PATH=[0,1]**

When set to 0 it will use the default path where tensors are encoded to ProtoText before being copied to a remote process. When set to 1 a more optimal path will be taken where only the tensor description is encoded while the actual tensor data is transferred directly from the source buffer to the destination buffer.
This path is disabled by default as it requires that the MPI library can directly access the pointer to the data. For CPU backed buffers this is no problem, however for GPU backed buffers this requires MPI libraries that are built with CUDA support (CUDA Aware). When using non-CUDA aware MPI libraries and GPU buffers you will get segmentation faults.

To use UDP extension, these are the environmnet variables that should be set:
	a) USE_UDP = [0,1]
	b) UDP_WORKERS = [0...n] where n is the total number of workers

## Overview

By using this protocol TensorFlow can take advantage of the high performance networking primitives that are offered via the MPI API. This enables TensorFlow to take advantage of high performance low latency networks such as Infiniband. These changes are largely transparent to the user who only has to change the offered protocol and launch the script using the 'mpirun'  launcher. For example:
    ```mpirun -np 2 python my_neuralnet.py ```
All enviornment variables can be set by '-x' flag as follows:
	```mpirun -np 2 -x MPI_DISABLED=0 -x USE_UDP=1 -x UDP_WORKERS=3 python my_neuralnet.py ```
