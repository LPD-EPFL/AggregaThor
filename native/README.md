# Classification

* `so` -> "Vanilla" shared object: build
* `py` -> "Python-interface" shared object: build and load (`ctypes.CDLL`)
* `op` -> "Custom ops" shared object: build, load (`tf.load_op_library`) and wrap exported ops

# Dependencies

In dependent SO directory, create symbolic link pointing to dependee SO directory.
