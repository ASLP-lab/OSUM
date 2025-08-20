export WENET_DIR=$PWD/../..
export BUILD_DIR=${WENET_DIR}/runtime/libtorch/build
export OPENFST_PREFIX_DIR=${BUILD_DIR}/../fc_base/openfst-subbuild/openfst-populate-prefix
export PATH=$PWD:${BUILD_DIR}/bin:${BUILD_DIR}/kaldi:${OPENFST_PREFIX_DIR}/bin:$PATH

export PYTHONIOENCODING=UTF-8
export PYTHONPATH=../../:$PYTHONPATH
