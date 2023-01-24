#!/bin/bash

echo "set ENVs"

export WC_GEN_STDPAR_SRC=/home/twang/myWork/wire-cell-gen-stdpar

#WC_GEN_STDPAR build directory 
[ -z $WC_GEN_STDPAR_BUILD ] && export WC_GEN_STDPAR_BUILD=$PWD
if [ $WC_GEN_STDPAR_BUILD == $WC_GEN_STDPAR_SRC ] ; then
	export WC_GEN_STDPAR_BUILD=${WC_GEN_STDPAR_SRC}/build
fi

alias wc_run="lar -n 1 -c ${WC_GEN_STDPAR_SRC}/example/sim.fcl  ${WC_GEN_STDPAR_SRC}/example/g4-1-event.root"
#alias wcb_configure="${WC_GEN_STDPAR_SRC}/configure.out ${WC_GEN_STDPAR_BUILD} "
#alias wcb_build="${WC_GEN_STDPAR_SRC}/wcb -o $WC_GEN_STDPAR_BUILD -t ${WC_GEN_STDPAR_SRC} build --notest "

alias wc-build-cmake="cmake ${WC_GEN_STDPAR_SRC}/.cmake-stdpar-cuda/ && make"
#alias wc-build-cmake-hip="${WC_GEN_STDPAR_SRC}/build-cmake.hip"

#echo "WC_STDPAR_GEN_BUILD directory: $WC_GEN_STDPAR_BUILD "
echo "WC_GEN_STDPAR_SRC directory: ${WC_GEN_STDPAR_SRC}"
#alias

#no-container
export WIRECELL_DATA=/home/zdong/PPS/git/wire-cell-data
export CUDA_DIR=/usr/local/cuda-11.2

export WC_GEN_STDPAR_LIB=$WC_GEN_STDPAR_BUILD

export WCT=$WIRECELL_FQ_DIR
export WCT_SRC=${WIRECELL_FQ_DIR}/wirecell-0.14.0

export PATH=${CUDA_DIR}/bin:$PATH

export WIRECELL_PATH=${WC_GEN_STDPAR}:${WC_GEN_STDPAR_SRC}/cfg:${WC_GEN_STDPAR_SRC}/example:${WIRECELL_DATA}:${WCT_SRC}/cfg:${WCT}/share/wirecell
export LD_LIBRARY_PATH=${WC_GEN_STDPAR_LIB}:${CUDA_DIR}/lib64:${WCT}/lib:$LD_LIBRARY_PATH
