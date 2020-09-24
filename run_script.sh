#!/bin/bash

POSITIONAL=()
while [[ $# -gt 0 ]]
do
key="$1"

case $key in
    -gpu|--gpu-id)
    GPU_ID="$2"
    shift # past argument
    shift # past value
    ;;
    *)    # unknown option
    POSITIONAL+=("$1") # save it in an array for later
    shift # past argument
    ;;
esac
done
set -- "${POSITIONAL[@]}" # restore positional parameters

export CUDA_VISIBLE_DEVICES="$GPU_ID"
export PYTHONPATH='.':$PYTHONPATH
python3 "${POSITIONAL[@]}"
