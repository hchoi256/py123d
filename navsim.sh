export NUSCENES_DATA_ROOT=/workspace/data/nuscenes
export PY123D_DATA_ROOT=/workspace/data/py123d_output

python src/py123d/script/run_conversion.py \
    datasets=[nuscenes] \
    datasets.nuscenes.splits="[nuscenes_val]" \
    log_writer=navsim_writer