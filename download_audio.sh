mkdir -p data/raw_data
python3 data/download/download.py \
        --dataset raw_30s \
        --type audio-low \
        --num_dir 3 \
        data/raw_data \
        --unpack \
        --remove
