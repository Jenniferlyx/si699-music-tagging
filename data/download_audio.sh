mkdir data/raw_data
python3 ../mtg-jamendo-dataset/scripts/download/download.py \
        --dataset raw_30s \
        --type audio-low \
        --num_dir 10 \
        data/raw_data \
        --unpack \
        --remove