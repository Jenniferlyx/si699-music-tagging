mkdir -p data/autotagging_moodtheme
python3 data/download/download.py \
        --dataset autotagging_moodtheme \
        --type audio-low \
        --num_dir $1 \
        data/autotagging_moodtheme \
        --unpack \
        --remove
