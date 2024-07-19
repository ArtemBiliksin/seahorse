wget -nc -P tmp/seahorse https://storage.googleapis.com/seahorse-public/seahorse_data.zip
unzip -n  tmp/seahorse/seahorse_data.zip seahorse seahorse_data/* -d tmp/seahorse

python src/download_seahorse.py \
    --languages de en es ru tr vi \
    --path_to_data tmp/seahorse/seahorse_data \
    --path_to_result tmp/seahorse/seahorse_data_with_source \
    --text_column article
