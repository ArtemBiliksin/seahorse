num=4
path_to_data="tmp/seahorse/ratings/google_seahorse_ratings_q${num}.tsv"

python src/metrics.py \
    --path_to_data $path_to_data \
    --true_col_name question${num} \
    --score_col_name "1_prob"
