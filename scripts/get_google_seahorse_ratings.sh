path_to_data=tmp/seahorse/seahorse_data_with_source/test.tsv
num=4
path_to_result=tmp/seahorse/ratings/google_seahorse_ratings_q${num}.tsv
model_name=google/seahorse-large-q${num}

python src/seahorse_ratings.py \
    --path_to_data $path_to_data \
    --model_name $model_name \
    --article_col_name article \
    --summary_col_name summary \
    --batch_size 1 \
    --path_to_result $path_to_result
