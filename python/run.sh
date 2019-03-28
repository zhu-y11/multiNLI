if [[ $# -ne 2 ]]; then
  echo "$0 base_path model_name"
  exit 1
fi

model=esim
base_path=$1
model_name=$2

python3 -u train_mnli.py $model $model_name \
  --train_embpath $base_path/$model_name/en.$model_name.txt \
  --test_embpath $base_path/$model_name/de.$model_name.txt \
  > $model_name.log 2>&1 &
