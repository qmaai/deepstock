set -eux
path='./data_full'
for file in $path"/"$(ls $path)
do
    python sr.py $file --check_all --save_result --render
done
