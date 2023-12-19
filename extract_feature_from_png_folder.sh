# path of dir containing pngs
folder_input=$1
folder_out_parent=$2
pretrain_weight=$3
pretrain_model=$4

# set -euxo pipefail # make sure failure will be reported
DIR_SCRIPTS=$(cd `dirname $0`; pwd)

#tmp_for_svs_path=`mktemp /tmp/aaa.XXXXXXXXX` # mktemp will automatically replace XXXXXXXXX
#echo $file_input > $tmp_for_svs_path # $tmp_for_svs_path is plain text without header

python $DIR_SCRIPTS/extract_feature_from_png_folder.py \
    $folder_input \
    $folder_out_parent \
    $pretrain_weight \
    $pretrain_model