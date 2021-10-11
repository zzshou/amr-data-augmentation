if [ "$#" -lt 1 ]; then
  echo "./predict_amr.sh <plain_text>"
  exit 2
fi

#echo ${1//.txt/.amr}
if [ -d "${1}" ] ; then
    echo "1 is a directory";
    for file in $(find $1 -type f -name "*.txt"); do
        echo $file
        PYTHONPATH="$(pwd)":"$PYTHON_PATH" \
            python bin/predict_amrs_from_plaintext.py \
            --texts ${file}  \
            --checkpoint amr-models/AMR2.parsing.pt \
            --beam-size 5 \
            --batch-size 500 \
            --device cuda:0 \
            --penman-linearization \
            --use-pointer-tokens \
            --pred_path ${file//.txt/.amr}
    done
else
    if [ -f "${1}" ]; then
        echo "${1} is a file";
        PYTHONPATH="$(pwd)":"$PYTHON_PATH" \
            python bin/predict_amrs_from_plaintext.py \
            --texts ${1}  \
            --checkpoint amr-models/AMR2.parsing.pt \
            --beam-size 5 \
            --batch-size 300 \
            --device cuda:0 \
            --penman-linearization \
            --use-pointer-tokens \
            --pred_path ${1//.txt/.amr}
    else
        echo "${1} is not valid";
        exit 1
    fi
fi

