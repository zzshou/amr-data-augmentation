# codes from plms-graph2text
#!/bin/bash

ROOT_DIR="$( cd "$( dirname "$( dirname "${BASH_SOURCE[0]}" )")" >/dev/null 2>&1 && pwd )"
UTILS_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
#echo $ROOT_DIR
#echo $UTILS_DIR

mkdir -p ${ROOT_DIR}/tmp_data
REPO_DIR=${ROOT_DIR}/tmp_data

DATA_FILE=${1}
#mkdir -p ${REPO_DIR}/tmp_amr
PREPROC_DIR=${REPO_DIR}
ORIG_AMR_FILE=${DATA_FILE}
#mkdir -p ${REPO_DIR}/question_amr
FINAL_AMR_DIR=$( dirname $DATA_FILE )

echo "processing ..."
python ${UTILS_DIR}/split_amr.py ${ORIG_AMR_FILE} ${PREPROC_DIR}/surface.txt ${PREPROC_DIR}/graphs.txt
python ${UTILS_DIR}/preproc_amr.py ${PREPROC_DIR}/graphs.txt ${PREPROC_DIR}/surface.txt ${FINAL_AMR_DIR}/nodes.pp.txt ${FINAL_AMR_DIR}/surface.pp.txt --mode LIN --triples-output ${FINAL_AMR_DIR}/triples.pp.txt

mv ${FINAL_AMR_DIR}/nodes.pp.txt ${DATA_FILE//.amr/.source}
mv ${FINAL_AMR_DIR}/surface.pp.txt ${DATA_FILE//.amr/.target}
#perl ${ROOT_DIR}/data-utils/tokenizer.perl -l en < ${ROOT_DIR}/question-pair-data/question2.target > ${ROOT_DIR}/question-pair-data/question2.target.tok

echo "done."

done

