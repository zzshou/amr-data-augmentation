# amr-data-augmentation
## requirements 
```
pip install -r requirements.txt
```

## parse the plain text to amr graph
```
cd amr-parser-spring
bash predict_amr.sh <plain_text_file_path>(../data/wiki_data/wiki.txt)
```
## preprocess amr graph, convert to source and target string
```
cd data-utils/preprocess
bash prepare_data.sh <amr_file_path>(../../data/wiki_data/wiki.amr)
```
## data augmentation (optional)
```
cd data_utils
python augment.py (modify parameters according to your requirements)
```
## generate text from amr graph
```
cd plms-graph2text
bash decode_AMR.sh <model-path> <checkpoint> <gpu_id> <source file> <output-name>
(bash decode_AMR.sh /path/to/t5-base amr-t5-base.ckpt 0 ../data/wiki-data/wiki.source wiki-perd-t5-base.txt)
```
