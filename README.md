# AMR-DA: Data Augmentation By Abstract Meaning Representation

We propose a new DA method called AMR-DA that uses the Abstract Meaning Representation (AMR, Banarescu et al., 2013) as the intermediate language. 
The figure shows an overview of our AMR-DA: AMR parser first transduces the sentence into an AMR graph, followed by an AMR graph exten- der to diversify graphs with different augmentation strategies; finally, the AMR generator synthesizes augmentations from AMR graphs.


## requirements 
```
pip install -r requirements.txt
```

## Parse the plain text to amr graph
```
cd amr-parser-spring
bash predict_amr.sh <plain_text_file_path>(../data/wiki_data/wiki.txt)
```
## Preprocess amr graph, convert to source and target string
```
cd data-utils/preprocess
bash prepare_data.sh <amr_file_path>(../../data/wiki_data/wiki.amr)
```
## Data augmentation (optional)
```
cd data_utils
python augment.py (modify parameters according to your requirements)
```
## Generate text from amr graph
```
cd plms-graph2text
bash decode_AMR.sh <model-path> <checkpoint> <gpu_id> <source file> <output-name>
(bash decode_AMR.sh /path/to/t5-base amr-t5-base.ckpt 0 ../data/wiki-data/wiki.source wiki-perd-t5-base.txt)
```
<!-- 
## Citation
If you make use of the code in this repository, please cite the following papers:
 -->
