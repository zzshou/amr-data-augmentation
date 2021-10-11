import os, sys
import json

sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), 'amr_parser_spring'))
# print(sys.path)

import torch
from amr_parser_spring.spring_amr.penman import encode
from amr_parser_spring.spring_amr.utils import instantiate_model_and_tokenizer
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

from data_utils.preprocess.amr import AMR
from data_utils.preprocess import preproc_amr
from data_utils.augment import *

from plms_graph2text.amr.finetune import Graph2TextModule, generic_train, SummarizationModule
from torch.utils.data import DataLoader
from plms_graph2text.amr.utils import Seq2SeqDataset
from plms_graph2text.amr.lightning_base import add_generic_args


def parse_sentence_to_amr(sentences, model, tokenizer, device, beam_size):
    x, _ = tokenizer.batch_encode_sentences(sentences, device=device)
    with torch.no_grad():
        model.amr_mode = True
        out = model.generate(**x, max_length=512, decoder_start_token_id=0, num_beams=beam_size)

    bgraphs = []
    for sent, tokk in zip(sentences, out):
        graph, status, (lin, backr) = tokenizer.decode_amr(tokk.tolist())
        bgraphs.append(graph)

    return bgraphs


def aug_amr_graph(amr_graph_str, if_del=True, if_syn=True, if_ant=True, if_pol=True, if_ins=True, if_swap=True,
                  if_rot=True, deln=1, synn=1, antn=1, insn=1, swapn=1, rotn=1):
    aug_sources = {}

    # amr -> sequence
    amr_graph = AMR(amr_graph_str)
    v2c = amr_graph.var2concept()
    tokens = amr_graph_str.split()
    new_tokens = preproc_amr.simplify(tokens, v2c)
    simplified_source = ' '.join(new_tokens)
    aug_sources['simplified'] = [simplified_source]  # simplified tokens

    # synonym replacement
    if if_syn:
        syn_aug = replace_syntonym(simplified_source, n=synn)
        if synn > 1:
            aug_sources['syn'] = syn_aug
        else:
            aug_sources['syn'] = [syn_aug]

    # antonym replacement
    if if_ant:
        ant_aug = replace_antonym(simplified_source, n=antn)
        if antn > 1:
            aug_sources['ant'] = ant_aug
        else:
            aug_sources['ant'] = [ant_aug]

    # polarity inversion
    if if_pol:
        pol_aug = convert_polarity(simplified_source)
        aug_sources['pol'] = [pol_aug]

    # random insertion (random choose item from edges and nodes)
    if if_ins:
        ins_aug = insert_edge(simplified_source, n=insn)
        aug_sources['ins'] = ins_aug

    # random swap (swap nodes in amr graph)
    if if_swap:
        swap_aug = swap_nodes(simplified_source, n=swapn)
        aug_sources['swap'] = swap_aug

    # perspective rotation
    if if_rot:
        rot_aug = psp_rot(simplified_source, n=rotn)
        aug_sources['rot'] = rot_aug

    # random deletion
    if if_del:
        del_aug = delete_edge(simplified_source, k=0.7, n=deln)
        aug_sources['del'] = del_aug

    return aug_sources


def parse_amr_to_sentence(amr_generator, trainer, args):
    dataset = Seq2SeqDataset(
        amr_generator.tokenizer,
        max_target_length=args.max_target_length,
        type_path='test',
        **amr_generator.dataset_kwargs
    )
    test_data = DataLoader(
        dataset,
        batch_size=args.eval_batch_size,
        collate_fn=dataset.collate_fn,
        shuffle=False,
        num_workers=args.num_workers,
        sampler=None,
    )
    res = trainer.test(amr_generator, test_dataloaders=test_data)
    return res


def main(sentences):
    parser = ArgumentParser(
        description="Script to predict AMR graphs given sentences. LDC format as input.",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    # args for amr-parser
    parser.add_argument('--parser_checkpoint', type=str, default="amr_parser_spring/amr-models/AMR2.parsing.pt",
                        help="Required. Checkpoint to restore.")
    parser.add_argument('--model', type=str, default='/home/data/zshou/corpus/bart-large',
                        help="Model config to use to load the model class.")
    parser.add_argument('--beam-size', type=int, default=2,
                        help="Beam size.")
    parser.add_argument('--batch-size', type=int, default=1000,
                        help="Batch size (as number of linearized graph tokens per batch).")
    parser.add_argument('--penman-linearization', default=True, action='store_true',
                        help="Predict using PENMAN linearization instead of ours.")
    parser.add_argument('--use-pointer-tokens', default=True, action='store_true')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help="Device. 'cpu', 'cuda', 'cuda:<n>'.")

    # args for augmentation
    parser.add_argument('--deln', type=int, default=2)
    parser.add_argument('--synn', type=int, default=2)
    parser.add_argument('--antn', type=int, default=2)

    # args for amr-generator
    parser.add_argument('--generator_checkpoint', type=str, default='plms_graph2text/amr-t5-large.ckpt')
    SummarizationModule.add_model_specific_args(parser)
    args = parser.parse_args()
    args.data_dir = 'tmp/aug.source'
    args.do_predict = True
    args.output_dir = 'tmp'
    args.model_name_or_path = '/home/data/zshou/corpus/t5-large'
    args.eval_batch_size = 4
    args.max_source_length = 512
    args.max_target_length = 384
    args.eval_beams = 5
    args.gpus = 1
    args.save_file_name = 'aug.pre.txt'

    # parser
    device = torch.device(args.device)
    amr_parser, tokenizer = instantiate_model_and_tokenizer(
        args.model,
        dropout=0.,
        attention_dropout=0,
        penman_linearization=args.penman_linearization,
        use_pointer_tokens=args.use_pointer_tokens,
    )
    amr_parser.load_state_dict(torch.load(args.parser_checkpoint, map_location='cpu')['model'])
    amr_parser.to(device)
    amr_parser.eval()

    # processing...
    amr_graphs = parse_sentence_to_amr(sentences, amr_parser, tokenizer, device, args.beam_size)
    aug_sources = []
    source_for_generator = []
    for i, g in enumerate(amr_graphs):
        eg = encode(g)

        # graph augmentation
        augs = aug_amr_graph(eg, deln=args.deln, synn=args.synn, antn=args.antn)
        aug_sources.append(json.dumps(augs))
        for type, a in augs.items():
            source_for_generator.extend(a)
    with open('tmp/aug.source', 'w') as wf, open('tmp/aug.dic', 'w') as dwf:
        dwf.writelines([aug + '\n' for aug in aug_sources])
        wf.writelines([aug + '\n' for aug in source_for_generator])

    # generator
    amr_generator = Graph2TextModule(args)
    amr_generator.load_state_dict(
        torch.load(args.generator_checkpoint, map_location='cpu')['state_dict'])
    trainer = generic_train(amr_generator, args)

    res = parse_amr_to_sentence(amr_generator, trainer, args)
    print(res)


if __name__ == "__main__":
    sentences = []
    input = 'A sad, superior human comedy played out on the back roads of life.'
    sentences.append(input)
    main(sentences)
