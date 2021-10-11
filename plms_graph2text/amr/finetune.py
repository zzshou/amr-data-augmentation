#!/usr/bin/env python

import argparse
import glob
import logging
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple
import torch.nn.functional as F

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from pytorch_lightning.utilities import rank_zero_info

from callbacks import Seq2SeqLoggingCallback, get_checkpoint_callback, get_early_stopping_callback
from transformers import MBartTokenizer, T5ForConditionalGeneration

# def shift_tokens_right(input_ids, pad_token_id):
#     """Shift input ids one token to the right, and wrap the last non pad token (usually <eos>)."""
#     prev_output_tokens = input_ids.clone()
#     index_of_eos = (input_ids.ne(pad_token_id).sum(dim=1) - 1).unsqueeze(-1)
#     prev_output_tokens[:, 0] = input_ids.gather(1, index_of_eos).squeeze()
#     prev_output_tokens[:, 1:] = input_ids[:, :-1]
#
#     # F.pad(a, (0, 1), mode='constant', value=1)
#     # tensor([[3., 2., 1.],
#     #         [5., 6., 1.]])
#
#     return prev_output_tokens

from utils import (
    ROUGE_KEYS,
    LegacySeq2SeqDataset,
    Seq2SeqDataset,
    assert_all_frozen,
    calculate_bleu,
    calculate_rouge,
    flatten_list,
    freeze_embeds,
    freeze_params,
    get_git_info,
    label_smoothed_nll_loss,
    lmap,
    pickle_save,
    save_git_info,
    save_json,
    use_task_specific_params,
)

from utils_graph2text import convert_text, eval_meteor, eval_bleu_sents, eval_bleu_sents_tok, eval_chrf

# need the parent dir module
sys.path.insert(2, str(Path(__file__).resolve().parents[1]))
from lightning_base import BaseTransformer, add_generic_args, generic_train  # noqa

logger = logging.getLogger(__name__)


class SummarizationModule(BaseTransformer):
    mode = "summarization"
    loss_names = ["loss"]
    metric_names = ROUGE_KEYS
    default_val_metric = "rouge2"

    def __init__(self, hparams, **kwargs):
        if hparams.sortish_sampler and hparams.gpus > 1:
            hparams.replace_sampler_ddp = False
        elif hparams.max_tokens_per_batch is not None:
            if hparams.gpus > 1:
                raise NotImplementedError("Dynamic Batch size does not work for multi-gpu training")
            if hparams.sortish_sampler:
                raise ValueError("--sortish_sampler and --max_tokens_per_batch may not be used simultaneously")

        super().__init__(hparams, num_labels=None, mode=self.mode, **kwargs)
        # use_task_specific_params(self.model, "summarization")
        # save_git_info(self.hparams.output_dir)
        # self.metrics_save_path = Path(self.output_dir) / "metrics.json"
        # self.hparams_save_path = Path(self.output_dir) / "hparams.pkl"
        # pickle_save(self.hparams, self.hparams_save_path)
        self.step_count = -2
        self.metrics = defaultdict(list)
        self.model_type = self.config.model_type
        self.vocab_size = self.config.tgt_vocab_size if self.model_type == "fsmt" else self.config.vocab_size

        if 't5' in hparams.model_name_or_path:
            self.model.config.prefix = 'translate Graph to English: '
        self.dataset_kwargs: dict = dict(
            data_dir=self.hparams.data_dir,
            max_source_length=self.hparams.max_source_length,
            prefix=self.model.config.prefix or "",
        )
        n_observations_per_split = {
            "train": self.hparams.n_train,
            "val": self.hparams.n_val,
            "test": self.hparams.n_test,
        }
        self.n_obs = {k: v if v >= 0 else None for k, v in n_observations_per_split.items()}

        self.target_lens = {
            "train": self.hparams.max_target_length,
            "val": self.hparams.val_max_target_length,
            "test": self.hparams.test_max_target_length,
        }
        assert self.target_lens["train"] <= self.target_lens["val"], f"target_lens: {self.target_lens}"
        assert self.target_lens["train"] <= self.target_lens["test"], f"target_lens: {self.target_lens}"
        if self.hparams.freeze_embeds:
            freeze_embeds(self.model)
        if self.hparams.freeze_encoder:
            freeze_params(self.model.get_encoder())
            assert_all_frozen(self.model.get_encoder())

        self.hparams.git_sha = get_git_info()["repo_sha"]
        self.num_workers = hparams.num_workers
        self.decoder_start_token_id = None  # default to config
        # print(self.tokenizer.__class__.__name__)
        # print(hasattr(self.tokenizer, "prepare_seq2seq_batch"))
        # exit()

        if self.model.config.decoder_start_token_id is None and isinstance(self.tokenizer, MBartTokenizer):
            self.decoder_start_token_id = self.tokenizer.lang_code_to_id[hparams.tgt_lang]
            self.model.config.decoder_start_token_id = self.decoder_start_token_id

        self.dataset_class = (
            Seq2SeqDataset if hasattr(self.tokenizer, "prepare_seq2seq_batch") else LegacySeq2SeqDataset
        )
        self.already_saved_batch = True
        self.eval_beams = self.model.config.num_beams if self.hparams.eval_beams is None else self.hparams.eval_beams
        if self.hparams.eval_max_gen_length is not None:
            self.eval_max_length = self.hparams.eval_max_gen_length
        else:
            self.eval_max_length = self.model.config.max_length
        self.val_metric = self.default_val_metric if self.hparams.val_metric is None else self.hparams.val_metric
        self.cal_metrics = False

    def save_readable_batch(self, batch: Dict[str, torch.Tensor]) -> Dict[str, List[str]]:
        """A debugging utility"""

        readable_batch = {
            k: self.tokenizer.batch_decode(v.tolist()) if "mask" not in k else v.shape for k, v in batch.items()
        }
        save_json(readable_batch, Path(self.output_dir) / "text_batch.json")

        tb = {}
        for k, v in batch.items():
            tb[k] = v.tolist()

        save_json(tb, Path(self.output_dir) / "tok_batch.json")

        self.already_saved_batch = True
        return readable_batch

    def forward(self, input_ids, **kwargs):
        return self.model(input_ids, **kwargs)

    def ids_to_clean_text(self, generated_ids: List[int]):
        gen_text = self.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        return lmap(str.strip, gen_text)

    def _step(self, batch: dict) -> Tuple:
        pad_token_id = self.tokenizer.pad_token_id
        src_ids, src_mask = batch["input_ids"], batch["attention_mask"]
        if isinstance(self.model, T5ForConditionalGeneration):
            tgt_ids = batch["labels"]
            decoder_input_ids = self.model._shift_right(tgt_ids)
        else:
            # tgt_ids = batch["labels"]
            # decoder_input_ids = shift_tokens_right(tgt_ids, pad_token_id)
            y = batch["labels"]
            decoder_input_ids = y[:, :-1].contiguous()
            tgt_ids = y[:, 1:].clone()
        if not self.already_saved_batch:  # This would be slightly better if it only happened on rank zero
            batch["decoder_input_ids"] = decoder_input_ids
            self.save_readable_batch(batch)

        outputs = self(src_ids, attention_mask=src_mask, decoder_input_ids=decoder_input_ids, use_cache=False)
        lm_logits = outputs[0]
        if self.hparams.label_smoothing == 0:
            # Same behavior as modeling_bart.py, besides ignoring pad_token_id
            ce_loss_fct = torch.nn.CrossEntropyLoss(ignore_index=pad_token_id)

            assert lm_logits.shape[-1] == self.vocab_size
            loss = ce_loss_fct(lm_logits.view(-1, lm_logits.shape[-1]), tgt_ids.view(-1))
        else:
            lprobs = torch.nn.functional.log_softmax(lm_logits, dim=-1)
            loss, nll_loss = label_smoothed_nll_loss(
                lprobs, tgt_ids, self.hparams.label_smoothing, ignore_index=pad_token_id
            )
        return (loss,)

    @property
    def pad(self) -> int:
        return self.tokenizer.pad_token_id

    def training_step(self, batch, batch_idx) -> Dict:
        loss_tensors = self._step(batch)

        logs = {name: loss for name, loss in zip(self.loss_names, loss_tensors)}
        # tokens per batch
        logs["tpb"] = batch["input_ids"].ne(self.pad).sum() + batch["labels"].ne(self.pad).sum()
        logs["bs"] = batch["input_ids"].shape[0]
        logs["src_pad_tok"] = batch["input_ids"].eq(self.pad).sum()
        logs["src_pad_frac"] = batch["input_ids"].eq(self.pad).float().mean()
        # TODO(SS): make a wandb summary metric for this
        return {"loss": loss_tensors[0], "log": logs}

    def validation_step(self, batch, batch_idx) -> Dict:
        return self._generative_step(batch)

    def validation_epoch_end(self, outputs, prefix="val") -> Dict:
        self.step_count += 1
        losses = {k: torch.stack([x[k] for x in outputs]).mean() for k in self.loss_names}
        loss = losses["loss"]
        if self.cal_metrics:
            generative_metrics = {
                k: np.array([x[k] for x in outputs]).mean() for k in self.metric_names + ["gen_time", "gen_len"]
            }
            metric_val = (
                generative_metrics[self.val_metric] if self.val_metric in generative_metrics else losses[
                    self.val_metric]
            )
            metric_tensor: torch.FloatTensor = torch.tensor(metric_val).type_as(loss)
            generative_metrics.update({k: v.item() for k, v in losses.items()})
            losses.update(generative_metrics)
            all_metrics = {f"{prefix}_avg_{k}": x for k, x in losses.items()}
            all_metrics["step_count"] = self.step_count
            if self.cal_metrics:
                self.metrics[prefix].append(all_metrics)  # callback writes this to self.metrics_save_path
        # print(outputs[0])
        preds = flatten_list([x["preds"] for x in outputs])

        # val_outputs_folder = "val_outputs"
        # os.system("mkdir -p " + os.path.join(self.hparams.output_dir, val_outputs_folder))

        if "preds" in outputs[0]:
            # tb_all = {}
            # idx_tb = 0
            # for output_batch in outputs:
            # a, b, c, e = output_batch["a"], output_batch["b"], output_batch["c"], output_batch["e"]
            #
            # for aa, bb, ee, cc in zip(a, b, e, c):
            #     tb_all[idx_tb] = {}
            #     tb_all[idx_tb]['input_ids'] = aa
            #     tb_all[idx_tb]['labels'] = bb
            #     tb_all[idx_tb]['decoder_input_ids'] = ee
            #     tb_all[idx_tb]['generated_ids'] = cc
            #     idx_tb += 1

            # file_debug = os.path.join(self.hparams.output_dir, val_outputs_folder,
            #                           "debug_" +
            #                           str(self.step_count) + ".json")
            # save_json(tb_all, file_debug)
            if self.hparams.save_file_name:
                output_test_predictions_file = os.path.join(self.hparams.output_dir, self.hparams.save_file_name)
                # output_test_predictions_detok_file = os.path.join(self.hparams.output_dir, val_outputs_folder,
                #                                                   "validation_predictions_detok_" +
                #                                                   str(self.step_count) + ".txt")
                # output_test_targets_file = os.path.join(self.hparams.output_dir, val_outputs_folder, "validation_targets_" +
                #                                         str(self.step_count) + ".txt")
                # write predictions and targets for later rouge evaluation.

                with open(output_test_predictions_file, "w") as p_writer:
                    # , open(output_test_predictions_detok_file, "w") as pd_writer, \
                    # open(output_test_targets_file, "w") as t_writer:
                    for output_batch in outputs:
                        p_writer.writelines(convert_text(s) + "\n" for s in output_batch["preds"])
                        # pd_writer.writelines(s + "\n" for s in output_batch["preds"])
                        # t_writer.writelines(convert_text(s) + "\n" for s in output_batch["target"])
                    p_writer.close()
                    # pd_writer.close()
                    # t_writer.close()

            # bleu_info = eval_bleu_sents(output_test_targets_file, output_test_predictions_file)
            # bleu_info_data = eval_bleu_sents_tok(output_test_predictions_detok_file, self.hparams.data_dir, 'val')
            # meteor_info = eval_meteor(output_test_targets_file, output_test_predictions_file)
            # chrf_info = eval_chrf(output_test_targets_file, output_test_predictions_file)
            #
            # rank_zero_info("number epoch: %s", self.step_count)
            # rank_zero_info("%s bleu_info: %s", self.step_count, bleu_info)
            # rank_zero_info("%s bleu_info_data: %s", self.step_count, bleu_info_data)
            # rank_zero_info("%s meteor_info: %s", self.step_count, meteor_info)
            # rank_zero_info("%s chrf_info: %s", self.step_count, chrf_info)

            # exit()

        return {
            # "log": all_metrics if self.cal_metrics else {},
            "preds": preds,
            # f"{prefix}_loss": loss,
            # f"{prefix}_{self.val_metric}": metric_tensor if self.cal_metrics else {},
        }

    def calc_generative_metrics(self, preds, target) -> Dict:
        return calculate_rouge(preds, target)

    def _generative_step(self, batch: dict) -> dict:
        t0 = time.time()

        generated_ids = self.model.generate(
            batch["input_ids"],
            attention_mask=batch["attention_mask"],
            use_cache=True,
            decoder_start_token_id=self.decoder_start_token_id,
            num_beams=self.eval_beams,
            no_repeat_ngram_size=0,
            min_length=0,
            max_length=self.eval_max_length,
            length_penalty=1.0
        )
        gen_time = (time.time() - t0) / batch["input_ids"].shape[0]
        preds: List[str] = self.ids_to_clean_text(generated_ids)
        target: List[str] = self.ids_to_clean_text(batch["labels"])

        y = batch["labels"]
        decoder_input_ids = y[:, :-1].contiguous()
        lm_labels = y[:, 1:].clone()

        a = self.tokenizer.batch_decode(batch["input_ids"].tolist())
        # b = self.tokenizer.batch_decode(batch["labels"].tolist())
        b = self.tokenizer.batch_decode(lm_labels.tolist())
        c = self.tokenizer.batch_decode(generated_ids)
        pad_token_id = self.tokenizer.pad_token_id
        tgt_ids = batch["labels"]
        # if isinstance(self.model, T5ForConditionalGeneration):
        #     decoder_input_ids = self.model._shift_right(tgt_ids)
        # else:
        #     decoder_input_ids = shift_tokens_right(tgt_ids, pad_token_id)

        e = self.tokenizer.batch_decode(decoder_input_ids.tolist())

        loss_tensors = self._step(batch)
        base_metrics = {name: loss for name, loss in zip(self.loss_names, loss_tensors)}
        base_metrics.update(preds=preds)
        if self.cal_metrics:
            rouge: Dict = self.calc_generative_metrics(preds, target)
            summ_len = np.mean(lmap(len, generated_ids))
            base_metrics.update(gen_time=gen_time, gen_len=summ_len, target=target, a=a, b=b, c=c, e=e,
                                **rouge)

        return base_metrics

    def test_step(self, batch, batch_idx):
        return self._generative_step(batch)

    def test_epoch_end(self, outputs):
        return self.validation_epoch_end(outputs, prefix="test")

    def get_dataset(self, type_path) -> Seq2SeqDataset:
        n_obs = self.n_obs[type_path]
        max_target_length = self.target_lens[type_path]
        dataset = self.dataset_class(
            self.tokenizer,
            type_path=type_path,
            # n_obs=n_obs,
            max_target_length=max_target_length,
            **self.dataset_kwargs,
        )
        return dataset

    def get_dataloader(self, type_path: str, batch_size: int, shuffle: bool = False) -> DataLoader:
        dataset = self.get_dataset(type_path)

        if self.hparams.sortish_sampler and type_path != "test":
            sampler = dataset.make_sortish_sampler(batch_size, distributed=self.hparams.gpus > 1)
            return DataLoader(
                dataset,
                batch_size=batch_size,
                collate_fn=dataset.collate_fn,
                shuffle=False,
                num_workers=self.num_workers,
                sampler=sampler,
            )

        elif self.hparams.max_tokens_per_batch is not None and type_path != "test":
            batch_sampler = dataset.make_dynamic_sampler(
                self.hparams.max_tokens_per_batch, distributed=self.hparams.gpus > 1
            )
            return DataLoader(
                dataset,
                batch_sampler=batch_sampler,
                collate_fn=dataset.collate_fn,
                # shuffle=False,
                num_workers=self.num_workers,
                # batch_size=None,
            )
        else:
            return DataLoader(
                dataset,
                batch_size=batch_size,
                collate_fn=dataset.collate_fn,
                shuffle=shuffle,
                num_workers=self.num_workers,
                sampler=None,
            )

    def train_dataloader(self) -> DataLoader:
        dataloader = self.get_dataloader("train", batch_size=self.hparams.train_batch_size, shuffle=True)
        return dataloader

    def val_dataloader(self) -> DataLoader:
        return self.get_dataloader("val", batch_size=self.hparams.eval_batch_size)

    def test_dataloader(self) -> DataLoader:
        return self.get_dataloader("test", batch_size=self.hparams.eval_batch_size)

    @staticmethod
    def add_model_specific_args(parser, root_dir=None):
        BaseTransformer.add_model_specific_args(parser, root_dir)
        add_generic_args(parser, root_dir)
        parser.add_argument(
            "--max_source_length",
            default=1024,
            type=int,
            help="The maximum total input sequence length after tokenization. Sequences longer "
                 "than this will be truncated, sequences shorter will be padded.",
        )
        parser.add_argument(
            "--max_target_length",
            default=56,
            type=int,
            help="The maximum total input sequence length after tokenization. Sequences longer "
                 "than this will be truncated, sequences shorter will be padded.",
        )
        parser.add_argument(
            "--val_max_target_length",
            default=384,  # these defaults are optimized for CNNDM. For xsum, see README.md.
            type=int,
            help="The maximum total input sequence length after tokenization. Sequences longer "
                 "than this will be truncated, sequences shorter will be padded.",
        )
        parser.add_argument(
            "--test_max_target_length",
            default=384,
            type=int,
            help="The maximum total input sequence length after tokenization. Sequences longer "
                 "than this will be truncated, sequences shorter will be padded.",
        )
        parser.add_argument("--freeze_encoder", action="store_true")
        parser.add_argument("--freeze_embeds", action="store_true")
        parser.add_argument("--sortish_sampler", action="store_true", default=False)
        parser.add_argument("--max_tokens_per_batch", type=int, default=None)
        parser.add_argument("--logger_name", type=str, choices=["default", "wandb", "wandb_shared"], default="default")
        parser.add_argument("--n_train", type=int, default=-1, required=False, help="# examples. -1 means use all.")
        parser.add_argument("--n_val", type=int, default=-1, required=False, help="# examples. -1 means use all.")
        parser.add_argument("--n_test", type=int, default=-1, required=False, help="# examples. -1 means use all.")
        parser.add_argument(
            "--task", type=str, default="summarization", required=False, help="# examples. -1 means use all."
        )
        parser.add_argument("--label_smoothing", type=float, default=0.0, required=False)
        parser.add_argument("--src_lang", type=str, default="", required=False)
        parser.add_argument("--tgt_lang", type=str, default="", required=False)
        parser.add_argument("--eval_beams", type=int, default=None, required=False)
        parser.add_argument("--checkpoint", type=str, default=None, required=False)
        parser.add_argument(
            "--val_metric", type=str, default=None, required=False, choices=["bleu", "rouge2", "loss", None]
        )
        parser.add_argument("--eval_max_gen_length", type=int, default=None, help="never generate more than n tokens")
        parser.add_argument("--save_top_k", type=int, default=1, required=False, help="How many checkpoints to save")
        parser.add_argument(
            "--early_stopping_patience",
            type=int,
            default=-1,
            required=False,
            help="-1 means never early stop. early_stopping_patience is measured in validation checks, not epochs. So val_check_interval will effect it.",
        )
        parser.add_argument("--save_file_name", type=str)
        return parser


class TranslationModule(SummarizationModule):
    mode = "translation"
    loss_names = ["loss"]
    metric_names = ["bleu"]
    default_val_metric = "bleu"

    def __init__(self, hparams, **kwargs):
        super().__init__(hparams, **kwargs)
        self.dataset_kwargs["src_lang"] = hparams.src_lang
        self.dataset_kwargs["tgt_lang"] = hparams.tgt_lang

    def calc_generative_metrics(self, preds, target) -> dict:
        return calculate_bleu(preds, target)


class Graph2TextModule(SummarizationModule):
    mode = "graph2text"
    loss_names = ["loss"]
    metric_names = ["bleu"]
    default_val_metric = "bleu"

    def __init__(self, hparams, **kwargs):
        super().__init__(hparams, **kwargs)

        rank_zero_info("parameters %s", hparams)

    def calc_generative_metrics(self, preds, target) -> dict:
        return calculate_bleu(preds, target)


def main(args, model=None) -> SummarizationModule:
    Path(args.output_dir).mkdir(exist_ok=True)
    if len(os.listdir(args.output_dir)) > 3 and args.do_train:
        raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    if model is None:
        model: SummarizationModule = Graph2TextModule(args)
    dataset = Path(args.data_dir).name
    if (
            args.logger_name == "default"
            or args.fast_dev_run
            or str(args.output_dir).startswith("/tmp")
            or str(args.output_dir).startswith("/var")
    ):
        logger = False  # don't pollute wandb logs unnecessarily
    elif args.logger_name == "wandb":
        from pytorch_lightning.loggers import WandbLogger

        project = os.environ.get("WANDB_PROJECT", dataset)
        logger = WandbLogger(name=model.output_dir.name, project=project)

    elif args.logger_name == "wandb_shared":
        from pytorch_lightning.loggers import WandbLogger

        logger = WandbLogger(name=model.output_dir.name, project=f"hf_{dataset}")

    if args.early_stopping_patience >= 0:
        es_callback = get_early_stopping_callback(model.val_metric, args.early_stopping_patience)
    else:
        es_callback = False

    lower_is_better = args.val_metric == "loss"
    trainer: pl.Trainer = generic_train(
        model,
        args,
        # logging_callback=Seq2SeqLoggingCallback(),
        checkpoint_callback=get_checkpoint_callback(
            args.output_dir, model.val_metric, args.save_top_k, lower_is_better
        ),
        early_stopping_callback=es_callback,
        logger=logger,
    )
    # pickle_save(model.hparams, model.output_dir / "hparams.pkl")
    if not args.do_predict:
        return model

    model.hparams.test_checkpoint = ""
    if not args.checkpoint:
        checkpoints = list(sorted(glob.glob(os.path.join(args.output_dir, "*.ckpt"), recursive=True)))
    else:
        checkpoints = [args.checkpoint]

    if checkpoints:
        model.hparams.test_checkpoint = checkpoints[-1]
        trainer.resume_from_checkpoint = checkpoints[-1]

        if args.do_predict and not args.do_train:
            checkpoint = checkpoints[-1]
            # print(checkpoint)
            # trainer.test(ckpt_path=checkpoints[-1])
            trainer.test(model, ckpt_path=checkpoint)
            return model

    trainer.logger.log_hyperparams(model.hparams)

    # test() without a model tests using the best checkpoint automatically
    trainer.test()
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser = SummarizationModule.add_model_specific_args(parser, os.getcwd())

    args = parser.parse_args()

    main(args)
