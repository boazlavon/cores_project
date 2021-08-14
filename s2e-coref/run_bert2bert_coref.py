from __future__ import absolute_import, division, print_function

import logging
import os
import shutil
import git
import torch

from transformers import AutoConfig, AutoTokenizer, CONFIG_MAPPING, LongformerConfig, BertConfig, BertTokenizer
from transformers import BertGenerationConfig, BertGenerationEncoder, BertGenerationDecoder, EncoderDecoderModel, EncoderDecoderConfig

from modeling import S2E
from data import get_dataset
from cores_tokens import get_cores_tokens, encode, W, C
from bert2bert_cli import parse_args
from training import train, set_seed
from eval import Evaluator
from utils import write_meta_data

logger = logging.getLogger(__name__)
def main():
    args = parse_args()

    transformers_logger = logging.getLogger("transformers")
    transformers_logger.setLevel(logging.ERROR)

    if args.predict_file is None and args.do_eval:
        raise ValueError(
            "Cannot do evaluation without an evaluation data file. Either supply a file to --eval_data_file "
            "or remove the --do_eval argument.")

    if args.output_dir and os.path.exists(args.output_dir) and \
            os.listdir(args.output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir))

    if args.overwrite_output_dir and os.path.isdir(args.output_dir):
        shutil.rmtree(args.output_dir)
    os.mkdir(args.output_dir)

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)

    with open(os.path.join(args.output_dir, 'args.txt'), 'w') as f:
        f.write(str(args))

    for key, val in vars(args).items():
        logger.info(f"{key} - {val}")

    try:
        write_meta_data(args.output_dir, args)
    except git.exc.InvalidGitRepositoryError:
        logger.info("didn't save metadata - No git repo!")


    logger.info("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, amp training: %s",
                args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.amp)

    # Set seed
    set_seed(args)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        # Barrier to make sure only the first process in distributed training download model & vocab
        torch.distributed.barrier()

    if args.config_name:
        config = AutoConfig.from_pretrained(args.config_name, cache_dir=args.cache_dir)
    elif args.model_name_or_path:
        config = AutoConfig.from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir)
    else:
        config = CONFIG_MAPPING[args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")

    cores_tokens = get_cores_tokens()
    logger.info(cores_tokens)
    if args.model_name_or_path and os.path.isdir(args.model_name_or_path):
        logger.info("Loading pre-trained tokenizer")
        tokenizer_path = os.path.join(args.model_name_or_path, 'tokenizer')
        tokenizer = BertTokenizer.from_pretrained(tokenizer_path, cache_dir=args.cache_dir)
        logger.info("pre-trained: tokenizer: {}".format(len(tokenizer)))
    elif args.tokenizer_name:
        logger.info("Building new tokenizer")
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, cache_dir=args.cache_dir)
        logger.info("tokenizer: {}".format(len(tokenizer)))
        tokenizer.add_special_tokens({'additional_special_tokens' : cores_tokens})
        logger.info("add special tokens: {}".format(str({'additional_special_tokens' : cores_tokens})))
        logger.info("tokenizer: {}".format(len(tokenizer)))
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported, but you can do it from another script, save it,"
            "and load it from here, using --tokenizer_name"
        )

    if args.model_name_or_path and os.path.isdir(args.model_name_or_path):
        logger.info("Loading pre-trained bert2bert model")
        model_path = os.path.join(args.model_name_or_path, 'model')
        bert2bert = EncoderDecoderModel.from_pretrained(model_path)
        print("pre-trained bert2bert: {}".format(str(bert2bert.get_input_embeddings())))
    else:
        logger.info("Building new bert2bert model")
        logger.info("Building BERT Encoder")
        encoder = BertGenerationEncoder.from_pretrained(args.config_name, cache_dir=args.cache_dir)
        logger.info("Buuilding BERT Decoder")
        decoder = BertGenerationDecoder.from_pretrained(args.config_name, add_cross_attention=True, is_decoder=True, cache_dir=args.cache_dir)
        # VERY IMPORTANT STEP!
        logger.info("resize token embeddings")
        encoder.resize_token_embeddings(len(tokenizer))
        decoder.resize_token_embeddings(len(tokenizer))
        logger.info("Building BERT to BERT model")
        bert2bert = EncoderDecoderModel(encoder=encoder, decoder=decoder)
        print("new bert2bert: {}".format(str(bert2bert.get_input_embeddings())))
    bert2bert.to(args.device)

    if True:
        logger.info("Running Training Example")
        input_example = ' '.join(W)
        _, output_example = encode(W,C,None)
    
        input_ids = tokenizer(input_example, add_special_tokens=True, return_tensors="pt").input_ids.to(args.device)
        logger.info("Input: str: {}\n IDs: {}".format(input_example, str(input_ids)))
        output_ids = tokenizer(output_example, return_tensors="pt").input_ids.to(args.device)
        logger.info("Output: str: {}\n IDs: {}".format(output_example, str(output_ids)))

        # train...
        loss = bert2bert(input_ids=input_ids, decoder_input_ids=output_ids, labels=output_ids)
        logger.info("Loss: {}".format(str(loss)))
        loss[0].backward()
        logger.info("Loss[0]: Backward... ")
        
        logger.info("Generate...")
        mask = torch.ones(input_ids.shape).to(args.device)
        outputs = bert2bert.generate(input_ids, attention_mask=mask)
        output_str = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        logger.info("Output: str: {}\n IDs: {}".format(output_str, str(outputs)))

    logger.info("Exit")
    exit(0)
    if args.local_rank == 0:
        # End of barrier to make sure only the first process in distributed training download model & vocab
        torch.distributed.barrier()

    logger.info("Training/evaluation parameters %s", args)

    evaluator = Evaluator(args, tokenizer)
    # Training
    if args.do_train:
        train_dataset = get_dataset(args, tokenizer, evaluate=False)

        global_step, tr_loss = train(args, train_dataset, model, tokenizer, evaluator)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    # Saving best-practices: if you use save_pretrained for the model and tokenizer,
    # you can reload them using from_pretrained()
    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        # Create output directory if needed
        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir)

        logger.info("Saving model checkpoint to %s", args.output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = model.module if hasattr(model,
                                                'module') else model  # Take care of distributed/parallel training
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(args.output_dir, 'training_args.bin'))

    # Evaluation
    results = {}

    if args.do_eval and args.local_rank in [-1, 0]:
        result = evaluator.evaluate(model, prefix="final_evaluation", official=True)
        results.update(result)
        return results

    return results


if __name__ == "__main__":
    main()
