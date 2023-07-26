import argparse
import random

import numpy as np
import torch

import pprint
import yaml


feature_types = {
    "RN50", "RN101", "RN50x4", "ViT", "butd", "raw_RN50", "raw_RN101", "raw_RN50x4", "raw_ViT"
}


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def is_interactive():
    import __main__ as main
    return not hasattr(main, '__file__')


def get_optimizer(optim, verbose=False):
    # Bind the optimizer
    if optim == 'rms':
        if verbose:
            print("Optimizer: Using RMSProp")
        optimizer = torch.optim.RMSprop
    elif optim == 'adam':
        if verbose:
            print("Optimizer: Using Adam")
        optimizer = torch.optim.Adam
    elif optim == 'adamw':
        if verbose:
            print("Optimizer: Using AdamW")
        # optimizer = torch.optim.AdamW
        optimizer = 'adamw'
    elif optim == 'adamax':
        if verbose:
            print("Optimizer: Using Adamax")
        optimizer = torch.optim.Adamax
    elif optim == 'sgd':
        if verbose:
            print("Optimizer: SGD")
        optimizer = torch.optim.SGD
    else:
        assert False, "Please add your optimizer %s in the list." % optim

    return optimizer


def parse_args(parse=True, **optional_kwargs):
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=9595, help='random seed')

    # Data Splits
    parser.add_argument("--train", default='train')
    parser.add_argument("--valid", default='valid')
    parser.add_argument("--test", default=None)
    parser.add_argument('--test_only', action='store_true')

    parser.add_argument('--submit', action='store_true')

    # Quick experiments
    parser.add_argument('--train_topk', type=float, default=-1)
    parser.add_argument('--valid_topk', type=float, default=-1)

    # Checkpoint
    parser.add_argument('--output', type=str, default='snap/test')
    parser.add_argument('--load', type=str, default=None, help='Load the model (usually the fine-tuned model).')
    parser.add_argument('--load_lxmert_qa', type=str, default=None)
    parser.add_argument('--from_scratch', action='store_true')
    parser.add_argument('--run_name', type=str, default="")

    # CPU/GPU
    parser.add_argument("--multiGPU", action='store_const', default=False, const=True)
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument("--distributed", action='store_true')
    parser.add_argument("--num_workers", default=0, type=int)
    parser.add_argument('--local_rank', type=int, default=-1)

    # Model Config
    parser.add_argument('--backbone', type=str, default='t5-base')
    parser.add_argument('--tokenizer', type=str, default=None)

    parser.add_argument('--feat_dim', type=float, default=2048)
    parser.add_argument('--pos_dim', type=float, default=4)
    parser.add_argument('--image_size', type=str, default="(448,448)")

    parser.add_argument('--use_vision', default=True, type=str2bool)
    parser.add_argument('--use_vis_order_embedding', default=True, type=str2bool)
    parser.add_argument('--use_vis_layer_norm', default=True, type=str2bool)
    parser.add_argument('--individual_vis_layer_norm', default=True, type=str2bool)
    parser.add_argument('--share_vis_lang_layer_norm', action='store_true')

    parser.add_argument('--n_boxes', type=int, default=36)
    parser.add_argument('--max_n_boxes', type=int, default=36)
    parser.add_argument('--max_text_length', type=int, default=20)
    
    parser.add_argument('--additional_visual_embedding_layers', type=int, default=0)

    parser.add_argument('--downsample', action="store_true")
    parser.add_argument('--oneddownsample', action="store_true")
    parser.add_argument('--expand_vis_embedding', action="store_true")
    parser.add_argument('--n_image_tokens', type=int, default=4)
    parser.add_argument('--vis_use_transformer', action="store_true")

    # Training
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--valid_batch_size', type=int, default=None)
    parser.add_argument('--optim', default='adamw')
    parser.add_argument('--warmup_ratio', type=float, default=0.05)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--clip_grad_norm', type=float, default=-1.0)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--vis_lr', type=float, default=1e-4)
    parser.add_argument('--vis_weight_decay', type=float, default=0.01)
    parser.add_argument('--adam_eps', type=float, default=1e-6)
    parser.add_argument('--adam_beta1', type=float, default=0.9)
    parser.add_argument('--adam_beta2', type=float, default=0.999)
    parser.add_argument('--epochs', type=int, default=12)
    parser.add_argument('--dropout', type=float, default=0.1)

    parser.add_argument("--losses", default='lm,obj,attr,feat', type=str)

    parser.add_argument('--log_train_accuracy', action='store_true')

    parser.add_argument('--n_ground', type=int, default=1)
    parser.add_argument("--wordMaskRate", dest='word_mask_rate', default=0.15, type=float)
    parser.add_argument("--objMaskRate", dest='obj_mask_rate',default=0.15, type=float)

    parser.add_argument('--encoder_prompt_len', type=int, default=0)
    parser.add_argument('--decoder_prompt_len', type=int, default=0)
    parser.add_argument('--use_single_prompt', action="store_true")
    parser.add_argument('--use_attn_prefix', action="store_true")
    parser.add_argument('--mid_dim', type=int, default=768)
    parser.add_argument('--use_adapter', action="store_true")
    parser.add_argument('--use_hyperformer', action="store_true")
    parser.add_argument('--use_compacter', action="store_true")
    parser.add_argument('--use_lradapter', action="store_true")
    parser.add_argument('--use_single_adapter', action="store_true")
    parser.add_argument('--efficient_unique_hyper_net', action="store_true")
    parser.add_argument('--unique_hyper_net', action="store_true")
    parser.add_argument('--unfreeze_vis_encoder', action="store_true")
    parser.add_argument('--unfreeze_vis_last_layer', action="store_true")
    parser.add_argument('--unfreeze_batch_norms', action="store_true")
    parser.add_argument('--projected_task_embedding_dim', default=-1, type=int, 
        help="projected_task_embedding_dim for hyperformer, -1 means using the default value in the config"
    )

    parser.add_argument('--share_down_sampler', action="store_true")
    parser.add_argument('--share_up_sampler', action="store_true")
    # Compacter
    parser.add_argument('--hypercomplex_division', type=int, default=4)
    parser.add_argument('--phm_rank', type=int, default=1)
    parser.add_argument('--shared_phm_rule', type=str2bool, default=True)
    parser.add_argument('--factorized_phm', type=str2bool, default=True)
    parser.add_argument('--add_adapter_cross_attn', type=str2bool, default=True)
    parser.add_argument('--low_rank_rank', type=int, default=1)
    parser.add_argument('--phm_init_range', type=float, default=0.01)
    parser.add_argument('--shared_phm_rule_over_tasks', action="store_true")

    parser.add_argument('--vis_pooling_output', action="store_true")
    parser.add_argument('--use_vis_adapter', action="store_true")
    parser.add_argument('--use_separate_optimizer_for_visual', action="store_true")
    parser.add_argument(
        '--use_adam_for_visual', action="store_true", help="Use SGD if false"
    )
    parser.add_argument('--freeze_ln_statistics', action="store_true")
    parser.add_argument('--freeze_bn_statistics', action="store_true")
    parser.add_argument('--add_layer_norm_before_adapter', action="store_true")
    parser.add_argument('--add_layer_norm_after_adapter', action="store_true")

    parser.add_argument('--vis_adapter_type', type=str, default="middle-bottleneck")
    parser.add_argument('--vis_reduction_factor', type=int, default=2)
    parser.add_argument('--reduction_factor', type=int, default=16)
    parser.add_argument('--use_data_augmentation', action="store_true")
    parser.add_argument('--deepspeed', type=str, default=None)
    parser.add_argument('--sparse_sample', action="store_true")
    parser.add_argument('--remove_bn_vis_adapter', action="store_true")
    parser.add_argument('--unfreeze_lm_head', action="store_true")
    parser.add_argument('--use_lm_head_adapter', action="store_true")


    # use lora
    parser.add_argument('--use_lora', action="store_true")
    parser.add_argument('--lora_dim', type=int, default=4)
    parser.add_argument('--lora_alpha', type=float, default=32)
    parser.add_argument('--use_single_lora', action="store_true")

    # Inference
    parser.add_argument('--num_beams', type=int, default=1)
    parser.add_argument('--gen_max_length', type=int, default=20)

    # Data
    parser.add_argument('--caption_only', action='store_true')
    parser.add_argument('--coco_only', action='store_true')
    parser.add_argument('--caption_cocoonly', default=True, type=str2bool)

    parser.add_argument('--do_lower_case', action='store_true')
    parser.add_argument('--oscar_tags', action='store_true')

    parser.add_argument('--prefix', type=str, default=None)

    parser.add_argument('--prompt', type=str, default="vqa: ")
    parser.add_argument('--post_prompt', type=str, default="")

    parser.add_argument('--feature_type', type=str, default="butd", choices=feature_types)

    # Pretraining
    parser.add_argument('--ground_upsample', type=int, default=1)
    parser.add_argument('--ground_weight', type=int, default=1)
    parser.add_argument('--itm_cocoonly', default=True, type=str2bool)
    parser.add_argument('--single_vqa_prefix', action='store_true')

    # COCO Caption
    parser.add_argument('--no_prefix', action='store_true')

    # VQA
    parser.add_argument("--raw_label", action='store_true')
    parser.add_argument("--answer_normalize", action='store_true')
    parser.add_argument("--classifier", action='store_true')
    parser.add_argument("--test_answerable", action='store_true')

    # Multitask
    parser.add_argument("--multitask_sampling", type=str, default='roundrobin')
    parser.add_argument("--tasks", type=str, default='')
    parser.add_argument("--use_tasks_prompts", action="store_true")
    parser.add_argument("--testing", action="store_true")
    parser.add_argument("--track_z", action="store_true")
    parser.add_argument("--lambda_z", type=float, default=0.001)

    # Etc.
    parser.add_argument('--comment', type=str, default='')
    parser.add_argument("--dry", action='store_true')


    parser.add_argument('--unfreeze_language_model', action="store_true")
    parser.add_argument('--unfreeze_layer_norms', action="store_true")

    # newly add 
    parser.add_argument('--no_decoder_adapter', action="store_true") # no_adapter_decoder
    parser.add_argument('--unfreeze_encoder_layer_norms', action="store_true")
    parser.add_argument('--unfreeze_decoder_layer_norms', action="store_true")
    parser.add_argument('--unfreeze_decoder_input_layer_norms', action="store_true")
    parser.add_argument('--unfreeze_decoder_self_attn_layer_norms', action="store_true")
    parser.add_argument('--unfreeze_decoder_encoder_attn_layer_norms', action="store_true")
    parser.add_argument('--unfreeze_decoder_ff_layer_norms', action="store_true")
    parser.add_argument('--unfreeze_bias', action="store_true")
    parser.add_argument('--unfreeze_encoder_bias', action="store_true")
    parser.add_argument('--unfreeze_decoder_bias', action="store_true")

    parser.add_argument('--use_encoder_adapter_gating_large_x', action="store_true") 

    parser.add_argument('--use_encoder_adapter_gating_large_x_lowrank', action="store_true") 
    parser.add_argument('--adapter_gating_down_dim', type=int, default=96) # adapter_gating_dim, for gating lowrank

    parser.add_argument('--use_encoder_gating_large_x_lowrank', action="store_true") 
    parser.add_argument('--gating_down_dim', type=int, default=96) # for gating_only lowrank
    parser.add_argument('--use_encoder_gating_large_x_lowrank_add_x2_deltay', action="store_true")

    parser.add_argument('--use_adapter_down_dim', action="store_true")
    parser.add_argument('--adapter_down_dim', type=int, default=96) 

    parser.add_argument('--use_decoder_enc_attn_value_parallel_adapter_down_dim', action="store_true")
    parser.add_argument('--decoder_enc_attn_value_parallel_adapter_down_dim', type=int, default=96) 

    parser.add_argument('--use_decoder_enc_attn_key_parallel_adapter_down_dim', action="store_true")
    parser.add_argument('--decoder_enc_attn_key_parallel_adapter_down_dim', type=int, default=96) 

    parser.add_argument('--use_decoder_enc_attn_key_value_adapter_down_dim', action="store_true") 
    parser.add_argument('--decoder_enc_attn_key_value_adapter_down_dim', type=int, default=96) 

    parser.add_argument('--use_decoder_enc_attn_adapter_down_dim', action="store_true")  # equals to no_decoder_self_attn_adapter + no_decoder_ff_adapter
    parser.add_argument('--decoder_enc_attn_adapter_down_dim', type=int, default=96) 

    parser.add_argument('--use_decoder_enc_attn_adapter_gating_large_x_lowrank', action="store_true")
    parser.add_argument('--decoder_enc_attn_adapter_gating_large_x_lowrank_down_dim', type=int, default=96)

    parser.add_argument('--use_encoder_adapter_gating_small_xy_cat', action="store_true") 
    parser.add_argument('--use_encoder_adapter_gating_middle_xy_add', action="store_true") 
    parser.add_argument('--use_encoder_adapter_gating_middle_ia3_add', action="store_true") 

    parser.add_argument('--use_encoder_adapter_gating_layernorm', action="store_true")
    parser.add_argument('--use_encoder_adapter_gating_l2norm', action="store_true")
    
    # replace element-wise product with element-wise add for encoder gating
    parser.add_argument('--use_encoder_adapter_gating_add', action="store_true")

    # change parallel Adapter to sequential Adapter
    parser.add_argument('--use_decoder_enc_attn_value_sequential_adapter_down_dim', action="store_true")
    parser.add_argument('--decoder_enc_attn_value_sequential_adapter_down_dim', type=int, default=96) 

    parser.add_argument('--use_decoder_enc_attn_value_residual_connection', action="store_true")

    parser.add_argument('--use_decoder_enc_attn_value_sequential_adapter_gating_large_x_lowrank', action="store_true")
    parser.add_argument('--decoder_enc_attn_value_sequential_adapter_gating_large_x_lowrank_down_dim', type=int, default=96)

    parser.add_argument('--use_decoder_enc_attn_value_parallel_adapter_gating_large_x_lowrank', action="store_true")
    parser.add_argument('--decoder_enc_attn_value_parallel_adapter_gating_large_x_lowrank_down_dim', type=int, default=96)

    parser.add_argument('--use_decoder_enc_attn_value_ia3', action="store_true")
    parser.add_argument('--use_decoder_enc_attn_value_ia3_add', action="store_true")
    parser.add_argument('--use_decoder_enc_attn_value_ia3_one_init', action="store_true")
    parser.add_argument('--use_decoder_self_attn_value_ia3', action="store_true")
    parser.add_argument('--use_decoder_self_attn_value_ia3_add', action="store_true")
    parser.add_argument('--use_decoder_self_attn_value_ia3_one_init', action="store_true")
    parser.add_argument('--use_decoder_ff_ia3', action="store_true")
    parser.add_argument('--use_decoder_ff_ia3_add', action="store_true")
    parser.add_argument('--use_decoder_ff_ia3_one_init', action="store_true")

    parser.add_argument('--no_encoder_attn_adapter', action="store_true") 
    parser.add_argument('--use_encoder_attn_value_parallel_adapter_down_dim', action="store_true")
    parser.add_argument('--encoder_attn_value_parallel_adapter_down_dim', type=int, default=96)

    parser.add_argument('--use_encoder_attn_value_ia3', action="store_true")
    parser.add_argument('--use_encoder_attn_value_ia3_add', action="store_true")
    parser.add_argument('--use_encoder_attn_value_ia3_one_init', action="store_true")

    parser.add_argument('--use_decoder_enc_attn_value_parallel_adapter_scaling', action="store_true")
    parser.add_argument('--decoder_enc_attn_value_parallel_adapter_scaling_factor', type=float, default=1.0) 
    
    parser.add_argument('--use_encoder_attn_adapter_scaling', action="store_true")
    parser.add_argument('--encoder_attn_adapter_scaling_factor', type=float, default=1.0) 
    parser.add_argument('--use_encoder_ff_adapter_scaling', action="store_true")
    parser.add_argument('--encoder_ff_adapter_scaling_factor', type=float, default=1.0) 

    parser.add_argument('--no_encoder_adapter', action="store_true") 
    parser.add_argument('--use_encoder_adapter_down_multihead', action="store_true")
    parser.add_argument('--use_encoder_adapter_up_multihead', action="store_true")
    parser.add_argument('--use_encoder_adapter_down_up_multihead', action="store_true")
    parser.add_argument('--use_encoder_adapter_down_up_pair_multihead', action="store_true")
    parser.add_argument('--encoder_adapter_multihead_num_head', type=int, default=1)

    parser.add_argument('--use_decoder_adapter_down_multihead', action="store_true") 
    parser.add_argument('--decoder_adapter_multihead_num_head', type=int, default=1)

    parser.add_argument('--use_decoder_enc_attn_value_parallel_adapter_down_multihead', action="store_true")
    parser.add_argument('--use_decoder_enc_attn_value_parallel_adapter_down_up_pair_multihead', action="store_true")
    parser.add_argument('--decoder_enc_attn_value_parallel_adapter_multihead_num_head', type=int, default=1)

    parser.add_argument('--use_decoder_self_attn_value_parallel_adapter_down_dim', action="store_true")
    parser.add_argument('--decoder_self_attn_value_parallel_adapter_down_dim', type=int, default=96) 

    parser.add_argument('--use_decoder_self_attn_adapter_down_dim', action="store_true")
    parser.add_argument('--decoder_self_attn_adapter_down_dim', type=int, default=96) 

    parser.add_argument('--use_decoder_ff_adapter_down_dim', action="store_true")
    parser.add_argument('--decoder_ff_adapter_down_dim', type=int, default=96) 

    parser.add_argument('--use_encoder_gating_scaling', action="store_true")
    parser.add_argument('--encoder_gating_scaling_factor', type=float, default=1.0) 

    parser.add_argument('--use_encoder_adapter_scaling', action="store_true")
    parser.add_argument('--encoder_adapter_scaling_factor', type=float, default=1.0) 

    parser.add_argument('--use_encoder_multihead_up_zero_init', action="store_true")
    parser.add_argument('--use_encoder_gating_large_x_lowrank_up_zero_init', action="store_true") 
    parser.add_argument('--use_decoder_enc_vpa_up_zero_init', action="store_true")

    parser.add_argument('--use_encoder_x2_scaling', action="store_true")
    parser.add_argument('--encoder_x2_scaling_factor', type=float, default=1.0) 

    parser.add_argument('--use_encoder_gating_small_up_zero_init', action="store_true") 
    parser.add_argument('--use_encoder_gating_middle_up_zero_init', action="store_true") 
    parser.add_argument('--use_encoder_gating_middle_ia3_one_init', action="store_true") 
    parser.add_argument('--use_encoder_gating_middle_ia3_zero_init', action="store_true") 

    parser.add_argument('--no_vis', action="store_true")
    parser.add_argument('--freeze_vis_emb', action="store_true")
    parser.add_argument('--use_vis_noise', action="store_true")

    # for LowRank visual projector
    parser.add_argument('--use_lowrank_visual_projector', action="store_true")
    parser.add_argument('--visual_projector_down_dim', type=int, default=96)
    parser.add_argument('--visual_projector_multihead_num_head', type=int, default=1)
    parser.add_argument('--use_visual_projector_gating_large_x_lowrank', action="store_true") 
    parser.add_argument('--visual_projector_gating_down_dim', type=int, default=96) 
    parser.add_argument('--use_visual_projector_residual_connection', action="store_true") 

    # for heatmap, store gate
    parser.add_argument('--use_store_gate_large', action="store_true")
    parser.add_argument("--store_gate_path", type=str)
    parser.add_argument('--infer_only', action="store_true")
    parser.add_argument("--load_trained_model_path", type=str)


    # Parse the arguments.
    if parse:
        args = parser.parse_args()
    # For interative engironmnet (ex. jupyter)
    else:
        args = parser.parse_known_args()[0]

    # Namespace => Dictionary
    kwargs = vars(args)
    kwargs.update(optional_kwargs)

    args = Config(**kwargs)

    # Bind optimizer class.
    verbose = False
    args.optimizer = get_optimizer(args.optim, verbose=verbose)

    # Set seeds
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    return args


class Config(object):
    def __init__(self, **kwargs):
        """Configuration Class: set kwargs as class attributes with setattr"""
        for k, v in kwargs.items():
            setattr(self, k, v)

    @property
    def config_str(self):
        return pprint.pformat(self.__dict__)

    def __repr__(self):
        """Pretty-print configurations in alphabetical order"""
        config_str = 'Configurations\n'
        config_str += self.config_str
        return config_str

    def save(self, path):
        with open(path, 'w') as f:
            yaml.dump(self.__dict__, f, default_flow_style=False)

    @classmethod
    def load(cls, path):
        with open(path, 'r') as f:
            kwargs = yaml.load(f)

        return Config(**kwargs)


if __name__ == '__main__':
    args = parse_args(True)
