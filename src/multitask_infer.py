
from trainer_base import TrainerBase
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import os
from pathlib import Path
from packaging import version
from tqdm import tqdm
import torch
import logging
from copy import deepcopy
from param import parse_args
import multitask_data
from utils import LossMeter, set_global_logging_level
import wandb
from vis_encoder import get_vis_encoder
from adapters import AdapterController, MetaLayersAdapterController

proj_dir = Path(__file__).resolve().parent.parent
_use_native_amp = False
_use_apex = False

# Check if Pytorch version >= 1.6 to switch between Native AMP and Apex
if version.parse(torch.__version__) < version.parse("1.6"):
    from transormers.file_utils import is_apex_available
    if is_apex_available():
        from apex import amp
    _use_apex = True
else:
    _use_native_amp = True
    from torch.cuda.amp import autocast

class Trainer(TrainerBase):
    def __init__(self, args, train_loader=None, val_loader=None, test_loader=None, train=True):
        super().__init__(
            args,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            train=train)

        if not self.verbose:
            set_global_logging_level(logging.ERROR, ["transformers"])

        from multitask_model import VLT5MultiTask, VLBartMultiTask

        model_kwargs = {}
        if 't5' in args.backbone:
            model_class = VLT5MultiTask
        elif 'bart' in args.backbone:
            model_class = VLBartMultiTask

        config = self.create_config()
        self.tokenizer = self.create_tokenizer()

        if 'bart' in self.args.tokenizer:
            num_added_toks = 0
            if config.use_vis_order_embedding:
                additional_special_tokens = [f'<extra_id_{i}>' for i in range(100-1, -1, -1)] + \
                        [f'<vis_extra_id_{i}>' for i in range(100-1, -1, -1)]
                special_tokens_dict = {'additional_special_tokens': additional_special_tokens}
                num_added_toks = self.tokenizer.add_special_tokens(special_tokens_dict)

                config.default_obj_order_ids = self.tokenizer.convert_tokens_to_ids([f'<vis_extra_id_{i}>' for i in range(100)])

        self.model = self.create_model(model_class, config, **model_kwargs)

        if 't5' in self.args.tokenizer:
            self.model.resize_token_embeddings(self.tokenizer.vocab_size)
        elif 'bart' in self.args.tokenizer:
            self.model.resize_token_embeddings(self.model.model.shared.num_embeddings + num_added_toks)

        self.model.tokenizer = self.tokenizer
        if 't5' in self.args.tokenizer or 'bart' in self.args.tokenizer:
            self.model.true_id = self.tokenizer('true', add_special_tokens=False).input_ids[0]
            self.model.false_id = self.tokenizer('false', add_special_tokens=False).input_ids[0]

        self.weight_initialization()
        
        # Load Checkpoint
        self.start_epoch = None
        if args.load is not None:
            ckpt_path = args.load + '.pth'
            self.load_checkpoint(ckpt_path)
        if self.args.from_scratch:
            self.init_weights()

        # GPU Options
        print(f'Model Launching at GPU {self.args.gpu}')
        if self.verbose:
            from time import time
            start = time()
        self.model = self.model.to(args.gpu)

        self.freeze_whole_model() # freeze whole parameters first
        self.unfreeze_parameters() # unfreeze selected parameters

        self.percent_updated_parameters = self.print_trainable_params_percentage(self.model)

        # Optimizer
        if train:
            self.optim, self.lr_scheduler = self.create_optimizer_and_scheduler()

            if self.args.fp16 and _use_native_amp:
                self.scaler = torch.cuda.amp.GradScaler()
            elif _use_apex:
                self.model, self.optim = amp.initialize(
                    self.model, self.optim, opt_level='O1', verbosity=self.verbose)

        if args.multiGPU:
            if args.distributed:
                self.model = DDP(self.model, device_ids=[args.gpu],
                                 find_unused_parameters=True
                                 )
        if self.verbose:
            print(f'It took {time() - start:.1f}s')

    def train(self):
        if self.verbose:
            vqa_loss_meter = LossMeter()
            quesid2ans = {}
            best_vqa_valid = 0.
            best_vqa_epoch = 0

            # gqa
            best_gqa_valid = 0
            best_gqa_epoch = 0

            # nlvr
            best_nlvr_valid = 0
            best_nlvr_epoch = 0

            # caption
            best_caption_valid = 0
            best_caption_epoch = 0

            # assert 't5'in self.args.backbone
            # self.setup_wandb()

            if 't5' in self.args.backbone:
                if self.args.use_vision:
                    project_name = "VLT5_multitask"
                else:
                    project_name = "T5_multitask"
            elif 'bart' in self.args.backbone:
                if self.args.use_vision:
                    project_name = "VLBart_multitask"
                else:
                    project_name = "Bart_multitask"

            wandb.init(project=project_name)
            wandb.run.name = self.args.run_name
            wandb.config.update(self.args)
            wandb.watch(self.model)
            wandb.log(
                {"percent of updated parameters (%)": self.percent_updated_parameters}
            )

            src_dir = Path(__file__).resolve().parent
            base_path = str(src_dir.parent)
            src_dir = str(src_dir)
            wandb.save(os.path.join(src_dir + "/*.py"), base_path=base_path)

        if self.args.distributed:
            dist.barrier()

        global_step = 0
        for epoch in range(self.args.epochs):
            if self.start_epoch is not None:
                epoch += self.start_epoch
            self.model.train()
            self.partial_eval()

            if self.args.distributed:
                self.train_loader.set_epoch(epoch)
            if self.verbose:
                pbar = tqdm(total=len(self.train_loader), ncols=250)

            epoch_results = {
                'loss': 0.,

            }

            task_counter = {
                'vqa': 0,
                'gqa': 0,
                'nlvr': 0,
                'caption': 0,
            }

            # vqa
            quesid2ans = {}
            train_acc = 0.
            # train_acc_steps = int(len(self.train_loader) * 0.05)
            # last_acc_step = 0



            for step_i, batch in enumerate(self.train_loader):

                # print(f'GPU{self.args.gpu} inside training loop')
                # print(batch)
                task = batch['task']
                # if self.verbose:
                #     print('task', task)
                task_counter[task] += 1

                batch['log_train_accuracy'] = self.args.log_train_accuracy

                # self.optim.zero_grad()
                if self.args.fp16 and _use_native_amp:
                    with autocast():
                        if self.args.distributed:
                            results = self.model.module.train_step(batch)
                        else:
                            results = self.model.train_step(batch)
                else:
                    if self.args.distributed:
                        results = self.model.module.train_step(batch)
                    else:
                        results = self.model.train_step(batch)

                loss = results['loss']

                if self.args.track_z:
                    reg_loss = 0
                    layer_num = 0
                    for name, sub_module in self.model.named_modules():
                        if isinstance(sub_module, (AdapterController)):
                            reg_loss += ((sub_module.adapters[task].z) ** 2).mean()
                            layer_num += 1

                        if isinstance(sub_module, (MetaLayersAdapterController)):
                            reg_loss += ((sub_module.z) ** 2).mean()
                            layer_num += 1

                    reg_loss = reg_loss / layer_num

                    loss = loss + self.args.lambda_z * reg_loss

                    # wandb.log(
                    #     {"Reg loss": reg_loss.item()},
                    #     step=global_step
                    # )

                # print(f'GPU{self.args.gpu} after loss')

                if self.args.fp16 and _use_native_amp:
                    self.scaler.scale(loss).backward()
                elif self.args.fp16 and _use_apex:
                    with amp.scale_loss(loss, self.optim) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

                # print(f'GPU{self.args.gpu} after backward')

                loss = loss.detach()

                # Update Parameters
                if self.args.clip_grad_norm > 0:
                    if self.args.fp16 and _use_native_amp:
                        self.scaler.unscale_(self.optim)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.args.clip_grad_norm)
                    elif self.args.fp16 and _use_apex:
                        torch.nn.utils.clip_grad_norm_(amp.master_params(
                            self.optim), self.args.clip_grad_norm)
                    else:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.args.clip_grad_norm)

                if self.args.fp16 and _use_native_amp:
                    self.scaler.step(self.optim)
                    self.scaler.update()
                else:
                    self.optim.step()

                if self.lr_scheduler:
                    self.lr_scheduler.step()
                for param in self.model.parameters(): # self.model.zero_grad()
                    param.grad = None

                global_step += 1

                for k, v in results.items():
                    if k in epoch_results:
                        epoch_results[k] += v.item()

                if self.lr_scheduler:
                    if version.parse(torch.__version__) >= version.parse("1.4"):
                        lr = self.lr_scheduler.get_last_lr()[0]
                    else:
                        lr = self.lr_scheduler.get_lr()[0]
                else:
                    try:
                        lr = self.optim.get_lr()[0]
                    except AttributeError:
                        lr = self.args.lr


                # self.train_step_post_hook(result)

                if self.verbose:
                    if task == 'vqa':
                        vqa_loss_meter.update(loss.item())

                    desc_str = f'Epoch {epoch} | LR {lr:.6f}'

                    desc_str += f" |"
                    if 'vqa' in self.args.tasks:
                        desc_str += f" VQA {task_counter['vqa']}"
                    if 'gqa' in self.args.tasks:
                        desc_str += f" GQA {task_counter['gqa']}"
                    if 'nlvr' in self.args.tasks:
                        desc_str += f" NLVR {task_counter['nlvr']}"
                    if 'caption' in self.args.tasks:
                        desc_str += f" COCO {task_counter['caption']}"

                    if len(vqa_loss_meter) > 0:
                        desc_str += f' | VQA Loss {vqa_loss_meter.val:4f}'

                    pbar.set_description(desc_str)
                    pbar.update(1)

                if self.args.distributed:
                    dist.barrier()

            if self.verbose:
                pbar.close()
                # self.save("Epoch%02d" % (epoch + 1))

            if self.verbose:
                # Validation
                log_str = ''
                wandb_log_dict = {}

                if 'vqa' in self.args.tasks:
                    # VQA
                    vqa_val_loader = self.val_loader['vqa']
                    score_dict = self.vqa_evaluate(vqa_val_loader)
                    valid_score = score_dict['topk_score'] * 100.
                    valid_score_raw = score_dict['overall']
                    if valid_score_raw > best_vqa_valid or epoch == 0:
                        best_vqa_valid = valid_score_raw
                        best_vqa_epoch = epoch
                        # self.save("VQA_BEST")
                    log_str += f"VQA"
                    log_str += "\nEpoch %d: Valid Raw %0.2f Topk %0.2f" % (epoch, valid_score_raw, valid_score)
                    log_str += "\nEpoch %d: Best Raw %0.2f\n" % (best_vqa_epoch, best_vqa_valid)
                    wandb_log_dict['VQA/Valid/score'] = valid_score
                    wandb_log_dict['VQA/Valid/raw_score'] = score_dict['overall']
                if 'gqa' in self.args.tasks:
                    # GQA
                    gqa_val_loader = self.val_loader['gqa']
                    valid_score = self.gqa_evaluate(gqa_val_loader) * 100
                    if valid_score > best_gqa_valid or epoch == 0:
                        best_gqa_valid = valid_score
                        best_gqa_epoch = epoch
                    wandb_log_dict['GQA/Valid/Acc'] = valid_score
                    log_str += f"GQA"
                    log_str += "\nEpoch %d: Valid %0.2f" % (epoch, valid_score)
                    log_str += "\nEpoch %d: Best %0.2f\n" % (best_gqa_epoch, best_gqa_valid)
                if 'nlvr' in self.args.tasks:
                    # NLVR
                    nlvr_val_loader = self.val_loader['nlvr']
                    valid_score_dict = self.nlvr_evaluate(nlvr_val_loader)
                    valid_acc = valid_score_dict['accuracy'] * 100.
                    if valid_acc > best_nlvr_valid or epoch == 0:
                        best_nlvr_valid = valid_acc
                        best_nlvr_epoch = epoch
                    wandb_log_dict['NLVR/Valid/Acc'] = valid_acc
                    log_str += f"NLVR"
                    log_str += "\nEpoch %d: Valid %0.2f" % (epoch, valid_acc)
                    log_str += "\nEpoch %d: Best %0.2f\n" % (best_nlvr_epoch, best_nlvr_valid)

                if 'caption' in self.args.tasks:
                    # COCO Caption
                    caption_val_loader = self.val_loader['caption']
                    valid_results = self.caption_evaluate(caption_val_loader)
                    valid_score = valid_results['CIDEr'] * 100
                    if valid_score > best_caption_valid or epoch == 0:
                        best_caption_valid = valid_score
                        best_caption_epoch = epoch
                    for score_name, score in valid_results.items():
                        wandb_log_dict[f'Caption/Valid/{score_name}'] = score * 100
                    log_str += f"COCO Caption"
                    log_str += "\nEpoch %d: Valid CIDEr %0.2f" % (epoch, valid_score)
                    log_str += "\nEpoch %d: Best %0.2f\n" % (best_caption_epoch, best_caption_valid)

                print('val wandb_log_dict = ',wandb_log_dict)
                wandb.log(wandb_log_dict, step=epoch)

                print(log_str)

            if self.args.distributed:
                dist.barrier()

        # Test Set
        if self.verbose:
            self.save("LAST")

            log_str = ''
            wandb_log_dict = {}

            if 'vqa' in self.args.tasks:
                # VQA
                vqa_test_loader = self.test_loader['vqa']
                evaluator = vqa_test_loader.evaluator
                dump_path = os.path.join(self.args.output, 'karpathy_test_predict.json')
                quesid2ans = self.vqa_predict(vqa_test_loader, dump_path)
                wandb.save(dump_path, base_path=self.args.output)

                acc_dict_all = evaluator.evaluate_raw(quesid2ans)
                acc_dict_answerable = evaluator.evaluate_raw(quesid2ans, is_topk_optimal=True)
                acc_dict_unanswerable = evaluator.evaluate_raw(quesid2ans, is_topk_optimal=False)

                wandb_log_dict['VQA/Test/overall'] = acc_dict_all['overall']
                wandb_log_dict['VQA/Test/topk_optimal'] = acc_dict_answerable['overall']
                wandb_log_dict['VQA/Test/topk_not_optimal'] = acc_dict_unanswerable['overall']

                if self.test_loader.get("vqa_submit", None):
                    vqa_submit_test_loader = self.test_loader['vqa_submit']
                    dump_path = os.path.join(self.args.output, 'vqa_submit.json')
                    self.vqa_predict(vqa_submit_test_loader, dump_path=dump_path)
                    wandb.save(dump_path, base_path=self.args.output)

            # if 'gqa' in self.args.tasks:
            #     gqa_test_loader = self.test_loader['gqa']
            #     dump_path = os.path.join(self.args.output, 'gqa_submit.json')
            #     self.gqa_predict(gqa_test_loader, dump_path=dump_path)
            #     wandb.save(dump_path, base_path=self.args.output)

            if 'nlvr' in self.args.tasks:
                # NLVR
                nlvr_test_loader = self.test_loader['nlvr']
                dump_path = os.path.join(self.args.output, 'nlvr_submit.csv')
                test_score_dict = self.nlvr_evaluate(nlvr_test_loader, dump_path=dump_path)
                wandb.save(dump_path, base_path=self.args.output)
                for score_name, score in test_score_dict.items():
                    wandb_log_dict[f'NLVR/Test/{score_name}'] = score * 100.
            if 'caption' in self.args.tasks:
                # COCO Caption
                caption_test_loader = self.test_loader['caption']
                test_results = self.caption_evaluate(caption_test_loader)
                for score_name, score in test_results.items():
                    wandb_log_dict[f'Caption/Test/{score_name}'] = score

            print(log_str) 
            print("test wandb_log_dict = ",wandb_log_dict)
            wandb.log(wandb_log_dict, step=self.args.epochs)

            wandb.log({'finished': True})

        if self.args.distributed:
            dist.barrier()
            exit()

    def vqa_predict(self, loader, dump_path=None):
        self.model.eval()
        with torch.no_grad():
            quesid2ans = {}

            gen_kwargs = {}
            gen_kwargs['num_beams'] = 1

            for i, batch in enumerate(tqdm(loader, ncols=150, desc="VQA Validation")):
                
                print("vqa batch = ", batch)
                print("vqa batch input_ids.shape = ", batch['input_ids'].shape)

                if self.args.distributed:
                    results = self.model.module.test_step(batch, **gen_kwargs)
                else:
                    results = self.model.test_step(batch, **gen_kwargs)

                pred_ans = results['pred_ans']
                ques_ids = batch['question_ids']

                for qid, ans in zip(ques_ids, pred_ans):
                    quesid2ans[qid] = ans

            if dump_path is not None:
                loader.evaluator.dump_result(quesid2ans, dump_path)
            return quesid2ans

    def vqa_evaluate(self, loader, dump_path=None):
        evaluator = loader.evaluator
        quesid2ans = self.vqa_predict(loader, dump_path)

        acc_dict = evaluator.evaluate_raw(quesid2ans)

        topk_score = evaluator.evaluate(quesid2ans)
        acc_dict['topk_score'] = topk_score

        return acc_dict

    def gqa_predict(self, loader, dump_path=None):
        self.model.eval()
        with torch.no_grad():
            quesid2ans = {}

            gen_kwargs = {}
            gen_kwargs['num_beams'] = 1

            if self.verbose:
                pbar = tqdm(total=len(loader), ncols=150, desc="GQA Validation")

            for i, batch in enumerate(loader):

                if self.args.distributed:
                    results = self.model.module.test_step(batch, **gen_kwargs)
                else:
                    results = self.model.test_step(batch, **gen_kwargs)

                pred_ans = results['pred_ans']
                ques_ids = batch['question_ids']

                for qid, ans in zip(ques_ids, pred_ans):
                    quesid2ans[qid] = ans

                if self.verbose:
                    pbar.update(1)

            if dump_path is not None:
                print('\nsave dump at', dump_path)
                loader.evaluator.dump_result(quesid2ans, dump_path)
            return quesid2ans

    def gqa_evaluate(self, loader, dump_path=None):
        evaluator = loader.evaluator
        quesid2ans = self.gqa_predict(loader, dump_path)
        return evaluator.evaluate(quesid2ans)

    def nlvr_predict(self, loader, dump_path=None):
        """
        Predict the answers to questions in a data split.
        :param eval_tuple: The data tuple to be evaluated.
        :param dump: The path of saved file to dump results.
        :return: A dict of question_id to answer.
        """
        self.model.eval()
        with torch.no_grad():
            quesid2ans = {}
            for i, batch in enumerate(tqdm(loader, ncols=150, desc="NLVR Prediction")):
                
                # print("nlvr batch = ", batch)
                # print("nlvr batch input_ids.shape = ", batch['input_ids'].shape)

                if self.args.distributed:
                    results = self.model.module.test_step(batch)
                else:
                    results = self.model.test_step(batch)

                pred_ans = results['pred_ans_id']
                ques_ids = batch['question_ids']

                for qid, ans in zip(ques_ids, pred_ans):
                    quesid2ans[qid] = ans

            if dump_path is not None:
                loader.evaluator.dump_result(quesid2ans, dump_path)
            return quesid2ans

    def nlvr_evaluate(self, loader, dump_path=None):
        evaluator = loader.evaluator
        quesid2ans = self.nlvr_predict(loader, dump_path)
        return evaluator.evaluate(quesid2ans)

    def caption_predict(self, loader, dump_path=None):
        self.model.eval()
        with torch.no_grad():

            predictions = []
            targets = []

            gen_kwargs = {}
            gen_kwargs['num_beams'] = self.args.num_beams
            gen_kwargs['max_length'] = self.args.gen_max_length

            for i, batch in enumerate(tqdm(loader, ncols=150, desc="Caption Prediction")):

                if self.args.distributed:
                    results = self.model.module.test_step(
                        batch,
                        **gen_kwargs)
                else:
                    results = self.model.test_step(
                        batch,
                        **gen_kwargs)

                predictions.extend(results['pred'])

                if 'targets' in batch:
                    targets.extend(batch['targets'])

            # if self.args.do_lower_case:
            # predictions = [sent.capitalize() for sent in predictions]

            results = {
                'predictions': predictions,
                'targets': targets
            }

            return results

    def caption_evaluate(self, loader, dump_path=None):
        evaluator = loader.evaluator
        results = self.caption_predict(loader, dump_path)

        predictions = results['predictions']
        if dump_path is None:
            targets = results['targets']
            eval_results = evaluator.evaluate(predictions, targets)
            return eval_results

    @torch.no_grad()
    def infer(self):
        if self.verbose:
            quesid2ans = {}

            if 't5' in self.args.backbone:
                if self.args.use_vision:
                    project_name = "VLT5_multitask"
                else:
                    project_name = "T5_multitask"
            elif 'bart' in self.args.backbone:
                if self.args.use_vision:
                    project_name = "VLBart_multitask"
                else:
                    project_name = "Bart_multitask"

            wandb.init(project=project_name)
            wandb.run.name = self.args.run_name
            wandb.config.update(self.args)
            wandb.watch(self.model)
            wandb.log(
                {"percent of updated parameters (%)": self.percent_updated_parameters}
            )

            src_dir = Path(__file__).resolve().parent
            base_path = str(src_dir.parent)
            src_dir = str(src_dir)
            wandb.save(os.path.join(src_dir + "/*.py"), base_path=base_path)

        if self.args.distributed:
            dist.barrier()

        # Test Set
        if self.verbose:
            print("Testing...")
            load_trained_model_path = os.path.join(self.args.load_trained_model_path, 'LAST')
            self.load(load_trained_model_path)
            print("Trained Model Loaded: ", load_trained_model_path)

            log_str = ''
            wandb_log_dict = {}

            if 'vqa' in self.args.tasks:
                # VQA
                vqa_test_loader = self.test_loader['vqa']
                evaluator = vqa_test_loader.evaluator
                dump_path = os.path.join(self.args.output, 'karpathy_test_predict.json')
                quesid2ans = self.vqa_predict(vqa_test_loader, dump_path)
                wandb.save(dump_path, base_path=self.args.output)

                acc_dict_all = evaluator.evaluate_raw(quesid2ans)
                acc_dict_answerable = evaluator.evaluate_raw(quesid2ans, is_topk_optimal=True)
                acc_dict_unanswerable = evaluator.evaluate_raw(quesid2ans, is_topk_optimal=False)

                wandb_log_dict['VQA/Test/overall'] = acc_dict_all['overall']
                wandb_log_dict['VQA/Test/topk_optimal'] = acc_dict_answerable['overall']
                wandb_log_dict['VQA/Test/topk_not_optimal'] = acc_dict_unanswerable['overall']

                if self.test_loader.get("vqa_submit", None):
                    vqa_submit_test_loader = self.test_loader['vqa_submit']
                    dump_path = os.path.join(self.args.output, 'vqa_submit.json')
                    self.vqa_predict(vqa_submit_test_loader, dump_path=dump_path)
                    wandb.save(dump_path, base_path=self.args.output)

            # if 'gqa' in self.args.tasks:
            #     gqa_test_loader = self.test_loader['gqa']
            #     dump_path = os.path.join(self.args.output, 'gqa_submit.json')
            #     self.gqa_predict(gqa_test_loader, dump_path=dump_path)
            #     wandb.save(dump_path, base_path=self.args.output)

            if 'nlvr' in self.args.tasks:
                # NLVR
                nlvr_test_loader = self.test_loader['nlvr']
                dump_path = os.path.join(self.args.output, 'nlvr_submit.csv')
                test_score_dict = self.nlvr_evaluate(nlvr_test_loader, dump_path=dump_path)
                wandb.save(dump_path, base_path=self.args.output)
                for score_name, score in test_score_dict.items():
                    wandb_log_dict[f'NLVR/Test/{score_name}'] = score * 100.

            if 'caption' in self.args.tasks:
                # COCO Caption
                caption_test_loader = self.test_loader['caption']
                test_results = self.caption_evaluate(caption_test_loader)
                for score_name, score in test_results.items():
                    wandb_log_dict[f'Caption/Test/{score_name}'] = score

            print(log_str) 
            print("test wandb_log_dict = ",wandb_log_dict)
            wandb.log(wandb_log_dict, step=self.args.epochs)

            wandb.log({'finished': True})

        if self.args.distributed:
            dist.barrier()
            exit()


def main_worker(gpu, args):
    # GPU is assigned
    args.gpu = gpu
    args.rank = gpu
    print(f'Process Launching at GPU {gpu}')

    if args.distributed:
        torch.cuda.set_device(args.gpu)
        dist.init_process_group(backend='nccl')

    feat_dim_dict = {
        "RN50": 2048,
        "RN101": 2048,
        "RN50x4": 2560,
        "ViT": 768
    }
    args.feat_dim = feat_dim_dict[args.feature_type]
    import vqa_clip_data as vqa_data
    import gqa_clip_data as gqa_data
    import nlvr_clip_data as nlvr_data
    import caption_clip_data as caption_data

    vqa_args = deepcopy(args)
    vqa_args.max_text_length = 20

    gqa_args = deepcopy(args)
    gqa_args.batch_size = int(args.batch_size * 100 / 60) # 100
    gqa_args.max_text_length = 20

    nlvr_args = deepcopy(args)
    nlvr_args.batch_size = int(args.batch_size * 20 / 60)

    caption_args = deepcopy(args)
    caption_args.batch_size = int(args.batch_size * 50 / 60)
    caption_args.max_text_length = 40
    caption_args.gen_max_length = 40

    if args.use_tasks_prompts:
        vqa_args.prompt = "vqa: "
        gqa_args.prompt = "gpa: "
        nlvr_args.prompt = "nlvr: "
        caption_args.prompt = "caption: "
    else:
        vqa_args.prompt = ""
        gqa_args.prompt = ""
        nlvr_args.prompt = ""
        caption_args.prompt = ""

    train_loaders = []

    if args.epochs > 0:
        if 'vqa' in args.tasks:
            print(f'Building VQA train loader at GPU {gpu}')
            vqa_train_loader = vqa_data.get_loader(
                vqa_args,
                split='karpathy_train', mode='train', batch_size=vqa_args.batch_size,
                distributed=args.distributed, gpu=args.gpu,
                workers=args.num_workers,
                topk=args.train_topk,
            )
            train_loaders.append(vqa_train_loader)
            # print(f'VQA train loader len: {len(vqa_train_loader)}')

        if 'gqa' in args.tasks:
            print(f'Building GQA train loader at GPU {gpu}')
            gqa_train_loader = gqa_data.get_loader(
                gqa_args,
                split='train,valid', mode='train', batch_size=gqa_args.batch_size,
                distributed=args.distributed, gpu=args.gpu,
                workers=args.num_workers,
                topk=args.train_topk,
            )
            train_loaders.append(gqa_train_loader)
            # print(f'GQA train loader len: {len(gqa_train_loader)}')

        if 'nlvr' in args.tasks:
            print(f'Building NLVR train loader at GPU {gpu}')
            nlvr_train_loader = nlvr_data.get_loader(
                nlvr_args,
                split='train', mode='train', batch_size=nlvr_args.batch_size,
                distributed=args.distributed, gpu=args.gpu,
                workers=args.num_workers,
                topk=args.train_topk,
            )
            train_loaders.append(nlvr_train_loader)
            # print(f'NLVR train loader len: {len(nlvr_train_loader)}')

        if 'caption' in args.tasks:
            print(f'Building COCO Caption train loader at GPU {gpu}')
            caption_train_loader = caption_data.get_loader(
                caption_args,
                split='karpathy_train', mode='train', batch_size=caption_args.batch_size,
                distributed=args.distributed, gpu=args.gpu,
                workers=args.num_workers,
                topk=args.train_topk,
            )
            train_loaders.append(caption_train_loader)

    train_loader = multitask_data.MultitaskLoader(
        train_loaders,
        sampling=args.multitask_sampling,
        verbose=gpu==0)

    val_num_workers = 4
    # Validation set
    if gpu == 0:
        val_loader = {}
        if args.epochs > 0:
            if 'vqa' in args.tasks:
                print(f'Building VQA val loader at GPU {gpu}')
                vqa_val_loader = vqa_data.get_loader(
                    vqa_args,
                    split='karpathy_val', mode='val', batch_size=vqa_args.batch_size,
                    distributed=False, gpu=args.gpu,
                    workers=val_num_workers,
                    topk=args.valid_topk,
                )
                val_loader['vqa'] = vqa_val_loader

            if 'gqa' in args.tasks:
                print(f'Building GQA val loader at GPU {gpu}')
                gqa_val_loader = gqa_data.get_loader(
                    gqa_args,
                    split='testdev', mode='val', batch_size=gqa_args.batch_size,
                    distributed=False, gpu=args.gpu,
                    workers=val_num_workers,
                    topk=args.valid_topk,
                )
                val_loader['gqa'] = gqa_val_loader

            if 'nlvr' in args.tasks:
                print(f'Building NLVR val loader at GPU {gpu}')
                nlvr_val_loader = nlvr_data.get_loader(
                    nlvr_args,
                    split='valid', mode='val', batch_size=nlvr_args.batch_size,
                    distributed=False, gpu=args.gpu,
                    workers=val_num_workers,
                    # topk=args.valid_topk,
                )
                val_loader['nlvr'] = nlvr_val_loader

            if 'caption' in args.tasks:
                print(f'Building COCO Caption val loader at GPU {gpu}')
                caption_val_loader = caption_data.get_loader(
                    caption_args,
                    split='karpathy_val', mode='val', batch_size=caption_args.batch_size,
                    distributed=False, gpu=args.gpu,
                    workers=val_num_workers,
                    topk=args.valid_topk,
                )
                val_loader['caption'] = caption_val_loader

        # Test set
        test_loader = {}
        if 'vqa' in args.tasks:
            print(f'Building VQA test loader at GPU {gpu}')
            vqa_test_loader = vqa_data.get_loader(
                vqa_args,
                split='karpathy_test', mode='val', batch_size=vqa_args.batch_size,
                distributed=False, gpu=args.gpu,
                workers=val_num_workers,
                topk=args.valid_topk,
            )
            test_loader['vqa'] = vqa_test_loader

            if args.testing:
                vqa_submit_test_loader = vqa_data.get_loader(
                    vqa_args,
                    split='test_4', mode='val', batch_size=vqa_args.batch_size,
                    distributed=False, gpu=args.gpu,
                    workers=val_num_workers,
                    topk=args.valid_topk,
                )
                test_loader['vqa_submit'] = vqa_submit_test_loader

        if 'gqa' in args.tasks and args.testing:
            print(f'Building GQA val loader at GPU {gpu}')
            gqa_val_loader = gqa_data.get_loader(
                args,
                split='submit', mode='val', batch_size=nlvr_args.batch_size,
                distributed=False, gpu=args.gpu,
                workers=val_num_workers,
                topk=args.valid_topk,
            )
            test_loader['gqa'] = gqa_val_loader

        if 'nlvr' in args.tasks:
            print(f'Building NLVR test loader at GPU {gpu}')
            nlvr_test_loader = nlvr_data.get_loader(
                nlvr_args,
                split='test', mode='val', batch_size=nlvr_args.batch_size,
                distributed=False, gpu=args.gpu,
                workers=val_num_workers,
                # topk=args.valid_topk,
            )
            test_loader['nlvr'] = nlvr_test_loader

        if 'caption' in args.tasks:
            print(f'Building COCO Caption test loader at GPU {gpu}')
            caption_test_loader = caption_data.get_loader(
                caption_args,
                split='karpathy_test', mode='val', batch_size=caption_args.batch_size,
                distributed=False, gpu=args.gpu,
                workers=val_num_workers,
                topk=args.valid_topk,
            )
            test_loader['caption'] = caption_test_loader

    else:
        val_loader = None
        test_loader = None

    trainer = Trainer(args, train_loader, val_loader, test_loader, train=True)

    if args.infer_only:
        trainer.infer()
    else:
        trainer.train()

if __name__ == "__main__":
    cudnn.benchmark = True
    args = parse_args()
    ngpus_per_node = torch.cuda.device_count()
    args.world_size = ngpus_per_node
    if args.local_rank in [0, -1]:
        print(args)

        comments = []
        if args.load is not None:
            ckpt_str = "_".join(args.load.split('/')[-3:])
            comments.append(ckpt_str)
        if args.comment != '':
            comments.append(args.comment)
        comment = '_'.join(comments)

        from datetime import datetime
        current_time = datetime.now().strftime('%b%d_%H-%M')
        run_name = f'{current_time}_GPU{args.world_size}'
        if len(comments) > 0:
            run_name += f'_{comment}'

        if args.run_name == "":
            args.run_name = run_name

    # if args.distributed:
    main_worker(args.local_rank, args)
