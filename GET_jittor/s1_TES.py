import argparse
import os
import math
import numpy as np

import jittor as jt
from jittor.dataset import DataLoader
import jittor.nn as nn
# from torch.optim import SGD, lr_scheduler
# from torch.utils.data import DataLoader
from tqdm import tqdm


from data.augmentations import get_transform
from data.get_datasets import get_datasets, get_class_splits

from util.general_utils import AverageMeter, init_experiment,distill_crit
from util.cluster_and_log_utils import log_accs_from_preds
from model import DINOHead, info_nce_logits, SupConLoss, DistillLoss, ContrastiveLearningViewGenerator, get_params_groups

from jclip import clip
# import torch.nn.functional as F

exp_root = 'dev_outputs'



class TES(nn.Module):
    def __init__(self,
                 device = None,
                 num_words = 7,  # context len
                 word_dim = 512,  # word embedding dim
                 words_drop_ratio = 0.2, 
                 ):
        super(TES, self).__init__()
        self.num_words = num_words
        self.word_dim = word_dim
        self.device = device
        
        self.projector = nn.Linear(in_features=512, out_features=num_words*word_dim)
        # transfer img embedding to text tokens
        self.tes_model,_ = clip.load_TES_CLIP("./jittor-ViT-B-16.pkl",n_ctx=self.num_words+2) #  num_words + sot + eot
        self.vlm_model,_ = clip.load_TES_CLIP("./jittor-ViT-B-16.pkl",n_ctx=77)
        
        for params in self.vlm_model.parameters():
            params.requires_grad = False  
              
        self.words_drop_ratio = words_drop_ratio
        
        for params in self.tes_model.parameters():
            params.requires_grad = False

    def text_encoder(self,text_descriptions):
        text_tokens = clip.tokenize(text_descriptions).to(self.device)
        text_features = self.vlm_model.encode_text(text_tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features
    
    def genPf(self,pseudo_words,drop_or_not):
        if pseudo_words.shape[0] == 0: 
            raise ValueError("pseudo_words.shape[0] == 0")
        valid_mask = self._drop_word(pseudo_words)
        pseudo_text, end_token_ids = self.tes_model.prepare_pseudo_text_tensor(
            pseudo_words,valid_mask=valid_mask,w_dropout=drop_or_not) 
        pseudo_feats = self.tes_model.encode_pseudo_text(pseudo_text, end_token_ids, text_pe=True)
        
        return pseudo_feats
        
    def _drop_word(self, pseudo_words):
        p = self.words_drop_ratio
        num_preds, num_words, _ = pseudo_words.shape
        mask = nn.dropout(jt.ones(num_preds, num_words),
                         p=p,
                         is_train=self.training)
        start_end_mask = jt.ones_like(mask[:, :1])
        # check empty
        is_empty = mask.sum(dim=-1) == 0.0
        mask[is_empty, 0] = 1.0       # TODO add random on this
        mask[mask > 0.0] = 1.0
        # add start and end token mask
        valid_mask = jt.concat([start_end_mask, mask, start_end_mask], dim=-1)

        return valid_mask
    
    
    def execute(self,x,drop_or_not=False):
        # [bs,512]
        img_feasts = self.vlm_model.encode_image(x).float()
        # [bs,7,512]
        pseudo_words = self.projector(img_feasts).view(-1, self.num_words, self.word_dim)
        # [bs,512]
        pseudo_text_features = self.genPf(pseudo_words, drop_or_not)
        pseudo_text_features = pseudo_text_features.float()

        
        return pseudo_text_features
    
def train_tes(model_clip, model_tes,train_loader, args,device):
    

    optimizer = jt.optim.SGD(list(model_tes.parameters()), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    
    


    exp_lr_scheduler = jt.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.epochs,
            eta_min=args.lr * 1e-3,
        )

    mseloss=nn.MSELoss(reduction='sum')
    
    # text_features of base classes
    text_descriptions = [f"a photo of a {label}" for label in args.base_names]
    
    base_Tfeats = model_tes.text_encoder(text_descriptions)
    base_Tfeats = base_Tfeats.float().detach()
    
    logit_scale = model_clip.logit_scale.exp()
    logit_scale =  logit_scale.detach()
    # # inductive
    # best_test_acc_lab = 0
    # # transductive
    # best_train_acc_lab = 0
    # best_train_acc_ubl = 0 
    # best_train_acc_all = 0
    

    drop_or_not = args.words_drop
    for epoch in range(args.epochs):
        loss_record = AverageMeter()
            
        model_clip.eval()
        model_tes.train()

        
     
        for batch_idx, batch in enumerate(train_loader):
            images, class_labels, uq_idxs, mask_lab = batch
            mask_lab = mask_lab[:, 0]

            # class_labels, mask_lab = class_labels.cuda(device=device), mask_lab.cuda(device=device).bool()
            class_labels, mask_lab = class_labels.cuda(), mask_lab.cuda().bool()
            
            # images = jt.concat(images, dim=0).cuda(non_blocking=True,device=device)
            images = jt.concat(images, dim=0).cuda()
            

           
            img_feats = model_clip.encode_image(images).float()
        
            pseudo_text_feats = model_tes(images, drop_or_not)
            pseudo_text_feats /= pseudo_text_feats.norm(dim=-1,keepdim=True)
            # text_feats_ln = ln_text(text_feats)
            img_feats /= img_feats.norm(dim=-1,keepdim=True)
    
            
            # align loss
            align_logits_img = logit_scale *  img_feats @ pseudo_text_feats.t()
            align_logits_text = logit_scale *  pseudo_text_feats @ img_feats.t()
        
            align_labels_text = jt.arange(pseudo_text_feats.shape[0])
            align_labels_img = jt.arange(img_feats.shape[0])
        
            align_loss_text = nn.CrossEntropyLoss()(align_logits_text, align_labels_text)
            align_loss_image = nn.CrossEntropyLoss()(align_logits_img, align_labels_img)
        
            align_loss = align_loss_text+align_loss_image
            
            # distill_loss
            mask_twoView = jt.concat([mask_lab,mask_lab])
            pseudo_text_feats_base = pseudo_text_feats[mask_twoView]
            # import pdb
            # pdb.set_trace()
            pseudo_text_feats_base = (pseudo_text_feats_base/pseudo_text_feats_base.norm(dim=-1,keepdim=True)) 
            label_base = jt.concat([class_labels[mask_lab], class_labels[mask_lab]])
            # import pdb
            # pdb.set_trace()
            distill_logits, distill_labels = distill_crit(stu_feats=pseudo_text_feats_base,tea_feats=base_Tfeats,labels=label_base, args=args, device=device)
        
            distill_loss = nn.CrossEntropyLoss()(distill_logits, distill_labels) + \
                            mseloss(pseudo_text_feats_base,base_Tfeats[label_base])/pseudo_text_feats_base.shape[0]

            pstr = ''

            pstr += f'align_loss: {align_loss.item():.4f} '
            pstr += f'distill_loss: {distill_loss.item():.4f} '


            loss = 0
            loss += align_loss + distill_loss

                
            # Train acc
            loss_record.update(loss.item(), class_labels.size(0))
            optimizer.zero_grad()
            optimizer.backward(loss)
            optimizer.step()
         

            if batch_idx % args.print_freq == 0:
                args.logger.info('Epoch: [{}][{}/{}]\t loss {:.5f}\t {}'
                            .format(epoch, batch_idx, len(train_loader), loss.item(), pstr))

        args.logger.info('Train Epoch: {} Avg Loss: {:.4f} '.format(epoch, loss_record.avg))

    
        # Step schedule
        exp_lr_scheduler.step()

        # save_dict = {
        #     'model_tes_projector': model_tes.projector.state_dict(),
        #     'optimizer': optimizer.state_dict(),
        #     'epoch': epoch + 1,
        # # }

        # for k in model_tes.projector.state_dict().keys():
        #     model_tes.projector.state_dict()[k] = model_tes.projector.state_dict()[k].float().cpu()
        jt.save( model_tes.projector.state_dict(), args.tes_model_path)
        args.logger.info("TES model saved to {}.".format(args.tes_model_path))



    

if __name__ == "__main__":
    
    

    parser = argparse.ArgumentParser(description='cluster', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--num_workers', default=8, type=int)

    parser.add_argument('--warmup_model_dir', type=str, default=None)
    parser.add_argument('--dataset_name', type=str, default='cub')
    parser.add_argument('--prop_train_labels', type=float, default=0.5)
    parser.add_argument('--use_ssb_splits', action='store_true', default=True)

    parser.add_argument('--grad_from_block', type=int, default=11)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=5e-5)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--exp_root', type=str, default=exp_root)
    parser.add_argument('--transform', type=str, default='imagenet')
    parser.add_argument('--n_views', default=2, type=int)
    
    parser.add_argument('--fp16', action='store_true', default=False)
    parser.add_argument('--print_freq', default=10, type=int)
    parser.add_argument('--exp_name', default='S1_TES', type=str)
    parser.add_argument('--exp_id', default='debug', type=str)
    
    parser.add_argument('--num_words', default=7, type=int)
    parser.add_argument('--word_dim', default=512, type=int)   
    parser.add_argument('--words_drop_ratio', default=0.5, type=float)
    parser.add_argument('--words_drop',  action='store_true', default=False)
    parser.add_argument('--lamC', default=1.0, type=float)
    
    parser.add_argument('--seed', default=None, type=int)
    parser.add_argument('--cuda_dev', type=int, default=0, help='GPU to select')
    
    
    
    # ----------------------
    # INIT
    # ----------------------
    args = parser.parse_args()
    

    
    device = "cuda"

    args = get_class_splits(args)
    
    args.num_labeled_classes = len(args.train_classes)
    args.num_unlabeled_classes = len(args.unlabeled_classes)
    
    # get classnames
    class_names_path = f"./dataset_class_name/{args.dataset_name}_name.npy"
        
    class_names = np.load(class_names_path)
    
    args.base_names = class_names[args.train_classes]

    init_experiment(args, runner_name=['GET'], exp_id=args.exp_id)
    
    
    args.interpolation = 3
    args.crop_pct = 0.875

    args.image_size = 224
    args.feat_dim = 512
    args.num_mlp_layers = 3
    args.mlp_out_dim = args.num_labeled_classes + args.num_unlabeled_classes
    
    args.logger.info('\n--------------stage1--------------'*10)
    
    args.logger.info(f'argsparser for stage 1:{args}')
    
    # --------------------
    # CONTRASTIVE TRANSFORM
    # --------------------
    train_transform, test_transform = get_transform(args.transform, image_size=args.image_size, args=args)
    # train_transform, test_transform = preprocess,preprocess
    train_transform = ContrastiveLearningViewGenerator(base_transform=train_transform, n_views=args.n_views)
    # --------------------
    # DATASETS
    # --------------------
    train_dataset, test_dataset, unlabelled_train_examples_test, datasets,train_examples_test = get_datasets(args.dataset_name,
                                                                                         train_transform,
                                                                                         test_transform,
                                                                                         args)
    
    # --------------------
    # SAMPLER
    # Sampler which balances labelled and unlabelled examples in each batch
    # --------------------
    label_len = len(train_dataset.labelled_dataset)
    unlabelled_len = len(train_dataset.unlabelled_dataset)
    sample_weights = [1 if i < label_len else label_len / unlabelled_len for i in range(len(train_dataset))]
    sample_weights = jt.float64(sample_weights)
    sampler = jt.dataset.RandomSampler(train_dataset)
    
    # --------------------
    # DATALOADERS
    # --------------------
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False,
                              sampler=sampler, drop_last=True)
    test_loader_unlabelled = DataLoader(unlabelled_train_examples_test,
                                        batch_size=256, shuffle=False)
    # train_test_loader = DataLoader(train_examples_test, num_workers=args.num_workers,
    #                                     batch_size=256, shuffle=False, pin_memory=False)
    
    
    

    # # load model
    model_tes = TES(device=device, num_words=args.num_words, 
                              word_dim=args.word_dim, words_drop_ratio=args.words_drop_ratio)
    
    model_clip, preprocess = clip.load("jittor-ViT-B-16.pkl",device)

    for m in model_clip.parameters():
        m.requires_grad = False

    args.logger.info("stage1-model_tes:")        
    for name, m in model_tes.named_parameters():
        if m.requires_grad ==True:
            args.logger.info(f"{name} requires_grad")
    
    args.logger.info("stage1-model_clip:")
    for name, m in model_clip.named_parameters():
        if m.requires_grad ==True:
            args.logger.info(f"{name} requires_grad")
    
            
    args.logger.info('stage1 model build')
    
    # train TES
    args.logger.info('Train TES')
    train_tes(model_clip,model_tes,train_loader, args,device)


    
        
    
    
  
