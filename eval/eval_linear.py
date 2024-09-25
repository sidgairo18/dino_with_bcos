# Copyright (c) Facebook, Inc. and its affiliates.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# some standard imports
import os
import argparse
import json
from pathlib import Path
import math

# pytorch imports
import torch
from torch import nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision import transforms as pth_transforms
from torchvision import models as torchvision_models

# external imports
import utils as main_utils
from image_dataloader_from_file import image_loader
from mlp_heads import *
#import vision_transformer as vits
# import bcos vit models
from bcos.models.vit_with_conv_stem import *
from bcos.modules import BcosLinear, BcosConv2d, LogitLayer
from bcos.modules import norms
from bcos.modules.losses import *

def eval_linear(args, eval_idx):
    if eval_idx == 0:
        main_utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(main_utils.get_sha()))
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    cudnn.benchmark = True

    # ============ preparing data ... ============
    if 'bcos' in args.arch:
        train_transform = pth_transforms.Compose([
            pth_transforms.RandomResizedCrop(224),
            pth_transforms.RandomHorizontalFlip(),
            pth_transforms.ToTensor()
            ])
        val_transform = pth_transforms.Compose([
            pth_transforms.Resize(256, interpolation=3),
            pth_transforms.CenterCrop(224),
            #pth_transforms.CenterCrop(args.global_image_size),
            pth_transforms.ToTensor(),
        ])
    else:
        train_transform = pth_transforms.Compose([
            pth_transforms.RandomResizedCrop(224),
            pth_transforms.RandomHorizontalFlip(),
            pth_transforms.ToTensor(),
            pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])
        val_transform = pth_transforms.Compose([
            pth_transforms.Resize(256, interpolation=3),
            pth_transforms.CenterCrop(224),
            #pth_transforms.CenterCrop(args.global_image_size),
            pth_transforms.ToTensor(),
            pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
    dataset_train = image_loader(os.path.join(args.data_path, "meta/train.txt"), os.path.join(args.data_path, "train"), input_transforms=train_transform)
    dataset_val = image_loader(os.path.join(args.data_path, "meta/val.txt"), os.path.join(args.data_path, "val"), input_transforms=val_transform)
    sampler = torch.utils.data.DistributedSampler(dataset_train, shuffle=False)
    train_loader = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    val_loader = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    print(f"Data loaded with {len(dataset_train)} train and {len(dataset_val)} val imgs.")
    
    # ============ building network ... ============
    # if the network is a Vision Transformer (i.e. vit_tiny, vit_small, vit_base)
    # ============ building network ... ============                                          
    if args.arch in ['bcos_vit_tiny', 'bcos_vit_small', 'bcos_vit_base']:                     
        bcos_configs = dict(                                                                  
                num_classes = 0,                                                              
                norm_layer = NoBias(DetachableLayerNorm),                                     
                act_layer = nn.Identity,                                                      
                channels = 6,                                                                 
                norm2d_layer = NoBias(DetachableGNLayerNorm2d),                               
                linear_layer = BcosLinear,                                                    
                conv2d_layer = BcosConv2d,
                use_cls_token = args.use_cls_token,
                pool_tokens_flag = args.pool_tokens_flag
                )                                                                             
        if args.arch == 'bcos_vit_tiny':                                                      
            raise Exception("Bcos ViT Tiny variant does not exist yet!")                      
        elif args.arch == 'bcos_vit_small':                                                   
            if args.global_image_size == 224:                                                 
                model = simple_vit_s_patch16_224(my_configs=bcos_configs)                     
            elif args.global_image_size == 192:                                               
                model = simple_vit_s_patch16_192(my_configs=bcos_configs)                     
            else:                                                                             
                raise Exception("Other global image sizes haven't been implemented yet!") 
        else:                                                                                 
            raise Exception("Bcos ViT Base variant does not exist yet!")                      
        embed_dim = model.embed_dim                                                           
        print("BCOS LOADED")                                                                  
    #elif args.arch in vits.__dict__.keys():                                                  
    elif args.arch in ["vit_tiny", "vit_small", "vit_base"]:                                  
        std_configs = dict(                                                                   
                num_classes = 0,                                                              
                norm_layer = nn.LayerNorm,                                                    
                act_layer = nn.GELU,                                                          
                channels = 3,                                                                 
                norm2d_layer = lambda x: nn.GroupNorm(1,x),                                   
                linear_layer = nn.Linear,                                                     
                conv2d_layer = nn.Conv2d,                                                     
                use_cls_token = args.use_cls_token,
                pool_tokens_flag = args.pool_tokens_flag
                )                                                                             
        if args.arch == 'vit_tiny':                                                           
            raise Exception("Std ViT Tiny variant does not exist yet!")                       
        elif args.arch == 'vit_small':                                                        
            if args.global_image_size == 224:                                                 
                model = simple_vit_s_patch16_224(my_configs=std_configs)                      
            elif args.global_image_size == 192:                                               
                model = simple_vit_s_patch16_192(my_configs=std_configs)                      
            else:                                                                             
                raise Exception("Other global image sizes haven't been implemented yet!") 
        else:                                                                                 
            raise Exception("Std ViT Base variant does not exist yet!")                       
        print("STD LOADED")                                                                   
                                                                                              
        """                                                                                   
        student = vits.__dict__[args.arch](                                                   
            patch_size=args.patch_size,                                                       
            drop_path_rate=args.drop_path_rate,  # stochastic depth                           
        )                                                                                     
        teacher = vits.__dict__[args.arch](patch_size=args.patch_size)                        
        """                                                                                   
        embed_dim = model.embed_dim
    elif args.arch in ['bcos_vit_tiny_conv', 'bcos_vit_small_conv', 'bcos_vit_base_conv']:
        bcos_configs = dict(                                                              
                num_classes = 0,                                                          
                norm_layer = NoBias(DetachableLayerNorm),                                 
                act_layer = nn.Identity,                                                  
                channels = 6,                                                             
                norm2d_layer = NoBias(DetachableGNLayerNorm2d),                           
                linear_layer = BcosLinear,                                                
                conv2d_layer = BcosConv2d,                                                
                use_cls_token = args.use_cls_token,
                pool_tokens_flag = args.pool_tokens_flag
                )                                                                         
        # have used multiple if / elif / else statements should get rid of this eventually.
        if args.arch == 'bcos_vit_tiny_conv':                                             
            if args.global_image_size == 224:                                             
                model = vitc_ti_patch1_14(my_configs=bcos_configs)                        
            elif args.global_image_size == 192:                                           
                model = vitc_ti_patch1_12(my_configs=bcos_configs)                        
            else:                                                                         
                raise Exception("Other global image sizes haven't been implemented yet!") 
                                                                                          
        elif args.arch == 'bcos_vit_small_conv':                                          
            if args.global_image_size == 224:                                             
                model = vitc_s_patch1_14(my_configs=bcos_configs)                         
            elif args.global_image_size == 192:                                           
                model = vitc_s_patch1_12(my_configs=bcos_configs)                         
            else:                                                                         
                raise Exception("Other global image sizes haven't been implemented yet!") 
        else:                                                                             
            if args.global_image_size == 224:                                             
                model = vitc_b_patch1_14(my_configs=bcos_configs)                         
            elif args.global_image_size == 192:                                           
                model = vitc_b_patch1_12(my_configs=bcos_configs)                         
            else:                                                                         
                raise Exception("Other global image sizes haven't been implemented yet!") 
        embed_dim = model.embed_dim                                                       
        print("BCOS WITH CONV LOADED")
    elif args.arch in ["vit_tiny_conv", "vit_small_conv", "vit_base_conv"]:               
        std_configs = dict(                                                               
                num_classes = 0,                                                          
                norm_layer = nn.LayerNorm,                                                
                act_layer = nn.GELU,                                                      
                channels = 3,                                                             
                norm2d_layer = lambda x: nn.GroupNorm(1,x),                               
                linear_layer = nn.Linear,                                                 
                conv2d_layer = nn.Conv2d,                                                 
                use_cls_token = args.use_cls_token,
                pool_tokens_flag = args.pool_tokens_flag
                )                                                                         
        if args.arch == 'vit_tiny_conv':                                                  
            if args.global_image_size == 224:                                             
                model = vitc_ti_patch1_14(my_configs=std_configs)                         
            elif args.global_image_size == 192:                                           
                model = vitc_ti_patch1_12(my_configs=std_configs)                         
            else:                                                                         
                raise Exception("Other global image sizes haven't been implemented yet!") 
        elif args.arch == 'vit_small_conv':                                               
            if args.global_image_size == 224:                                             
                model = vitc_s_patch1_14(my_configs=std_configs)                          
            elif args.global_image_size == 192:                                           
                model = vitc_s_patch1_12(my_configs=std_configs)                          
            else:                                                                         
                raise Exception("Other global image sizes haven't been implemented yet!") 
        else:                                                                             
            if args.global_image_size == 224:                                             
                model = vitc_b_patch1_14(my_configs=std_configs)                          
            elif args.global_image_size == 192:                                           
                model = vitc_b_patch1_12(my_configs=std_configs)                          
            else:                                                                         
                raise Exception("Other global image sizes haven't been implemented yet!") 
        print("STD WITH CONV LOADED")                                                     
        embed_dim = model.embed_dim 
    else:
        print(f"Unknow architecture: {args.arch}")
        sys.exit(1)
    #return embed_dim
    model.cuda()
    model.eval()
    # load weights to evaluate
    main_utils.load_pretrained_weights(model, args.pretrained_weights, args.checkpoint_key, args.arch, args.patch_size)
    print(f"Model {args.arch} built.")

    # set output_dir
    if args.output_dir == "." or args.output_dir == "" or args.output_dir is None:
        args.output_dir = args.arch+"_"+str(args.global_image_size)
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    else:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    print("Output dir is {}".format(args.output_dir))

    if args.head_type == 'bcos':
        hid_channels = embed_dim if args.num_layers > 1 else None
        linear_classifier = BcosLinearClsHead(
                num_classes = args.num_labels,
                in_channels = embed_dim,
                hid_channels = hid_channels,
                num_layers = args.num_layers,
                with_last_bn = False,
                with_avg_pool = not args.pool_tokens_flag
                )
        print("BCOS MLP head setup")
    else:
        hid_channels = embed_dim if args.num_layers > 1 else None
        linear_classifier = StdLinearClsHead(
                num_classes = args.num_labels,
                in_channels = embed_dim,
                hid_channels = hid_channels,
                num_layers = args.num_layers,
                with_last_bn = False,
                with_avg_pool = not args.pool_tokens_flag,
                with_bias = args.with_bias
                )
    linear_classifier = linear_classifier.cuda()
    linear_classifier = nn.parallel.DistributedDataParallel(linear_classifier, device_ids=[args.gpu])

    # ============ preparing data ... ============

    if args.evaluate:
        main_utils.load_pretrained_linear_weights(linear_classifier, args.arch, args.patch_size)
        test_stats = validate_network(val_loader, linear_classifier)
        print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
        return

    print(f"Data loaded with {len(dataset_train)} train and {len(dataset_val)} val imgs.")

    # set loss
    if args.loss_type == "ce":
        loss = nn.CrossEntropyLoss()
    elif args.loss_type == "bce":
        loss = BinaryCrossEntropyLoss()
    elif args.loss_type == "uobce":
        loss = UniformOffLabelsBCEWithLogitsLoss()

    # set optimizer
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(
            linear_classifier.parameters(),
            args.lr * (args.batch_size_per_gpu * main_utils.get_world_size()) / 256., # linear scaling rule
            momentum=0.9,
            weight_decay=0, # we do not apply weight decay
        )
    else:
        optimizer = torch.optim.AdamW(linear_classifier.parameters(), 
                lr=args.lr * (args.batch_size_per_gpu * main_utils.get_world_size()) / 256.)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=0)

    # get flag strings
    pool_tokens_flag_string = "with-pool-tokens" if args.pool_tokens_flag else "no-pool-tokens"
    with_bias_string = "with-bias" if args.with_bias else "no-bias"
    use_cls_token_string = "with-cls-token" if args.use_cls_token else "no-cls-token"
    fname = str(args.pretrained_weights.split('/')[-1].split('.')[0]+'_'+args.head_type+"-linear_"+str(args.num_layers)+"_"+with_bias_string+"_"+args.optimizer+"_lr-"+str(args.lr)+"_"+args.loss_type+"_"+pool_tokens_flag_string+"_"+use_cls_token_string)

    # Optionally resume from a checkpoint
    to_restore = {"epoch": 0, "best_acc": 0.}
    main_utils.restart_from_checkpoint(
        #os.path.join(args.output_dir, "checkpoint.pth.tar"),
        os.path.join(args.output_dir, fname+"_mlp_checkpoint.pth.tar"),
        run_variables=to_restore,
        state_dict=linear_classifier,
        optimizer=optimizer,
        scheduler=scheduler,
    )
    start_epoch = to_restore["epoch"]
    best_acc = to_restore["best_acc"]
    

    for epoch in range(start_epoch, args.epochs):
        train_loader.sampler.set_epoch(epoch)

        train_stats = train(model, linear_classifier, optimizer, train_loader, epoch, args, loss)
        scheduler.step()

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch}
        if epoch % args.val_freq == 0 or epoch == args.epochs - 1:
            test_stats = validate_network(val_loader, model, linear_classifier, args, loss)
            print(f"Accuracy at epoch {epoch} of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
            best_acc = max(best_acc, test_stats["acc1"])
            print(f'Max accuracy so far: {best_acc:.2f}%')
            log_stats = {**{k: v for k, v in log_stats.items()},
                         **{f'test_{k}': v for k, v in test_stats.items()}}
        if main_utils.is_main_process():
            with (Path(args.output_dir) / str(fname+"_linear_log.txt")).open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
            save_dict = {
                "epoch": epoch + 1,
                "state_dict": linear_classifier.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_acc": best_acc,
            }
            torch.save(save_dict, os.path.join(args.output_dir, fname+"_mlp_checkpoint.pth.tar"))
            
            """
            with (Path(args.output_dir) / fname+"_log_acc.txt").open("a") as f:
                f.write("Top-1 test accuracy: {acc:1f}".format(acc=best_acc) + "\n")
    print("Training of the supervised linear classifier on frozen features completed.\n"
                "Top-1 test accuracy: {acc:.1f}".format(acc=best_acc))
    return train_features, test_features, train_labels, test_labels
            """

def train(model, linear_classifier, optimizer, loader, epoch, args, train_loss):
    linear_classifier.train()
    metric_logger = main_utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', main_utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    for (inp, target) in metric_logger.log_every(loader, 20, header):
        # move to gpu
        inp = inp.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        
        # forward
        with torch.no_grad():
            output = model(inp)

        if not args.pool_tokens_flag:
            B, N, dim = output.shape
            output = output.reshape(B, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2)
        output = linear_classifier(output)

        # compute cross entropy loss
        loss = train_loss(output, target)

        # compute the gradients
        optimizer.zero_grad()
        loss.backward()

        #print("GRADIENT",linear_classifier.module.linear.linear.weight.grad.abs().mean())
        #print("GRADIENT",linear_classifier.module.linear.weight.grad.abs().mean())

        # step
        optimizer.step()

        # log 
        torch.cuda.synchronize()
        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def validate_network(val_loader, model, linear_classifier,args, train_loss):
    linear_classifier.eval()
    metric_logger = main_utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    for inp, target in metric_logger.log_every(val_loader, 20, header):
        # move to gpu
        inp = inp.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # forward
        with torch.no_grad():
            output = model(inp)
        if not args.pool_tokens_flag:
            B, N, dim = output.shape
            output = output.reshape(B, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2)
        output = linear_classifier(output)
        
        loss = train_loss(output, target)

        if linear_classifier.module.num_classes >= 5:
            acc1, acc5 = main_utils.accuracy(output, target, topk=(1, 5))
        else:
            acc1, = main_utils.accuracy(output, target, topk=(1,))

        batch_size = inp.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        if linear_classifier.module.num_classes >= 5:
            metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    if linear_classifier.module.num_classes >= 5:
        print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))
    else:
        print('* Acc@1 {top1.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, losses=metric_logger.loss))
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Evaluation with linear classification on ImageNet')
    parser.add_argument('--n_last_blocks', default=4, type=int, help="""Concatenate [CLS] tokens
        for the `n` last blocks. We use `n=4` when evaluating ViT-Small and `n=1` with ViT-Base.""")
    parser.add_argument('--avgpool_patchtokens', default=False, type=main_utils.bool_flag,
        help="""Whether ot not to concatenate the global average pooled features to the [CLS] token.
        We typically set this to False for ViT-Small and to True with ViT-Base.""")
    # the above 2 are not used at the moment.
    parser.add_argument('--arch', default='vit_small', type=str, help='Architecture')
    parser.add_argument('--patch_size', default=16, type=int, help='Patch resolution of the model.')
    parser.add_argument('--global_image_size', default=224, type=int, help='Input image resolution of the model.')
    parser.add_argument('--pretrained_weights', default='', type=str, help="Path to pretrained weights to evaluate.")
    parser.add_argument("--checkpoint_key", default="teacher", type=str, help='Key to use in the checkpoint (example: "teacher")')
    parser.add_argument('--epochs', default=100, type=int, help='Number of epochs of training.')
    parser.add_argument("--lr", default=0.001, type=float, help="""Learning rate at the beginning of
        training (highest LR used during training). The learning rate is linearly scaled
        with the batch size, and specified here for a reference batch size of 256.
        We recommend tweaking the LR depending on the checkpoint evaluated.""")
    parser.add_argument('--optimizer', default='sgd', type=str, help='Optimizer to use. sgd or adam')
    parser.add_argument('--batch_size_per_gpu', default=128, type=int, help='Per-GPU batch-size')
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")
    parser.add_argument('--data_path', default='/path/to/imagenet/', type=str)
    parser.add_argument('--num_workers', default=10, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument('--val_freq', default=1, type=int, help="Epoch frequency for validation.")
    parser.add_argument('--output_dir', default=".", help='Path to save logs and checkpoints')
    parser.add_argument('--num_labels', default=1000, type=int, help='Number of labels for linear classifier')
    parser.add_argument('--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')
    parser.add_argument('--init_checkpoints', default='', type=str, help="Checkpoints dir.")
    parser.add_argument('--use_cls_token', default=False, type=main_utils.bool_flag, help="Whether or not to use the cls token")
    parser.add_argument('--pool_tokens_flag', default=True, type=main_utils.bool_flag, help="Whether global average pool the tokens or not.")
    parser.add_argument('--loss_type', default='ce', type=str, help="Loss type bce or cross entropy")
    parser.add_argument('--head_type', default='std', type=str, help="Type of classifier head std mlp or bcos mlp")
    parser.add_argument('--num_layers', default=1, type=int, help="Number of layers in the mlp head")
    parser.add_argument('--with_bias', default=False, type=main_utils.bool_flag, help="Whether to use a bias in the mlp head or not")

    args = parser.parse_args()
    
    eval_linear(args, 0)
