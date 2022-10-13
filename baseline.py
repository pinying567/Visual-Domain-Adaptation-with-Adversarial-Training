import numpy as np
import argparse
import os
import json
import torch.utils.data as data
from torchvision import transforms
import random
import torch
from torch import nn
from torch.nn import functional as F

from dataset import Dataset
from DANN import FeatureExtractor, Classifier, Discriminator
from util import averageMeter, accuracy, lr_decay, get_lambda

def main():
    global save_dir, logger
    
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    # setup directory to save logfiles, checkpoints, and output csv
    save_dir = args.save_dir
    if 'train' in args.phase and not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    # setup logger
    logger = None
    if 'train' in args.phase:
        logger = open(os.path.join(save_dir, 'train.log'), 'a')
        logfile = os.path.join(save_dir, 'training_log.json')
        log = {'train': []}
        logger.write('{}\n'.format(args))
        
        # setup data loader for training images
        trans_train = transforms.Compose([
            transforms.RandomRotation(15),
            transforms.Resize((28, 28)),            
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        if args.train_dataset == 'usps':
            img_dir = os.path.join(args.data_root, 'digits/usps/train')
            csv_path = os.path.join(args.data_root, 'digits/usps/train.csv')
        elif args.train_dataset == 'svhn':
            img_dir = os.path.join(args.data_root, 'digits/svhn/train')
            csv_path = os.path.join(args.data_root, 'digits/svhn/train.csv')
        else: # mnistm
            img_dir = os.path.join(args.data_root, 'digits/mnistm/train')
            csv_path = os.path.join(args.data_root, 'digits/mnistm/train.csv')
            
        dataset_train = Dataset(img_dir, trans_train, csv_path)
        train_loader = data.DataLoader(dataset_train, shuffle=True, drop_last=False, pin_memory=True, batch_size=args.batch_size)

        print('train: {}'.format(dataset_train.__len__()))
        logger.write('train: {}\n'.format(dataset_train.__len__()))
    
    # setup data loader for testing images
    if args.phase == 'train_da':
        trans_val = transforms.Compose([
            transforms.RandomRotation(15),
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    else:
        trans_val = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    
    if args.test_dir:
        img_dir = args.test_dir
        dataset_val = Dataset(img_dir, trans_val)
    else:
        if args.phase == 'train_da':
            if args.test_dataset == 'usps':
                img_dir = os.path.join(args.data_root, 'digits/usps/train')
                csv_path = os.path.join(args.data_root, 'digits/usps/train.csv')
            elif args.test_dataset == 'svhn':
                img_dir = os.path.join(args.data_root, 'digits/svhn/train')
                csv_path = os.path.join(args.data_root, 'digits/svhn/train.csv')
            else: # mnistm
                img_dir = os.path.join(args.data_root, 'digits/mnistm/train')
                csv_path = os.path.join(args.data_root, 'digits/mnistm/train.csv')
        else:
            if args.test_dataset == 'usps':
                img_dir = os.path.join(args.data_root, 'digits/usps/test')
                csv_path = os.path.join(args.data_root, 'digits/usps/test.csv')
            elif args.test_dataset == 'svhn':
                img_dir = os.path.join(args.data_root, 'digits/svhn/test')
                csv_path = os.path.join(args.data_root, 'digits/svhn/test.csv')
            else: # mnistm
                img_dir = os.path.join(args.data_root, 'digits/mnistm/test')
                csv_path = os.path.join(args.data_root, 'digits/mnistm/test.csv')
        dataset_val = Dataset(img_dir, trans_val, csv_path)
    
    print('test: {}'.format(dataset_val.__len__()))    
    if 'train' in args.phase:
        logger.write('test: {}\n'.format(dataset_val.__len__()))
        val_loader = data.DataLoader(dataset_val, shuffle=True, drop_last=False, pin_memory=True, batch_size=args.batch_size)
    else:
        val_loader = data.DataLoader(dataset_val, shuffle=False, drop_last=False, pin_memory=True, batch_size=args.batch_size)
    
    # setup model
    F = FeatureExtractor().cuda()
    C = Classifier().cuda()
    D = Discriminator().cuda()
    if 'train' in args.phase:
        logger.write('FeatureExtractor: {}\n'.format(F))
        logger.write('Classifier: {}\n'.format(C))
        logger.write('Discriminator: {}\n'.format(D))
        
    # setup optimizer
    opt_F = torch.optim.Adam(F.parameters(), lr=args.lr)
    opt_C = torch.optim.Adam(C.parameters(), lr=args.lr)
    opt_D = torch.optim.Adam(D.parameters(), lr=args.lr_d)
    if 'train' in args.phase:
        logger.write('opt_F: {}\n'.format(opt_F))
        logger.write('opt_C: {}\n'.format(opt_C))
        logger.write('opt_D: {}\n'.format(opt_D))
    
    # setup loss function
    celoss = nn.CrossEntropyLoss(reduction='mean')
    bceloss = nn.BCELoss(reduction='mean')
    
    # load checkpoint
    start_ep = 0
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint)
        F.load_state_dict(checkpoint['F_state'])
        C.load_state_dict(checkpoint['C_state'])
        if not args.load_cls_only:
            D.load_state_dict(checkpoint['D_state'])
            opt_F.load_state_dict(checkpoint['opt_F_state'])
            opt_C.load_state_dict(checkpoint['opt_C_state'])
            opt_D.load_state_dict(checkpoint['opt_D_state'])
            start_ep = checkpoint['epoch']
        print("Loaded checkpoint '{}' (epoch: {})".format(args.checkpoint, checkpoint['epoch']))
        if 'train' in args.phase:
            logger.write("Loaded checkpoint '{}' (epoch: {})\n".format(args.checkpoint, checkpoint['epoch']))
            
    if args.phase == 'train':
    
        # start training
        print('Start training from epoch {}'.format(start_ep))
        logger.write('Start training from epoch {}\n'.format(start_ep))
        
        for epoch in range(start_ep, args.epochs):
            
            acc, loss = train(train_loader, F, C, opt_F, opt_C, epoch, celoss)
            log['train'].append([epoch + 1, acc, loss])
            
            if (epoch + 1) % 10 == 0:
                # save checkpoint
                state = {
                    'epoch': epoch + 1,
                    'acc': acc,
                    'F_state': F.state_dict(),
                    'C_state': C.state_dict(),
                    'opt_F_state': opt_F.state_dict(),
                    'opt_C_state': opt_C.state_dict(),
                }
                checkpoint = os.path.join(save_dir, 'ep-{}.pkl'.format(epoch + 1))
                torch.save(state, checkpoint)
                print('[Checkpoint] {} is saved.'.format(checkpoint))
                logger.write('[Checkpoint] {} is saved.\n'.format(checkpoint))
                json.dump(log, open(logfile, 'w'))
            
            if (epoch + 1) % args.step == 0:
                lr_decay(opt_F, decay_rate=args.gamma)
                lr_decay(opt_C, decay_rate=args.gamma)
        
        # save last model
        state = {
            'epoch': epoch + 1,
            'acc': acc,
            'F_state': F.state_dict(),
            'C_state': C.state_dict(),
            'opt_F_state': opt_F.state_dict(),
            'opt_C_state': opt_C.state_dict(),
        }
        checkpoint = os.path.join(save_dir, 'last_checkpoint.pkl')
        torch.save(state, checkpoint)
        print('[Checkpoint] {} is saved.'.format(checkpoint))
        logger.write('[Checkpoint] {} is saved.\n'.format(checkpoint))
        print('Training is done.')
        logger.write('Training is done.\n')
        logger.close()
        
    elif args.phase == 'train_da':
        
        # start training
        print('Start training from epoch {}'.format(start_ep))
        logger.write('Start training from epoch {}\n'.format(start_ep))
        
        best_dloss = 0
        for epoch in range(start_ep, args.epochs):
            
            acc_c, acc_d, loss_c, loss_d = train_da(train_loader, val_loader, F, C, D, opt_F, \
                                                    opt_C, opt_D, epoch, celoss, bceloss)
            log['train'].append([epoch + 1, acc_c, acc_d, loss_c, loss_d])
            
            if (epoch + 1) % 10 == 0:
                with torch.no_grad():# peek
                    acc_val, loss_val = val(val_loader, F, C, celoss)
                # save checkpoint
                state = {
                    'epoch': epoch + 1,
                    'acc_c': acc_c,
                    'acc_d': acc_d,
                    'F_state': F.state_dict(),
                    'C_state': C.state_dict(),
                    'D_state': D.state_dict(),
                    'opt_F_state': opt_F.state_dict(),
                    'opt_C_state': opt_C.state_dict(),
                    'opt_D_state': opt_D.state_dict(),
                }
                checkpoint = os.path.join(save_dir, 'ep-{}.pkl'.format(epoch + 1))
                torch.save(state, checkpoint)
                print('[Checkpoint] {} is saved.'.format(checkpoint))
                logger.write('[Checkpoint] {} is saved.\n'.format(checkpoint))
                json.dump(log, open(logfile, 'w'))
            
            if (epoch + 1) % args.step == 0:
                lr_decay(opt_F, decay_rate=args.gamma)
                lr_decay(opt_C, decay_rate=args.gamma)
                lr_decay(opt_D, decay_rate=args.gamma)
        
        # save last model
        state = {
            'epoch': epoch + 1,
            'acc_c': acc_c,
            'acc_d': acc_d,
            'F_state': F.state_dict(),
            'C_state': C.state_dict(),
            'D_state': D.state_dict(),
            'opt_F_state': opt_F.state_dict(),
            'opt_C_state': opt_C.state_dict(),
            'opt_D_state': opt_D.state_dict(),
        }
        checkpoint = os.path.join(save_dir, 'last_checkpoint.pkl')
        torch.save(state, checkpoint)
        print('[Checkpoint] {} is saved.'.format(checkpoint))
        logger.write('[Checkpoint] {} is saved.\n'.format(checkpoint))
        print('Training is done.')
        logger.write('Training is done.\n')
        logger.close()
        
    else:
        with torch.no_grad():
            acc_val, loss_val = val(val_loader, F, C, celoss, save_result=True)
        
        print('Testing is done.')

        
def train(data_loader, F, C, opt_F, opt_C, epoch, celoss):
    
    ACC = averageMeter()
    losses = averageMeter()
    
    # setup training mode
    F.train()
    C.train()
    
    for (step, value) in enumerate(data_loader):
        
        image = value[0].cuda()
        target = value[2].cuda(non_blocking=True)
        
        # forward
        output = C(F(image))
        
        # compute loss
        loss = celoss(output, target)
        losses.update(loss.item(), image.size(0))
        
        # compute accuracy
        prec1 = accuracy(output, target, topk=(1,))
        ACC.update(prec1[0].item(), image.size(0))
        
        # backward
        opt_F.zero_grad()
        opt_C.zero_grad()
        loss.backward()
        opt_F.step()
        opt_C.step()
    
    # logging
    curr_lr_f = opt_F.param_groups[0]['lr']
    curr_lr_c = opt_C.param_groups[0]['lr']
    print('Epoch: [{}/{}]\t' \
        'LR_F: [{:.6g}] ' \
        'LR_C: [{:.6g}] ' \
        'Loss: {loss.avg:.4f}\t' \
        'ACC: {ACC.avg:.4f}'.format(epoch + 1, args.epochs, curr_lr_f, curr_lr_c, loss=losses, ACC=ACC)
    )
    logger.write('Epoch: [{}/{}]\t' \
        'LR_F: [{:.6g}] ' \
        'LR_C: [{:.6g}] ' \
        'Loss: {loss.avg:.4f}\t' \
        'ACC: {ACC.avg:.4f}\n'.format(epoch + 1, args.epochs, curr_lr_f, curr_lr_c, loss=losses, ACC=ACC)
    )
    
    return ACC.avg, losses.avg
    
    
def train_da(train_loader, val_loader, F, C, D, opt_F, opt_C, opt_D, epoch, celoss, bceloss):
    
    ACC_c = averageMeter()
    ACC_d = averageMeter()
    c_losses = averageMeter()
    d_losses = averageMeter()
    losses = averageMeter()
    
    # setup training mode
    F.train()
    C.train()
    D.train()
    for (images, _, labels) in train_loader:
        
        """ train domain classifier (discriminator) """
        images = images.cuda()
        labels = labels.cuda(non_blocking=True)
        
        # load target domain data
        images_tgt, _, _ = next(iter(val_loader))
        images_tgt = images_tgt.cuda()
        
        # compute feature
        feat_src = F(images)
        feat_tgt = F(images_tgt)
        
        # compute discriminator score
        d_src = D(feat_src.detach())
        d_tgt = D(feat_tgt.detach())
        
        # compute domain classification loss
        dc_loss = (bceloss(d_src, torch.ones_like(d_src)) + bceloss(d_tgt, torch.zeros_like(d_tgt))) / 2
        d_losses.update(dc_loss.item(), images.size(0) + images_tgt.size(0))
        
        # compute domain classification accuracy
        acc_1 = torch.ge(d_src, 0.5).sum().float() / images.size(0)
        acc_0 = torch.lt(d_tgt, 0.5).sum().float() / images_tgt.size(0)
        acc_d = (acc_0 + acc_1) / 2
        ACC_d.update(acc_d.item(), images.size(0) + images_tgt.size(0))
        
        # backward
        D.zero_grad() 
        dc_loss.backward()
        opt_D.step()
        
        """ train classifier """
        # compute feature
        feat_src = F(images)
        feat_tgt = F(images_tgt)
        
        # compute classification score
        y_src = C(feat_src)
        
        # compute classification accuracy
        prec1 = accuracy(y_src, labels, topk=(1,))
        ACC_c.update(prec1[0].item(), images.size(0))
        
        # compute classification loss
        closs = celoss(y_src, labels)
        c_losses.update(closs.item(), images.size(0))
        
        # compute discriminator score
        d_src = D(feat_src)
        d_tgt = D(feat_tgt)
        
        # compute domain adversarial loss
        dc_loss = (bceloss(d_src, torch.ones_like(d_src)) + bceloss(d_tgt, torch.zeros_like(d_tgt))) / 2     
        
        # compute total loss
        lamda = args.lamda * get_lambda(epoch + 1, args.epochs)
        loss = closs - lamda * dc_loss
        losses.update(loss.item(), images.size(0))
        
        # backward
        F.zero_grad()
        C.zero_grad()
        D.zero_grad()
        loss.backward()
        opt_F.step()
        opt_C.step()
    
    # logging
    curr_lr_f = opt_F.param_groups[0]['lr']
    curr_lr_c = opt_C.param_groups[0]['lr']
    curr_lr_d = opt_D.param_groups[0]['lr']
    print('Epoch: [{}/{}] ' \
        'LR_FC: [{:.6g}] ' \
        'LR_D: [{:.6g}] ' \
        'lambda: [{:.3g}] ' \
        'Loss: {loss.avg:.4f} '
        'Loss_D: {d_loss.avg:.4f} ' \
        'Loss_C: {c_loss.avg:.4f} ' \
        'ACC_d: {ACC_d.avg:.4f} ' \
        'ACC_c: {ACC_c.avg:.4f}'.format(epoch + 1, args.epochs, curr_lr_f, curr_lr_d, lamda, loss=losses, \
                                        d_loss=d_losses, c_loss=c_losses, ACC_d=ACC_d, ACC_c=ACC_c)
    )
    logger.write('Epoch: [{}/{}] ' \
        'LR_FC: [{:.6g}] ' \
        'LR_D: [{:.6g}] ' \
        'lambda: [{:.3g}] ' \
        'Loss: {loss.avg:.4f} '
        'Loss_D: {d_loss.avg:.4f} ' \
        'Loss_C: {c_loss.avg:.4f} ' \
        'ACC_d: {ACC_d.avg:.4f} ' \
        'ACC_c: {ACC_c.avg:.4f}\n'.format(epoch + 1, args.epochs, curr_lr_f, curr_lr_d, lamda, loss=losses, \
                                        d_loss=d_losses, c_loss=c_losses, ACC_d=ACC_d, ACC_c=ACC_c)
    )
    
    return ACC_c.avg, ACC_d.avg, c_losses.avg, d_losses.avg


def val(data_loader, F, C, celoss, save_result=False):

    losses = averageMeter()
    ACC = averageMeter()

    # setup evaluation mode
    F.eval()
    C.eval()
    
    if save_result:
        fnames = []
        predictions = []
    
    if args.save_feat:
        all_feat = []
        all_label = []
    
    for (step, value) in enumerate(data_loader):
        
        image = value[0].cuda()
        if len(value) > 2:
            target = value[2].cuda(non_blocking=True)
        
        # forward
        feat = F(image)
        output = C(feat)
        
        # accumulate image_id & predictions
        if save_result:
            fnames.extend(value[1])
            pred_batch = torch.max(output, dim=1)[1].data.cpu().numpy()
            predictions = np.concatenate((predictions, pred_batch), axis=0)
        
        if args.save_feat:
            all_feat.extend(feat.data.cpu().numpy())
            if len(value) > 2:
                all_label.extend(value[2])
        
        if len(value) > 2:
            # compute loss
            loss = celoss(output, target)
            losses.update(loss.item(), image.size(0))
            
            # compute accuracy
            prec1 = accuracy(output, target, topk=(1,))
            ACC.update(prec1[0].item(), image.size(0))
    
    if losses.count > 0:
        # logging                        
        print('[Val] Loss {loss.avg:.4f}\tACC {ACC.avg:.4f}'.format(loss=losses, ACC=ACC))
        
    # write results to csv file
    if save_result:
        if args.out_csv:
            csv_file = args.out_csv
        else:
            csv_file = os.path.join(save_dir, 'test_pred.csv')
        with open(csv_file, 'w') as csv:
            csv.write('image_name,label\n')
            for i in range(len(fnames)):
                csv.write('{},{}\n'.format(fnames[i], int(predictions[i])))
        print("Output prediction saved to '{}'".format(csv_file))
    
    if args.save_feat:
        np.save(os.path.join(save_dir, '{}_feat.npy'.format(args.test_dataset)), np.asarray(all_feat))
        np.save(os.path.join(save_dir, '{}_label.npy'.format(args.test_dataset)), np.asarray(all_label))
    
    return ACC.avg, losses.avg


if __name__ == '__main__':
    global args
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=128, help='input batch size (default: 128)')
    parser.add_argument('--epochs', type=int, default=90, help='number of epochs to train (default: 90)')
    parser.add_argument('--lr', type=float, default=1e-2, help='base learning rate (default: 1e-2)')
    parser.add_argument('--lr_d', type=float, default=1e-2, help='base learning rate of discriminator (default: 1e-2)')
    parser.add_argument('--step', type=int, default=30, help='learning rate decay step (default: 30)')
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--gamma', type=float, default=0.1, help='learning rate step gamma (default: 0.1)')
    parser.add_argument('--lamda', type=float, default=0.1, help='weight for domain adversarial loss (default: 0.1)')
    parser.add_argument('--test_dir', type=str, default='', help='testing image directory')
    parser.add_argument('--checkpoint', type=str, default='', help='pretrained model')
    parser.add_argument('--save_dir', type=str, default='checkpoint/usps-usps', help='directory to save logfile, checkpoint and output csv')
    parser.add_argument('--out_csv', type=str, default='', help='path to output prediction file (csv)')
    parser.add_argument('--data_root', type=str, default='hw3_data', help='data root')
    parser.add_argument('--save_feat', type=bool, default=False, help='save features and corresponding labels for TSNE plot')
    parser.add_argument('--train_dataset', type=str, default='usps', help='train digits datasets (usps/mnistm/svhn)')
    parser.add_argument('--test_dataset', type=str, default='usps', help='test digits datasets (usps/mnistm/svhn)')
    parser.add_argument('--load_cls_only', type=bool, default=True, help='only load the parameters of the classifier')
    parser.add_argument('--phase', type=str, default='train', help='phase (train/train_adda/test)')
    
    args = parser.parse_args()
    print(args)
    
    main()