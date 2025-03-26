import torch
import torch.utils.data as Data
import numpy as np
import time
import random

from scipy.ndimage import zoom
from sklearn.manifold import TSNE
from collections import defaultdict
from pathlib import Path
import matplotlib.pyplot as plt
from torchvision import utils as vutils
import albumentations as A
import deepspeed

from MS_models.utils.metrics import DiceLoss, DiceBCELoss, dice_score, precision, recall
from MS_models.utils.val import test_single_volume, calculate_metric_percase, calculate_metric_parcase2
from MS_models.utils.util import TwoStreamBatchSampler, TwoStreamBatchSampler_L
from MS_models.ConvBlock import ConvEncoder, Decoder, Atts
from MS_models.ConvBlock3 import ConvEncoder53
from MS_models.MSDdatagenerator import SemiTask012D, SemiTask022D, SemiTask072D,SemiTask082D,SemiTask083D,SemiTask092D
from MS_models.TACEdatagenerator import SemiTACESeg
from MS_models.utils.util import get_train_transforms, get_val_transforms


from torch.utils.tensorboard import SummaryWriter


class MS(torch.nn.Module):
    def __init__(self, args):
        super(MS, self).__init__()
        self.args = args
        self.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu' )
        self.writer = SummaryWriter(Path(args.root_path) / args.train_tb)
        # 一组原型，triplet loss，预测值和相似度相减
        # model 加载
        self.encoder = ConvEncoder53(inshape=args.img_shape, in_ch=args.in_channel).to(self.device)
        self.decoder = Decoder(inshape=args.img_shape).to(self.device)
        
        layser_shapes = []
        inputs = [torch.zeros(args.img_shape).unsqueeze(0).repeat(args.in_channel,1,1)] * args.batch_size
        inputs = torch.stack(inputs, dim=0).to(self.device)
        for i in range(self.encoder.stage_num):
            inputs = self.encoder.encode_blocks[i](inputs)
            layser_shapes.append(list(inputs.shape))
        self.atts = Atts(T_num=1, layer_shapes=layser_shapes).to(self.device)
        

        # initialize optimizer
        if True:
            self.optimizerD = torch.optim.Adam(self.decoder.parameters(), lr=3e-4, betas=(args.beta1, 0.999))
            self.optimizerE = torch.optim.Adam(filter(lambda p: p.requires_grad, self.encoder.parameters()), lr=3e-4, betas=(args.beta1, 0.999))
            self.optimizerT = torch.optim.Adam(self.atts.parameters(), lr=3e-4, betas=(args.beta1, 0.999))
        else:
            self.optimizerD = torch.optim.RMSprop(self.decoder.parameters(), lr = args.lrG)
            self.optimizerE = torch.optim.RMSprop( filter(lambda p: p.requires_grad, self.encoder.parameters()), lr = args.lrG)
            self.optimizerT = torch.optim.RMSprop( self.atts.parameters(), lr = args.lrG)

        self.seg_loss = DiceBCELoss()

        self.now_epoch = 0
        self.recoder = defaultdict(list)
        self.global_step = 0

        exp_dir = Path(args.model_dir) / 'models'
        self.args.log_save_dir = exp_dir
        self.args.save_dir = exp_dir

    def load_model(self, load_path, is_best=True):
        print('load saved model')

        ckp = torch.load(load_path, map_location=torch.device(self.device))
        
        self.encoder.load_state_dict(ckp['encoder'])
        self.decoder.load_state_dict(ckp['decoder'])
        self.atts.load_state_dict(ckp['atts'])

        if is_best:
            return
        
        self.optimizerE.load_state_dict(ckp['optimizerE'])
        self.optimizerD.load_state_dict(ckp['optimizerD'])
        self.optimizerT.load_state_dict(ckp['optimizerT'])
        self.now_epoch = ckp['now_epoch']

        # for key in cpk.keys():
        #     obj = getattr(self, key)
        #     if isinstance(obj, torch.nn.Module) or isinstance(obj, torch.optim.Optimizer):
        #         obj.load_state_dict(cpk[key])
        #     else:
        #         setattr(self, key, cpk[key])

    def save_model(self, epoch=0, is_best=False):
        if is_best:
            ckp = {}
            ckp['encoder'] = self.encoder.state_dict()
            ckp['decoder'] = self.decoder.state_dict()
            ckp['atts'] = self.atts.state_dict()
            torch.save(ckp, Path(self.args.root_path) / self.args.save_dir / 'MS_best.pt')
            return 
        
        ckp = {}
        ckp['encoder'] = self.encoder.state_dict()
        ckp['decoder'] = self.decoder.state_dict()
        ckp['atts'] = self.atts.state_dict()

        ckp['optimizerE'] = self.optimizerE.state_dict()
        ckp['optimizerD'] = self.optimizerD.state_dict()
        ckp['optimizerT'] = self.optimizerT.state_dict()
        ckp['now_epoch'] = epoch
        torch.save(ckp, Path(self.args.root_path) / self.args.save_dir / ('MS_%04d.pt' % epoch))

    def write_tb(self, epoch):
        for k, v in self.recoder.items():
            loss_np = np.array(v)
            self.writer.add_scalar(f'train/{k}', loss_np.mean(), epoch)

    def print_loss(self, epoch_info, is_write=False):
        count = 1
        loss_info = ''
        for k, v in self.recoder.items():
            loss_info = loss_info + '{}:{:.4e},'.format(k, np.array(v).mean())
            if count % 5 == 0:
                loss_info = loss_info + '\n'
            count += 1

        print(loss_info)
        if is_write:
            with open(str(Path(self.args.log_save_dir) / 'log_loss.txt'), 'a') as f:  # 设置文件对象
                print(epoch_info, flush=True, file = f)
                print(loss_info, flush=True, file = f)
            
    def clear_recoder(self):
        self.recoder = defaultdict(list)
    
    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def forward(self, labeled, unlabeled, mask):
        self.skips, self.pos_loss, self.neg_loss, self.contra_loss = self.encoder.forward1(labeled, mask)
        skips = self.atts(self.skips)
        self.seg_outputs = self.decoder(skips)

        self.un_skips = self.encoder.forward1(unlabeled)
        un_skips = self.atts(self.un_skips)
        self.un_seg_outputs = self.decoder(un_skips)
        
    def infer(self, image):
        skips = self.encoder.forward1(image)
        skips = self.atts(skips)
        seg_outputs = self.decoder(skips)
        return seg_outputs[-1]

    def backward_labeled(self, mask):
        seg_loss = 0

        # seg_loss = seg_loss + self.seg_loss(self.seg_outputs[-1], mask)
        for output in self.seg_outputs:
            seg_loss = seg_loss + self.seg_loss(output, mask)
        
        total_loss = self.args.w_seg*seg_loss + self.args.w_pos*self.pos_loss+ \
                        self.args.w_neg*self.neg_loss + self.args.w_contra*self.contra_loss
        
        total_loss.backward(retain_graph=True)
        
        self.recoder['pos_loss'].append( self.args.w_pos * self.pos_loss.item())
        self.recoder['neg_loss'].append( self.args.w_neg * self.neg_loss.item())
        self.recoder['seg_loss'].append( self.args.w_seg * seg_loss.item())
        self.recoder['contra_loss'].append( self.args.w_contra * self.contra_loss.item())
        # self.recoder['un_pos_loss'].append(self.args.w_un_pos * un_pos_loss.item())
        # self.recoder['un_neg_loss'].append(self.args.w_un_neg * un_neg_loss.item())
        self.recoder['supervised_loss'].append(total_loss.item())

    def backward_unlabeled(self):
        # sim and smooth loss
        un_pos_loss, un_neg_loss = self.encoder.forward2(self.un_skips, self.un_seg_outputs[-1])
        total_loss = self.args.w_un_pos*un_pos_loss + self.args.w_un_neg*un_neg_loss
        total_loss.backward()

        self.recoder['un_pos_loss'].append(self.args.w_un_pos * un_pos_loss.item())
        self.recoder['un_neg_loss'].append(self.args.w_un_neg * un_neg_loss.item())
        self.recoder['un_pos+neg_loss'].append(total_loss.item())

    def optimize(self, epoch, images, mask, un_images, un_mask):
        self.global_step += 1
        self.forward(images, un_images, mask)

        self.optimizerE.zero_grad()
        self.optimizerT.zero_grad()
        self.optimizerD.zero_grad()

        # backward labeled
        self.backward_labeled(mask)

        # backward unlabeled
        self.set_requires_grad([self.encoder], False)
        self.backward_unlabeled()
        
        # torch.nn.utils.clip_grad_norm_(list(self.atts.parameters()), max_norm=10, norm_type=2)
        # torch.nn.utils.clip_grad_norm_(list(self.decoder.parameters()), max_norm=10, norm_type=2)
        
        self.optimizerE.step()
        self.optimizerT.step()
        self.optimizerD.step()

        self.set_requires_grad([self.encoder], True)
        num_classes = self.args.out_channel-1
        if self.global_step % self.args.tb_save_freq == 0:
            self.tb_save_step += 1
            slice_id = self.args.img_shape[0]//2 if len(self.args.img_shape)==3 else 0
            images = images/255 if images.max()>50 else images
            self.writer.add_image('train/image', images[0,:,:,:], self.tb_save_step)
            self.writer.add_image('train/image_true_seg', mask[0,:,:,:]/num_classes, self.tb_save_step)
            self.writer.add_image('train/unimage', un_images[0,:,:,:]/num_classes, self.tb_save_step)
            self.writer.add_image('train/unimage_true_seg', un_mask[0,:,:,:]/num_classes, self.tb_save_step)
            for i in range(len(self.seg_outputs)):
                self.writer.add_image(f'train/seg_{i}', self.seg_outputs[i][0,:,:,:]/num_classes, self.tb_save_step)
                self.writer.add_image(f'train/unseg_{i}', self.un_seg_outputs[i][0,:,:,:]/num_classes, self.tb_save_step)
 
    def train_model(self):
        
        transform =  A.Compose([
            A.Rotate(limit=35, p=0.3),
            A.HorizontalFlip(p=0.3),
            A.VerticalFlip(p=0.3),
            # A.CoarseDropout(p=0.3, max_holes=10, max_height=16, max_width=16),
            A.ElasticTransform(p=0.3),
        ])

        tr_transforms = get_train_transforms(patch_size=self.args.img_shape, rotation_for_DA={'x':(-3.141592653589793, 3.141592653589793), 'y': (0, 0), 'z': (0, 0)}, mirror_axes=(0,1))
        val_transforms = get_val_transforms()

        # 定义train setc
        # brats_train = BraTSRegDataset1(Path('E:\\datasets\\BRAST2018\\MICCAI_BraTS_2018_Data_Training\\HGG'), 
        #                             fixed_seqs=['t1','t21','t1ce1'], moving_seqs=['t1','t21','t1ce1'], 
        #                             resize=(64, 64, 64), label_flag=4, flag=0, transform=transform)

        # task02_train = SemiTask022D(Path('E:\\datasets\\MSD\\Task02_Heart\\Task02\\train'), partition=0.1, size=self.args.img_shape, transform=transform)
        # task07_train = SemiTask072D(Path('E:\\datasets\\MSD\Task07_Pancreas\\Task07\\train'), partition=0.3, size=self.args.img_shape, transform=transform)
        # task08_train = SemiTask082D(Path('E:\\datasets\\MSD\\Task08_HepaticVessel\\Task08\\train'), partition=self.args.semi_partition, size=self.args.img_shape, transform=transform)
        # task08_train = SemiTask083D(Path('E:\\datasets\\MSD\\Task08_HepaticVessel\\Task08\\train'), partition=0.3, size=self.args.img_shape, transform=transform)
        # task09_train = SemiTask092D(Path('E:\\datasets\\MSD\\Task09_Spleen\\Task09\\train'), partition=0.3, size=self.args.img_shape, transform=transform)
        tace_train = SemiTACESeg(Path('E:\datasets\\arcade\syntax\\train'), partition=0.3, size=self.args.img_shape, transform=transform)

        total_slices = len(tace_train)
        # labeled_slice = patients_to_slices(args.root_path, args.labeled_num)
        labeled_slice = tace_train.gt_num
        print("Total silices is: {}, labeled slices is: {}".format(total_slices, labeled_slice))

        labeled_idxs = list(range(0, labeled_slice))
        unlabeled_idxs = list(range(labeled_slice, total_slices))
        
        # batch_sampler = TwoStreamBatchSampler_L(labeled_idxs, unlabeled_idxs, self.args.batch_size, self.args.labeled_bs)
        batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, self.args.batch_size, self.args.batch_size-self.args.labeled_bs)

        self.train_loader = Data.DataLoader(tace_train, batch_sampler=batch_sampler, pin_memory=True, num_workers=0)

        # 定义 validate set
        tace_val = SemiTACESeg(Path('E:\datasets\\arcade\syntax\\val'), partition=1, size=self.args.img_shape)
        # task02_val = SemiTask022D(Path('E:\\datasets\\MSD\\Task02_Heart\\Task02\\val'), size=self.args.img_shape)
        # task07_val = SemiTask072D(Path('E:\\datasets\\MSD\\Task07_Pancreas\\Task07\\val'),size=self.args.img_shape)
        # task08_val = SemiTask082D(Path('E:\\datasets\\MSD\\Task08_HepaticVessel\\Task08\\val'), size=self.args.img_shape)
        # task08_val = SemiTask083D(Path('E:\\datasets\\MSD\\Task08_HepaticVessel\\Task08\\val'), size=self.args.img_shape)
        # task09_val = SemiTask092D(Path('E:\\datasets\\MSD\\Task09_Spleen\\Task09\\val'), size=self.args.img_shape)
        self.val_loader = Data.DataLoader(tace_val, batch_size=16, shuffle=False, num_workers=0)

        self.global_step = (self.now_epoch+1) * len(self.train_loader)
        self.tb_save_step = self.global_step // self.args.tb_save_freq
        self.best_performance = 0
        best_dice_mean = 0

        with open(str(Path(self.args.log_save_dir) /'log_loss.txt'), 'a') as f:  # 设置文件对象
            print('\nDate:{}'.format(time.strftime( '%Y-%m-%d %H:%M:%S',time.localtime(time.time()))), flush=True, file = f)
        
        self.args.epoch = self.args.max_iter // len(self.train_loader) if self.args.max_iter is not None else self.args.epoch

        for epoch in range(self.now_epoch+1, self.args.epoch+1):
            epoch_info = "Epoch:{}/{} - ".format(epoch, self.args.epoch) + time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
            print(epoch_info)
            for sampled_batch in self.train_loader:
                volume_batch, label_batch = sampled_batch['image'].to(self.device).float(), sampled_batch['mask'].to(self.device).float()

                labeled_image, labeled_mask = volume_batch[:self.args.labeled_bs], label_batch[:self.args.labeled_bs]
                unlabeled_image = volume_batch[self.args.labeled_bs:]

                self.optimize(epoch, labeled_image, labeled_mask, unlabeled_image, label_batch[self.args.labeled_bs:])
                
                if self.global_step % 500 == 0:
                    self.val_model(epoch)

                # if self.global_step % 2000 == 0:
                #     self.tsne(epoch, self.val_loader)
                    
            # 打印信息
            self.write_tb(epoch) 
            self.print_loss(epoch_info, is_write=True)
            self.clear_recoder()

            if epoch % self.args.save_per_epoch == 0:
                self.save_model(epoch=epoch)

            if self.args.max_iter>0 and self.global_step >= self.args.max_iter:
                self.save_model(epoch=epoch)
                break
    
    def val_model(self, epoch, val_loader=None):
        val_loader = val_loader if val_loader is not None else self.val_loader
        num_class = 2
        metric_list = 0.0
        gap = len(val_loader)

        self.encoder.eval()
        self.atts.eval()
        self.decoder.eval()
        
        num_classes = self.args.out_channel-1
        for i_batch, sampled_batch in enumerate(val_loader):
            # metric_i = test_single_volume(sampled_batch["image"], sampled_batch["label"].unsqueeze(1), model, classes=num_classes, patch_size=self.args.img_shape)
            image = sampled_batch["image"].float().to(self.device)
            label = sampled_batch["mask"].numpy()
            
            with torch.no_grad():
                out = self.infer(image).round()
                prediction = out.cpu().detach().numpy()

            tmp_metric_list = []
            for i in range(1, num_class):
                tmp_metric_list.append(calculate_metric_percase(prediction==i, label==i))
            # return metric_list
            
            if random.random()<0.1:
                ran_id = int(random.random() * image.shape[2])
                image = image/255 if image.max()>50 else image
                self.writer.add_image('val/image', image[0,:,:,:], epoch*gap+i_batch)
                self.writer.add_image('val/true_mask', label[0,:,:,:]/num_classes, epoch*gap+i_batch)          
                self.writer.add_image('val/pred_mask', prediction[0,:,:,:]/num_classes, epoch*gap+i_batch)     

            metric_list += np.array(tmp_metric_list)
        metric_list = metric_list / len(val_loader)
        
        for class_i in range(num_class-1):
            self.writer.add_scalar('info/val_{}_dice'.format(class_i+1), metric_list[class_i, 0], self.global_step)
            self.writer.add_scalar('info/val_{}_hd95'.format(class_i+1), metric_list[class_i, 1], self.global_step)

        performance = np.mean(metric_list, axis=0)[0]

        mean_hd95 = np.mean(metric_list, axis=0)[1]
        self.writer.add_scalar('info/val_mean_dice', performance, self.global_step)
        self.writer.add_scalar('info/val_mean_hd95', mean_hd95, self.global_step)

        if performance > self.best_performance:
            ckp = {}
            self.best_performance = performance
            save_mode_path = self.args.save_dir / 'iter_{}_dice_{}.pth'.format(self.global_step, round(self.best_performance, 4))
            save_best = self.args.save_dir / 'best_model.pth'
            
            ckp['encoder'] = self.encoder.state_dict()
            ckp['atts'] = self.atts.state_dict()
            ckp['decoder'] = self.decoder.state_dict()
            torch.save(ckp, str(save_mode_path))
            torch.save(ckp, str(save_best))

        # logging.info(
        #     'iteration %d : mean_dice : %f mean_hd95 : %f' % (iter_num, performance, mean_hd95))

        print('\niteration %d : mean_dice : %f mean_hd95 : %f' % (self.global_step, performance, mean_hd95))

        self.encoder.train()
        self.atts.train()
        self.decoder.train()

    def test_model(self, epoch=None, test_loader=None):
        test_loader = test_loader if test_loader is not None else self.test_loader
        self.encoder.eval()
        self.atts.eval()
        self.decoder.eval()

        dice_ = []
        dice2_ = []
        prec_ = []
        recall_ = []
        jc_ = []
        hd_ = []
        asd_ = []
        
        step = 0
        gap = len(test_loader)
        img_ls = []
        lab_ls = []
        name_ls = []
        slice_ls = []
        seg_ls = []

        with torch.no_grad():
            dices = []
            for sampled_batch in test_loader:
                volume_batch, label_batch = sampled_batch['image'], sampled_batch['mask']
                volume_batch, label_batch = volume_batch.to(self.device).float(), label_batch.to(self.device).float()

                img_ls = img_ls + list(volume_batch.chunk(volume_batch.shape[0], dim=0))
                lab_ls = lab_ls + list(label_batch.chunk(label_batch.shape[0], dim=0))
                name_ls = name_ls + [x for x in sampled_batch['name']]
                # slice_ls = slice_ls + [int(x) for x in sampled_batch['slice_id']]

                out_seg = self.infer(volume_batch)
                out_seg = out_seg.round()

                seg_ls = seg_ls + list(out_seg.chunk(out_seg.shape[0], 0))

                # ran_id = int(random.random() * volume_batch.shape[0])
                # self.writer.add_image('test/image', volume_batch[ran_id,0:1,:,:], epoch*gap+step)
                # self.writer.add_image('test/true_mask', label_batch[ran_id,0:1,:,:], epoch*gap+step)          
                # self.writer.add_image('test/pred_mask', out_seg[ran_id,0:1,:,:], epoch*gap+step) 

                step = step + 1

                for i in range(out_seg.shape[0]):
                    dice_.append(dice_score(label_batch[i:i+1], out_seg[i:i+1]).cpu().numpy())
                    prec_.append(precision(label_batch[i:i+1], out_seg[i:i+1]).cpu().numpy())
                    recall_.append(recall(label_batch[i:i+1], out_seg[i:i+1]).cpu().numpy())
                    dice, jc, hd, asd = calculate_metric_parcase2(out_seg.cpu().numpy(), label_batch.cpu().numpy())
                    dice2_.append(dice)
                    jc_.append(jc)
                    hd_.append(hd)
                    asd_.append(asd)
                
            if epoch is None:
                for b in range(len(img_ls)):
                    # out_dir = Path(self.args.root_path) / self.args.test_jpg_outpath / (name_ls[b])
                    out_dir = Path(self.args.root_path) / self.args.test_jpg_outpath
                    out_dir.mkdir(exist_ok=True)
                    vutils.save_image(img_ls[b][0], str(out_dir / f'{name_ls[b]}_img.jpg'), normalize=True)
                    vutils.save_image(lab_ls[b][0], str(out_dir / f'{name_ls[b]}_lab.jpg'), normalize=True)
                    vutils.save_image(seg_ls[b][0], str(out_dir / f'{name_ls[b]}_seg_{dice_[b].round(4)}.jpg'), normalize=True) 
            
            dice_np = np.array(dice_)
            prec_np = np.array(prec_)
            recall_np = np.array(recall_)
            dice2_np = np.array(dice2_)
            jc_np = np.array(jc_)
            hd_np = np.array(hd_)
            asd_np = np.array(asd_)

            dice_info = f'\nepoch:{epoch}\n'

            dice_info = dice_info + 'Dice1:{},{}; \nDice2:{},{}; \nPrecision:{},{}; \nRecall:{},{};\nJaccard:{},{};\nHD95:{},{};\nASD:{},{}'.format(
                                    dice_np.mean(), dice_np.std(), dice2_np.mean(), dice2_np.std(), prec_np.mean(), prec_np.std(), recall_np.mean(), recall_np.std(),
                                    jc_np.mean(), jc_np.std(), hd_np.mean(), hd_np.std(), asd_np.mean(), asd_np.std()) 
            
            print(dice_info)
            # with open(str('test_acc.txt'), 'a') as f:  # 设置文件对象
            with open(str(Path(self.args.log_save_dir) / 'log_test.txt'), 'a') as f:
                print(dice_info, flush=True, file = f)

        self.encoder.train()
        self.atts.train()
        self.decoder.train()

        return dice_np.mean()
    
    def tsne(self, epoch, tsne_loader):
        dictX = defaultdict(list)
        dictY = defaultdict(list)

        self.encoder.eval()
        
        for i_batch, sampled_batch in enumerate(tsne_loader):
            # metric_i = test_single_volume(sampled_batch["image"], sampled_batch["label"].unsqueeze(1), model, classes=num_classes, patch_size=self.args.img_shape)
            volume_batch, label_batch = sampled_batch['image'].to(self.device).float(), sampled_batch['mask'].to(self.device).float()

            labeled_image, labeled_mask = volume_batch[:self.args.labeled_bs], label_batch[:self.args.labeled_bs]

            with torch.no_grad():
                rep_ls, rep_lab_ls = self.encoder.tsne_forward(labeled_image, labeled_mask)
                for d in range(len(rep_ls)):
                    dictX[d].append(rep_ls[d].cpu().numpy())
                    dictY[d].append(rep_lab_ls[d].int().cpu().numpy())

        for d in dictX.keys():
            
            X = np.concatenate(dictX[d])
            Y = np.concatenate(dictY[d])
            
            ind = random.sample(range(X.shape[0]), max(X.shape[0]//100, 1000))
            tmpX = X[ind]
            tmpY = Y[ind]
            
            label = np.unique(tmpY).tolist()
            label = [str(int(lab)) for lab in label]

            ts = TSNE(n_components=2)
            # 训练模型
            X_tsne = ts.fit_transform(tmpX)
            plt.figure()
            handle = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=tmpY, label="t-SNE")
            plt.legend(handles=handle.legend_elements()[0],labels=label,title="Classes")
            plt.savefig(Path(self.args.root_path)/'tsne'/f'{epoch}_{d}.png', dpi=300)
            plt.close()

        self.encoder.train()
