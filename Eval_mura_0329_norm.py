#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   Eval_mura.py    
@Contact :   lvxingvir@gmail.com
@License :   (C)Copyright Xing and UCSD

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
03/16/2021 19:16  Xing        1.0         None
'''

import argparse
import os
import sys
import pandas as pd
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

# import matplotlib
# matplotlib.use('Qt5Agg')

import cv2
import matplotlib.pyplot as plt



# from config_bdpt import input_size, root, proposalN, channels
from config_bdpt import input_size, root, proposalN, channels,set
from utils.read_dataset import read_dataset
from utils.auto_laod_resume import auto_load_resume
from utils.utils import TrainClock, save_args, AverageMeter, AUCMeter,calculate_accuracy
# from networks.model_bdpt import MainNet
# from networks.model import MainNet
from networks.model_bdpt_mil import MainNet


def parse_arg():
    parser = argparse.ArgumentParser(description='Eval_mura')
    parser.add_argument('--ensamble', help='using ensambled model', default=True, type=bool)
    parser.add_argument('--dataset', help='using ensambled model', default=r'', type=str)
    parser.add_argument('--pth_dir', help='main path', default=r'C:/Users/Xing/Projects/AirGo/MMAL-Net/checkpoint/', type=str)
    parser.add_argument('--root', help='data_root', default=r'E:\Xing\Data\MURA-v1.1', type=str)
    parser.add_argument('--bs', help='batch size', default=1, type=int)
    parser.add_argument('--model_name', help='model name',
                        default={'All':'mura_bp_bimodel_corr_mil_0326_test'}, type=dict)
    args = parser.parse_args()
    return args


class Eval_mura(object):
    def __init__(self, args):
        self.ensb = args.ensamble
        self.dataset = args.dataset
        self.pth_dir = args.pth_dir
        self.model_dict = {}
        self.model_name = args.model_name
        self.bs = args.bs
        self.root = args.root
        self.set = set

        for key in self.model_name:
            self.model_dict[key] = os.path.join(self.pth_dir,self.model_name[key],'window_best_model.pth')


        os.environ['CUDA_VISIBLE_DEVICES'] = '3'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = MainNet(proposalN=proposalN, num_classes=7, channels=channels)
        self.model.to(self.device)

        _,self.testloader = read_dataset(input_size, self.bs, self.root, self.set)

    def model_eval(self):
        criterion = nn.BCELoss()
        losses_val = AverageMeter('loss')
        accuracies_val = AverageMeter('acc')

        set = self.set

        self.model.eval()

        y = []
        pred = []
        bd_part = []
        ID_lst = []

        study_out = {}
        study_label = {}

        # fileio.maybe_make_new_dir(result_path)
        with torch.no_grad():
            for i, data in tqdm(enumerate(self.testloader)):

                if i<10000:

                    if set == 'CUB':
                        x, label, boxes, _ = data
                    elif set == 'Mura_bp':
                        x, label_bp, label, meta_data = data
                    else:
                        x, xs, label, meta_data = data

                    encounter = meta_data['encounter']
                    bd_key = str.capitalize(meta_data['study_type'][0])
                    # bd_key = 'All'

                    # pth_path = self.model_dict[bd_key]
                    pth_path = self.model_dict['All']
                    # 加载checkpoint
                    if os.path.exists(pth_path):
                        epoch = auto_load_resume(self.model, pth_path, status='test')
                    else:
                        sys.exit('There is not a pth exist.')

                    image_val = x.to(self.device)
                    # images_val = xs.to(self.device)
                    # images_val = F.interpolate(images_val, size=[448, 448])
                    targets_val = label.float().to(self.device)
                    # outputs_logits = self.model(image_val, images_val, epoch, i, 'test', self.device)[1]
                    outputs_logits = self.model(image_val, epoch, i, 'test', self.device)[1]
                    outputs_val = F.sigmoid(outputs_logits)

                    #         print(targets_val.shape,outputs_val.shape)

                    #         targets_val = data['label'].float().cuda()
                    #         images_val = data['image'].float().cuda()
                    #         outputs_val = model(images_val)
                    loss_val = criterion(outputs_val, targets_val)

                    #         add_gl_image_index(images_val, patches_val, outputs_val, targets_val,  writer, subset='val', epoch=0, index = j)

                    acc_val = calculate_accuracy(outputs_val, targets_val)
                    losses_val.update(loss_val.item(), targets_val.size(0))
                    accuracies_val.update(acc_val, targets_val.size(0))
                    y.append(targets_val.cpu().numpy())
                    pred.append(outputs_val[0][0].cpu().numpy())
                    bd_part.append(bd_key)
                    ID_lst.append(encounter[0])

                    for j in range(len(outputs_logits)):
                        if study_out.get(encounter[j], -1) == -1:
                            study_out[encounter[j]] = [outputs_val[j].item()]
                            study_label[encounter[j]] = targets_val[j].item()
                        else:
                            study_out[encounter[j]] += [outputs_val[j].item()]

            print('val_loss: ', losses_val.avg, 'val_acc: ', accuracies_val.avg)

            self.df= pd.DataFrame({
                'ID':ID_lst,
                'Bodypart':bd_part,
                'y':y,
                'pred':pred
            })

            st_pred = []
            st_y = []
            index = []
            for key in study_out.keys():
                index.append(key)
                st_pred.append(np.mean(study_out[key]))
                st_y.append(study_label[key])

        return np.asarray(y).squeeze(),np.asarray(pred).squeeze(),np.asarray(st_y).squeeze(),np.asarray(st_pred).squeeze()

    def draw_roc(self,y,pred,title = ''):
        from sklearn import metrics
        # % matplotlib inline

        fpr, tpr, thresholds = metrics.roc_curve(y, pred)

        roc_auc = metrics.auc(fpr, tpr)

        plt.figure()
        lw = 2
        plt.plot(fpr, tpr, color='darkorange',
                 lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(title)
        plt.legend(loc="lower right")
        # plt.show()
        plt.savefig(fname = title + '.png',dpi=300)

    def draw_kappa(self,y,pred,title = ''):
        import numpy
        from sklearn.metrics import cohen_kappa_score,confusion_matrix
        mali = []
        beni = []
        cohen = []
        tt = numpy.arange(0, 1, 0.01)
        for thres in tt:
            #     print(thres)
            pred_t = pred > thres
            pred_t.astype(int)
            cm = confusion_matrix(y, pred_t)
            mali.append(cm[1, 1] / sum(cm[1, :]))
            beni.append(cm[0, 0] / sum(cm[0, :]))
            cohen.append(cohen_kappa_score(y, pred_t))
        # print(thresholds)
        plt.figure(), plt.plot(tt, mali, label='TPR'), plt.plot(tt, beni, label='TNR'), plt.plot(tt, cohen,
                                                                                                 label='kappa')
        plt.xlabel('Thresholds')
        plt.ylabel('TPR/TNR')

        plt.legend(loc="lower right")

        net_max = cohen.index(max(cohen))
        plt.title('TPR/TNR with Thresholds \n (with max_cohen_kappa_score of {} at {})'.format(max(cohen), tt[net_max]))
        plt.scatter(tt[net_max], cohen[net_max], color='b')
        # plt.show()
        plt.savefig(fname = title + '.png',dpi=300)

    @staticmethod
    def plot_confusion_matrix(cm,
                              target_names,
                              title='Confusion matrix',
                              cmap=None,
                              normalize=True):
        """
        given a sklearn confusion matrix (cm), make a nice plot

        Arguments
        ---------
        cm:           confusion matrix from sklearn.metrics.confusion_matrix

        target_names: given classification classes such as [0, 1, 2]
                      the class names, for example: ['high', 'medium', 'low']

        title:        the text to display at the top of the matrix

        cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                      see http://matplotlib.org/examples/color/colormaps_reference.html
                      plt.get_cmap('jet') or plt.cm.Blues

        normalize:    If False, plot the raw numbers
                      If True, plot the proportions

        Usage
        -----
        plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                                  # sklearn.metrics.confusion_matrix
                              normalize    = True,                # show proportions
                              target_names = y_labels_vals,       # list of names of the classes
                              title        = best_estimator_name) # title of graph

        Citiation
        ---------
        http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

        """
        import matplotlib.pyplot as plt
        import numpy as np
        import itertools

        accuracy = np.trace(cm) / float(np.sum(cm))
        misclass = 1 - accuracy

        if cmap is None:
            cmap = plt.get_cmap('Blues')

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        plt.figure(figsize=(4, 3))
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()

        if target_names is not None:
            tick_marks = np.arange(len(target_names))
            plt.xticks(tick_marks, target_names, rotation=45)
            plt.yticks(tick_marks, target_names)

        #     plt.figure(figsize=(4, 3))
        #     plt.imshow(cm, interpolation='nearest', cmap=cmap)
        #     plt.title(title)
        #     plt.colorbar()

        thresh = cm.max() / 1.5 if normalize else cm.max() / 2
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            if normalize:
                plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black")
            else:
                plt.text(j, i, "{:,}".format(cm[i, j]),
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
        # plt.show()
        plt.savefig(fname = title + '.png',dpi=300)

    def draw_confusionmatrix(self,y,pred,title = ''):
        from sklearn.metrics import confusion_matrix

        threshold = 0.5

        pred_t = pred > threshold
        pred_t.astype(int)

        cm = confusion_matrix(y, pred_t)

        self.plot_confusion_matrix(cm, ('Normal', 'Abnormal'), title=title, normalize=False)
        self.plot_confusion_matrix(cm, ('Normal', 'Abnormal'), title=title+'_norm', normalize=True)

    def rendering(self,y,pred,title=''):

        self.draw_roc(y, pred, title+'_roc')

        self.draw_kappa(y, pred, title+'_kappa')

        self.draw_confusionmatrix(y, pred, title+'_cm')



    def __call__(self, *args, **kwargs):

        y,pred,st_y,st_pred = self.model_eval()

        print('eval done! saving results...')

        self.df.to_csv('eval_result_window_0329.csv')

        self.rendering(y,pred,title='Normal Window 0329')

        self.rendering(st_y, st_pred, title='Study Window 0329')





if __name__ == "__main__":
    args = parse_arg()

    Eval_mura(args)()
