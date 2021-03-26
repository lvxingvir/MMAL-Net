import torch
from torch import nn
import torch.nn.functional as F
from networks import resnet
from config import pretrain_path, coordinates_cat, iou_threshs, window_nums_sum, ratios, N_list
# from config_appm_eval import pretrain_path, coordinates_cat, iou_threshs, window_nums_sum, ratios, N_list
import numpy as np
from utils.AOLM import AOLM

from torchvision import transforms,utils


def nms(scores_np, proposalN, iou_threshs, coordinates):
    if not (type(scores_np).__module__ == 'numpy' and len(scores_np.shape) == 2 and scores_np.shape[1] == 1):
        raise TypeError('score_np is not right')

    windows_num = scores_np.shape[0]
    indices_coordinates = np.concatenate((scores_np, coordinates), 1)

    indices = np.argsort(indices_coordinates[:, 0])
    indices_coordinates = np.concatenate((indices_coordinates, np.arange(0,windows_num).reshape(windows_num,1)), 1)[indices]                  #[339,6]
    indices_results = []

    res = indices_coordinates

    while res.any():
        indice_coordinates = res[-1]
        indices_results.append(indice_coordinates[5])

        if len(indices_results) == proposalN:
            return np.array(indices_results).reshape(1,proposalN).astype(np.int)
        res = res[:-1]

        # Exclude anchor boxes with selected anchor box whose iou is greater than the threshold
        start_max = np.maximum(res[:, 1:3], indice_coordinates[1:3])
        end_min = np.minimum(res[:, 3:5], indice_coordinates[3:5])
        lengths = end_min - start_max + 1
        intersec_map = lengths[:, 0] * lengths[:, 1]
        intersec_map[np.logical_or(lengths[:, 0] < 0, lengths[:, 1] < 0)] = 0
        iou_map_cur = intersec_map / ((res[:, 3] - res[:, 1] + 1) * (res[:, 4] - res[:, 2] + 1) +
                                      (indice_coordinates[3] - indice_coordinates[1] + 1) *
                                      (indice_coordinates[4] - indice_coordinates[2] + 1) - intersec_map)
        res = res[iou_map_cur <= iou_threshs]

    while len(indices_results) != proposalN:
        indices_results.append(indice_coordinates[5])

    return np.array(indices_results).reshape(1, -1).astype(np.int)

class APPM(nn.Module):
    def __init__(self):
        super(APPM, self).__init__()
        self.avgpools = [nn.AvgPool2d(ratios[i], 1) for i in range(len(ratios))]

    def forward(self, proposalN, x, ratios, window_nums_sum, N_list, iou_threshs, DEVICE='cuda'):
        batch, channels, _, _ = x.size()
        avgs = [self.avgpools[i](x) for i in range(len(ratios))]

        # feature map sum
        fm_sum = [torch.sum(avgs[i], dim=1) for i in range(len(ratios))]

        all_scores = torch.cat([fm_sum[i].view(batch, -1, 1) for i in range(len(ratios))], dim=1)
        windows_scores_np = all_scores.data.cpu().numpy()
        window_scores = torch.from_numpy(windows_scores_np).to(DEVICE).reshape(batch, -1)

        # nms
        proposalN_indices = []
        for i, scores in enumerate(windows_scores_np):
            indices_results = []
            for j in range(len(window_nums_sum)-1):
                indices_results.append(nms(scores[sum(window_nums_sum[:j+1]):sum(window_nums_sum[:j+2])], proposalN=N_list[j], iou_threshs=iou_threshs[j],
                                           coordinates=coordinates_cat[sum(window_nums_sum[:j+1]):sum(window_nums_sum[:j+2])]) + sum(window_nums_sum[:j+1]))
            # indices_results.reverse()
            proposalN_indices.append(np.concatenate(indices_results, 1))   # reverse

        proposalN_indices = np.array(proposalN_indices).reshape(batch, proposalN)
        proposalN_indices = torch.from_numpy(proposalN_indices).type(torch.long).to(DEVICE)
        proposalN_windows_scores = torch.cat(
            [torch.index_select(all_score, dim=0, index=proposalN_indices[i]) for i, all_score in enumerate(all_scores)], 0).reshape(
            batch, proposalN)

        return proposalN_indices, proposalN_windows_scores, window_scores

class MainNet(nn.Module):
    def __init__(self, proposalN, num_classes, channels):
        # nn.Module子类的函数必须在构造函数中执行父类的构造函数
        super(MainNet, self).__init__()
        self.num_classes = num_classes
        self.proposalN = proposalN
        self.pretrained_model = resnet.resnet50(pretrained=True, pth_path=pretrain_path)
        self.rawcls_net = nn.Linear(channels, num_classes)
        self.APPM = APPM()

    def forward(self, x, epoch, batch_idx, status='test', DEVICE='cuda'):
        fm, embedding, conv5_b = self.pretrained_model(x)
        batch_size, channel_size, side_size, _ = fm.shape
        assert channel_size == 2048

        # raw branch
        raw_logits = self.rawcls_net(embedding)

        #SCDA
        coordinates = torch.tensor(AOLM(fm.detach(), conv5_b.detach()))

        local_imgs = torch.zeros([batch_size, 3, 448, 448]).to(DEVICE)  # [N, 3, 448, 448]

        for i in range(batch_size):
            [x0, y0, x1, y1] = coordinates[i]
            local_imgs[i:i + 1] = F.interpolate(x[i:i + 1, :, x0:(x1+1), y0:(y1+1)], size=(448, 448),
                                                mode='bilinear', align_corners=True)  # [N, 3, 224, 224]
        local_imgs = x
        local_fm, local_embeddings, _ = self.pretrained_model(local_imgs.detach())  # [N, 2048]
        local_logits = self.rawcls_net(local_embeddings)  # [N, 200]

        proposalN_indices, proposalN_windows_scores, window_scores \
            = self.APPM(self.proposalN, local_fm.detach(), ratios, window_nums_sum, N_list, iou_threshs, DEVICE)

        if status == "train":
            # window_imgs cls
            window_imgs = torch.zeros([batch_size, self.proposalN, 3, 224, 224]).to(DEVICE)  # [N, 4, 3, 224, 224]
            for i in range(batch_size):
                for j in range(self.proposalN):
                    [x0, y0, x1, y1] = coordinates_cat[proposalN_indices[i, j]]
                    window_imgs[i:i + 1, j] = F.interpolate(local_imgs[i:i + 1, :, x0:(x1 + 1), y0:(y1 + 1)], size=(224, 224),
                                                                mode='bilinear',
                                                                align_corners=True)  # [N, 4, 3, 224, 224]

            window_imgs = window_imgs.reshape(batch_size * self.proposalN, 3, 224, 224)  # [N*4, 3, 224, 224]
            _, window_embeddings, _ = self.pretrained_model(window_imgs.detach())  # [N*4, 2048]
            proposalN_windows_logits = self.rawcls_net(window_embeddings)  # [N* 4, 200]
        else:
            proposalN_windows_logits = torch.zeros([batch_size * self.proposalN, self.num_classes]).to(DEVICE)

        return proposalN_windows_scores, proposalN_windows_logits, proposalN_indices, \
               window_scores, coordinates, raw_logits, local_logits, local_imgs
class MainNet_2input(nn.Module):
    def __init__(self, proposalN, num_classes, channels):
        # nn.Module子类的函数必须在构造函数中执行父类的构造函数
        super(MainNet_2input, self).__init__()
        self.num_classes = num_classes
        self.proposalN = proposalN
        self.raw_model = resnet.resnet50(pretrained=True, pth_path=pretrain_path)
        # self.pretrained_model = resnet.resnet50(pretrained=True, pth_path=pretrain_path)
        self.window_model = resnet.resnet50(pretrained=True, pth_path=pretrain_path)
        self.rawcls_net = nn.Linear(channels, num_classes)
        self.localcls_net = nn.Linear(channels, num_classes)
        self.window_net = nn.Linear(channels, num_classes)
        self.APPM = APPM()

        self.attention = nn.Sequential(
            nn.Linear(channels, 128),
            nn.Tanh(),
            nn.Linear(128, num_classes)

        )

    def forward(self, x_1,x_2, epoch, batch_idx, status='test', DEVICE='cuda'):
        fm, embedding, conv5_b = self.raw_model(x_1)
        batch_size, channel_size, side_size, _ = fm.shape
        assert channel_size == 2048

        # raw branch
        raw_logits = self.rawcls_net(embedding)


        # #SCDA
        coordinates = []
        # coordinates = torch.tensor(AOLM(fm.detach(), conv5_b.detach()))
        #
        # local_imgs = torch.zeros([batch_size, 3, 448, 448]).to(DEVICE)  # [N, 3, 448, 448]
        #
        # for i in range(batch_size):
        #     [x0, y0, x1, y1] = coordinates[i]
        #     local_imgs[i:i + 1] = F.interpolate(x_1[i:i + 1, :, x0:(x1+1), y0:(y1+1)], size=(448, 448),
        #                                         mode='bilinear', align_corners=True)  # [N, 3, 224, 224]

        local_imgs = x_1
        # local_windws = nn.Unfold(x_2,)
        local_fm, local_embeddings, _ = self.window_model(local_imgs.detach())  # [N, 2048]
        local_logits = self.localcls_net(local_embeddings)  # [N, 200]
        # proposalN_windows_logits = self.localcls_net(local_embeddings)  # [N, 200]

        proposalN_indices, proposalN_windows_scores, window_scores \
            = self.APPM(self.proposalN, local_fm.detach(), ratios, window_nums_sum, N_list, iou_threshs, DEVICE)

        # if status == "train":
        if True:
            wsz = 224
            # window_imgs cls
            window_imgs = torch.zeros([batch_size, self.proposalN+4, 3, wsz, wsz]).to(DEVICE)  # [N, 4, 3, 224, 224]
            # window_imgs = torch.zeros([batch_size, self.proposalN, 3, wsz, wsz]).to(DEVICE)
            for i in range(batch_size):
                for j in range(self.proposalN):
                    [x0, y0, x1, y1] = coordinates_cat[proposalN_indices[i, j]]
                    window_imgs[i:i + 1, j] = F.interpolate(local_imgs[i:i + 1, :, x0:(x1 + 1), y0:(y1 + 1)], size=(wsz, wsz),
                                                                mode='bilinear',
                                                                align_corners=True)  # [N, 4, 3, 224, 224]
                for k in range(4):
                    x0 = k%2
                    y0 = k//2
                    # print(224*x0,224*(x0+1),224*y0,224*(y0+1))
                    window_imgs[i:i + 1, self.proposalN+k] = F.interpolate(local_imgs[i:i+1,:,224*x0:224*(x0+1),224*y0:224*(y0+1)],size=(wsz, wsz),
                                                                mode='bilinear',
                                                                align_corners=True)

            window_imgs = window_imgs.reshape(batch_size * (self.proposalN+4), 3, wsz, wsz)  # [N*4, 3, 224, 224]
            # window_imgs = window_imgs.reshape(batch_size * self.proposalN, 3, wsz, wsz)  # [N*4, 3, 224, 224]

            _, window_embeddings, _ = self.window_model(window_imgs.detach())  # [N*4, 2048]

            proposalN_windows_logits = self.attention(window_embeddings).reshape(raw_logits.size()[0], 1, -1)
            # proposalN_windows_logits = torch.transpose(proposalN_windows_logits,1,0)
            proposalN_windows_logits = F.softmax(proposalN_windows_logits, dim=2)

            M = torch.bmm(proposalN_windows_logits, window_embeddings.reshape(raw_logits.size()[0], self.proposalN+4, -1))
            # M = torch.bmm(proposalN_windows_logits,
            #               window_embeddings.reshape(raw_logits.size()[0], self.proposalN, -1))
            windows_logits = self.window_net(M)  # [N* 4, 200]



            # proposalN_windows_logits = self.window_net(window_embeddings)  # [N* 4, 200]
            # local_logits = self.rawcls_net(window_embeddings)  # [N* 4, 200]
        # else:
        #     # proposalN_windows_logits = torch.zeros([batch_size * self.proposalN, self.num_classes]).to(DEVICE)
        #
        #     window_imgs = torch.zeros([batch_size, self.proposalN, 3, 224, 224]).to(DEVICE)  # [N, 4, 3, 224, 224]
        #     for i in range(batch_size):
        #         for j in range(self.proposalN):
        #             [x0, y0, x1, y1] = coordinates_cat[proposalN_indices[i, j]]
        #             window_imgs[i:i + 1, j] = F.interpolate(local_imgs[i:i + 1, :, x0:(x1 + 1), y0:(y1 + 1)],
        #                                                     size=(224, 224),
        #                                                     mode='bilinear',
        #                                                     align_corners=True)  # [N, 4, 3, 224, 224]
        #
        #     window_imgs = window_imgs.reshape(batch_size * self.proposalN, 3, 224, 224)  # [N*4, 3, 224, 224]
        #     _, window_embeddings, _ = self.raw_model(window_imgs.detach())  # [N*4, 2048]
        #     proposalN_windows_logits = self.rawcls_net(window_embeddings)  # [N* 4, 200]

        return proposalN_windows_scores, windows_logits, proposalN_indices, \
               window_scores, coordinates, raw_logits, local_logits, local_imgs