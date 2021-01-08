import os
import glob
import torch
import torch.nn.functional as F
from tqdm import tqdm
from tensorboardX import SummaryWriter
from config_bdpt import max_checkpoint_num, proposalN, eval_trainset, set
from utils.eval_model_bdpt import eval


def train(model,
          trainloader,
          testloader,
          criterion_bp,
          criterion_ls,
          optimizer,
          scheduler,
          save_path,
          start_epoch,
          end_epoch,
          save_interval):

    best_acc = 0

    for epoch in range(start_epoch + 1, end_epoch + 1):
        model.train()

        print('Training %d epoch' % epoch)

        lr = next(iter(optimizer.param_groups))['lr']

        for i, data in enumerate(tqdm(trainloader)):
            if set == 'CUB':
                images, labels, _, _ = data
            elif set == 'Mura_bp':
                images, labels_bp, labels = data
            else:
                images, labels = data
            images, labels_bp,labels = images.cuda(), labels_bp.cuda(), labels.float().cuda()

            optimizer.zero_grad()

            proposalN_windows_score, proposalN_windows_logits, indices, \
            window_scores, _, raw_logits, local_logits, _ = model(images, epoch, i, 'train')

            # raw_logits = F.sigmoid(raw_logits)   # for mura
            local_logits = F.sigmoid(local_logits)  # for mura
            proposalN_windows_logits = F.sigmoid(proposalN_windows_logits) # for mura

            raw_loss = criterion_bp(raw_logits, labels_bp)  #float for mura

            local_loss = criterion_ls(local_logits, labels) # float for mura
            windowscls_loss = criterion_ls(proposalN_windows_logits,
                               labels.unsqueeze(1).repeat(1, proposalN).view(-1))

            if epoch < 2:
                total_loss = raw_loss
            else:
                total_loss = 0.5*raw_loss + local_loss + 0.5*windowscls_loss

            total_loss.backward()

            optimizer.step()

        scheduler.step()

        # evaluation every epoch
        if eval_trainset and epoch%10==0:
            raw_loss_avg, windowscls_loss_avg, total_loss_avg, raw_accuracy, local_accuracy, local_loss_avg\
                = eval(model, trainloader, criterion_bp,criterion_ls,'train', save_path, epoch)

            print(
                'Train set: raw accuracy: {:.2f}%, local accuracy: {:.2f}%'.format(
                    100. * raw_accuracy, 100. * local_accuracy))

            # tensorboard
            with SummaryWriter(log_dir=os.path.join(save_path, 'log'), comment='train') as writer:

                writer.add_scalar('Train/learning rate', lr, epoch)
                writer.add_scalar('Train/raw_accuracy', raw_accuracy, epoch)
                writer.add_scalar('Train/local_accuracy', local_accuracy, epoch)
                writer.add_scalar('Train/raw_loss_avg', raw_loss_avg, epoch)
                writer.add_scalar('Train/local_loss_avg', local_loss_avg, epoch)
                writer.add_scalar('Train/windowscls_loss_avg', windowscls_loss_avg, epoch)
                writer.add_scalar('Train/total_loss_avg', total_loss_avg, epoch)

        # eval testset
        raw_loss_avg, windowscls_loss_avg, total_loss_avg, raw_accuracy, local_accuracy, \
        local_loss_avg\
            = eval(model, testloader, criterion_bp,criterion_ls, 'test', save_path, epoch)

        print(
            'Test set: raw accuracy: {:.2f}%, local accuracy: {:.2f}%'.format(
                100. * raw_accuracy, 100. * local_accuracy))

        # tensorboard
        with SummaryWriter(log_dir=os.path.join(save_path, 'log'), comment='test') as writer:
            writer.add_scalar('Test/raw_accuracy', raw_accuracy, epoch)
            writer.add_scalar('Test/local_accuracy', local_accuracy, epoch)
            writer.add_scalar('Test/raw_loss_avg', raw_loss_avg, epoch)
            writer.add_scalar('Test/local_loss_avg', local_loss_avg, epoch)
            writer.add_scalar('Test/windowscls_loss_avg', windowscls_loss_avg, epoch)
            writer.add_scalar('Test/total_loss_avg', total_loss_avg, epoch)

        # save checkpoint
        if (epoch % save_interval == 0) or (epoch == end_epoch):
            print('Saving checkpoint')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'learning_rate': lr,
            }, os.path.join(save_path, 'epoch' + str(epoch) + '.pth'))

        if best_acc<local_accuracy:
            print('Saving checkpoint')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'learning_rate': lr,
            }, os.path.join(save_path, 'best_model.pth'))
            best_acc = local_accuracy

        # Limit the number of checkpoints to less than or equal to max_checkpoint_num,
        # and delete the redundant ones
        checkpoint_list = [os.path.basename(path) for path in glob.glob(os.path.join(save_path, '*.pth'))]
        if len(checkpoint_list) == max_checkpoint_num + 1:
            idx_list = [int(name.replace('epoch', '').replace('.pth', '')) for name in checkpoint_list]
            min_idx = min(idx_list)
            os.remove(os.path.join(save_path, 'epoch' + str(min_idx) + '.pth'))

