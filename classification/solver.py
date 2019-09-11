from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from model.build_gen import *
from datasets.dataset_read import dataset_read

# import tensorboard
from tensorboardX import SummaryWriter


# Training settings
class Solver(object):
    def __init__(self, args, batch_size=64, source='svhn', target='mnist', learning_rate=0.0002, interval=100, optimizer='adam'
                 , num_k=4, all_use=False, checkpoint_dir=None, save_epoch=10):

        self.batch_size = batch_size
        self.source = source
        self.target = target
        self.num_k = num_k
        self.checkpoint_dir = checkpoint_dir
        self.save_epoch = save_epoch
        self.use_abs_diff = args.use_abs_diff
        self.all_use = all_use
        if self.source == 'svhn':
            self.scale = True
        else:
            self.scale = False
        print('dataset loading')
        self.datasets, self.dataset_test = dataset_read(source, target, self.batch_size, scale=self.scale,
                                                        all_use=self.all_use)
        print('load finished!')
        self.G = Generator(source=source, target=target)
        self.C1 = Classifier(source=source, target=target)
        self.C2 = Classifier(source=source, target=target)
        self.C3 = Classifier(source=source, target=target)

        if args.eval_only:
            self.G.torch.load(
                '%s/%s_to_%s_model_epoch%s_G.pt' % (self.checkpoint_dir, self.source, self.target, args.resume_epoch))
            self.G.torch.load(
                '%s/%s_to_%s_model_epoch%s_G.pt' % (
                    self.checkpoint_dir, self.source, self.target, self.checkpoint_dir, args.resume_epoch))
            self.G.torch.load(
                '%s/%s_to_%s_model_epoch%s_G.pt' % (self.checkpoint_dir, self.source, self.target, args.resume_epoch))

        self.G.cuda()
        self.C1.cuda()
        self.C2.cuda()
        self.C3.cuda()
        self.interval = interval
        self.writer = SummaryWriter()

        self.set_optimizer(which_opt=optimizer, lr=learning_rate)
        self.lr = learning_rate
        # self.bn_f = nn.BatchNorm1d(10, affine=False).cuda()

    def set_optimizer(self, which_opt='momentum', lr=0.001, momentum=0.9):
        if which_opt == 'momentum':
            self.opt_g = optim.SGD(self.G.parameters(),
                                   lr=lr, weight_decay=0.0005,
                                   momentum=momentum)

            self.opt_c1 = optim.SGD(self.C1.parameters(),
                                    lr=lr, weight_decay=0.0005,
                                    momentum=momentum)
            self.opt_c2 = optim.SGD(self.C2.parameters(),
                                    lr=lr, weight_decay=0.0005,
                                    momentum=momentum)
            self.opt_c3 = optim.SGD(self.C3.parameters(),
                                    lr=lr, weight_decay=0.0005,
                                    momentum=momentum)

        if which_opt == 'adam':
            self.opt_g = optim.Adam(self.G.parameters(),
                                    lr=lr, weight_decay=0.0005)

            self.opt_c1 = optim.Adam(self.C1.parameters(),
                                     lr=lr, weight_decay=0.0005)
            self.opt_c2 = optim.Adam(self.C2.parameters(),
                                     lr=lr, weight_decay=0.0005)
            self.opt_c3 = optim.Adam(self.C3.parameters(),
                                     lr=lr, weight_decay=0.0005)

    def reset_grad(self):
        self.opt_g.zero_grad()
        self.opt_c1.zero_grad()
        self.opt_c2.zero_grad()
        self.opt_c3.zero_grad()

    def ent(self, output):
        return - torch.mean(output * torch.log(output + 1e-6))

    def discrepancy(self, out1, out2):
        return torch.mean(torch.abs(F.softmax(out1, dim=1) - F.softmax(out2, dim=1)))

    def sort_rows(self, matrix, num_rows):
        # matrix_T = array_ops.transpose(matrix, [1, 0])
        matrix_T = torch.transpose(matrix, 1, 0)
        # sorted_matrix_T = nn_ops.top_k(matrix_T, num_rows)[0]
        sorted_matrix_T, _ = matrix_T.topk(k=num_rows, dim=1)

        return torch.transpose(sorted_matrix_T, 1, 0)
        # return array_ops.transpose(sorted_matrix_T, [1, 0])

    def discrepancy_slice_wasserstein(self, p1, p2):
        # s = array_ops.shape(p1)
        s = p1.shape
        if p1.shape[1] > 1:
        # if p1.get_shape().as_list()[1] > 1:
            # For data more than one-dimensional, perform multiple random projection to 1-D
            # proj = random_ops.random_normal([array_ops.shape(p1)[1], 128])
            proj = torch.randn(p1.shape[1], 128).cuda()

            temp = torch.sum(proj.pow(2), dim=0)
            proj *= torch.rsqrt(temp.reshape(1, temp.shape[0])).cuda()
            # proj *= math_ops.rsqrt(math_ops.reduce_sum(math_ops.square(proj), 0, keep_dims=True))
            # proj *= torch.rsqrt(torch.sum(proj.pow(2), dim=0)).cuda()

            p1 = p1.mm(proj)
            p2 = p2.mm(proj)
            # p1 = math_ops.matmul(p1, proj)
            # p2 = math_ops.matmul(p2, proj)

        p1 = self.sort_rows(p1, s[0])
        p2 = self.sort_rows(p2, s[0])

        wdist = torch.mean((p1-p2).pow(2))
        # wdist = math_ops.reduce_mean(math_ops.square(p1 - p2))
        return wdist
        # return torch.mean(wdist)
        # return math_ops.reduce_mean(wdist)

    def train(self, epoch, record_file=None):
        criterion = nn.CrossEntropyLoss().cuda()
        self.G.train()
        self.C1.train()
        self.C2.train()
        self.C3.train()
        torch.cuda.manual_seed(1)

        for batch_idx, data in enumerate(self.datasets):
            img_t = data['T']
            img_s = data['S']
            label_s = data['S_label']
            if img_s.size()[0] < self.batch_size or img_t.size()[0] < self.batch_size:
                break
            img_s = img_s.cuda()
            img_t = img_t.cuda()
            imgs = Variable(torch.cat((img_s, img_t), dim=0))
            label_s = Variable(label_s.long().cuda())

            img_s = Variable(img_s)
            img_t = Variable(img_t)

            # step A: training the model on source domain
            self.reset_grad()
            feat_s = self.G(img_s)
            output_s1 = self.C1(feat_s)
            output_s2 = self.C2(feat_s)
            output_s3 = self.C3(feat_s)


            loss_s1 = criterion(output_s1, label_s)
            loss_s2 = criterion(output_s2, label_s)
            loss_s3 = criterion(output_s3, label_s)
            loss_s = loss_s1 + loss_s2 + loss_s3
            loss_s.backward()
            self.opt_g.step()
            self.opt_c1.step()
            self.opt_c2.step()
            self.opt_c3.step()

            # step B: using the test data to update the parameters of the classifier
            self.reset_grad()
            feat_s = self.G(img_s)
            output_s1 = self.C1(feat_s)
            output_s2 = self.C2(feat_s)
            output_s3 = self.C3(feat_s)

            feat_t = self.G(img_t)
            output_t1 = self.C1(feat_t)
            output_t2 = self.C2(feat_t)
            output_t3 = self.C3(feat_t)

            loss_s1 = criterion(output_s1, label_s)
            loss_s2 = criterion(output_s2, label_s)
            loss_s3 = criterion(output_s3, label_s)
            loss_s = loss_s1 + loss_s2 + loss_s3

            loss_dis1 = self.discrepancy(output_t1, output_t2)
            loss_dis2 = self.discrepancy(output_t1, output_t3)
            loss_dis3 = self.discrepancy(output_t2, output_t3)
            loss_dis = loss_dis1 + loss_dis2 + loss_dis3
            # loss_dis = self.discrepancy_slice_wasserstein(F.softmax(output_t1), F.softmax(output_t2))
            loss = loss_s - loss_dis
            loss.backward()
            self.opt_c1.step()
            self.opt_c2.step()
            self.opt_c3.step()

            # step C: using the target data to update the generator
            self.reset_grad()
            for i in xrange(self.num_k):
                feat_t = self.G(img_t)
                output_t1 = self.C1(feat_t)
                output_t2 = self.C2(feat_t)
                output_t3 = self.C3(feat_t)
                # loss_dis = self.discrepancy_slice_wasserstein(F.softmax(output_t1), F.softmax(output_t2))
                # loss_dis = self.discrepancy(output_t1, output_t2)
                loss_dis1 = self.discrepancy(output_t1, output_t2)
                loss_dis2 = self.discrepancy(output_t1, output_t3)
                loss_dis3 = self.discrepancy(output_t2, output_t3)
                loss_dis = loss_dis1 + loss_dis2 + loss_dis3
                loss_dis.backward()
                self.opt_g.step()
                self.reset_grad()

            if batch_idx > 500: 
                return batch_idx

            if batch_idx % self.interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss1: {:.6f}\t Loss2: {:.6f}\t Loss3: {:.6f}\t Discrepancy: {:.6f}'.format(
                    epoch, batch_idx, 500, (100. * batch_idx * self.batch_size)/ 70000, loss_s1.data, loss_s2.data, loss_s3.data, loss_dis.data))
                if record_file:
                    record = open(record_file, 'a')
                    record.write('%s %s %s %s\n' % (loss_dis.data, loss_s1.data, loss_s2.data, loss_s3.data))
                    record.close()

        return batch_idx

    def train_onestep(self, epoch, record_file=None):
        criterion = nn.CrossEntropyLoss().cuda()
        self.G.train()
        self.C1.train()
        self.C2.train()
        torch.cuda.manual_seed(1)

        for batch_idx, data in enumerate(self.datasets):
            img_t = data['T']
            img_s = data['S']
            label_s = data['S_label']
            if img_s.size()[0] < self.batch_size or img_t.size()[0] < self.batch_size:
                break
            img_s = img_s.cuda()
            img_t = img_t.cuda()
            label_s = Variable(label_s.long().cuda())
            img_s = Variable(img_s)
            img_t = Variable(img_t)
            self.reset_grad()
            feat_s = self.G(img_s)
            output_s1 = self.C1(feat_s)
            output_s2 = self.C2(feat_s)
            loss_s1 = criterion(output_s1, label_s)
            loss_s2 = criterion(output_s2, label_s)
            loss_s = loss_s1 + loss_s2
            loss_s.backward(retain_graph=True)
            # loss_s.backward(retain_variables=True)
            feat_t = self.G(img_t)
            self.C1.set_lambda(1.0)
            self.C2.set_lambda(1.0)
            output_t1 = self.C1(feat_t, reverse=True)
            output_t2 = self.C2(feat_t, reverse=True)
            loss_dis = -self.discrepancy(output_t1, output_t2)
            #loss_dis.backward()
            self.opt_c1.step()
            self.opt_c2.step()
            self.opt_g.step()
            self.reset_grad()
            if batch_idx > 500:
                return batch_idx

            if batch_idx % self.interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss1: {:.6f}\t Loss2: {:.6f}\t  Discrepancy: {:.6f}'.format(
                    epoch, batch_idx, 100,
                    100. * batch_idx / 70000, loss_s1.data, loss_s2.data, loss_dis.data))
                if record_file:
                    record = open(record_file, 'a')
                    record.write('%s %s %s\n' % (loss_dis.data, loss_s1.data, loss_s2.data))
                    record.close()
        return batch_idx

    def test(self, epoch, record_file=None, save_model=False):
        self.G.eval()
        self.C1.eval()
        self.C2.eval()
        self.C3.eval()
        test_loss = 0
        correct1 = 0
        correct2 = 0
        correct3 = 0
        correct4 = 0
        size = 0
        with torch.no_grad():
            for batch_idx, data in enumerate(self.dataset_test):
                img = data['T']
                label = data['T_label']
                img, label = img.cuda(), label.long().cuda()
                # img, label = Variable(img, volatile=True), Variable(label)
                feat = self.G(img)
                output1 = self.C1(feat)
                output2 = self.C2(feat)
                output3 = self.C3(feat)
                test_loss += F.nll_loss(output1, label).data
                output_ensemble = output1 + output2 + output3
                # output_ensemble = output1 + output2
                pred1 = output1.data.max(1)[1]
                pred2 = output2.data.max(1)[1]
                pred3 = output3.data.max(1)[1]
                pred_ensemble = output_ensemble.data.max(1)[1]
                k = label.data.size()[0]
                correct1 += pred1.eq(label.data).cpu().sum()
                correct2 += pred2.eq(label.data).cpu().sum()
                correct3 += pred3.eq(label.data).cpu().sum()
                correct4 += pred_ensemble.eq(label.data).cpu().sum()
                size += k
        test_loss = test_loss / size
        print('\nTest set: Average loss: {:.4f}, Accuracy C1: {}/{} ({:.0f}%) Accuracy C2: {}/{} ({:.0f}%) Accuracy C3: {}/{} ({:.0f}%) \
                Accuracy Ensemble: {}/{} ({:.0f}%) \n'.format(test_loss, correct1, size, 100. * correct1 / size, \
                correct2, size, 100. * correct2 / size, correct3, size, 100. * correct3 / size, correct4, size, 100. * correct4 / size))

        # print('\nTest set: Average loss: {:.4f}, Accuracy C1: {}/{} ({:.0f}%) Accuracy C2: {}/{} ({:.0f}%) Accuracy Ensemble \
        #         : {}/{} ({:.0f}%) \n'.format(test_loss, correct1, size, 100. * correct1 / size, correct2, size, 
        #         100. * correct2 / size, correct4, size, 100. * correct4 / size))

        if save_model and epoch % self.save_epoch == 0:
            torch.save(self.G,
                       '%s/%s_to_%s_model_epoch%s_G.pt' % (self.checkpoint_dir, self.source, self.target, epoch))
            torch.save(self.C1,
                       '%s/%s_to_%s_model_epoch%s_C1.pt' % (self.checkpoint_dir, self.source, self.target, epoch))
            torch.save(self.C2,
                       '%s/%s_to_%s_model_epoch%s_C2.pt' % (self.checkpoint_dir, self.source, self.target, epoch))
            torch.save(self.C3,
                       '%s/%s_to_%s_model_epoch%s_C3.pt' % (self.checkpoint_dir, self.source, self.target, epoch))
        if record_file:
            record = open(record_file, 'a')
            print('recording %s', record_file)
            # record.write('%s %s %s \n' % (float(correct1) / size, float(correct2) / size, float(correct4) / size))
            record.write('%s %s %s %s \n' % (float(correct1) / size, float(correct2) / size, \
                    float(correct3) / size, float(correct4) / size))
            record.close()

        self.writer.add_scalar('Test/test_loss', test_loss, epoch)
        self.writer.add_scalar('Test/ACC_C1', 100. * correct1 / size, epoch)
        self.writer.add_scalar('Test/ACC_C2', 100. * correct2 / size, epoch)
        self.writer.add_scalar('Test/ACC_C3', 100. * correct3 / size, epoch)
        self.writer.add_scalar('Test/ACC_en', 100. * correct4 / size, epoch)
