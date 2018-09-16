import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import os
import matplotlib.pyplot as plt 
import skimage.transform

def selu(x):
    alpha = 1.6732632423543772848170429916717
    scale = 1.0507009873554804934193349852946
    return scale * F.elu(x, alpha)


class Attn(nn.Module):

    def __init__(self):
        super(Attn, self).__init__()
        self.lr = 0.0001
        self.batch_size = 16
        self.cuda_exist = torch.cuda.is_available()
        print('cuda exist', self.cuda_exist)

        self.conv1 = nn.Conv2d(4, 24, 3, stride=2, padding=1)
        self.batchNorm1 = nn.BatchNorm2d(24)
        self.conv2 = nn.Conv2d(24, 24, 3, stride=2, padding=1)
        self.batchNorm2 = nn.BatchNorm2d(24)
        self.conv3 = nn.Conv2d(24, 24, 3, stride=2, padding=1)
        self.batchNorm3 = nn.BatchNorm2d(24)
        self.conv4 = nn.Conv2d(24, 24, 3, stride=2, padding=1)
        self.batchNorm4 = nn.BatchNorm2d(24)

        self.num_heads = 2
        self.w1 = nn.ModuleList([nn.Linear(26, 256) for _ in range(self.num_heads)])
        self.w2 = nn.ModuleList([nn.Linear(256, 256) for _ in range(self.num_heads)])
        self.w3 = nn.ModuleList([nn.Linear(256, 1) for _ in range(self.num_heads)])

        self.f_fc1 = nn.Linear(26 * self.num_heads, 256)
        self.f_fc2 = nn.Linear(256, 256)
        self.f_fc3 = nn.Linear(256, 6)
        
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)

        # prepare coord tensor
        self.coord_tensor = torch.FloatTensor(self.batch_size, 36, 2)
        if self.cuda_exist:
            self.coord_tensor = self.coord_tensor.cuda()
        self.coord_tensor = Variable(self.coord_tensor)
        np_coord_tensor = np.zeros((self.batch_size, 36, 2))
        for i in range(36):
            np_coord_tensor[:,i,:] = np.array(self.cvt_coord(i))
        self.coord_tensor.data.copy_(torch.from_numpy(np_coord_tensor))

        print('two heads')
        self.plot_num = 0
        self.total_plot = 100


    def cvt_coord(self, i):
        return [(i/6-2.5)/2.5, (i%6-2.5)/2.5]

        
    def visual_pass(self, img):
        """convolution"""
        x = self.conv1(img)
        x = F.relu(x)
        x = self.batchNorm1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.batchNorm2(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.batchNorm3(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = self.batchNorm4(x)

        """g"""
        mb = x.size()[0]
        n_channels = x.size()[1]
        d = x.size()[2]

        x_flat = x.view(mb,n_channels,d*d).permute(0,2,1)

        x_flat = torch.cat([x_flat, self.coord_tensor[:mb]], dim=2)

        x_flat2 = x_flat.view(mb*d*d, 26)

        ### plot probs
        prob_summ = np.zeros((d, d))

        for i in range(self.num_heads):
            scores = self.w3[i](selu(self.w2[i](selu(self.w1[i](x_flat2))))) # bsize*36 x 1
            scores = scores.squeeze(1).view(mb, d * d) # bsize x 36

            probs = F.softmax(scores).unsqueeze(1) # bsize x 1 x 36
            obj = torch.bmm(probs, x_flat).squeeze(1) # bsize x 26

            ### plot probs
            prob = probs[0][0].view(d, d).data.numpy()
            prob_summ += prob

        prob_summ /= self.num_heads

        img_plt = img[0].permute(1, 2, 0).data.numpy()
        img_plt = np.mean(img_plt, axis=2, keepdims=False) # average four frames
        prob_vis = skimage.transform.pyramid_expand(prob_summ, upscale=14)

        # plot four sublots
        f, (axx_arr) = plt.subplots(2, 2)
        axx_arr[0, 0].imshow(img_plt, cmap='gray')
        axx_arr[0, 0].set_title('original')

        axx_arr[0, 1].set_title('original + mask')
        axx_arr[0, 1].imshow(img_plt, cmap='gray')
        axx_arr[0, 1].imshow(prob_vis, alpha=0.6, cmap='gray')

        axx_arr[1, 0].set_title('mask')  
        axx_arr[1, 0].imshow(prob_vis, cmap='gray')

        axx_arr[1, 1].set_title('raw probabilities 6x6')  
        axx_arr[1, 1].imshow(prob_summ, cmap='gray')
        plt.tight_layout()

        f.savefig('results/attention'+str(self.plot_num))
        plt.close()
        ### end plotting
        
        print('result %d / %d' % (self.plot_num, self.total_plot))
        self.plot_num += 1

        return self.plot_num == self.total_plot
    
    def visualize_(self, input_img):
        # input_img     | N, H, W, C
        # label         | N
        input_img = input_img.transpose(0, 3, 1, 2) / 255.
        input_img = torch.FloatTensor(input_img)
        if self.cuda_exist:
            input_img = input_img.cuda()
        input_img = Variable(input_img)

        last_vis = self.visual_pass(input_img)
        return last_vis
