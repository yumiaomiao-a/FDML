from xception import Xception
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from efficientnet_pytorch import EfficientNet


# Filter Module
class Filter(nn.Module):
    def __init__(self, size, band_start, band_end, use_learnable=True, norm=False):
        super(Filter, self).__init__()
        self.use_learnable = use_learnable
        self.base = nn.Parameter(torch.tensor(generate_filter(band_start, band_end, size)), requires_grad=False)
        if self.use_learnable:
            self.learnable = nn.Parameter(torch.randn(size, size), requires_grad=True)
            self.learnable.data.normal_(0., 0.1)
            # Todo
            # self.learnable = nn.Parameter(torch.rand((size, size)) * 0.2 - 0.1, requires_grad=True)

        self.norm = norm
        if norm:
            self.ft_num = nn.Parameter(torch.sum(torch.tensor(generate_filter(band_start, band_end, size))),
                                       requires_grad=False)

    def forward(self, x):
        if self.use_learnable:
            filt = self.base + norm_sigma(self.learnable)
        else:
            filt = self.base

        if self.norm:
            y = x * filt / self.ft_num
        else:
            y = x * filt
        return y


# FAD Module
class FAD_Head(nn.Module):
    def __init__(self, size):
        super(FAD_Head, self).__init__()
        # init DCT matrix
        self._DCT_all = nn.Parameter(torch.tensor(DCT_mat(size)).float(), requires_grad=False)
        self._DCT_all_T = nn.Parameter(torch.transpose(torch.tensor(DCT_mat(size)).float(), 0, 1), requires_grad=False)
        # transpose
        # define base filters and learnable
        # 0 - 1/16 || 1/16 - 1/8 || 1/8 - 1
        low_filter = Filter(size, 0, size // 2.82)
        middle_filter = Filter(size, size // 2.82, size // 2)
        high_filter = Filter(size, size // 2, size * 2)
        all_filter = Filter(size, 0, size * 2)
        self.filters = nn.ModuleList([low_filter,middle_filter,high_filter, all_filter])

    def forward(self, x):
        # DCT
        x_freq = self._DCT_all @ x @ self._DCT_all_T  # @代表常规的数学上定义的矩阵相乘

        # 4 kernels
        y_list = []
        for i in range(4):
            x_pass = self.filters[i](x_freq)  # [N, 3, 299, 299] #选其中一个滤波器
            y = self._DCT_all_T @ x_pass @ self._DCT_all  # [N, 3, 299, 299] #转成RGB
            y_list.append(y)
        out = torch.cat(y_list, dim=1)  # [N, 12, 299, 299]
        return out



class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers  #3
        h = [hidden_dim] * (num_layers - 1) #81*2
        self.layers = nn.ModuleList(nn.Linear(n, k)
                                    for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class FDML(nn.Module):
    def __init__(self, num_classes=2, img_width=299, img_height=299):
        super(FDML, self).__init__()
        assert img_width == img_height
        img_size = img_width
        self.num_classes = num_classes
        self.MixBlock_1 = MixBlock(728)
        self.MixBlock_2 = MixBlock(728)
        self.MixBlock_3 = MixBlock(2048)
        self.mlp = MLP(3584, 3584, 3584, 3)
        self.mlp2 = MLP(81, 81, 81, 3)
        self.maxpooling = nn.MaxPool2d(2,stride=2,padding=1)
        self.my_mix_module_1 = mix_module(56,56,56,56,1369,1369)
        self.my_mix_module_2 = mix_module(160,160,160,160,324,324)
        self.my_mix_module_3 = mix_module(1792,1792,1792,1792,81,81)
        self.FAD_head = FAD_Head(img_size)
        self.init_xcep_FAD()
        self.init_xcep()
        self.init_efficient_FAD()
        self.init_efficient()
        self.fc2048 = torch.nn.Sequential(
            torch.nn.Linear(3584, 500),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.5),
            torch.nn.Linear(500, num_classes))
        self.fc2048_2 = torch.nn.Sequential(
            torch.nn.Linear(3584, 500),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.5),
            torch.nn.Linear(500, num_classes))
        self.fc100 = torch.nn.Sequential(
            torch.nn.Linear(81, num_classes))
        self.fc100_2 = torch.nn.Sequential(
            torch.nn.Linear(81, num_classes))

        # classifier
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(2048,num_classes)
        self.dp = nn.Dropout(p=0.2)
        self.sigmoid = nn.Sigmoid()

    def init_xcep_FAD(self):
        self.FAD_xcep = Xception(self.num_classes)
        # To get a good performance, using ImageNet-pretrained Xception model is recommended
        state_dict = get_xcep_state_dict()
        conv1_data = state_dict['conv1.weight'].data
        self.FAD_xcep.load_state_dict(state_dict, False)
        # copy on conv1
        # let new conv1 use old param to balance the network
        self.FAD_xcep.conv1 = nn.Conv2d(12, 32, 3, 2, 0, bias=False)
        for i in range(4):
            self.FAD_xcep.conv1.weight.data[:, i * 3:(i + 1) * 3, :, :] = conv1_data / 4.0

    def init_efficient_FAD(self):
        self.FAD_efficient = EfficientNet.from_pretrained('efficientnet-b4')#,num_classes=2)
        self.FAD_efficient._conv_stem = nn.Conv2d(12,48,kernel_size=(3, 3), stride=(2, 2), bias=False)

    def init_efficient(self):
        self.efficient = EfficientNet.from_pretrained('efficientnet-b4')#,num_classes=2)

    def _norm_fea(self, fea):
        f = self.relu(fea)
        f = F.adaptive_avg_pool2d(f, (1, 1))
        f = f.view(f.size(0), -1)
        return f

    def _norm_fea_spa(self, fea):
        f = self.relu(fea)
        f = torch.mean(f,dim=1,keepdim=True)
        f = f.view(f.size(0),-1)
        return f

    def forward(self, x,x_aug):
        fea_rgb_1,fea_rgb_2,fea_rgb_3 = self.efficient(x_aug)
        fea_FAD = self.FAD_head(x)
        fea_FAD_1,fea_FAD_2,fea_FAD_3 = self.FAD_efficient(fea_FAD)

        s1 = self.my_mix_module_1(fea_FAD_1, fea_rgb_1, 37)
        s2 = self.my_mix_module_2(fea_FAD_2, fea_rgb_2, 18)
        s3 = self.my_mix_module_3(fea_FAD_3, fea_rgb_3, 9)

        s1 = nn.functional.interpolate(s1,(9,9),mode='bilinear',align_corners=True)
        s2 = nn.functional.interpolate(s2, (9, 9), mode='bilinear', align_corners=True)
        s_fus = torch.cat((s1,s2,s3),dim=1)

        # channel-wise distangling
        fu = torch.cat((fea_rgb_3,fea_FAD_3),dim=1)
        fea_fusion = self._norm_fea(fu)
        a1 = self.mlp(fea_fusion)
        a = self.sigmoid(a1)  # channel-wise attention vector, which is used to disentanglement fea_fusion
        y1 = torch.mul(a, fea_fusion)
        y2 = torch.mul((1 - a), fea_fusion)  # forgery features
        f1 = self.fc2048(y1)
        f2 = self.fc2048_2(y2)
        # f1 = self.sigmoid(f1)
        # f2 = self.sigmoid(f2)


        # spatial-wise distangling
        fus = self._norm_fea_spa(s_fus)
        b1 = self.mlp2(fus)
        b = self.sigmoid(b1) # spatial-wise attention vector, which is used to disentanglement fus
        z1 = torch.mul(b, fus)
        z2 = torch.mul((1 - b), fus)  ## forgery features
        f3 = self.fc100(z1)
        f4 = self.fc100_2(z2)
        # f3 = self.sigmoid(f3)
        # f4 = self.sigmoid(f4)

        return f1,f2,f3,f4, y1, y2, z1, z2


class mix_module(nn.Module):
    def __init__(self,cin,cout,selfattin,selfattout,selfattin_spa,selfattout_spa):
        super(mix_module,self).__init__()
        self.cin = cin
        self.cout = cout
        self.selfattin = selfattin
        self.selfattout = selfattout
        self.selfattin_spa = selfattin_spa
        self.selfattout_spa = selfattout_spa

        self.relu = nn.ReLU()
        self.SpatialAttention = SpatialAttention()
        self.sigmoid = nn.Sigmoid()

        self.conv1 = nn.Sequential(nn.Conv2d(2*cout,cout,kernel_size= (1,1),stride= (1,1),bias = False),
                                   nn.BatchNorm2d(cout),
                                   nn.ReLU(),
                                   nn.Conv2d(cout, cout, kernel_size=(1, 1), stride=(1, 1), bias=False),
                                   nn.BatchNorm2d(cout),
                                   nn.Sigmoid())

        self.conv2 = nn.Sequential(nn.Conv2d(2*cout, cout, kernel_size=(1, 1), stride=(1, 1), bias=False),
                                   nn.BatchNorm2d(cout),
                                   nn.ReLU(),
                                   nn.Conv2d(cout, cout, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False),
                                   nn.BatchNorm2d(cout),
                                   nn.Sigmoid())

        self.SelfAttention = SelfAttention(num_attention_heads=4, input_size=selfattin, hidden_size=selfattout, hidden_dropout_prob=0.5)

        self.SelfAttention_spa = SelfAttention(num_attention_heads=1, input_size=selfattin_spa, hidden_size=selfattout_spa, hidden_dropout_prob=0.5)


    def forward(self,f1,f2,sqrt):

        ######################## top branch
        f1 = self.relu(f1)
        f2 = self.relu(f2)
        # f11 = F.adaptive_avg_pool2d(f1,(1,1))
        # f12 = F.adaptive_max_pool2d(f1,(1,1))
        # ff1 = torch.cat((f11,f12),dim=1)
        # g = self.conv1(ff1)
        # h = g.mul(f1)
        # fus1 = torch.cat((h,f2),dim=1)
        # m = self.conv2(fus1)
        # n = self.SpatialAttention(m)
        # f3 = m.mul(n)

        ######################## bottom branch
        f_down_org = torch.add(f1,f2)
        N, C, H, W = f_down_org.size()
        f_down = f_down_org.view(N, C, H * W).transpose(2,1)
        f_selfatt = self.SelfAttention(f_down)
        b,n,d = f_selfatt.size()
        f_selfatt = f_selfatt.reshape(b,d,sqrt,sqrt)
        # print('//--------------',f_selfatt.shape)
        # f_down_org = self.sigmoid(f_down_org)
        f_selfatt = self.sigmoid(f_selfatt)
        # f4 = torch.mul(f_down_org,f_selfatt)

        f_spa = f_down_org.view(N, C, H * W)
        f_selfatt2 = self.SelfAttention_spa(f_spa)
        b, n, d = f_selfatt2.size()
        f_selfatt2 = f_selfatt2.reshape(b, n, sqrt, sqrt)
        f_selfatt2 = self.sigmoid(f_selfatt2)
        # f5 = torch.mul(f_down_org,f_selfatt2)
        f_out = torch.mul(f_selfatt,f_selfatt2)
        # f_out = f_selfatt
        return f_out


################################# multi-head self-attention module
import math
class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class SelfAttention(nn.Module):
    def __init__(self, num_attention_heads, input_size, hidden_size, hidden_dropout_prob):
        super(SelfAttention, self).__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, num_attention_heads))
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = hidden_size

        self.query = nn.Linear(input_size, self.all_head_size)
        self.key = nn.Linear(input_size, self.all_head_size)
        self.value = nn.Linear(input_size, self.all_head_size)

        # self.attn_dropout = nn.Dropout(attention_probs_dropout_prob)
        self.attn_dropout = nn.Dropout(hidden_dropout_prob)

        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = LayerNorm(hidden_size, eps=1e-12)
        self.out_dropout = nn.Dropout(hidden_dropout_prob)


    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, input_tensor):
        #input_tensor=size(batch, n, input size)
        #mixed_query_layer = size(batch,n,all_head_size)
        mixed_query_layer = self.query(input_tensor)
        mixed_key_layer = self.key(input_tensor)
        mixed_value_layer = self.value(input_tensor)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # print('-------',attention_scores.shape)  #torch.Size([16, 4, 100, 100])

        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        # [batch_size heads seq_len seq_len] scores
        # [batch_size 1 1 seq_len]
        # attention_scores = attention_scores + attention_mask
        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        # Fixme
        attention_probs = self.attn_dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        hidden_states = self.dense(context_layer)
        hidden_states = self.out_dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class SpatialAttention(nn.Module):
    def __init__(self,kernel =3):
        super(SpatialAttention,self).__init__()
        self.conv1 = nn.Conv2d(2,1, kernel_size=kernel, padding = kernel//2,bias = False)
        self.sigmoid = nn.Sigmoid()
    def forward(self,x):
        avg_out = torch.mean(x,dim=1,keepdim=True)
        max_out, _ = torch.max(x,dim=1,keepdim=True)
        x = torch.cat([avg_out,max_out],dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


# utils
def DCT_mat(size):
    m = [[(np.sqrt(1. / size) if i == 0 else np.sqrt(2. / size)) * np.cos((j + 0.5) * np.pi * i / size) for j in
          range(size)] for i in range(size)]
    return m


def generate_filter(start, end, size):
    return [[0. if i + j > end or i + j < start else 1. for j in range(size)] for i in range(size)]


def norm_sigma(x):
    return 2. * torch.sigmoid(x) - 1.


def get_xcep_state_dict(pretrained_path='pretrained/xception-b5690688.pth'):
    # load Xception
    state_dict = torch.load(pretrained_path)

    for name, weights in state_dict.items():
        if 'pointwise' in name:
            state_dict[name] = weights.unsqueeze(-1).unsqueeze(-1)  # unsqueeze(-1)在倒数第一维度add one dimension
    state_dict = {k: v for k, v in state_dict.items() if 'fc' not in k}
    return state_dict


# def new_xcep_features(self, input):
#
#     x = self.conv2(input)  # input :[149, 149, 6]  conv2:[in_filter:32]
#     x = self.bn2(x)
#     x = self.relu(x)
#
#     x = self.block1(x)
#     x = self.block2(x)
#     x = self.block3(x)
#     x = self.block4(x)
#     x = self.block5(x)
#     x = self.block6(x)
#     x = self.block7(x)
#     x = self.block8(x)
#     x = self.block9(x)
#     x = self.block10(x)
#     x = self.block11(x)
#     x = self.block12(x)
#
#     x = self.conv3(x)
#     x = self.bn3(x)
#     x = self.relu(x)
#
#     x = self.conv4(x)
#     x = self.bn4(x)
#     return x
#
#
#
#
# def fea_0_7(self, x):
#     x = self.conv1(x)
#     x = self.bn1(x)
#     x = self.relu(x)
#
#     x = self.conv2(x)
#     x = self.bn2(x)
#     x = self.relu(x)
#
#     x = self.block1(x)
#     x = self.block2(x)
#     x = self.block3(x)
#     x = self.block4(x)
#     x = self.block5(x)
#     x = self.block6(x)
#     x = self.block7(x)
#     return x
#
#
# def fea_8_12(self, x):
#     x = self.block8(x)
#     x = self.block9(x)
#     x = self.block10(x)
#     x = self.block11(x)
#     x = self.block12(x)
#
#     x = self.conv3(x)
#     x = self.bn3(x)
#     x = self.relu(x)
#
#     x = self.conv4(x)
#     x = self.bn4(x)
#     return x


class MixBlock(nn.Module):
    # An implementation of the cross attention module in F3-Net
    # Haven't added into the whole network yet
    def __init__(self,c_in):
        super(MixBlock, self).__init__()
        self.c_in = c_in


        self.FAD_query = nn.Conv2d(c_in, c_in, (1, 1))
        self.LFS_query = nn.Conv2d(c_in, c_in, (1, 1))

        self.FAD_key = nn.Conv2d(c_in, c_in, (1, 1))
        self.LFS_key = nn.Conv2d(c_in, c_in, (1, 1))

        self.softmax = nn.Softmax(dim=-1)
        self.relu = nn.ReLU()

        self.FAD_gamma = nn.Parameter(torch.zeros(1))
        self.LFS_gamma = nn.Parameter(torch.zeros(1))
        # Parameter是将一个不可训练的tensor转换成可训练的类型parameter，并绑定到这个module里面

        self.FAD_conv = nn.Conv2d(c_in, c_in, (1, 1), groups=c_in)
        self.FAD_bn = nn.BatchNorm2d(c_in)
        self.LFS_conv = nn.Conv2d(c_in, c_in, (1, 1), groups=c_in)
        self.LFS_bn = nn.BatchNorm2d(c_in)

    def forward(self, x_FAD, x_LFS):
        B, C, W, H = x_FAD.size()
        assert W == H

        q_FAD = self.FAD_query(x_FAD).view(-1, W, H)  # [BC, W, H]
        q_LFS = self.LFS_query(x_LFS).view(-1, W, H)
        M_query = torch.cat([q_FAD, q_LFS], dim=2)  # [BC, W, 2H]

        k_FAD = self.FAD_key(x_FAD).view(-1, W, H).transpose(1, 2)  # [BC, H, W]
        k_LFS = self.LFS_key(x_LFS).view(-1, W, H).transpose(1, 2)
        M_key = torch.cat([k_FAD, k_LFS], dim=1)  # [BC, 2H, W]

        energy = torch.bmm(M_query, M_key)  # [BC, W, W] # torch.bmm这个函数是将最后两维矩阵相乘
        attention = self.softmax(energy).view(B, C, W, W)

        att_LFS = x_LFS * attention * (torch.sigmoid(self.LFS_gamma) * 2.0 - 1.0)
        y_FAD = x_FAD + self.FAD_bn(self.FAD_conv(att_LFS))

        att_FAD = x_FAD * attention * (torch.sigmoid(self.FAD_gamma) * 2.0 - 1.0)
        y_LFS = x_LFS + self.LFS_bn(self.LFS_conv(att_FAD))

        y= torch.mul(y_FAD, y_LFS)
        return y



if __name__ == "__main__":
    import cv2
    from torchvision import datasets, transforms

    mean = [0.485,0.456,0.406]
    std = [0.229,0.224,0.225]

    img1 = cv2.imread('./train/fake/0008/00051.jpg')
    img11 = cv2.imread('./3-00051.jpg')
    img2 = cv2.imread('./train/real/0008/00051.jpg')
    img22 = cv2.imread('./3real-00051.jpg')

    to_tensor = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Scale([299,299]),
        # transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    to_tensor2 = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Scale([299,299]),
        # # transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    img1 = to_tensor(img1)
    img1 = torch.unsqueeze(img1,0)

    img111 = img11#cv2.resize(img11,(299,299))
    img222 = img22#cv2.resize(img22,(299,299))
    # img111 = to_tensor2(img11)
    # img222 = to_tensor2(img22)

    # img222 = img222.transpose(2,0,1)
    # img111 = img111.transpose(2,0,1)
    # print('///',img222.shape)

    # imgres = img111-img222
    gray1 = cv2.cvtColor(img111,cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img222,cv2.COLOR_BGR2GRAY)
    imgres = cv2.absdiff(gray1,gray2)
    # imgres = cv2.cvtColor

    # ret,binary = cv2.threshold(imgres,20,255,cv2.THRESH_BINARY)
    # imgres = binary
    # imgres = imgres[0:1,:,:]
    imgres = imgres/255

    model = FDML()
    print(model)
    # dic = torch.load('./save_result_eff/loss/1111/df_aug_bs12_5.pth')
    # new_state_dict = {}
    # for k,v in dic.items():
    #     new_state_dict[k[7:]] = v
    # model.load_state_dict(new_state_dict)

    s1, s2, s3, s4, y1, y2, z1,z2 = model(img1,img1)

    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name,":",param.size())