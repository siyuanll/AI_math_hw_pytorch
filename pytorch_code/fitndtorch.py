import shutil
import platform
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt
import matplotlib
import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np
import os
import sys
import time
import pickle
import warnings
from datetime import datetime
warnings.filterwarnings("ignore")
matplotlib.use('Agg')

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
Leftp = 0.18
Bottomp = 0.18
Widthp = 0.88 - Leftp
Heightp = 0.9 - Bottomp
pos = [Leftp, Bottomp, Widthp, Heightp]


def mkdir(fn):  # Create a directory
    if not os.path.isdir(fn):
        os.mkdir(fn)


def save_fig(pltm, fntmp, fp=0, ax=0, isax=0, iseps=0, isShowPic=0):  # Save the figure
    if isax == 1:
        pltm.rc('xtick', labelsize=18)
        pltm.rc('ytick', labelsize=10)
        ax.set_position(pos, which='both')
    fnm = '%s.png' % (fntmp)
    pltm.savefig(fnm)
    if iseps:
        fnm = '%s.eps' % (fntmp)
        pltm.savefig(fnm, format='eps', dpi=600)
    if fp != 0:
        fp.savefig("%s.pdf" % (fntmp), bbox_inches='tight')
    if isShowPic == 1:
        pltm.show()
    elif isShowPic == -1:
        return
    else:
        pltm.close()


# All parameters
R = {}
R['times'] = 0.5 #initial
R['input_dim'] = 1
R['output_dim'] = 1
R['ActFuc'] = 1  # 0: ReLU; 1: Tanh; 2:Sin; 3:x**50; 4:Sigmoid
R['hidden_units'] = [100,100]

R['learning_rate'] = 2e-4
R['learning_rateDecay'] = 5e-8

plot_epoch = 500
R['train_size'] = 100

R['test_size'] = 100
R['x_start'] = -5
R['x_end'] = 5
R['device'] = "0"
R['asi'] = 0
R['tuning_points'] = []
R['check_epoch'] = 10  # find the tuning point
R['tuning_ind'] = []
Ry = {}
Ry['y_all'] = []
Rw = {}
Rw['weight_R'] = []
lenarg = np.shape(sys.argv)[
    0]  # Sys.argv[ ]其实就是一个列表，里边的项为用户输入的参数，关键就是要明白这参数是从程序外部输入的，而非代码本身的什么地方，要想看到它的效果就应该将程序保存了，从外部来运行程序并给出参数。
if lenarg > 1:
    ilen = 1
    while ilen < lenarg:
        if sys.argv[ilen] == '-m':
            R['hidden_units'] = [np.int32(sys.argv[ilen + 1])]
        if sys.argv[ilen] == '-g':
            R['device'] = np.int32(sys.argv[ilen + 1])
        if sys.argv[ilen] == '-t':
            R['times'] = np.float32(sys.argv[ilen+1])
        if sys.argv[ilen] == '-s':
            R['train_size'] = np.int32(sys.argv[ilen+1])
        # if sys.argv[ilen]=='-lr':
        #     R['learning_rate']=np.float32(sys.argv[ilen+1])
        # if sys.argv[ilen]=='-dir':
        #     sBaseDir=sys.argv[ilen+1]
        ilen = ilen + 2

R['hidden_units'] = [200, 200, 200, 100]
R['batch_size'] = R['train_size']
R['astddev'] = 1 / (R['hidden_units'][0] ** R['times'])
R['bstddev'] = 1 / (R['hidden_units'][0] ** R['times'])
R['full_net'] = [R['input_dim']] + R['hidden_units'] + [R['output_dim']]

if R['input_dim'] == 1:
    R['test_inputs'] = np.reshape(np.linspace(R['x_start'] - 0.5, R['x_end'] + 0.5, num=R['test_size'],
                                              endpoint=True), [R['test_size'], 1])
    R['train_inputs'] = np.reshape(np.linspace(R['x_start'], R['x_end'], num=R['train_size'],
                                               endpoint=True), [R['train_size'], 1])
else:
    R['test_inputs'] = np.random.rand(
        R['test_size'], R['input_dim']) * (R['x_end'] - R['x_start']) + R['x_start']
    R['train_inputs'] = np.random.rand(
        R['train_size'], R['input_dim']) * (R['x_end'] - R['x_start']) + R['x_start']


def ReLU(x):
    return x * (x > 0)


# def get_y(xs, sampleNo):  # Function to fit
#     tmp = np.sin(xs)+np.sin(6*xs)
#     return tmp

def func0(xx):
    y_sin = np.sin(xx)+2*np.sin(3*xx)+3*np.sin(5*xx)
    return y_sin


def get_y(xx, alpha=1):
    y_sin = func0(xx)
    if alpha == 0:
        return y_sin
    out_y = np.round(y_sin/alpha)
    out_y2 = out_y * alpha
    return out_y2


test_inputs = R['test_inputs']
train_inputs = R['train_inputs']
R['y_true_train'] = get_y(R['train_inputs'])
# Make a folder to save all output
BaseDir_neu = 'test'
if platform.system() == 'Windows':
    # device_n="0"
    BaseDir0 = '../../../nn/%s' % (sBaseDir0)
    # BaseDir = '../../../nn/%s'%(sBaseDir)
else:
    # device_n="0"
    # BaseDir0 = sBaseDir0
    # BaseDir = sBaseDir
    matplotlib.use('Agg')
# mkdir(BaseDir0)
# BaseDir = '%s/%s' % (BaseDir0, example_folder)
# mkdir(BaseDir)
# BaseDir_a = '%s/%s' % (BaseDir, R['times'])
# mkdir(BaseDir_a)
# BaseDir_neu = '%s/%s' % (BaseDir_a, neu_ind_folder)
mkdir(BaseDir_neu)
subFolderName = '%s'%(datetime.now().strftime("%y%m%d%H%M%S")) 
#subFolderName = '%s'%(int(np.absolute(np.random.normal([1])*100000))//int(1)) 

# subFolderName = '%s' % (
#     int(np.absolute(np.random.normal([1]) * 100000)) // int(1))
FolderName = '%s/%s/' % (BaseDir_neu, subFolderName)
mkdir(FolderName)

# mkdir('%smodel/'%(FolderName))
# print(subFolderName)

if not platform.system() == 'Windows':
    shutil.copy(__file__, '%s%s' % (FolderName, os.path.basename(__file__)))

device = torch.device("cuda:%s" % (
    R['device']) if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
print(device)


def weights_init(m):  # Initialization weight
    if isinstance(m, nn.Linear):
        m.weight.data.normal_(0, R['astddev'])
        m.bias.data.normal_(0, R['bstddev'])


class Act_op(nn.Module):  # Custom activation function
    def __init__(self):
        super(Act_op, self).__init__()

    def forward(self, x):
        # return x ** 50  # or F.relu(x) * F.relu(1-x)
        return (F.relu(x))**3


def getWini(hidden_units=[10, 20, 40], input_dim=1, output_dim_final=1, astddev=0.05, bstddev=0.05):
    hidden_num = len(hidden_units)
    # print(hidden_num)
    add_hidden = [input_dim] + hidden_units

    w_Univ0 = []
    b_Univ0 = []

    for i in range(hidden_num):
        input_dim = add_hidden[i]
        output_dim = add_hidden[i + 1]
        ua_w = np.float32(np.random.normal(
            loc=0.0, scale=astddev, size=[input_dim, output_dim]))
        ua_b = np.float32(np.random.normal(
            loc=0.0, scale=bstddev, size=[output_dim]))
        w_Univ0.append(np.transpose(ua_w))
        b_Univ0.append(np.transpose(ua_b))
    ua_w = np.float32(np.random.normal(loc=0.0, scale=astddev, size=[
                      hidden_units[hidden_num - 1], output_dim_final]))
    ua_b = np.float32(np.random.normal(
        loc=0.0, scale=bstddev, size=[output_dim_final]))
    w_Univ0.append(np.transpose(ua_w))
    b_Univ0.append(np.transpose(ua_b))
    return w_Univ0, b_Univ0



def my_fft(data, freq_len=40, x_input=np.zeros(10), kk=0, min_f=0, max_f=np.pi/3, isnorm=1):
    second_diff_input = np.mean(np.diff(np.diff(np.squeeze(x_input))))
    if abs(second_diff_input) < 1e-10:
        datat = np.squeeze(data)
        datat_fft = np.fft.fft(datat)
        freq_len=min(freq_len,len(datat_fft))
        print(freq_len)
        ind2 = range(freq_len)
        fft_coe = datat_fft[ind2]
        if isnorm == 1:
            return_fft = np.absolute(fft_coe)
        else:
            return_fft = fft_coe
    else:
        return_fft = get_ft_multi(
            x_input, data, kk=kk, freq_len=freq_len, min_f=min_f, max_f=max_f, isnorm=isnorm)
    return return_fft


def get_ft_multi(x_input, data, kk=0, freq_len=100, min_f=0, max_f=np.pi/3, isnorm=1):
    n = x_input.shape[1]
    if np.max(abs(kk)) == 0:
        k = np.linspace(min_f, max_f, num=freq_len, endpoint=True)
        kk = np.matmul(np.ones([n, 1]), np.reshape(k, [1, -1]))
    tmp = np.matmul(np.transpose(data), np.exp(-1J * (np.matmul(x_input, kk))))
    if isnorm == 1:
        return_fft = np.absolute(tmp)
    else:
        return_fft = tmp
    return np.squeeze(return_fft)


def SelectPeakIndex(FFT_Data, endpoint=True):
    D1 = FFT_Data[1:-1]-FFT_Data[0:-2]
    D2 = FFT_Data[1:-1]-FFT_Data[2:]
    D3 = np.logical_and(D1 > 0, D2 > 0)
    tmp = np.where(D3 == True)
    sel_ind = tmp[0]+1
    if endpoint:
        if FFT_Data[0]-FFT_Data[1] > 0:
            sel_ind = np.concatenate([[0], sel_ind])
        if FFT_Data[-1]-FFT_Data[-2] > 0:
            Last_ind = len(FFT_Data)-1
            sel_ind = np.concatenate([sel_ind, [Last_ind]])
    return sel_ind


R['n_fixed'] = 0

min_n = np.min([R['n_fixed'], R['hidden_units'][0]])
R['n_fixed'] = min_n
# print(min_n)
w_Univ0, b_Univ0 = getWini(hidden_units=R['hidden_units'], input_dim=R['input_dim'], output_dim_final=R['output_dim'],
                           astddev=R['astddev'], bstddev=R['bstddev'])

print(np.shape(w_Univ0[0]))
print(np.shape(b_Univ0[0]))


class Network(nn.Module):  # DNN 0: ReLU; 1: Tanh; 2:Sin; 3:x**50; 4:Sigmoid
    def __init__(self):
        super(Network, self).__init__()
        self.block3 = nn.Sequential()
        self.block = nn.Sequential()
        for i in range(len(R['full_net']) - 2):
            d_linear = nn.Linear(R['full_net'][i], R['full_net'][i + 1])
            print('weight1: start')
            print(np.shape(d_linear.weight.data.numpy()))
            print('weight1: end')
            d_linear.weight.data = torch.nn.Parameter(torch.tensor(w_Univ0[i]))
            d_linear.bias.data = torch.nn.Parameter(torch.tensor(b_Univ0[i]))
            # print(d_linear.weight)
            print('weight2: start')
            print(np.shape(d_linear.weight.data.numpy()))
            print('weight2: end')
            self.block3.add_module('linear' + str(i), d_linear)

            self.block.add_module('linear' + str(i), d_linear)
            if R['ActFuc'] == 0:
                self.block.add_module('relu' + str(i), nn.ReLU())
                self.block3.add_module('relu' + str(i), nn.ReLU())
            elif R['ActFuc'] == 1:
                self.block.add_module('tanh' + str(i), nn.Tanh())
                self.block3.add_module('tanh' + str(i), nn.Tanh())
            elif R['ActFuc'] == 3:
                self.block.add_module('relu3' + str(i), Act_op())
                self.block3.add_module('relu3' + str(i), Act_op())
        i = len(R['full_net']) - 2
        d_linear = nn.Linear(
            R['full_net'][i], R['full_net'][i + 1], bias=False)
        d_linear.weight.data = torch.nn.Parameter(torch.tensor(w_Univ0[i]))
        # d_linear.bias.data = torch.nn.Parameter(torch.tensor(b_Univ0[i]))
        self.block.add_module('linear' + str(i), d_linear)
        if R['asi']:
            self.block2 = nn.Sequential()
            for i in range(len(R['full_net']) - 2):
                d_linear = nn.Linear(R['full_net'][i], R['full_net'][i + 1])
                print('weight1: start')
                print(np.shape(d_linear.weight.data.numpy()))
                print('weight1: end')
                d_linear.weight.data = torch.nn.Parameter(
                    torch.tensor(w_Univ0[i]))
                d_linear.bias.data = torch.nn.Parameter(
                    torch.tensor(b_Univ0[i]))
                # print(d_linear.weight)
                print('weight2: start')
                print(np.shape(d_linear.weight.data.numpy()))
                print('weight2: end')
                # d_linear.weight.data = torch.tensor(w_Univ0[i])
                # d_linear.bias.data = torch.tensor(b_Univ0[i])
                self.block2.add_module('linear2' + str(i), d_linear)
                if R['ActFuc'] == 0:
                    self.block2.add_module('relu2' + str(i), nn.ReLU())
                elif R['ActFuc'] == 1:
                    self.block2.add_module('tanh2' + str(i), nn.Tanh())
                elif R['ActFuc'] == 2:
                    self.block2.add_module('sin2' + str(i), nn.sin())
                elif R['ActFuc'] == 3:
                    self.block2.add_module('**502' + str(i), Act_op())
                elif R['ActFuc'] == 4:
                    self.block2.add_module('sigmoid2' + str(i), nn.sigmoid())
            i = len(R['full_net']) - 2
            d_linear = nn.Linear(
                R['full_net'][i], R['full_net'][i + 1], bias=False)
            d_linear.weight.data = torch.nn.Parameter(
                torch.tensor(-w_Univ0[i]))
            d_linear.bias.data = torch.nn.Parameter(torch.tensor(-b_Univ0[i]))
            self.block2.add_module('linear2' + str(i), d_linear)

    def forward(self, x):
        if R['asi']:
            out = self.block(x) + self.block2(x)
        else:
            out = self.block(x)
        return out

    def hidden(self, x):
        out = self.block3(x)
        return out


class Model():
    def __init__(self):

        # y_train = net_(torch.FloatTensor(train_inputs).to(device))
        y_train = net_(torch.FloatTensor(train_inputs).to(device))
        loss_train = float(
            criterion(y_train.cpu(), torch.FloatTensor(R['y_true_train'])).cpu())
        y_test = net_(torch.FloatTensor(test_inputs).to(device))
        # loss_test = float(criterion(y_test.cpu(), torch.FloatTensor(R['y_true_test'])).cpu())

        nametmp = '%smodel/' % (FolderName)
        mkdir(nametmp)
        torch.save(net_.state_dict(), "%smodel.ckpt" % (nametmp))

        R['y_train'] = y_train.cpu().detach().numpy()
        R['y_test'] = y_test.cpu().detach().numpy()
        # self.record_weight()

        R['loss_train'] = [loss_train]

    def run_onestep(self,optimizer):

        y_test = net_(torch.FloatTensor(test_inputs).to(device))
        # loss_test = float(criterion(y_test, torch.FloatTensor(R['y_true_test']).to(device)).cpu())
        y_train = net_(torch.FloatTensor(train_inputs).to(device))
        loss_train = float(criterion(y_train, torch.FloatTensor(
            R['y_true_train']).to(device)).cpu())

        R['y_train'] = y_train.cpu().detach().numpy()
        R['y_test'] = y_test.cpu().detach().numpy()
        R['loss_train'].append(loss_train)

        # optimizer = torch.optim.SGD(
        #     net_.parameters(), lr=R['learning_rate'], momentum=0.)


        for i in range(R['train_size'] // R['batch_size'] + 1):  # bootstrap

            mask = np.random.choice(
                R['train_size'], R['batch_size'], replace=False)
            y_train = net_(torch.FloatTensor(train_inputs[mask]).to(device))
            loss = criterion(y_train, torch.FloatTensor(
                R['y_true_train'][mask]).to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        R['learning_rate'] = R['learning_rate'] * (1 - R['learning_rateDecay'])

    # def record_weight(self):
    #     if R['hidden_units'][0] == 1:
    #         tmp_w1 = np.squeeze(net_.block[0].weight.cpu().detach().numpy())
    #         tmp_b1 = np.squeeze(net_.block[0].bias.cpu().detach().numpy())
    #         tmp_w2 = np.squeeze(net_.block[2].weight.cpu().detach().numpy())
    #         tmp_w = [tmp_w1, tmp_b1, tmp_w2]
    #     else:
    #         tmp_w1 = np.squeeze(net_.block[0].weight.cpu().detach().numpy())
    #         tmp_b1 = np.squeeze(net_.block[0].bias.cpu().detach().numpy())
    #         tmp_w2 = np.squeeze(net_.block[2].weight.cpu().detach().numpy())
    #         # tmp_w2=np.squeeze(net_.block[2].weight.cpu().detach().numpy())[0:R['n_fixed']]
    #         tmp_w = np.concatenate((tmp_w1, tmp_b1, tmp_w2), axis=0)
    #     Rw['weight_R'].append(tmp_w)

    def run(self, step_n=1):

        # Load paremeters
        nametmp = '%smodel/model.ckpt' % (FolderName)
        net_.load_state_dict(torch.load(nametmp))
        net_.eval()
        optimizer = torch.optim.Adam(net_.parameters(),lr=2e-4)

        for epoch in range(step_n):

            self.run_onestep(optimizer)
            # self.record_weight()
            Ry['y_all'].append(R['y_train'])

            if epoch % plot_epoch == 0:

                print('time elapse: %.3f' % (time.time() - t0))
                print('model, epoch: %d, train loss: %f' %
                      (epoch, R['loss_train'][-1]))
                self.plot_loss()
                self.plot_y(name='%s' % (epoch))
                self.save_file()

            if R['loss_train'][-1] < 1e-5:
                break

    def plot_weight(self):
        weight_R = np.stack(Rw['weight_R'])
        plt.figure()
        for i_sub in range(R['n_fixed']):
            # print(i_sub)
            for ji in range(3):
                # print('%s'%(3*i_sub+ji))
                ax = plt.subplot(R['n_fixed'], 3, 3 * i_sub + ji + 1)
                ax.plot(abs(weight_R[:, ji * R['n_fixed'] + i_sub]))
                plt.title('%s' % (3 * i_sub + ji))
                ax.set_xscale('log')
                ax.set_yscale('log')
                ax.set_ylim([5e-2, 1e1])
                # ax.axis('off')
                # ax.text(-0.5,1,'%.2f'%(output_weight[i_sub]))

        # ax.set_xscale('log')
        # ax.set_yscale('log')
        # plt.legend(fontsize=18)
        # plt.title('loss',fontsize=15)
        # fntmp = '%shiddeny%s'%(FolderName,epoch)
        fntmp = '%sweightevolve' % (FolderName)
        save_fig(plt, fntmp, iseps=0)

    def plot_loss(self):

        plt.figure()
        ax = plt.gca()
        # y1 = R['loss_test']
        y2 = np.asarray(R['loss_train'])
        # plt.plot(y1,'ro',label='Test')
        plt.plot(y2, 'k-', label='Train')
        if len(R['tuning_ind']) > 0:
            plt.plot(R['tuning_ind'], y2[R['tuning_ind']], 'r*')
        ax.set_xscale('log')
        ax.set_yscale('log')
        plt.legend(fontsize=18)
        plt.title('loss', fontsize=15)
        fntmp = '%sloss' % (FolderName)
        save_fig(plt, fntmp, ax=ax, isax=1, iseps=0)

    def plot_tuning(self):
        plt.figure()
        ax = plt.gca()
        y2 = R['y_true_train']
        plt.plot(train_inputs, y2, 'b*', label='True')
        for iit in range(len(R['y_tuning'])):
            plt.plot(test_inputs, R['y_tuning'][iit], '-',
                     label='%.3f' % (R['loss_tuning'][iit]))
        plt.title('turn points', fontsize=15)
        plt.legend(fontsize=18)
        fntmp = '%sturn' % (FolderName)
        save_fig(plt, fntmp, ax=ax, isax=1, iseps=0)

    def plot_y(self, name=''):

        if R['input_dim'] == 2:
            X = np.arange(R['x_start'], R['x_end'], 0.1)
            Y = np.arange(R['x_start'], R['x_end'], 0.1)
            X, Y = np.meshgrid(X, Y)
            xy = np.concatenate(
                (np.reshape(X, [-1, 1]), np.reshape(Y, [-1, 1])), axis=1)
            Z = np.reshape(get_y(xy), [len(X), -1])

            fp = plt.figure()
            ax = fp.gca(projection='3d')
            surf = ax.plot_surface(
                X, Y, Z - np.min(Z), cmap=cm.coolwarm, linewidth=0, antialiased=False)
            ax.zaxis.set_major_locator(LinearLocator(5))
            ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
            fp.colorbar(surf, shrink=0.5, aspect=5)
            ax.scatter(train_inputs[:, 0], train_inputs[:, 1],
                       R['y_train'] - np.min(R['y_train']))
            fntmp = '%s2du%s' % (FolderName, name)
            save_fig(plt, fntmp, ax=ax, isax=1, iseps=0)

        if R['input_dim'] == 1:
            plt.figure()
            ax = plt.gca()
            y1 = R['y_test']
            y2 = R['y_true_train']
            plt.plot(test_inputs, y1, 'r-', label='Test')
            plt.plot(train_inputs, y2, 'b*', label='True')
            plt.title('g2u', fontsize=15)
            plt.legend(fontsize=18)
            fntmp = '%su_m%s' % (FolderName, name)
            fntmp = '%su_m%s' % (FolderName, '')
            save_fig(plt, fntmp, ax=ax, isax=1, iseps=0)

    def save_file(self):
        with open('%s/objs.pkl' % (FolderName), 'wb') as f:
            pickle.dump(R, f, protocol=4)
        with open('%s/objsy.pkl' % (FolderName), 'wb') as f:
            pickle.dump(Ry, f, protocol=4)
        with open('%s/objsw.pkl' % (FolderName), 'wb') as f:
            pickle.dump(Rw, f, protocol=4)
        text_file = open("%s/Output.txt" % (FolderName), "w")
        for para in R:
            if np.size(R[para]) > 20:
                continue
            text_file.write('%s: %s\n' % (para, R[para]))
        text_file.write('loss end: %s\n' % (R['loss_train'][-1]))
        # text_file.write('weight ini: %s\n' % (Rw['weight_R'][0]))
        text_file.close()


t0 = time.time()
net_ = Network().to(device)
# net_.apply(weights_init)
print(net_)

criterion = nn.MSELoss(reduction='mean').to(device)
model = Model()
model.run(3000)

y_pred=R['y_train']
y_fft = my_fft(R['y_true_train'])/R['train_size']
plt.semilogy(y_fft+1e-5, label='real')
idx = SelectPeakIndex(y_fft, endpoint=False)
plt.semilogy(idx, y_fft[idx]+1e-5, 'o')
y_fft_pred = my_fft(y_pred)/R['train_size']
plt.semilogy(y_fft_pred+1e-5, label='train')
plt.semilogy(idx, y_fft_pred[idx]+1e-5, 'o')
plt.legend()
plt.xlabel('freq idx')
plt.ylabel('freq')
plt.savefig(FolderName + 'fft.png')

y_pred_epoch = np.squeeze(Ry['y_all'])
idx1 = idx[:3]
abs_err = np.zeros([len(idx1), len(Ry['y_all'])])
y_fft = my_fft(R['y_true_train'])
tmp1 = y_fft[idx1]
for i in range(len(y_pred_epoch)):
    tmp2 = my_fft(y_pred_epoch[i])[idx1]
    abs_err[:, i] = np.abs(tmp1 - tmp2)/(1e-5 + tmp1)

plt.figure()
plt.pcolor(abs_err, cmap='RdBu', vmin=0.1, vmax=1)
plt.colorbar()
plt.savefig(FolderName + '/hot.png')
