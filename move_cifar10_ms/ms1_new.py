import mindspore as ms
import mindspore.nn as nn  
import numpy as np
import time
import mindspore.ops as ops
import mindspore.dataset as ds
from mindvision.classification.dataset import Cifar10
from mindvision.dataset import DownLoad
from mindspore.common.initializer import initializer, HeNormal
from mindspore import dtype as mstype
from mindspore.ops import stop_gradient
from mindspore.ops import functional as F

import utils

from meters import flush_scalar_meters
from fn_get_meters import get_meters
from ms_resnet import resnet20, resnet32, resnet44, resnet56, resnet110, resnet1202

def train_loop(epoch, net, train_net1,train_net2, k_loss2, k_loss3, train_meters, val_meters,
               batch_size, data_val, topk):
    print('\nEpoch: %d' % epoch)
    #model.train()
    train_net1.set_train()
    train_net2.set_train()
    i_num = 0
    loss_show = [0,0,0,0]
    top1 = [0, 0, 0, 0]
    top5 = [0, 0, 0, 0]
    step = 0 
    #import ipdb;ipdb.set_trace()
    onehot = ops.OneHot(axis=-1)
    for i, (batch_x, batch_y) in enumerate(data_train):
        i_num = i_num + 1
        train_net1.set_train()
        train_net2.set_train()
        if step == 0:
            start_time = time.time()
            depth, on_value, off_value = num_class, ms.Tensor(1.0, ms.float32), ms.Tensor(0.0, ms.float32)
            one_hot = onehot(batch_y, depth, on_value, off_value)
            one_hot = ops.cast(one_hot, ms.int32)
            _, features = net(batch_x)
            base_data_tensor = features
            base_data_target = one_hot
            print("base_data_tensor: ", ops.shape(base_data_tensor)) 
            print("base_data_target: ", ops.shape(base_data_target)) 
            _, K_ni = utils.GP_new_cal(base_data_tensor,batch_size)
            build_end = time.time()
            print('build: ', build_end - start_time)
            loss, correct = train_net1(batch_x, batch_y)
            train_meters[str(3 + 1)]['loss'].cache(stop_gradient(loss))
            for k in topk:
                correct_k = ops.cast(correct[:k],ms.float32).sum(axis=0)
                error_list = list(1. - stop_gradient(correct_k))  #error_list = list(1. - correct_k.cpu().detach().numpy())
                train_meters[str(3 + 1)]['top{}_error'.format(k)].cache_list(error_list)
                loss_show[3] = stop_gradient(loss)
        else:
#            print(i)
            loss, k_loss2, k_loss3, correct = train_net2(batch_x, batch_y, base_data_tensor, base_data_target,
                                                K_ni, err, k_loss2, k_loss3, batch_size)
            # print("loss=", loss)
            train_meters[str(3 + 1)]['loss'].cache(stop_gradient(loss))
            for k in topk:
                correct_k = ops.cast(correct[:k],ms.float32).sum(axis=0)
                error_list = list(1.-stop_gradient(correct_k))  #error_list = list(1.-correct_k.cpu().detach().numpy())
                train_meters[str(3 + 1)]['top{}_error'.format(k)].cache_list(error_list)
            loss_show[3] = stop_gradient(loss)
        if step % 200 == 0:  #200
            err, loss_test = utils.test(epoch, i, net, train_net1, train_net2, data_val, topk)
            f1 = open('./test_err_2GP_loss1_32.txt', 'r+')
            f1.read()
            f1.write('\n')
            f1.write(str(loss_test))
            f1.close()
            f2 = open('./test_err_2GP_err_32.txt', 'r+')
            f2.read()
            f2.write('\n')
            f2.write(str(err))
            f2.close()
        results = flush_scalar_meters(train_meters[str(3 + 1)])
        top1[3] = top1[3] + results["top1_error"]
        loss_show[3] = loss_show[3] + results["loss"]
        step = step + 1
    end_time = time.time()
    print('all: ', end_time - start_time)
    print("-------------------")
    print('AVG_TRAIN_LOSS3: ', loss_show[3] / i_num)
    print('AVG_Top1_error3: ',top1[3] / i_num)
    f3 = open('./train_err_2GP_loss_32.txt', 'r+')
    f3.read()
    f3.write('\n')
    f3.write(str(loss_show[3] / i_num))
    f3.close()
    f4 = open('./train_err_2GP_err_32.txt', 'r+')
    f4.read()
    f4.write('\n')
    f4.write(str(top1[3] / i_num))
    f4.close()
    return k_loss2, k_loss3

if __name__ == '__main__':
    ms.set_context(mode=ms.GRAPH_MODE, device_target="GPU",device_id=1)
    train_meters = get_meters('train')
    val_meters = get_meters('val')
    topk = [1, 5]
    dict_own = {}
    batch_size = 128 #128
    num_class = 10

    dl_path_cifar10 = "./cifar10_data"
    dl_url_cifar10 = "https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/notebook/datasets/cifar-10-binary.tar.gz"
    dl = DownLoad()
    # 下载CIFAR-10数据集并解压
    dl.download_and_extract_archive(url=dl_url_cifar10, download_path=dl_path_cifar10)
    DATA_DIR_CIFAR10 = "./cifar10_data/cifar-10-batches-bin/"

    data_train = utils.create_dataset(DATA_DIR_CIFAR10, do_train=True, batch_size=batch_size)
    data_val = utils.create_dataset(DATA_DIR_CIFAR10, do_train=False, batch_size=batch_size)
    print('train_batch_total', data_train.get_dataset_size())
    print('val_batch_total', data_val.get_dataset_size())
    
    start_time = time.time()
    network = resnet20()
    for _, cell in network.cells_and_names():
        if isinstance(cell, nn.Dense) or isinstance(cell, nn.Conv2d): 
            cell.weight.set_data(initializer(HeNormal(), cell.weight.shape, cell.weight.dtype))
    
    # milestone = [150, 200, 250]
    # learning_rates = [0.1, 0.01, 0.001]
    # lr = nn.piecewise_constant_lr(milestone, learning_rates)
    # optimizer = nn.SGD(filter(lambda p: p.requires_grad, network.get_parameters()), learning_rate=lr, momentum=0.9, weight_decay=0.0001)
    optimizer = nn.SGD(filter(lambda p: p.requires_grad, network.get_parameters()), learning_rate=0.1, momentum=0.9, weight_decay=0.0001)
    # optimizer.init_learning_rate = 0.001

    
    start_time = time.time()
    k_loss2 = 1
    k_loss3 = 1
    loss_net1 = utils.CustomWithLossCell1(network) # 包含损失函数的Cell
    train_network1 = utils.CustomTrainOneStepCell1(loss_net1, optimizer)
    loss_net2 = utils.CustomWithLossCell2(network) # 包含损失函数的Cell
    train_network2 = utils.CustomTrainOneStepCell2(loss_net2, optimizer)
    train_network1.set_train()
    train_network2.set_train()

    for epoch in range(250):
        tmp_start = time.time()
        # print("lr=", optimizer.get_lr())
        print("lr=", optimizer.learning_rate.data.asnumpy())
        k_loss2, k_loss3 = train_loop(epoch, network, train_network1, train_network2, k_loss2, k_loss3,
                                      train_meters, val_meters, batch_size, data_val, topk=topk)
        utils.save(epoch, network)
        if epoch == 150:
            ops.assign(optimizer.learning_rate, ms.Tensor(0.01, ms.float32))
        if epoch == 200:
            ops.assign(optimizer.learning_rate, ms.Tensor(0.001, ms.float32))
    #    test(epoch)
        end_time = time.time()
        print("this_epoch: ", end_time-tmp_start)
        print("total: ", end_time - start_time, "\n\n\n")
    print("ok")