import mindspore as ms
import mindspore.nn as nn  
import time
import mindspore.ops as ops
from mindvision.dataset import DownLoad
from mindspore.common.initializer import initializer, HeNormal
from mindspore.ops import stop_gradient

import utils

from meters import flush_scalar_meters
from fn_get_meters import get_meters
from ms_resnet import resnet20, resnet32, resnet44, resnet56, resnet110, resnet1202

def train_loop(epoch, net, train_net, k_loss2, k_loss3, train_meters, data_test, data_val, topk, num_class, GP_num):
    print('\nEpoch: %d' % epoch)
    #model.train()
    train_net.set_train()
    i_num = 0
    loss_show = 0
    top1 = 0
    err = 0.99

    onehot = ops.OneHot(axis=-1)
    stack = ops.Stack()
    ms_topk = ops.TopK()
    val_tensor_list, val_target_list, val_K_ni_list = utils.val_data_build(net, train_net, data_val,
                                                                           onehot, num_class, GP_num)
    
    val_tensor_list = stack(val_tensor_list)
    val_target_list = stack(val_target_list)
    val_K_ni_list = stack(val_K_ni_list)

    for i, data in enumerate(data_train):
        batch_x, batch_y = data[0], data[2]
        
        if i % 200 == 0:
            err, loss_test = utils.test(epoch, i, net, train_net, data_test, topk)
            f1 = open('./logs/test_err_2GP_loss1_32.txt', 'r+')
            f1.read()
            f1.write('\n')
            f1.write(str(loss_test))
            f1.close()
            f2 = open('./logs/test_err_2GP_err_32.txt', 'r+')
            f2.read()
            f2.write('\n')
            f2.write(str(err))
            f2.close()
        i_num = i_num + 1
        
        train_net.set_train(True)
        network.set_train(True)
        
        i_num = i_num + 1

        loss, output = train_net(batch_x, batch_y, val_tensor_list, val_target_list, val_K_ni_list, GP_num, k_loss2, k_loss3, err)

        train_meters[str(3 + 1)]['loss'].cache(stop_gradient(loss))
        # topk
        _, pred = ms_topk(output, max(topk))
        pred = pred.T
        correct = ops.equal(pred, batch_y.view(1, -1).expand_as(pred))
        for k in topk:
            correct_k = ops.cast(correct[:k], ms.float32).sum(axis=0)
            error_list = list(1.-stop_gradient(correct_k))
            train_meters[str(3 + 1)]['top{}_error'.format(k)].cache_list(error_list)
        results = flush_scalar_meters(train_meters[str(3 + 1)])
        top1 = top1 + results["top1_error"]
        loss_show = loss_show + results["loss"]
        
    print("-------------------")
    print('AVG_TRAIN_LOSS3: ', loss_show / i_num)
    print('AVG_Top1_error3: ', top1 / i_num)
    f3 = open('./logs/train_err_2GP_loss_32.txt', 'r+')
    f3.read()
    f3.write('\n')
    f3.write(str(loss_show / i_num))
    f3.close()
    f4 = open('./logs/train_err_2GP_err_32.txt', 'r+')
    f4.read()
    f4.write('\n')
    f4.write(str(top1 / i_num))
    f4.close()
    end_time = time.time()
    print('this_train_epoch_time: ', end_time - start_time)
    return k_loss2, k_loss3

if __name__ == '__main__':
    ms.set_context(mode=ms.GRAPH_MODE, device_target="GPU", device_id=1)
    
    train_meters = get_meters('train')
    val_meters = get_meters('val')
    topk = [1, 5]
    dict_own = {}
    batch_size = 128
    val_batch_size = 70
    num_class = 100
    GP_num = 70

    dl_path_cifar100 = "./cifar100_data"
    dl_url_cifar100 = "https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/notebook/datasets/cifar-100-binary.tar.gz"
    dl = DownLoad()
    # 下载CIFAR-100数据集并解压
    dl.download_and_extract_archive(url=dl_url_cifar100, download_path=dl_path_cifar100)
    DATA_DIR_CIFAR100 = "./cifar100_data/cifar-100-binary"
    DATA_VAL_CIFAR100 = "./cifar100_val_all"

    data_train = utils.create_dataset(DATA_DIR_CIFAR100, do_train=True, batch_size=batch_size)
    data_test = utils.create_dataset(DATA_DIR_CIFAR100, do_train=False, batch_size=batch_size)
    data_val = utils.data_loader_val(DATA_VAL_CIFAR100, batch_size=val_batch_size)
    print('train_batch_total', data_train.get_dataset_size())
    print('test_batch_total', data_test.get_dataset_size())
    print('val_batch_total', data_val.get_dataset_size())
    
    start_time = time.time()
    network = resnet20(100)
    for _, cell in network.cells_and_names():
        if isinstance(cell, nn.Dense) or isinstance(cell, nn.Conv2d): 
            cell.weight.set_data(initializer(HeNormal(), cell.weight.shape, cell.weight.dtype))
            
    optimizer = nn.SGD(network.trainable_params(), learning_rate=0.1, momentum=0.9, weight_decay=0.0001)
    
    start_time = time.time()
    k_loss2 = 1
    k_loss3 = 1
    loss_net = utils.CustomWithLossCell(network) # 包含损失函数的Cell
    train_network = utils.CustomTrainOneStepCell(loss_net, optimizer)
    train_network.set_train()

    for epoch in range(200):
        tmp_start = time.time()
        print("lr=", optimizer.learning_rate.data.asnumpy())
        k_loss2, k_loss3 = train_loop(epoch, network, train_network, k_loss2, k_loss3,
                                      train_meters, data_test, data_val,
                                      topk=topk, num_class=num_class, GP_num=GP_num)
        utils.save(epoch, network)
        if epoch == 99:
            ops.assign(optimizer.learning_rate, ms.Tensor(0.01, ms.float32))
        if epoch == 149:
            ops.assign(optimizer.learning_rate, ms.Tensor(0.001, ms.float32))

        end_time = time.time()
        print("this_epoch: ", end_time-tmp_start)
        print("total: ", end_time - start_time, "\n\n\n")
    print("ok")