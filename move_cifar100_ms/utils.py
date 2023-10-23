import time
import mindspore as ms
import numpy as np
import mindspore.dataset as ds
from mindvision.classification.dataset import Cifar10
from mindvision.dataset import DownLoad
from mindspore.common.initializer import initializer, HeNormal
from mindspore import dtype as mstype
from mindspore.ops import stop_gradient
from mindspore.ops import functional as F
    
from mindspore import ops, nn
from mindspore.parallel._utils import (_get_device_num, _get_gradients_mean,
                                       _get_parallel_mode)

class CustomWithLossCell(nn.Cell):
    def __init__(self, backbone):
        super(CustomWithLossCell, self).__init__(auto_prefix=False)
        self._backbone = backbone
        self._loss_fn = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
        self.criterion1 = ops.KLDivLoss(reduction="sum")
        self.softmax = ops.Softmax(axis=1)
        self.zeros = ops.Zeros()
        self.cat = ops.Concat(axis=0)
        self.cast = ops.Cast()
        self.uniformreal = ops.UniformReal()
        self.reduce_mean = ops.ReduceMean()
        self.matmul = ops.MatMul()
        self.norm = ops.LpNorm(axis=1, p=2)

    
    def construct(self, input, target, val_tensor_list, val_target_list, val_K_ni_list, GP_num, k_loss2, k_loss3, err):
        loss, output = self.forward_loss(input, target, val_tensor_list, val_target_list, val_K_ni_list, GP_num, k_loss2, k_loss3, err) 
        return loss, output
        
    def own_KL(self, input0, label0):           
        input = self.softmax(input0)
        label = self.softmax(label0)
        result_KL_temp = self.criterion1(input, label)
        result_KL =  result_KL_temp / ops.shape(input)[0] 
        return result_KL

    def GP_loss_use_cal(self, old_data_use, K_ni, input, GP_num):
        B, _ = input.shape
        old_data_use1 = stop_gradient(old_data_use)
        total_num, _ = old_data_use1.shape
        sum_here = self.uniformreal((B, total_num))

        for i in range(B):
            sum_here[i] = 2.718281828 ** (
                -(1 / GP_num) * self.norm((input[i, :].expand_as(old_data_use1) - old_data_use1)))

        K_u = self.matmul(sum_here, K_ni)
        K_var_sum = 0
        K_var = - (self.matmul(self.matmul(sum_here, K_ni), sum_here.T)) + 1.
        
        for i in range(B):
            K_var_sum = K_var_sum + K_var[i, i]
        K_var_sum = K_var_sum / B
        
        return K_u, K_var_sum, B
    
    def forward_loss(self, input, target, val_tensor_list, val_target_list, val_K_ni_list, GP_num, k_loss2, k_loss3, err):
        B_num, _, _, _ = input.shape
        output, features = self._backbone(input)
        _, channel = features.shape
        current_num = target.item(0)
        tmp = self.zeros((1, channel), ms.float32)
        tmp[0, :] = features[0, :]
        K_var_all = 0
        K_u, K_var, _ = self.GP_loss_use_cal(val_tensor_list[current_num], val_K_ni_list[current_num], tmp, GP_num)
        K_var_all = K_var_all + K_var
        val_target_list = self.cast(val_target_list, ms.float32)
        result_V_T = self.matmul(K_u, val_target_list[current_num])
        for j in range(B_num - 1):
            current_num = target.item(0)
            tmp = self.zeros((1, channel), ms.float32)
            tmp[0, :] = features[j + 1, :]
            K_u, K_var, _ = self.GP_loss_use_cal(val_tensor_list[current_num], val_K_ni_list[current_num], tmp, GP_num)
            K_var = stop_gradient(K_var)
            K_var_all = K_var_all + K_var
            val_target_list[current_num] = stop_gradient(val_target_list[current_num])
            result_V_T_tmp = stop_gradient(self.matmul(K_u, val_target_list[current_num]))
            result_V_T = self.cat((result_V_T, result_V_T_tmp))
        K_var_avg = K_var_all / B_num
        k3 = 1. / (1. + K_var_avg)
        
        result_V_T1 = stop_gradient(result_V_T)
        result_V_T2 = result_V_T
        
        loss1 = self.reduce_mean(self._loss_fn(output, target))
        loss2 = self.reduce_mean(self.own_KL(output, result_V_T1))
        loss3 = self.reduce_mean(self._loss_fn(result_V_T2, target))
        
        k3 = stop_gradient(k3)
        err = stop_gradient(err)

        loss = loss1 / (1. + (1. - err)) + k3 * (1. - err) * (k_loss2 * loss2 + k_loss3 * loss3) / (1. + (1. - err)) / 2.

        output = stop_gradient(output)
        return loss, output

class CustomTrainOneStepCell(nn.Cell):
    def __init__(self, network, optimizer):
        super(CustomTrainOneStepCell, self).__init__(auto_prefix=False)
        self.network = network  # 带loss的网络结构
        self.network.set_grad()   # PYNATIVE模式时需要，如果为True，则在执行正向网络时，将生成需要计算梯度的反向网络。
        self.optimizer = optimizer   # 优化器，用于参数更新
        self.weights = self.optimizer.parameters    # 获取优化器的参数
        self.grad = ops.GradOperation(get_by_list=True)   # 获取所有输入和参数的梯度

        # 并行计算相关逻辑
        self.reducer_flag = False
        self.grad_reducer = ops.identity
        self.parallel_mode = _get_parallel_mode()
        self.reducer_flag = self.parallel_mode in (ms.ParallelMode.DATA_PARALLEL, ms.ParallelMode.HYBRID_PARALLEL)
        if self.reducer_flag:
            self.mean = _get_gradients_mean()
            self.degree = _get_device_num()
            self.grad_reducer = nn.DistributedGradReducer(self.weights, self.mean, self.degree)

    def construct(self, *inputs):
        loss, output = self.network(*inputs)    # 运行正向网络，获取loss
        grads = self.grad(self.network, self.weights)(*inputs) # 获得所有Parameter自由变量的梯度
        # grads = grad_op(grads)    # 可以在这里加对梯度的一些计算逻辑，如梯度裁剪
        grads = self.grad_reducer(grads)  # 梯度聚合
        loss = F.depend(loss, self.optimizer(grads))
        return loss, output
    
def test(epoch, i, network, train_net, data_test, topk):
    print('\nTest: ', epoch, 'i_num: ', i)
    #model.eval()
    train_net.set_train(False)
    network.set_train(False)
    i_num = 0
    top1 = 0
    loss1_all = 0
    reduce_sum = ops.ReduceSum()
    reduce_mean = ops.ReduceMean()
    reshape = ops.Reshape()
    criterion = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
    for data in data_test:
        batch_x, batch_y = data[0], data[2]
        B,_,_,_ = ops.shape(batch_x)
        i_num = i_num + 1
        output, _ = network(batch_x)
        loss1 = stop_gradient(reduce_mean(criterion(output, batch_y)))
        loss1_all = loss1_all + loss1
        _, pred = ops.top_k(output, max(topk))
        pred = pred.T
        correct = ops.equal(pred, reshape(batch_y,(1,-1)).expand_as(pred))
        correct_k_list = ops.cast(correct[:1],ms.float32).sum(axis=0)
        correct_k = reduce_sum(correct_k_list)
        error = 1.0 - (correct_k / B)
        top1 = top1 + error
    print( 'OUT_TEST_Top1_error3: ',top1 / i_num)
    train_net.set_train(True)
    network.set_train(True)
    return (top1 / i_num), (loss1_all / i_num)

def create_dataset(data_home, do_train, batch_size):
    normalize = ds.vision.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225], is_hwc=False)
    if do_train:
        cifar_ds = ds.Cifar100Dataset(dataset_dir=data_home, shuffle=True, usage='train')
    else:
        cifar_ds = ds.Cifar100Dataset(dataset_dir=data_home, shuffle=False, usage='test')  #shuffle=False

    if do_train:
        transform_data = ds.transforms.Compose([ds.vision.RandomCrop(32, 4), 
                                                ds.vision.RandomHorizontalFlip(), 
                                                ds.vision.ToTensor(), 
                                                normalize])
    else:
        transform_data = ds.transforms.Compose([ds.vision.ToTensor(),
                                                normalize])

    # Transformation on label  
    transform_label = ds.transforms.TypeCast(mstype.int32)

    # Apply map operations on images

    cifar_ds = cifar_ds.map(operations=transform_label,
                            python_multiprocessing=False, input_columns="fine_label")
    cifar_ds = cifar_ds.map(operations=transform_data,
                            python_multiprocessing=False, input_columns="image")
    cifar_ds = cifar_ds.batch(batch_size, drop_remainder=False)
    return cifar_ds

def data_loader_val(data_home, batch_size):
    normalize = ds.vision.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225], is_hwc=False)

    cifar_ds = ds.ImageFolderDataset(dataset_dir=data_home, shuffle=False, decode=True)
    
    # Transformation on image
    transform_data = ds.transforms.Compose([ds.vision.ToTensor(),
                                            normalize])

    # Transformation on label  
    transform_label = ds.transforms.TypeCast(mstype.int32)

    # Apply map operations on images
    cifar_ds = cifar_ds.map(operations=transform_data,
                            python_multiprocessing=False, input_columns="image")
    cifar_ds = cifar_ds.map(operations=transform_label,
                            python_multiprocessing=False, input_columns="label")
    cifar_ds = cifar_ds.batch(batch_size=batch_size, drop_remainder=False)
    return cifar_ds

def val_data_build(network, train_net, data_val, onehot, num_class, GP_num):
    train_net.set_train(False)
    network.set_train(False)
    
    start_time = time.time()
    base_data = []
    base_target = []
    use_tensor = []
    use_target = []
    use_K_ni = []
    class_np = np.load('./checkpoint/5near_list_local_5.npy', allow_pickle=True)
    for i, (batch_x, batch_y) in enumerate(data_val):
        depth, on_value, off_value = num_class, ms.Tensor(1.0, ms.float32), ms.Tensor(0.0, ms.float32)
        one_hot = onehot(batch_y, depth, on_value, off_value)
        one_hot = ops.cast(one_hot, ms.int32)
        _, features = network(batch_x)
        base_data.append(features.asnumpy())
        base_target.append(one_hot.asnumpy())
    for k in range(num_class):
        _, long = class_np.shape
        for i in range(long):
            base_data_tensor1 = base_data[class_np[k][i]]
            base_data_target1 = base_target[class_np[k][i]]
            if i == 0:
                base_data_tensor = base_data_tensor1
                base_data_target = base_data_target1
            else:
                base_data_tensor = np.append(base_data_tensor, base_data_tensor1, axis=0)
                base_data_target = np.append(base_data_target, base_data_target1, axis=0)

        K_ni = GP_new_cal_np(base_data_tensor, GP_num=GP_num)
        base_data_tensor = ms.Tensor.from_numpy(base_data_tensor)
        base_data_target = ms.Tensor.from_numpy(base_data_target)
        
        use_tensor.append(base_data_tensor)
        use_target.append(base_data_target)
        use_K_ni.append(K_ni)
    end_time = time.time()
    print('build_val_time: ', end_time - start_time)
    train_net.set_train(True)
    network.set_train(True)
    return use_tensor, use_target, use_K_ni

def GP_new_cal_np(input, GP_num):
        batch_size = input.shape[0]
        sum = np.zeros([batch_size, batch_size], dtype=np.float32)
        for j in range(batch_size):
            sum_a = np.tile(input[j, :], (350, 1)) - input
            sum_b = np.linalg.norm(x=sum_a, ord=2, axis=1)
            sum_c = -(1 / GP_num) * sum_b
            sum_temp = 2.718281828 ** sum_c
            sum[j] =  sum_temp
        K_ni = np.linalg.inv(sum)
        K_ni = ms.Tensor.from_numpy(K_ni)
        return K_ni

# Save checkpoint.
def save(epoch, net):
    print('Saving..')
    ms.save_checkpoint(net, './checkpoint/5/ckpt_top5_res20_' + str(epoch) + '.ckpt')
    