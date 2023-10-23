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

class CustomWithLossCell1(nn.Cell):
    def __init__(self, backbone):
        super(CustomWithLossCell1, self).__init__(auto_prefix=False)
        self._backbone = backbone
        self._loss_fn = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
        self.topk = [1, 5]
        self.reduce_mean = ops.ReduceMean()
        self.reshape = ops.Reshape()

    def construct(self, input, target):
        #global train_meters
        loss = self.forward_loss(input, target)
        return loss
    
    def forward_loss(self, input, target):
        """forward model and return loss"""
        output,_ = self._backbone(input)

        loss = self.reduce_mean(self._loss_fn(output, target))

        _, pred = ops.top_k(output, max(self.topk))
        pred = pred.T
        
        correct = ops.equal(pred, self.reshape(target,(1,-1)).expand_as(pred))
        #    tar = torch.max(pred,1)

        return loss, correct

class CustomWithLossCell2(nn.Cell):
    def __init__(self, backbone):
        super(CustomWithLossCell2, self).__init__(auto_prefix=False)
        self._backbone = backbone
        self._loss_fn = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
        self.criterion1 = ops.KLDivLoss(reduction="sum")
        self.topk = [1, 5]
        self.softmax = ops.Softmax(axis=1)
        self.zeros = ops.Zeros()
        self.uniformreal = ops.UniformReal()
        self.reshape = ops.Reshape()
        self.ArgMaxWithValue = ops.ArgMaxWithValue(1)
        self.reduce_sum = ops.ReduceSum()
        self.reduce_mean = ops.ReduceMean()

    
    def construct(self, input, target, base_data_tensor, base_data_target, K_ni, err, k_loss2, k_loss3, batch_size):
        loss, k_loss2, k_loss3, correct = self.forward_loss_2(input, target,
                                                    base_data_tensor, base_data_target, K_ni, batch_size,
                                                    k_loss2, k_loss3, err) 
        k_loss2 = stop_gradient(k_loss2)
        k_loss3 = stop_gradient(k_loss3)
        return loss, k_loss2, k_loss3, correct
        
    def own_KL(self, input0, label0):   
        B, C = ops.shape(input0)
        input = self.softmax(input0*10)
        label = self.softmax(label0*10)
        result_KL_temp = self.criterion1(input, label)
        result_KL = result_KL_temp / (ops.shape(input[0])) 
        return result_KL

    def GP_loss_use_cal(self, old_data_use, K_ni, input, batch_size):
        B, channel_num = ops.shape(input)
        input_3 = self.zeros((channel_num, B, batch_size),ms.float32)   #128
        input_old = self.zeros((channel_num, batch_size,B),ms.float32)  #128
        old_data_use1 = stop_gradient(old_data_use)
        total_num,_ = ops.shape(old_data_use1)
        sum_here = self.uniformreal((B, total_num))

        old_data_use1_new = ops.transpose(old_data_use1,(1,0))
        old_data_use1_new = ops.expand_dims(old_data_use1_new,2)
        old_data_use2 = old_data_use1_new.expand_as(input_old)
        old_data_use2 = ops.transpose(old_data_use2,(2,1,0))

        input_new = ops.transpose(input,(1,0))
        input_new = ops.expand_dims(input_new,2)
        input_use = input_new.expand_as(input_3)  
        input_use1 = ops.transpose(input_use,(1,2,0))
    #    for i in range(B):
        sum_here = 2.718281828 ** (-(1 / 200) * ops.norm((input_use1 - old_data_use2), 2, 2))
    #    print('sum_here: ', ops.shape(sum_here))
        K_a = sum_here
        K_u = ops.matmul(K_a,K_ni)
        K_var_sum = 0
        K_var = - (ops.matmul(ops.matmul(K_a,K_ni), K_a.T)) + 1.
        for i in range(B):
            K_var_sum = K_var_sum + K_var[i, i]
            # K_var_sum = K_var_sum + float(K_var[i, i])
        K_var_sum = K_var_sum / B
    #    print("K_var: ", K_var_sum)
        return K_u, K_var_sum, B
    
    def forward_loss_2(self, input, target, old_data_use, old_target, K_ni, batch_size, k_loss2, k_loss3, err):
        output, features = self._backbone(input)
        K_u, K_var, input_batch_size = self.GP_loss_use_cal(old_data_use, K_ni, features, batch_size)
        result_V_T1 = stop_gradient(ops.matmul(K_u,old_target))
        result_V_T2 = ops.matmul(K_u,old_target)
        _, y = self.ArgMaxWithValue(result_V_T1)
        x1 = ops.equal(target,y)
        x2 = self.reduce_sum(x1)
        x3 = x2 / input_batch_size
        k3 = 1 / (1 + K_var)
        loss1 = self.reduce_mean(self._loss_fn(output, target))
        loss2 = self.reduce_mean(self.own_KL(output, result_V_T1))
        temp = self._loss_fn(result_V_T2, target)
        loss3 = self.reduce_mean(temp)
        #loss3 = reduce_mean(criterion(result_V_T2, target)) 
        loss = (loss1 + k3 * (1. -err) * (k_loss2 * loss2 + k_loss3 * loss3) / 2.) / (1. + (1. - err))
        k_loss2 = loss1 / loss2
        k_loss3 = loss1 / loss3

        _, pred = ops.top_k(output, max(self.topk))
        pred = pred.T

        correct = ops.equal(pred, self.reshape(target,(1,-1)).expand_as(pred))

        return loss, k_loss2, k_loss3, correct

class CustomTrainOneStepCell1(nn.Cell):
    def __init__(self, network, optimizer):
        super(CustomTrainOneStepCell1, self).__init__(auto_prefix=False)
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
        loss, correct = self.network(*inputs)    # 运行正向网络，获取loss
        grads = self.grad(self.network, self.weights)(*inputs) # 获得所有Parameter自由变量的梯度
        # grads = grad_op(grads)    # 可以在这里加对梯度的一些计算逻辑，如梯度裁剪
        grads = self.grad_reducer(grads)  # 梯度聚合
        loss = F.depend(loss, self.optimizer(grads))
        return loss, correct
    
class CustomTrainOneStepCell2(nn.Cell):
    def __init__(self, network, optimizer):
        super(CustomTrainOneStepCell2, self).__init__(auto_prefix=False)
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
        loss, k_loss2, k_loss3, correct = self.network(*inputs)    # 运行正向网络，获取loss
        grads = self.grad(self.network, self.weights)(*inputs) # 获得所有Parameter自由变量的梯度
        # grads = grad_op(grads)    # 可以在这里加对梯度的一些计算逻辑，如梯度裁剪
        grads = self.grad_reducer(grads)  # 梯度聚合
        # print("grad=", grads)
        loss = F.depend(loss, self.optimizer(grads))
        return loss, k_loss2, k_loss3, correct

def test(epoch, i, network, train_net1, train_net2, data_val, topk):
    print('\nTest: ', epoch, 'i_num: ', i)
    #model.eval()
    train_net1.set_train(False)
    train_net2.set_train(False)
    network.set_train(False)
    i_num = 0
    top1 = 0
    loss1_all = 0
    reduce_sum = ops.ReduceSum()
    reduce_mean = ops.ReduceMean()
    reshape = ops.Reshape()
    criterion = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
    for batch_x, batch_y in data_val:
        B,_,_,_ = ops.shape(batch_x)
        i_num = i_num + 1
        output, _ = network(batch_x)
        # loss1 = float(stop_gradient(reduce_mean(criterion(output, batch_y))))
        loss1 = stop_gradient(reduce_mean(criterion(output, batch_y)))
        loss1_all = loss1_all + loss1
        _, pred = ops.top_k(output, max(topk))
        pred = pred.T
        correct = ops.equal(pred, reshape(batch_y,(1,-1)).expand_as(pred))
        correct_k_list = ops.cast(correct[:1],ms.float32).sum(axis=0)
        correct_k = reduce_sum(correct_k_list)
        # error = 1.0 - (float(correct_k) / int(B))
        error = 1.0 - (correct_k / B)
        top1 = top1 + error
    print( 'OUT_TEST_Top1_error3: ',top1 / i_num)
    network.set_train(True)
    return (top1 / i_num), (loss1_all / i_num)

def create_dataset(data_home, do_train, batch_size):
    normalize = ds.vision.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225],is_hwc=False)
    if do_train:
        cifar_ds = ds.Cifar10Dataset(dataset_dir=data_home, shuffle=True, usage='train')
    else:
        cifar_ds = ds.Cifar10Dataset(dataset_dir=data_home, shuffle=True, usage='test')  #shuffle=False

    if do_train:
        transform_data = ds.transforms.Compose([ds.vision.RandomHorizontalFlip(),
                                    ds.vision.RandomCrop(32, 4),
                                    ds.vision.ToTensor(),
                                    normalize])
    else:
        transform_data = ds.transforms.Compose([ds.vision.ToTensor(),
                                                normalize])

    # Transformation on label  
    transform_label = ds.transforms.TypeCast(mstype.int32)

    # Apply map operations on images

    cifar_ds = cifar_ds.map(operations=transform_label,
                            python_multiprocessing=False, input_columns="label")
    cifar_ds = cifar_ds.map(operations=transform_data,
                            python_multiprocessing=False, input_columns="image")
    cifar_ds = cifar_ds.batch(batch_size, drop_remainder=False)
    return cifar_ds

def GP_new_cal(input, batch_size1):
        batch_size,channel_num = ops.shape(input)
        zeros = ops.Zeros()
        input_3 = zeros((channel_num, batch_size, batch_size),ms.float32)
        uniformreal = ops.UniformReal()
        sum = uniformreal((batch_size, batch_size))
        input_new = ops.transpose(input,(1,0))
        input_new = ops.expand_dims(input_new,2)
        input_use = input_new.expand_as(input_3)
        input_use1 = ops.transpose(input_use,(1,2,0))
        input_use2 = ops.transpose(input_use,(2,1,0))
    #    for j in range(batch_size):
        sum = stop_gradient(2.718281828 ** (-(1 / 200) * ops.norm((input_use1 - input_use2), 2, 2)))
    #    print('sum_here11: ', ops.shape(sum))
        #    print("sum: ", sum)
        matrix_inverse = ops.MatrixInverse(adjoint=False)
        K_ni = stop_gradient(matrix_inverse(sum))
    #    print(sum)
        return sum, K_ni
    
# Save checkpoint.
def save(epoch, net):
    print('Saving..')
    ms.save_checkpoint(net, './checkpoint/cifar10_32/ckpt_2_' + str(epoch) + '.ckpt')