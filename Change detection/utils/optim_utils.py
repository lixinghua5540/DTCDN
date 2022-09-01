import torch.optim as optim

optimizer_name =  'SGD'
lr_mode = 'exp'
gamma2 = 0.9
base_lr= 1e-2
momentun=0
dampening=0
# eps =
# betas =
def optimizer(net):

    if optimizer_name=='SGD':
        optimizer = optim.SGD(net.parameters(), lr=base_lr,momentum=momentun, dampening=dampening,\
                              weight_decay=1e-4)
    # elif optimizer_name=='Adam':
    #     optimizer = optim.Adam(net.parameters(), lr=base_lr, betas=conf['betas'], eps=conf['eps'],
    #              weight_decay=c.WEIGHT_DECAY)
    return optimizer
def scheduler(optimizer):

    # if lr_mode=='stepwise':
    #     '''
    #     step_size(int)- 学习率下降间隔数，若为 30，则会在 30、 60、 90…个 step 时，将学习率调整为 lr*gamma。
    #     gamma(float)- 学习率调整倍数，默认为 0.1 倍，即下降 10 倍。
    #     last_epoch(int)- 上一个 epoch 数，这个变量用来指示学习率是否需要调整。当last_epoch 符合设定的间隔时，就会对学习率进行调整。当为-1 时，学习率设置为初始值。
    #     '''
    #     return optim.lr_scheduler.StepLR(optimizer, step_size=conf['step_size'], gamma=conf['gamma1'], last_epoch=-1)
    if lr_mode=='exp':
        '''
        gamma- 学习率调整倍数的底，指数为 epoch，即 gamma**epoch
        '''
        return optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma2, last_epoch=-1)
    # elif  lr_mode=='cos':
    #     '''
    #     T_max(int)- 一次学习率周期的迭代次数，即 T_max 个 epoch 之后重新设置学习率。
    #     eta_min(float)- 最小学习率，即在一个周期中，学习率最小会下降到 eta_min，默认值为 0。
    #     '''
    #     return optim.lr_scheduler.CosineAnnealingLR(optimizer, gamma=conf['T_max'], eta_min=0, last_epoch=-1)
    # elif lr_mode=='plateau':
    #     '''
    #     mode(str)- 模式选择，有 min 和 max 两种模式， min 表示当指标不再降低(如监测loss)， max 表示当指标不再升高(如监测 accuracy)。
    #     factor(float)- 学习率调整倍数(等同于其它方法的 gamma)，即学习率更新为 lr = lr * factor
    #     patience(int)- 忍受该指标多少个 step 不变化，当忍无可忍时，调整学习率。
    #     verbose(bool)- 是否打印学习率信息， print(‘Epoch {:5d}: reducing learning rate of group {} to {:.4e}.’.format(epoch, i, new_lr))
    #     threshold_mode(str)- 选择判断指标是否达最优的模式，有两种模式， rel 和 abs。
    #     当 threshold_mode == rel，并且 mode == max 时， dynamic_threshold = best * ( 1 +threshold )；
    #     当 threshold_mode == rel，并且 mode == min 时， dynamic_threshold = best * ( 1 -threshold )；
    #     当 threshold_mode == abs，并且 mode== max 时， dynamic_threshold = best + threshold ；
    #     当 threshold_mode == rel，并且 mode == max 时， dynamic_threshold = best - threshold；
    #     threshold(float)- 配合 threshold_mode 使用。
    #     cooldown(int)- “冷却时间“，当调整学习率之后，让学习率调整策略冷静一下，让模型再训练一段时间，再重启监测模式。
    #     min_lr(float or list)- 学习率下限，可为 float，或者 list，当有多个参数组时，可用 list 进行设置。
    #     eps(float)- 学习率衰减的最小值，当学习率变化小于 eps 时，则不调整学习率。
    #     '''
    #     return optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode=conf['p_mode'], factor=conf['gamma3'], patience=conf['patience'],
    #                                             verbose=True, threshold=conf['p_thre'], threshold_mode=conf['threshold_mode'], cooldown=0,
    #                                             min_lr=conf['min_lr'], eps=1e-08)
    else:
        return optimizer

