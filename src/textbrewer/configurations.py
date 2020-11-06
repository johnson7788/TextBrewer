import json
import os
from typing import Union, List, Optional, Dict
from .presets import *

class Config:
    """TrainingConfig和DistillationConfig的基类。"""
    def __init__(self,**kwargs):
        pass

    @classmethod
    def from_json_file(cls, json_filename):
        """Construct configurations from a json file."""
        with open(json_filename,'r') as f:
            json_data = json.load(f)
        return cls.from_dict(json_data)

    @classmethod
    def from_dict(cls, dict_object):
        """Construct configurations from a dict."""
        config = cls(**dict_object)
        return config

    def __str__(self):
        str = ""
        for k,v in self.__dict__.items():
            str += f"{k} : {v}\n"
        return str

    def __repr__(self):
        classname = self.__class__.__name__
        return classname +":\n"+self.__str__()


class TrainingConfig(Config):
    """
    生成训练配置文件

    Args:
        gradient_accumulation_steps (int): 优化之前会累积梯度以减少GPU内存使用量. It calls ``optimizer.step()`` every `gradient_accumulation_steps` backward steps.
        ckpt_frequency (int): 每个epoch存储几个模型权重
        ckpt_epoch_frequency (int): 每个几个epoch存储一个模型权重
        ckpt_steps (int):  if *num_steps* is passes to ``distiller.train()``, 每 **ckpt_steps**保存模型, 同时忽略 `ckpt_frequency` and `ckpt_epoch_frequency` .
        log_dir (str): 保存tensorboard日志文件的目录. Set it to ``None`` to disable tensorboard.
        output_dir (str): 保存模型权重的目录.
        device (str or torch.device): 在CPU或GPU上训练.
        fp16 (bool): if ``True``, 使用Apex实现混合精度训练.
        fp16_opt_level(str): 纯或混合精度优化级别。可接受的值为“ O0”，“ O1”，“ O2”和“ O3”。有关详细信息，请参见Apex文档。.
        data_parallel (bool): If ``True``, 将模型包装 ``torch.nn.DataParallel``.
        local_rank (int): 当前进程的本地rank. 非负值表示我们处于分布式训练模式 ``DistributedDataParallel``.
    Note:
        * 为了进行数据并行(DP)训练，您可以自己在TextBrewer外部使用``torch.nn.DataParallel''包装模型，或者通过将** data_parallel **设置为``True''将工作留给TextBrewer。
        * 要同时启用数据并行训练和混合精度训练，应将** data_parallel **设置为``True``，并且不要自己包装模型。
        * 在一些实验中，我们观察到``torch.nn.DataParallel``的速度降低了。
        * 要执行分布式数据并行(DDP)训练，应在初始化TrainingConfig之前调用torch.distributed.init_process_group。并在初始化蒸馏器时传递“原始”(未包装)模型。
        * DP和DDP是互斥的。
    Example::

        # 通常只需要设置log_dir和output_dir并保留其他默认值即可
        train_config = TrainingConfig(log_dir=my_log_dir, output_dir=my_output_dir)
        
        # 在每个epoch结束时存储模型
        train_config = TrainingConfig(ckpt_frequency=1, ckpt_epoch_frequency=1)
        # 在每个epoch中两次存储模型(在中间和末尾)
        train_config = TrainingConfig(ckpt_frequency=2, ckpt_epoch_frequency=1)
        # 每两个epoch存储一次模型
        train_config = TrainingConfig(ckpt_frequency=1, ckpt_epoch_frequency=2)

    """
    def __init__(self,gradient_accumulation_steps = 1,
                 ckpt_frequency = 1,
                 ckpt_epoch_frequency = 1,
                 ckpt_steps = None,
                 log_dir = None,
                 output_dir = './saved_models',
                 device = 'cuda',
                 fp16 = False,
                 fp16_opt_level = 'O1',
                 data_parallel = False,
                 local_rank = -1
                 ):
        super(TrainingConfig, self).__init__()

        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.ckpt_frequency = ckpt_frequency
        self.ckpt_epoch_frequency = ckpt_epoch_frequency
        self.ckpt_steps = ckpt_steps
        self.log_dir = log_dir
        self.output_dir = output_dir
        self.device = device
        self.fp16 = fp16
        self.fp16_opt_level = fp16_opt_level
        self.data_parallel = data_parallel
        # 分布式训练设置
        self.local_rank = local_rank
        # 本地设置，创建输出文件夹
        if self.local_rank == -1 or torch.distributed.get_rank() == 0:
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)


class IntermediateMatch:
    def __init__(self,layer_T: Union[int,List[int]], layer_S: Union[int,List[int]],
                 weight: float, loss: str, feature: str, proj: Optional[List] = None):
        self.layer_T = layer_T
        self.layer_S = layer_S
        self.feature = feature
        self.weight = weight
        self.loss = loss
        self.proj = proj
        assert feature in FEATURES
        if proj:
            assert proj[0] in PROJ_MAP.keys()
            assert type(proj[1]) is int and type(proj[2]) is int
            if len(proj)==3:
                self.proj.append(dict())   # ['linear', dim_T, dim_S, {...}]
            else:
                assert type(proj[3]) is dict

    def __str__(self):
        str = ""
        for k,v in self.__dict__.items():
            str += f"{k} : {v}, "
        return str[:-2]

    def __repr__(self):
        classname = self.__class__.__name__
        return '\n'+classname +": "+self.__str__()

    @classmethod
    def from_dict(cls,dict_object):
        if dict_object is None:
            return None
        else:
            return cls(**dict_object)


class DistillationConfig(Config):
    """
    与蒸馏方法有关的配置。它定义了要优化的总损失
    
    .. math::

        \mathcal{L}_{total}=  \mathcal{L}_{KD} * w_{KD} + \mathcal{L}_{hl} * w_{hl} + sum(\\textrm{intermediate_losses})
    
    where

        * :math:`\mathcal{L}_{KD}` 是logit的KD损失, :math:`w_{KD}` 是其权重;
        * :math:`\mathcal{L}_{hl}` 是adaptor返回的``损失''之和， :math:`w_{hl}` 是其权重;
        * intermediate_losses is defined via `intermediate_matches`.
    
    Args:
        temperature (float) : 蒸馏温度。在计算KD损失时，teacher-student模型的logits将除以温度。温度通常在1到10的范围内。我们发现高于1的温度通常会带来更好的性能。
        temperature_scheduler: 动态调节温度。有关所有可用选项，请参见：data：`〜textbrewer.presets.TEMPERATURE_SCHEDULER`。
        kd_loss_type (str): KD loss 函数对于adaptor返回的logits, 可以是``'ce'`` 或 ``'mse'``. See :data:`~textbrewer.presets.KD_LOSS_MAP`.
        kd_loss_weight (float): KD损失的权重.
        hard_label_weight (float): adaptor返回的``losses''项目和的权重。`losses``可能包括真实标签上的损失和其他用户定义的损失。
        kd_loss_weight_scheduler: 动态调整KD损失权重。有关所有可用选项，请参见：data：`〜textbrewer.presets.WEIGHT_SCHEDULER`。
        hard_label_weight_scheduler: 动态调整``losses``总和的权重。有关所有可用选项，请参见：data：`〜textbrewer.presets.WEIGHT_SCHEDULER`。
        probability_shift (bool): 如果为``True''，则将ground-truth标签的logit和teacher预测的最大logit进行切换，以使ground-truth标签的logit最大。需要adaptor返回的``labels`''项。
        is_caching_logits (bool): 如果为``True''，则将teacher模型的批次和输出logit缓存在内存中，以便这些logit仅计算一次。它将加快蒸馏过程。 **仅适用于：class：`〜textbrewer.BasicDistiller`和：class：`〜textbrewer.MultiTeacherDistiller`，并且仅当使用dinumers的``train()''方法以num_steps=None`。它适用于中小型数据集。
        intermediate_matches (`List[Dict]`) : 中间特征匹配的配置。列表中的每个元素都是字典，代表一对匹配的配置。

    “intermediate_matches”中的字典包含以下键:

        * '**layer_T**': `layer_T` (*int*): 选择teacher模型的第layer_T层。
        * '**layer_S**': `layer_S` (*int*): 选择Student模型的第layer_S层。

        .. Note::
        
            1. “ layer_T”和“ layer_S”指示adaptor返回的字典中“attention”或“hidden”列表中的层，而不是模型中的实际层。
            2. 如果损失是fst <textbrewer.losses.fsp_loss>或nst <textbrewer.losses.mmd_loss>，则必须分别从teacher和student中选择两层。在这种情况下，“ layer_T”和“ layer_S”是两个整数的列表。请参见下面的示例。

        * '**feature**': `feature` (*str*): 中间层的特征。可以是:

            * '**attention**' : attention matrix, 形状是 (*batch_size*, *num_heads*, *length*, *length*) or (*batch_size*, *length*, *length*)
            * '**hidden**'：hidden states, 形状是 (*batch_size*, *length*, *hidden_dim*).

        * '**loss**' : `loss` (*str*) : loss function. See :data:`~textbrewer.presets.MATCH_LOSS_MAP` for available losses. 当前有: ``'attention_mse'``, ``'attention_ce'``, ``'hidden_mse'``, ``'nst'``, etc.
        * '**weight**': `weight` (float) : 损失的权重.
        * '**proj**' : `proj` (*List*, optional) : 如果teacher和student的特征尺寸相同，则为可选；否则是必需的。它是匹配teacher和student中间特征尺寸的映射特征。它是一个包含以下元素的列表：
            * **proj[0]** (*str*): 映射函数, can be ``'linear'``, ``'relu'``, ``'tanh'``. See :data:`~textbrewer.presets.PROJ_MAP`.
            * **proj[1]** (*int*): student模型的特征维度.
            * **proj[2]** (*int*): teacher模型的特征维度.
            * **proj[3]** (*dict*): 可选，提供诸如学习率之类的配置。如果未提供，学习率和优化器配置将遵循优化器的默认配置，否则将使用此处指定的配置。

    Example::

        from textbrewer import DistillationConfig

        # 配置简单：使用默认值，或尝试不同的温度
        distill_config = DistillationConfig(temperature=8)

        # 添加中间特征匹配
        # 在此设置下，adaptor_T/S的返回字典results_T/S 应包含“hidden”键。
        # 将计算teacher的results_T['hidden'][10]与student的results_S['hidden'][3]之间的mse损失
        distill_config = DistillationConfig(
            temperature=8,
            intermediate_matches = [{'layer_T':10, 'layer_S':3, 'feature':'hidden', 'loss':'hidden_mse', 'weight':1}]
        )

        # 多个非中间特征匹配。teacher和student的hidden_​​dim分别为768和384。
        distill_config = DistillationConfig(
            temperature = 8, 
            intermediate_matches = [ \\
            {'layer_T':0,  'layer_S':0, 'feature':'hidden','loss': 'hidden_mse', 'weight' : 1,'proj':['linear',384,768]},
            {'layer_T':4,  'layer_S':1, 'feature':'hidden','loss': 'hidden_mse', 'weight' : 1,'proj':['linear',384,768]},
            {'layer_T':8,  'layer_S':2, 'feature':'hidden','loss': 'hidden_mse', 'weight' : 1,'proj':['linear',384,768]},
            {'layer_T':12, 'layer_S':3, 'feature':'hidden','loss': 'hidden_mse', 'weight' : 1,'proj':['linear',384,768]}]
        )

    """
    def __init__(self,temperature=4,
                      temperature_scheduler = 'none',
                      hard_label_weight=0,
                      hard_label_weight_scheduler = 'none',
                      kd_loss_type='ce',
                      kd_loss_weight=1,
                      kd_loss_weight_scheduler = 'none',
                      probability_shift = False,
                      intermediate_matches:Optional[List[Dict]]=None,
                      is_caching_logits = False):
        super(DistillationConfig, self).__init__()

        self.temperature = temperature
        self.temperature_scheduler = None
        if temperature_scheduler is not 'none':
            assert temperature_scheduler in TEMPERATURE_SCHEDULER, \
                    "Invalid temperature_scheduler"
            self.temperature_scheduler = TEMPERATURE_SCHEDULER[temperature_scheduler]

        self.hard_label_weight = hard_label_weight
        self.hard_label_weight_scheduler = None
        if hard_label_weight_scheduler is not 'none':
            assert hard_label_weight_scheduler in WEIGHT_SCHEDULER, \
                    "Invalid hard_label_weight_scheduler"
            self.hard_label_weight_scheduler = WEIGHT_SCHEDULER[hard_label_weight_scheduler]

        self.kd_loss_type = kd_loss_type
        self.kd_loss_weight = kd_loss_weight
        self.kd_loss_weight_scheduler = None
        if kd_loss_weight_scheduler is not 'none':
            assert kd_loss_weight_scheduler in WEIGHT_SCHEDULER, \
                    "Invalid kd_loss_weight_scheduler"
            self.kd_loss_weight_scheduler = WEIGHT_SCHEDULER[kd_loss_weight_scheduler]

        self.probability_shift = probability_shift

        self.intermediate_matches:[List[IntermediateMatch]] = []
        if intermediate_matches:
            self.intermediate_matches = [IntermediateMatch.from_dict(im) for im in intermediate_matches]

        self.is_caching_logits = is_caching_logits