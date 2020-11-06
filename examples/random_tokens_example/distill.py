import textbrewer
from textbrewer import GeneralDistiller
from textbrewer import TrainingConfig, DistillationConfig
from transformers import BertForSequenceClassification, BertConfig, AdamW
from transformers import get_linear_schedule_with_warmup
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

# device
device = torch.device('cpu')

# 定义teacher模型和student模型配置, 默认num_labels=2
bert_config = BertConfig.from_json_file('bert_config/bert_config.json')
bert_config_T3 = BertConfig.from_json_file('bert_config/bert_config_T3.json')

# 把output_hidden_states设置为True, 所以distill_config的配置中才能使用hidden的feature
bert_config.output_hidden_states = True
bert_config_T3.output_hidden_states = True

# 默认 num_labels = 2
# teacher应使用预训练的权重进行初始化，并在下游任务上进行微调。为了演示，我们在这里省略了这些步骤
teacher_model = BertForSequenceClassification(bert_config)

student_model = BertForSequenceClassification(bert_config_T3)

teacher_model.to(device=device)
student_model.to(device=device)


# 定义Dataset
class DictDataset(Dataset):
    def __init__(self, all_input_ids, all_attention_mask, all_labels):
        assert len(all_input_ids) == len(all_attention_mask) == len(all_labels)
        self.all_input_ids = all_input_ids
        self.all_attention_mask = all_attention_mask
        self.all_labels = all_labels

    def __getitem__(self, index):
        return {'input_ids': self.all_input_ids[index],
                'attention_mask': self.all_attention_mask[index],
                'labels': self.all_labels[index]}

    def __len__(self):
        return self.all_input_ids.size(0)


# 准备随机数据用测试【100，128】， 100条数据，每条数据有128个字
all_input_ids = torch.randint(low=0, high=100, size=(100, 128))
# 为了测试简单设置所有attention都为1
all_attention_mask = torch.ones_like(all_input_ids)
# 为100个样本生成标签
all_labels = torch.randint(low=0, high=2, size=(100,))
# 构造DictDataset
dataset = DictDataset(all_input_ids, all_attention_mask, all_labels)
eval_dataset = DictDataset(all_input_ids, all_attention_mask, all_labels)
# 构造dataloader
dataloader = DataLoader(dataset, batch_size=32)
num_epochs = 10
# 训练steps是epoch * dataloader的长度
num_training_steps = len(dataloader) * num_epochs
# Optimizer and learning rate scheduler
optimizer = AdamW(student_model.parameters(), lr=1e-4)
# 学习率scheduler
scheduler_class = get_linear_schedule_with_warmup
# arguments dict except 'optimizer'， 前10%作为num_warmup_steps
scheduler_args = {'num_warmup_steps': int(0.1 * num_training_steps), 'num_training_steps': num_training_steps}

# display model parameters statistics, result是模型参数信息
print("\nteacher_model's parametrers:")
result, _ = textbrewer.utils.display_parameters(teacher_model, max_level=3)
print(result)

print("student_model's parametrers:")
result, _ = textbrewer.utils.display_parameters(student_model, max_level=3)
print(result)


def simple_adaptor(batch, model_outputs):
    # The second element of model_outputs is the logits before softmax
    # The third element of model_outputs is hidden states
    return {'logits': model_outputs[1],
            'hidden': model_outputs[2],
            'inputs_mask': batch['attention_mask']}


# Define callback function
def predict(model, eval_dataset, step, device):
    '''
    eval_dataset: 验证数据集
    '''
    model.eval()
    pred_logits = []
    label_ids = []
    dataloader = DataLoader(eval_dataset, batch_size=32)
    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels']
        with torch.no_grad():
            logits, _ = model(input_ids=input_ids, attention_mask=attention_mask)
            cpu_logits = logits.detach().cpu()
        for i in range(len(cpu_logits)):
            pred_logits.append(cpu_logits[i].numpy())
            label_ids.append(labels[i])
    model.train()
    pred_logits = np.array(pred_logits)
    label_ids = np.array(label_ids)
    y_p = pred_logits.argmax(axis=-1)
    accuracy = (y_p == label_ids).sum() / len(label_ids)
    print("Number of examples: ", len(y_p))
    print("Acc: ", accuracy)


from functools import partial

# 定义callback func，先提供dataset和device参数
callback_fun = partial(predict, eval_dataset=eval_dataset, device=device)  # fill other arguments

# 初始化训练配置文件和 distiller蒸馏配置文件
train_config = TrainingConfig(device=device)
# 蒸馏模型的配置
distill_config = DistillationConfig(
    temperature=8,
    hard_label_weight=0,
    kd_loss_type='ce',
    probability_shift=False,
    intermediate_matches=[
        {'layer_T': 0, 'layer_S': 0, 'feature': 'hidden', 'loss': 'hidden_mse', 'weight': 1},
        {'layer_T': 8, 'layer_S': 2, 'feature': 'hidden', 'loss': 'hidden_mse', 'weight': 1},
        {'layer_T': [0, 0], 'layer_S': [0, 0], 'feature': 'hidden', 'loss': 'nst', 'weight': 1},
        {'layer_T': [8, 8], 'layer_S': [2, 2], 'feature': 'hidden', 'loss': 'nst', 'weight': 1}]
)

print("train_config:")
print(train_config)

print("distill_config:")
print(distill_config)
# 初始化distiller
distiller = GeneralDistiller(
    train_config=train_config, distill_config=distill_config,
    model_T=teacher_model, model_S=student_model,
    adaptor_T=simple_adaptor, adaptor_S=simple_adaptor)

#开始蒸馏
with distiller:
    distiller.train(optimizer, dataloader, num_epochs=num_epochs,
                    scheduler_class=scheduler_class, scheduler_args=scheduler_args, callback=callback_fun)
