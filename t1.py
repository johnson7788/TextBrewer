import textbrewer
from textbrewer import GeneralDistiller
from textbrewer import TrainingConfig, DistillationConfig
# 展示模型参数量的统计
print("\nteacher_model's parametrers:")
result, _ = textbrewer.utils.display_parameters(teacher_model,max_level=3)
print (result)


print("student_model's parametrers:")
result, _ = textbrewer.utils.display_parameters(student_model,max_level=3)
print (result)

# 定义adaptor用于解释模型的输出
def simple_adaptor(batch, model_outputs):
    # model输出的第二、三个元素分别是logits和hidden states
    return {'logits': model_outputs[1], 'hidden': model_outputs[2]}

# 蒸馏与训练配置
# 匹配教师和学生的embedding层；同时匹配教师的第8层和学生的第2层
distill_config = DistillationConfig(
    intermediate_matches=[
     {'layer_T':0, 'layer_S':0, 'feature':'hidden', 'loss': 'hidden_mse','weight' : 1},
     {'layer_T':8, 'layer_S':2, 'feature':'hidden', 'loss': 'hidden_mse','weight' : 1}])
train_config = TrainingConfig()


#初始化distiller
distiller = GeneralDistiller(
    train_config=train_config, distill_config = distill_config,
    model_T = teacher_model, model_S = student_model,
    adaptor_T = simple_adaptor, adaptor_S = simple_adaptor)

# 开始蒸馏
with distiller:
    distiller.train(optimizer, dataloader, num_epochs=1, scheduler_class=scheduler_class, scheduler_args = scheduler_args, callback=None)