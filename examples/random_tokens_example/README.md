# Simple Example

此可运行样本演示了TextBrewer的用法。

teacher是基于BERT的。student是3层BERT。

任务是文本分类。我们生成一些随机tokenID和标签作为输入。

因此，这个简单的样本仅用于教学目的。

我们还将使用toolkit提供的实用工具列出模型参数的摘要。

## Requirements

PyTorch >= 1.0

transformers >= 2.0

tensorboard

## Run
```python
python distill.py
```

## Screenshots

![screenshot1](screenshots/screenshot1.png)

![screenshot2](screenshots/screenshot2.png)