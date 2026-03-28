# 基于卷积神经网络的暴力行为视频识别研究

本项目用于完成“暴力 / 非暴力视频识别”任务。整体流程是：

1. 准备原始视频数据
2. 抽帧并整理成训练集 / 验证集
3. 训练基于 CNN 的分类模型
4. 对单个视频进行预测
5. 在测试集上评估效果
6. 搜索更合适的判定阈值

---

## 1. 项目文件说明

```text
.
├── collect_data.py                # 通过摄像头采集图片样本
├── prepare_dataset_from_videos.py # 从原始视频中抽帧，生成 train/val 数据集
├── dataset.py                     # 读取图像数据集并生成 DataLoader
├── train.py                       # 训练模型
├── eval.py                        # 在测试集上评估模型
├── predict.py                     # 对单个视频进行预测
├── threshold_search.py            # 搜索最佳阈值
├── utils.py                       # 设备、模型保存与加载等工具函数
├── requirements.txt               # 依赖库
└── models/
    ├── cnn_backbone.py           # ResNet18 特征提取骨干网络
    └── behavior_model.py         # 暴力行为二分类模型
```

---

## 2. 环境配置

建议使用 Python 3.8 及以上版本。

```bash
pip install -r requirements.txt
```

依赖包包括：

- `torch`：深度学习框架
- `torchvision`：图像处理与预训练模型
- `opencv-python`：视频读取、抽帧、摄像头采集
- `numpy`：数值运算
- `tqdm`：进度条（当前版本中可选）

---

## 3. 数据集目录约定

这个项目对目录结构有明确要求。训练、验证和测试的目录名字不要改错。

### 3.1 训练集 / 验证集

```text
data/
├── train/
│   ├── normal/
│   └── abnormal/
└── val/
    ├── normal/
    └── abnormal/
```

### 3.2 测试集

```text
test/
├── Violence/
└── NonViolence/
```

### 3.3 类别含义

在当前代码里，标签顺序是：

- `abnormal` = 暴力行为
- `normal` = 非暴力行为

测试集目录中：

- `Violence/` 对应 `abnormal`
- `NonViolence/` 对应 `normal`

这个对应关系很重要，评估和阈值搜索都依赖它。

---

## 4. 从原始视频生成数据集

脚本：`prepare_dataset_from_videos.py`

这个脚本的作用是：

- 读取原始视频
- 按设定间隔抽帧
- 将视频按训练集 / 验证集划分
- 生成项目训练所需的图像文件夹结构

### 4.1 你需要填写的变量

脚本顶部标注了“你只需要改这里”。这些变量的含义如下：

| 变量名 | 作用 | 应该填写什么 |
|---|---|---|
| `RAW_ROOT` | 原始视频数据集根目录 | 原始数据集所在文件夹路径，里面必须有 `Violence` 和 `NonViolence` 两个子文件夹 |
| `OUT_ROOT` | 输出数据目录 | 生成后的数据集保存位置，建议填项目里的 `data` 目录 |
| `VAL_RATIO` | 验证集比例 | 0 到 1 之间的小数，例如 `0.2` 表示 20% 做验证集 |
| `FRAME_EVERY_N` | 抽帧间隔 | 每隔多少帧取一帧，数值越小，生成图片越多 |
| `MAX_FRAMES_PER_VIDEO` | 单个视频最多抽取帧数 | 用于防止一个视频生成过多图片 |
| `IMG_SIZE` | 图片尺寸 | 输出图片宽高，当前模型使用 `224` 最合适 |

### 4.2 原始视频目录格式

原始视频必须整理成下面这样：

```text
RAW_ROOT/
├── Violence/
│   ├── xxx.mp4
│   └── yyy.mp4
└── NonViolence/
    ├── aaa.mp4
    └── bbb.mp4
```

### 4.3 参数建议

- `VAL_RATIO = 0.2`：常用配置，80% 训练，20% 验证
- `FRAME_EVERY_N = 10`：抽帧较密，适合短视频和样本较少的情况
- `FRAME_EVERY_N = 20`：抽帧较稀，适合视频较长或数据量较大时减少冗余
- `MAX_FRAMES_PER_VIDEO = 80`：避免单个长视频生成过多帧图像
- `IMG_SIZE = 224`：与 ResNet18 输入兼容

### 4.4 运行方式

```bash
python prepare_dataset_from_videos.py
```

### 4.5 脚本执行后会生成什么

脚本会自动生成：

```text
data/
├── train/
│   ├── normal/
│   └── abnormal/
└── val/
    ├── normal/
    └── abnormal/
```

### 4.6 关键说明

这个脚本是“按视频划分 train / val”，不是“按图片划分”。
这样做的好处是：同一个视频抽出来的帧不会同时出现在训练集和验证集里，避免数据泄漏。

---

## 5. 摄像头采集样本

脚本：`collect_data.py`

这个脚本用于通过电脑摄像头手动采集图片样本。

### 5.1 内置变量说明

| 变量名 | 作用 | 填写建议 |
|---|---|---|
| `BASE_DIR` | 样本保存根目录 | 默认是 `data/train`，表示采集到训练集目录下 |
| `NORMAL_DIR` | 非暴力样本保存目录 | 自动由 `BASE_DIR` 生成，通常不需要手动改 |
| `ABNORMAL_DIR` | 暴力样本保存目录 | 自动由 `BASE_DIR` 生成，通常不需要手动改 |

### 5.2 按键说明

- `n`：保存当前画面为 `normal` 图片
- `a`：保存当前画面为 `abnormal` 图片
- `q`：退出采集

### 5.3 运行方式

```bash
python collect_data.py
```

### 5.4 适合什么场景

- 手动补充少量训练样本
- 调试摄像头是否正常工作
- 快速采集“正常 / 异常”两个类别的图片

---

## 6. 数据读取方式

脚本：`dataset.py`

这个文件负责把 `data/train` 和 `data/val` 目录中的图片读成 PyTorch 的 `DataLoader`。

### 6.1 核心函数

```python
def get_dataloader(data_dir, batch_size=16, is_train=True)
```

### 6.2 参数说明

| 参数名 | 作用 | 应该填什么 |
|---|---|---|
| `data_dir` | 数据集目录 | 例如 `data/train` 或 `data/val` |
| `batch_size` | 每批训练样本数 | 显存小就填小一点，如 `8`；显存够可用 `16`、`32` |
| `is_train` | 是否训练模式 | 训练集传 `True`，验证集传 `False` |

### 6.3 图像预处理

当前统一做了以下处理：

1. `Resize((224, 224))`：缩放到 224×224
2. `ToTensor()`：转成张量
3. `Normalize(mean, std)`：使用 ImageNet 标准归一化

这说明模型是按 ResNet18 的输入习惯设计的。

---

## 7. 模型结构

### 7.1 `models/cnn_backbone.py`

这里使用的是 **ResNet18** 作为特征提取器。

它做了两件事：

- 去掉 ResNet18 最后分类层
- 输出一个图像特征向量

### 7.2 `models/behavior_model.py`

这个模型结构很简单：

- `backbone`：提取图像特征
- `classifier`：把特征映射到 2 个类别

输出类别顺序为：

- `0` → `abnormal`
- `1` → `normal`

---

## 8. 训练模型

脚本：`train.py`

### 8.1 可修改的超参数

```python
num_epochs = 10
batch_size = 16
learning_rate = 1e-4
num_classes = 2
```

### 8.2 每个变量的作用

| 变量名 | 作用 | 该怎么设置 |
|---|---|---|
| `num_epochs` | 训练轮数 | 训练不够就增大，过拟合就减小 |
| `batch_size` | 每批训练图片数 | 显存越小，值越小 |
| `learning_rate` | 学习率 | 影响收敛速度，常见范围 `1e-3` 到 `1e-5` |
| `num_classes` | 分类数量 | 当前固定为 `2`，不要改错 |

### 8.3 训练数据来源

脚本默认读取：

- `data/train`
- `data/val`

要求目录必须符合第 3 节的结构。

### 8.4 运行方式

```bash
python train.py
```

### 8.5 训练过程中会发生什么

- 每轮输出训练损失和训练准确率
- 每轮输出验证损失和验证准确率
- 当验证集准确率更高时，自动保存模型为 `best_model.pth`

---

## 9. 单个视频预测

脚本：`predict.py`

用于对一个视频做暴力 / 非暴力预测。

### 9.1 命令行参数说明

| 参数名 | 作用 | 应该填什么 |
|---|---|---|
| `--video` | 待预测视频路径 | 直接填写视频文件完整路径 |
| `--weights` | 模型权重文件 | 默认 `best_model.pth` |
| `--frames` | 抽取帧数 | 视频越长，可适当增大；常用 `15` |
| `--show` | 是否显示抽样帧 | 加上这个参数就会弹出预览窗口 |

### 9.2 预测逻辑

1. 从视频中均匀抽取 `frames` 帧
2. 把每一帧缩放到 224×224
3. 输入模型做逐帧分类
4. 对所有帧的 softmax 概率求平均，作为视频级结果

### 9.3 运行方式

```bash
python predict.py --video your_video.mp4 --weights best_model.pth --frames 15
```

如果要查看抽帧效果：

```bash
python predict.py --video your_video.mp4 --weights best_model.pth --frames 15 --show
```

### 9.4 输出结果含义

程序会打印：

- 视频路径
- 抽取帧数
- 预测类别：`abnormal` 或 `normal`
- 置信度
- 两类概率值

---

## 10. 测试集评估

脚本：`eval.py`

这个脚本用于在测试集上统计模型效果，并输出混淆矩阵、分类报告和错误样本列表。

### 10.1 命令行参数说明

| 参数名 | 作用 | 应该填什么 |
|---|---|---|
| `--test_dir` | 测试集根目录 | 默认 `test`，内部应有 `Violence` 和 `NonViolence` |
| `--weights` | 模型权重文件 | 一般用训练得到的 `best_model.pth` |
| `--frames` | 每个视频抽取多少帧 | 默认 `15` |
| `--threshold` | 暴力判定阈值 | 不填时使用模型直接预测；填写后以 `abnormal` 概率与阈值比较 |

### 10.2 `threshold` 的作用

当你填写 `--threshold` 时，评估不再单纯依赖 `argmax`，而是按下面规则判断：

- `abnormal_prob >= threshold` → 判为 `abnormal`
- 否则 → 判为 `normal`

这个参数适合在你希望“减少漏检”或“减少误报”时做调节。

### 10.3 运行方式

```bash
python eval.py --test_dir test --weights best_model.pth --frames 15
```

带阈值版本：

```bash
python eval.py --test_dir test --weights best_model.pth --frames 15 --threshold 0.6
```

### 10.4 输出文件

运行结束后会在 `outputs/` 下生成：

- `confusion_matrix_test.png`：混淆矩阵图
- `classification_report_test.txt`：分类报告
- `wrong_samples_test.csv`：预测错误样本清单

---

## 11. 阈值搜索

脚本：`threshold_search.py`

这个脚本用于在一系列阈值中搜索一个更合适的 `abnormal` 判定阈值。

### 11.1 命令行参数说明

| 参数名 | 作用 | 应该填什么 |
|---|---|---|
| `--test_dir` | 测试集目录 | 默认 `test` |
| `--model_path` | 模型权重路径 | 默认 `best_model.pth` |
| `--frames` | 每个视频抽帧数 | 默认 `15` |
| `--min_t` | 最小阈值 | 一般从 `0.1` 开始 |
| `--max_t` | 最大阈值 | 一般到 `0.95` 左右 |
| `--step` | 扫描步长 | 越小越精细，但越慢 |
| `--recall_min` | 召回率最低要求 | 用于控制“不能漏掉太多暴力样本” |
| `--out_dir` | 输出目录 | 默认 `outputs` |

### 11.2 搜索策略

脚本会遍历一组阈值，并计算：

- Accuracy
- Precision
- Recall
- F1
- FP rate
- FN rate

当前选择策略是：

- 先筛选出 `Recall >= recall_min` 的阈值
- 再从中选 `F1` 最大的那个

如果没有任何阈值满足召回率要求，就退回到全局 F1 最优值。

### 11.3 运行方式

```bash
python threshold_search.py --test_dir test --model_path best_model.pth --frames 15
```

### 11.4 输出文件

会在 `outputs/` 下生成：

- `threshold_curves.png`：各指标随阈值变化的曲线
- `threshold_fp_fn.png`：误报率和漏报率曲线
- `threshold_best.txt`：最佳阈值结果摘要

---

## 12. 常用完整流程

### 12.1 第一步：准备原始视频

确认原始目录是：

```text
RAW_ROOT/
├── Violence/
└── NonViolence/
```

### 12.2 第二步：抽帧生成训练集

修改 `prepare_dataset_from_videos.py` 顶部的：

- `RAW_ROOT`
- `OUT_ROOT`
- `VAL_RATIO`
- `FRAME_EVERY_N`
- `MAX_FRAMES_PER_VIDEO`
- `IMG_SIZE`

然后运行：

```bash
python prepare_dataset_from_videos.py
```

### 12.3 第三步：训练模型

```bash
python train.py
```

训练完成后会得到：

```text
best_model.pth
```

### 12.4 第四步：单视频预测

```bash
python predict.py --video your_video.mp4 --weights best_model.pth --frames 15
```

### 12.5 第五步：测试集评估

```bash
python eval.py --test_dir test --weights best_model.pth --frames 15
```

### 12.6 第六步：阈值搜索

```bash
python threshold_search.py --test_dir test --model_path best_model.pth --frames 15
```

---

## 13. 参数怎么选

### 13.1 `frames` 选多少合适

- `10~15`：速度快，适合大多数普通视频
- `20~30`：更稳定，但推理更慢

### 13.2 `batch_size` 选多少合适

- 显存较小：`8`
- 常规显卡：`16`
- 显存充足：`32`

### 13.3 `FRAME_EVERY_N` 选多少合适

- 数据量少：`10`
- 数据量多：`20`
- 想生成更多图片：调小
- 想减少图片数量：调大

### 13.4 `threshold` 选多少合适

- 阈值越低，越容易判为暴力，召回通常更高，误报也可能更高
- 阈值越高，越保守，误报可能更低，但更容易漏掉暴力

一般建议先用 `threshold_search.py` 搜索一个更合适的值。

---

## 14. 常见问题

### 14.1 提示 `Cannot open camera`

说明摄像头没打开，常见原因：

- 电脑没有摄像头
- 摄像头被别的软件占用
- 系统权限没开

### 14.2 提示 `Cannot open video`

说明视频路径不正确，或者视频文件损坏。

### 14.3 提示 `FileNotFoundError`

通常是以下原因之一：

- 模型权重文件不存在
- 测试集目录没有 `Violence` / `NonViolence`
- 数据集目录结构不符合要求

### 14.4 训练效果不理想

可以优先尝试：

- 增加训练数据
- 调整 `FRAME_EVERY_N`
- 增大 `num_epochs`
- 调整 `learning_rate`
- 增大 `frames`

---

## 15. 已知注意事项

1. `prepare_dataset_from_videos.py` 里的 `RAW_ROOT` 和 `OUT_ROOT` 是需要你手动改成自己电脑路径的。
2. 训练和评估时，目录名称必须和代码一致，尤其是 `normal`、`abnormal`、`Violence`、`NonViolence`。
3. 这个项目是“按帧做 CNN 分类，再对视频级结果做平均”，不是 3D CNN，也不是 LSTM 时序模型。
4. 当前代码中 `threshold_search.py` 和 `predict.py` 的参数调用方式需要保持一致；如果你运行时遇到参数名报错，优先检查 `predict_video(...)` 的参数名是否与当前脚本一致。

---

## 16. 简洁版使用顺序

```bash
# 1. 抽帧生成数据集
python prepare_dataset_from_videos.py

# 2. 训练模型
python train.py

# 3. 单视频预测
python predict.py --video your_video.mp4 --weights best_model.pth

# 4. 测试集评估
python eval.py --test_dir test --weights best_model.pth

# 5. 搜索阈值
python threshold_search.py --test_dir test --model_path best_model.pth
```

---

## 17. 如果你是第一次使用，最少只需要改这几个地方

### `prepare_dataset_from_videos.py`
- `RAW_ROOT`
- `OUT_ROOT`
- `VAL_RATIO`
- `FRAME_EVERY_N`
- `MAX_FRAMES_PER_VIDEO`
- `IMG_SIZE`

### `train.py`
- `num_epochs`
- `batch_size`
- `learning_rate`

### `predict.py` / `eval.py` / `threshold_search.py`
- `--video`
- `--weights`
- `--frames`
- `--test_dir`
- `--threshold`

### 作者 苍崎青子
## 联系邮箱：3215592140@qq.com
##  本项目只用于学术研究和项目学习，仅供参考
