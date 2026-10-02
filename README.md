# 基于卷积神经网络的中国交通标志识别系统

一个面向毕业设计的端到端中国交通标志分类系统。项目基于 TSRD（Chinese Traffic Sign Recognition Database）训练轻量卷积神经网络，包含数据下载与校验、标注框裁剪、类别均衡训练、独立测试集评估、FastAPI 推理接口、响应式 Web 页面、SQLite 历史记录和 58 类验证样例。

> 当前 `tsrd-v1` 模型在 TSRD 官方独立测试集 1,994 张图片上取得 **93.88% Accuracy** 和 **0.9352 Macro-F1**。测试集中实际出现 54 类，Macro-F1 按这 54 个有支持样本的类别计算。

![系统运行效果](docs/system-preview.png)

## 功能与亮点

- 面向中国道路标志，覆盖 TSRD 全部 58 类限速、禁令、指示和警告标志
- 采用 MobileNetV3-Small 轻量 CNN，通过 ImageNet 预训练进行迁移学习
- 根据官方边界框裁剪训练图像，减少建筑、车辆和其他标牌等背景干扰
- 使用分层训练/验证划分与类别均衡采样，缓解 2～446 张/类的长尾分布
- 使用旋转、仿射、透视、光照、色彩和轻度模糊增强，不进行会改变方向语义的水平翻转
- 支持图片拖拽上传、Top-3 预测、类别分组、置信度、候选间隔和推理耗时展示
- 结合首选置信度与 Top-2 间隔进行不确定性拒识，低可靠结果会明确提示重新上传
- SQLite 保存识别历史，提供查询和清空接口
- 提供 API 测试、真实浏览器端到端测试及 58 类可直接上传的验证样例

## 识别类别

| 类别组 | ID 范围 | 代表类别 |
|---|---:|---|
| 限速标志 | 0–7 | 限速 5、15、30、40、50、60、70、80 km/h |
| 禁令标志 | 8–19、52–57 | 禁止转弯、禁止超车、停车让行、禁止停车、禁止驶入、减速让行 |
| 指示标志 | 20–32 | 直行、左右转、靠左/右行驶、环岛、机动车/非机动车行驶 |
| 警告标志 | 33–51 | 信号灯、行人、儿童、急转弯、陡坡、铁路道口、施工、连续弯路 |

完整类别映射见 [`app/labels.py`](app/labels.py)，可视化样例见 [`sample_images/contact-sheet.png`](sample_images/contact-sheet.png)。

## 模型指标

| 指标 | 结果 |
|---|---:|
| 官方独立测试集 | 1,994 张 |
| 测试集实际覆盖类别 | 54 / 58 |
| Accuracy | 93.88% |
| Macro-F1 | 0.9352 |
| 最佳训练轮次 | 17 |
| 输入尺寸 | 96 × 96 |
| 参数量 | 1,577,306 |
| 权重大小 | 约 6.1 MB |

完整训练历史、逐类别召回率、样本支持数和混淆矩阵见 [`reports/tsrd_training_report.json`](reports/tsrd_training_report.json)。类别 9、18、19、33 在官方测试集中没有样本，因此不能从该测试集计算这四类的独立召回率；项目仍训练并支持输出这些类别。

## 系统架构

```text
浏览器上传单个中国交通标志
              │
              ▼
FastAPI 文件校验 ──► Pillow 解码、缩放与 ImageNet 归一化
              │
              ▼
MobileNetV3-Small CNN ──► 58 类 Softmax 概率
              │
              ├──► Top-3、置信度、候选间隔
              ├──► 可靠识别 / 建议重试
              ├──► SQLite 识别历史
              ▼
          响应式网页展示
```

## 技术栈

| 模块 | 技术 |
|---|---|
| 数据集 | TSRD，58 类中国交通标志 |
| 深度学习 | PyTorch、Torchvision、MobileNetV3-Small |
| 图像处理 | Pillow、NumPy |
| 后端 | FastAPI、Uvicorn |
| 前端 | HTML、CSS、原生 JavaScript |
| 数据库 | SQLite |
| 测试 | Pytest、HTTPX、Playwright |

## 快速启动

### 1. 克隆项目

```bash
git clone https://github.com/Lyh2422/TrafficSignRecognition_CNN.git
cd TrafficSignRecognition_CNN
```

### 2. 创建虚拟环境

推荐 Python 3.11 或更高版本。

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-ml.txt
```

Windows PowerShell：

```powershell
.venv\Scripts\Activate.ps1
pip install -r requirements-ml.txt
```

### 3. 启动系统

仓库已经包含中国模型权重 `models/tsrd_cnn.pt`，不需要重新训练即可识别。

```bash
uvicorn app.main:app --reload
```

- 系统页面：<http://127.0.0.1:8000>
- Swagger 接口文档：<http://127.0.0.1:8000/docs>
- 健康检查：<http://127.0.0.1:8000/api/health>

页面右上角显示 `TSRD · 58` 即表示中国 CNN 权重加载成功。如果显示 `DEMO MODE`，请检查 PyTorch 是否安装以及权重文件是否存在。

## 如何测试识别

`sample_images/` 包含 58 张当前模型已验证识别正确的中国交通标志图片。推荐先测试：

- `04-speed-limit-50-km-h.png`：限速 50 km/h
- `26-keep-right.png`：靠右侧行驶
- `35-pedestrians.png`：注意行人
- `48-road-works.png`：施工
- `52-stop.png`：停车让行
- `55-no-entry.png`：禁止驶入
- `56-give-way.png`：减速让行

完整预期结果和置信度见 [`sample_images/manifest.json`](sample_images/manifest.json)。

用户图片建议满足：

- 图片中只有一个中国交通标志；
- 标志已经裁剪并位于画面中央；
- 标志主体约占画面 60%～90%；
- 尽量正面、清晰，避免严重反光、遮挡、倾斜和运动模糊；
- 类别属于系统定义的 58 类。

当前系统是**交通标志分类器**，不是街景目标检测器。完整道路照片中的标志通常很小，应先裁剪标志区域再上传。

## 数据准备与重新训练

### 1. 准备 TSRD 数据

原始数据不提交到仓库。执行以下命令会下载约 199 MB 的压缩包，校验 SHA-256、进行安全解压，并验证官方训练/测试样本数量：

```bash
python -m scripts.prepare_tsrd
```

生成目录：

```text
data/tsrd/full_classes/
├── train/                         # 4,170 张
├── test/                          # 1,994 张
├── TsignRecgTrainAnnotation.txt
└── TsignRecgTestAnnotation.txt
```

如果所在网络无法访问默认地址，可通过 `--url` 指定同一数据压缩包的镜像地址。

### 2. 训练模型

```bash
python -m scripts.train --epochs 30 --batch-size 96 --device auto
```

训练流程包括：

1. 读取文件名前三位类别编号并校验 58 类标签；
2. 根据官方标注框裁剪交通标志，并保留少量上下文边缘；
3. 对训练集进行分层训练/验证划分，确保稀有类别进入验证集；
4. 使用 `WeightedRandomSampler` 平衡不同类别的训练频率；
5. 应用不会改变标志语义的数据增强；
6. 使用 AdamW、OneCycleLR 和 Label Smoothing 微调 MobileNetV3-Small；
7. 按验证集 Macro-F1 保存最佳权重并执行早停；
8. 最后在官方独立测试集上计算 Accuracy、Macro-F1、逐类别召回率和混淆矩阵。

生成文件：

```text
models/tsrd_cnn.pt
reports/tsrd_training_report.json
reports/tsrd_training_history.csv
```

设备参数支持 `auto`、`mps`、`cuda` 和 `cpu`。项目还保留了自研基础 CNN，可用于毕业论文中的对比实验：

```bash
python -m scripts.train --architecture custom_cnn --no-pretrained
```

### 3. 独立评估

```bash
python -m scripts.evaluate --device auto
```

## API

| 方法 | 路径 | 说明 |
|---|---|---|
| `GET` | `/api/health` | 查看服务状态、数据集、模型元数据和类别数量 |
| `POST` | `/api/predict` | 上传图片，返回 Top-3、可靠性判定和推理信息 |
| `GET` | `/api/history?limit=20` | 查询最近识别记录 |
| `DELETE` | `/api/history` | 清空识别历史 |

上传示例：

```bash
curl -X POST http://127.0.0.1:8000/api/predict \
  -F "file=@sample_images/55-no-entry.png"
```

核心响应字段：

```json
{
  "prediction": {
    "id": 55,
    "name_zh": "禁止驶入",
    "name_en": "No entry",
    "category": "禁令",
    "confidence": 0.949306
  },
  "recognized": true,
  "decision": "recognized",
  "confidence_margin": 0.91,
  "mode": "cnn"
}
```

系统仅在首选置信度不低于 0.45、且首选与第二候选差值不低于 0.20 时标记为“可靠识别”。在官方测试集上，组合阈值覆盖约 88.5% 的样本，被接受结果的准确率约为 98.07%。拒识只是风险提示，不代表绝对正确或错误。

## 自动测试

```bash
pip install -r requirements-dev.txt
pytest -q
```

测试范围包括：

- 58 类标签完整性和连续性；
- TSRD 文件名标签解析与官方标注框裁剪；
- 图片上传、格式校验、Top-3 和历史记录；
- 中国“禁止驶入”真实样例的模型集成测试；
- 桌面端上传流程和移动端响应式布局。

浏览器端到端测试：

```bash
playwright install chromium
TSR_DB_PATH=.artifacts/browser-smoke.db uvicorn app.main:app --port 8000
```

另一个终端执行：

```bash
python tests/browser_smoke.py
```

## 目录结构

```text
├── app/
│   ├── main.py                 # FastAPI 路由与上传校验
│   ├── model.py                # MobileNetV3 与基础 CNN
│   ├── predictor.py            # 预处理、Top-3 与不确定性拒识
│   ├── database.py             # SQLite 识别历史
│   └── labels.py               # TSRD 58 类中英文标签
├── scripts/
│   ├── prepare_tsrd.py         # 下载、校验和解压数据集
│   ├── tsrd_dataset.py         # 数据读取与标注框裁剪
│   ├── train.py                # 均衡训练、验证和官方测试
│   ├── evaluate.py             # 独立模型评估
│   └── export_test_samples.py  # 导出 58 类验证样例
├── static/                     # Web 页面、样式和交互
├── sample_images/              # 58 类中国标志验证图片
├── models/tsrd_cnn.pt          # 最佳中国模型权重
├── reports/                    # 训练历史与完整评估结果
└── tests/                      # API、数据集和浏览器测试
```

## 毕业设计可展开内容

- 对比基础自研 CNN 与 MobileNetV3 迁移学习的准确率、参数量和推理耗时；
- 分析类别不平衡对少样本交通标志的影响；
- 对比使用/不使用官方标注框裁剪的泛化结果；
- 展示数据增强、类别均衡采样和不确定性拒识的消融实验；
- 通过混淆矩阵分析方向相近、图案相似类别的错误原因；
- 后续增加 YOLO 检测模块，形成“街景检测 + CNN 分类”的两阶段系统。

## 适用边界

- 当前模型支持定义内的 58 类中国交通标志，不覆盖全部现行国标标志；
- 输入应是裁剪后的单个标志，暂不支持直接定位完整街景中的多个标志；
- 极端模糊、遮挡、夜间强反光和训练集外类别仍可能误判；
- 识别结果适合教学与实验，不应直接用于真实车辆的安全决策。

## 数据集与参考

- [Chinese Traffic Sign Recognition Database 官方页面](https://nlpr.ia.ac.cn/pal/trafficdata/recognition.html)
- [TSRD 数据镜像与数据卡](https://huggingface.co/datasets/EvanDu/Traffic_Sign_Recogntion_Database)
- [MobileNetV3 论文](https://arxiv.org/abs/1905.02244)
- [Torchvision MobileNetV3 文档](https://docs.pytorch.org/vision/stable/models/mobilenetv3.html)
