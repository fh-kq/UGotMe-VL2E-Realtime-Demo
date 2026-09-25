# UGotMe / VL2E 实时情绪识别 Demo

这是 [UGotMe / VisionLanguageEmotion 作者仓库](https://github.com/lipzh5/VisionLanguageEmotion)的 fork，增加了在 Windows 上运行的摄像头演示，以及我们训练的 VL2E 主模型权重。论文为 [UGotMe: An Embodied System for Affective Human-Robot Interaction](https://arxiv.org/abs/2410.18373)。作者原始使用说明保留在 [UPSTREAM_README.md](UPSTREAM_README.md)。

**权重来源**：Release 中的 `best.pth` 是我们在 MELD 上重新训练得到的第 14 轮最佳验证模型，**不是论文作者发布的权重**。本地 2610 条 MELD 测试样本上的 weighted F1 为 66.62%；论文报告 67.29%。这是单次训练、使用 MTCNN 人脸预处理的近似复现，不能把测试集分数当作摄像头演示的准确率。

## 运行摄像头 Demo（Windows）

1. 安装 Python 3.11。在本仓库目录打开 PowerShell。
2. 建立环境并安装依赖：

   ```powershell
   py -3.11 -m venv .venv
   .\.venv\Scripts\python.exe -m pip install torch torchvision
   .\.venv\Scripts\python.exe -m pip install -r requirements-demo.txt
   .\.venv\Scripts\python.exe -m pip install --no-deps facenet-pytorch==2.6.0
   ```

   有 NVIDIA GPU 的电脑可以按 [PyTorch 官方安装说明](https://pytorch.org/get-started/locally/)选择匹配的版本。本机验证使用 Python 3.11.16、PyTorch 2.11.0+cu128、torchvision 0.26.0+cu128。CPU 也可作为运行设备，但速度可能较慢。

3. 首次联网预存两个基础模型。Demo 启动时按离线模式读取 Hugging Face 缓存，因此要先运行：

   ```powershell
   .\.venv\Scripts\python.exe -c "from huggingface_hub import snapshot_download; snapshot_download('princeton-nlp/sup-simcse-roberta-large')"
   .\.venv\Scripts\python.exe -c "from facenet_pytorch import InceptionResnetV1; InceptionResnetV1(pretrained='casia-webface')"
   ```

4. 从本仓库 **Releases** 下载 `best.pth`（约 1.66 GB），原样放在 `results\trained_full\best.pth`；这是模型文件，**不用解压**。可对照 [SHA256SUMS.txt](SHA256SUMS.txt) 验证下载是否完整。不要把权重作为普通 Git 文件提交。
5. 启动并打开浏览器：

   ```powershell
   .\.venv\Scripts\python.exe vl2e_live_demo.py
   ```

   浏览器访问 <http://127.0.0.1:8766>，允许摄像头访问。如果画面不对，请在页面中改选实际摄像头。手动输入当前说话内容，可选填对话前文。停止服务请在终端按 `Ctrl+C`。

要分析上传的视频而非摄像头，请运行 `vl2e_demo.py`，浏览器访问 <http://127.0.0.1:8765>。

## 能做什么、不能做什么

- 模型输入为手动提供的对话文字和人脸视频，输出七类情绪的 softmax 分数；这些分数未经概率校准。
- 摄像头 Demo 约每 3 秒处理最近几秒画面；它是循环推理演示，不是逐帧情绪识别。
- 当前用 MTCNN 检测人脸，遇到多人脸时选择面积最大者；没有说话人定位、自动语音转写或麦克风输入。
- 摄像头画面发给本机运行的服务，不上传到远程服务器。无需 MELD 数据集即可运行 Demo。
- 本仓库保留作者模型实现及必要的兼容性修改。作者仓库目前未提供明确的 LICENSE 文件；在 GitHub fork 之外再分发或商用原作者代码时，请先确认使用权限。

如需复现训练过程，请参看[原作者说明](UPSTREAM_README.md)。本仓库 Release 只分发我们训练的模型，不包含 MELD 原始视频，也不包含作者发布的 checkpoint。
