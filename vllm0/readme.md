# V0：最小生成

V0 每生成一个 token 都对完整历史做前向。用它观察输入长度和输出长度增加时，重复计算如何影响延迟。

[版本目录](../README.md) · [Benchmark](bench/README.md)

## 阅读源码

`src/engine.py` → `src/models/model.py` → `src/models/loading/` → `src/models/sampler.py`

本版代码、CLI 和模型加载均在本目录中。`src/` 实现推理，`run.py` 演示生成，`bench/` 测量性能。

## 启动

在仓库根目录激活环境，再进入本版目录：

```bash
source .venv/bin/activate
export MODEL="$HOME/huggingface/Qwen/Qwen3-0.6B"
cd vllm-v0
python run.py --model "$MODEL" --prompt "Hello, my name is"
```

模型参数默认读取 `config.yaml`；`python run.py --help` 列出命令行参数。

## 执行实验

```bash
python bench/run.py e2e --model "$MODEL" --prompt-lengths 32 128 512 --output-tokens 32
```

实验场景、指标定义和结果文件见 [bench/README.md](bench/README.md)。

## 实现说明

- [01-overview](docs/01-overview.md)
- [02-model](docs/02-model.md)
- [03-weights](docs/03-weights.md)
- [04-sampler](docs/04-sampler.md)
- [05-engine](docs/05-engine.md)
