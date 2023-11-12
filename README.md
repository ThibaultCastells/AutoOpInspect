<div align="center">

# AutoOpInspect
</div>

AutoOpInspect is a Python package designed to streamline the inspection and profiling of operators within PyTorch models. It is a helpful tool for developers, researchers, and enthusiasts working in the machine learning field. Whether you are debugging, trying to improve the performance of your PyTorch models, or trying to collect operators info for a project, AutoOpInspect can help you!

<div align="center">
  <img src="./assets/AutoOpInspect_logo.png" width="500"/>
</div>


## Core Features

Here's what AutoOpInspect can offer:

- Operators Data Collection:
  - Automatically collects data of your model's operators such as input and output dimensions, number of parameters, class type, and more.
  - Gain comprehensive insights into the individual operators, aiding in meticulous analysis and optimization of your PyTorch models.
- Operators inference speed evaluation:
  - Automatically and individually measures the inference speed of each operator in your PyTorch models, providing you with detailed performance metrics.
  - This invaluable data assists developers in identifying bottlenecks and optimizing the performance of models for faster inference times on the device of your choice, ensuring peak operational efficiency.
- Automated Dummy Input/Output Data Generation:
  - Effortlessly generate dummy input or output data for any operator within your model.
  - Allows for rapid testing and verification of individual operators, speeding up the debugging process and ensuring the stability of your models.
- Model visualization
  - Offers a clear overview of all the operators in your model along with their detailed information, by printing the model structure in a readable format.
  - Facilitates a deeper understanding of the model's architecture, helping in fine-tuning and making informed adjustments during the development stage.
  - New in v1.0: barplot the inference speed or number of modules in your terminal, for an even better visualization experience!

## Installation

You can install AutoOpInspect using pip, with the following command:

``` bash
pip install auto_op_inspect
```

## Usage

Below are some examples demonstrating how to use the AutoOpInspect package:

### Basic Usage

Create an OpsInfoProvider instance using a PyTorch model and input data:

``` python
from AutoOpInspect import OpsInfoProvider
import torchvision
import torch

model = torchvision.models.vgg11()
input_data = [torch.randn(1, 3, 224, 224)] # make a list of inputs (supports multiple inputs)
ops_info_provider = OpsInfoProvider(model, input_data)
```

You can also specify a target operator to inspect, if you do not need to inspect the whole model:

``` python
target_module = model.features
ops_info_provider = OpsInfoProvider(model, input_data, target=target_module)
```

### Measuring inference speed

You can measure the inference speed of a single module, by specifying an operator.
If `operator` is None (default), then the benchmark will be done through all operators.

``` python
operator = model.features[6]
ops_info_provider.benchmark_speed(operator=operator, device='cpu', iterations=100)
print(ops_info_provider[operator].speed)
```

Before you begin measuring the inference speed, ensure that no other applications are running to limit the potential interference with the benchmark results. This precautionary measure helps in acquiring a more accurate measurement of the inference speed.
You might notice slight variations in the inference speed across different runs; this is normal and can be attributed to a variety of factors including system load, CPU/GPU thermal throttling, etc. If you need more reliable results, it is recommended to run the benchmark several times and consider the average value.
Note that multiple-gpu is not supported.

### Getting Dummy Data

Retrieve dummy input and output data for any operator:

``` python
operator = model.features[6]
dummy_input, dummy_output = ops_info_provider.get_dummy(operator, mode='both')
# available modes are input, output and both
```

### Visualize the model

``` python
print(ops_info_provider)
```

result:

```
Layer (type)                    Input Shape               Output Shape              Param #       Inference (ms)      Other
================================================================================================================================
target_module (VGG)            [[1, 3, 224, 224]]        [[1, 1000]]               132.86M       45.65834                 
├─ features (Sequential)       [[1, 3, 224, 224]]        [[1, 512, 7, 7]]          9.22M         33.99795                 
│ ├─ features.0 (Conv2d)       [[1, 3, 224, 224]]        [[1, 64, 224, 224]]       1,792         1.69590         (3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
│ ├─ features.1 (ReLU)         [[1, 64, 224, 224]]       [[1, 64, 224, 224]]       0             0.12630         (inplace=True)
│ ├─ features.2 (MaxPool2d)    [[1, 64, 224, 224]]       [[1, 64, 112, 112]]       0             1.50131         (kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
│ ├─ features.3 (Conv2d)       [[1, 64, 112, 112]]       [[1, 128, 112, 112]]      73,856        4.50531         (64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
│ ├─ features.4 (ReLU)         [[1, 128, 112, 112]]      [[1, 128, 112, 112]]      0             0.08341         (inplace=True)
│ ├─ features.5 (MaxPool2d)    [[1, 128, 112, 112]]      [[1, 128, 56, 56]]        0             0.77252         (kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
│ ├─ features.6 (Conv2d)       [[1, 128, 56, 56]]        [[1, 256, 56, 56]]        295,168       2.98192         (128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
│ ├─ features.7 (ReLU)         [[1, 256, 56, 56]]        [[1, 256, 56, 56]]        0             0.05805         (inplace=True)
│ ├─ features.8 (Conv2d)       [[1, 256, 56, 56]]        [[1, 256, 56, 56]]        590,080       5.55548         (256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
│ ├─ features.9 (ReLU)         [[1, 256, 56, 56]]        [[1, 256, 56, 56]]        0             0.05683         (inplace=True)
│ ├─ features.10 (MaxPool2d)   [[1, 256, 56, 56]]        [[1, 256, 28, 28]]        0             0.42715         (kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
│ ├─ features.11 (Conv2d)      [[1, 256, 28, 28]]        [[1, 512, 28, 28]]        1.18M         2.52301         (256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
│ ├─ features.12 (ReLU)        [[1, 512, 28, 28]]        [[1, 512, 28, 28]]        0             0.04940         (inplace=True)
│ ├─ features.13 (Conv2d)      [[1, 512, 28, 28]]        [[1, 512, 28, 28]]        2.36M         5.03365         (512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
│ ├─ features.14 (ReLU)        [[1, 512, 28, 28]]        [[1, 512, 28, 28]]        0             0.05053         (inplace=True)
│ ├─ features.15 (MaxPool2d)   [[1, 512, 28, 28]]        [[1, 512, 14, 14]]        0             0.26663         (kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
│ ├─ features.16 (Conv2d)      [[1, 512, 14, 14]]        [[1, 512, 14, 14]]        2.36M         1.48468         (512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
│ ├─ features.17 (ReLU)        [[1, 512, 14, 14]]        [[1, 512, 14, 14]]        0             0.02102         (inplace=True)
│ ├─ features.18 (Conv2d)      [[1, 512, 14, 14]]        [[1, 512, 14, 14]]        2.36M         1.54018         (512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
│ ├─ features.19 (ReLU)        [[1, 512, 14, 14]]        [[1, 512, 14, 14]]        0             0.01807         (inplace=True)
│ ├─ features.20 (MaxPool2d)   [[1, 512, 14, 14]]        [[1, 512, 7, 7]]          0             0.11825         (kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
├─ avgpool (AdaptiveAvgPool2d) [[1, 512, 7, 7]]          [[1, 512, 7, 7]]          0             0.05351         (output_size=(7, 7))
├─ classifier (Sequential)     [[1, 25088]]              [[1, 1000]]               123.64M       10.39421                 
│ ├─ classifier.0 (Linear)     [[1, 25088]]              [[1, 4096]]               102.76M       8.57219         (in_features=25088, out_features=4096, bias=True)
│ ├─ classifier.1 (ReLU)       [[1, 4096]]               [[1, 4096]]               0             0.00115         (inplace=True)
│ ├─ classifier.2 (Dropout)    [[1, 4096]]               [[1, 4096]]               0             0.00173         (p=0.5, inplace=False)
│ ├─ classifier.3 (Linear)     [[1, 4096]]               [[1, 4096]]               16.78M        1.26286         (in_features=4096, out_features=4096, bias=True)
│ ├─ classifier.4 (ReLU)       [[1, 4096]]               [[1, 4096]]               0             0.00115         (inplace=True)
│ ├─ classifier.5 (Dropout)    [[1, 4096]]               [[1, 4096]]               0             0.00162         (p=0.5, inplace=False)
│ ├─ classifier.6 (Linear)     [[1, 4096]]               [[1, 1000]]               4.10M         0.14087         (in_features=4096, out_features=1000, bias=True)
```

From v1.0, you can also visualize the model with a barplot, direcly in you terminal. This is a great way to visualize large models.
Here is an example with the Unet of [Stable Diffusion v1-5](https://huggingface.co/runwayml/stable-diffusion-v1-5), from Diffusers 0.20.2 implementation, on cpu:

``` python
ops_info_provider.barplot_speed(mode = 'sum')
```

result:
``` 
┌──────────────────────────────────────────────────────────────── Operator Speed in ms (sum) ────────────────────────────────────────────────────────────────┐
│   LoRACompatibleConv : █████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████  1385.092 │
│ LoRACompatibleLinear : █████████████████████████████████                                                                                           386.313 │
│               Linear : ██████████████                                                                                                              169.798 │
│                 SiLU : ███                                                                                                                          39.761 │
│            GroupNorm : ██                                                                                                                           23.157 │
│               Conv2d : █                                                                                                                            16.170 │
│            LayerNorm : █                                                                                                                            13.915 │
│              Dropout :                                                                                                                               0.121 │
│            Timesteps :                                                                                                                               0.023 │
└────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
``` 

``` python
ops_info_provider.barplot_speed(mode = 'mean')
```

result:
``` 
┌─────────────────────────────────────────────────────────────── Operator Speed in ms (mean) ────────────────────────────────────────────────────────────────┐
│   LoRACompatibleConv : █████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████    14.428 │
│               Conv2d : ███████████████████████████████████████████████████████████████████                                                           8.085 │
│ LoRACompatibleLinear : ███████████████████████████████████████████████████████████                                                                   7.154 │
│                 SiLU : █████████████                                                                                                                 1.657 │
│               Linear : ██████████                                                                                                                    1.306 │
│            GroupNorm : ███                                                                                                                           0.380 │
│            LayerNorm : ██                                                                                                                            0.290 │
│            Timesteps :                                                                                                                               0.023 │
│              Dropout :                                                                                                                               0.002 │
└────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
``` 

``` python
ops_info_provider.barplot_quantity()
```

result:
``` 
┌──────────────────────────────────────────────────────────────────── Operator Quantity ─────────────────────────────────────────────────────────────────────┐
│               Linear : █████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████       130 │
│   LoRACompatibleConv : █████████████████████████████████████████████████████████████████████████████████████████                                        96 │
│              Dropout : █████████████████████████████████████████████████████████████████                                                                70 │
│            GroupNorm : ████████████████████████████████████████████████████████                                                                         61 │
│ LoRACompatibleLinear : ██████████████████████████████████████████████████                                                                               54 │
│            LayerNorm : ████████████████████████████████████████████                                                                                     48 │
│                 SiLU : ██████████████████████                                                                                                           24 │
│               Conv2d : █                                                                                                                                 2 │
│            Timesteps :                                                                                                                                   1 │
└────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
``` 

## Contributing

We welcome contributions to the AutoOpInspect project. Whether it's reporting issues, improving documentation, or contributing code, your input is valuable.