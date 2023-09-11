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

### Specifying a Target Module

You can specify a target operator to inspect, if you do not need to inspect the whole model:

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
target_module (VGG)             [[1, 3, 224, 224]]        [[1, 1000]]               132.86M       43.66851
├─ features (Sequential)        [[1, 3, 224, 224]]        [[1, 512, 7, 7]]          9.22M         32.90186
│ ├─ features.0 (Conv2d)        [[1, 3, 224, 224]]        [[1, 64, 224, 224]]       1,792         1.93931
│ ├─ features.1 (ReLU)          [[1, 64, 224, 224]]       [[1, 64, 224, 224]]       0             0.13679
│ ├─ features.2 (MaxPool2d)     [[1, 64, 224, 224]]       [[1, 64, 112, 112]]       0             1.49747
│ ├─ features.3 (Conv2d)        [[1, 64, 112, 112]]       [[1, 128, 112, 112]]      73,856        4.56279
│ ├─ features.4 (ReLU)          [[1, 128, 112, 112]]      [[1, 128, 112, 112]]      0             0.08741
│ ├─ features.5 (MaxPool2d)     [[1, 128, 112, 112]]      [[1, 128, 56, 56]]        0             0.83432
│ ├─ features.6 (Conv2d)        [[1, 128, 56, 56]]        [[1, 256, 56, 56]]        295,168       2.99812
│ ├─ features.7 (ReLU)          [[1, 256, 56, 56]]        [[1, 256, 56, 56]]        0             0.06622
│ ├─ features.8 (Conv2d)        [[1, 256, 56, 56]]        [[1, 256, 56, 56]]        590,080       5.51739
│ ├─ features.9 (ReLU)          [[1, 256, 56, 56]]        [[1, 256, 56, 56]]        0             0.06624
│ ├─ features.10 (MaxPool2d)    [[1, 256, 56, 56]]        [[1, 256, 28, 28]]        0             0.52735
│ ├─ features.11 (Conv2d)       [[1, 256, 28, 28]]        [[1, 512, 28, 28]]        1.18M         2.33528
│ ├─ features.12 (ReLU)         [[1, 512, 28, 28]]        [[1, 512, 28, 28]]        0             0.05863
│ ├─ features.13 (Conv2d)       [[1, 512, 28, 28]]        [[1, 512, 28, 28]]        2.36M         5.01996
│ ├─ features.14 (ReLU)         [[1, 512, 28, 28]]        [[1, 512, 28, 28]]        0             0.04898
│ ├─ features.15 (MaxPool2d)    [[1, 512, 28, 28]]        [[1, 512, 14, 14]]        0             0.27941
│ ├─ features.16 (Conv2d)       [[1, 512, 14, 14]]        [[1, 512, 14, 14]]        2.36M         1.62717
│ ├─ features.17 (ReLU)         [[1, 512, 14, 14]]        [[1, 512, 14, 14]]        0             0.01928
│ ├─ features.18 (Conv2d)       [[1, 512, 14, 14]]        [[1, 512, 14, 14]]        2.36M         1.37824
│ ├─ features.19 (ReLU)         [[1, 512, 14, 14]]        [[1, 512, 14, 14]]        0             0.01863
│ ├─ features.20 (MaxPool2d)    [[1, 512, 14, 14]]        [[1, 512, 7, 7]]          0             0.15563
├─ avgpool (AdaptiveAvgPool2d)  [[1, 512, 7, 7]]          [[1, 512, 7, 7]]          0             0.05168
├─ classifier (Sequential)      [[1, 25088]]              [[1, 1000]]               123.64M       10.25144
│ ├─ classifier.0 (Linear)      [[1, 25088]]              [[1, 4096]]               102.76M       8.51207
│ ├─ classifier.1 (ReLU)        [[1, 4096]]               [[1, 4096]]               0             0.00121
│ ├─ classifier.2 (Dropout)     [[1, 4096]]               [[1, 4096]]               0             0.00159
│ ├─ classifier.3 (Linear)      [[1, 4096]]               [[1, 4096]]               16.78M        1.17055
│ ├─ classifier.4 (ReLU)        [[1, 4096]]               [[1, 4096]]               0             0.00118
│ ├─ classifier.5 (Dropout)     [[1, 4096]]               [[1, 4096]]               0             0.00162
│ ├─ classifier.6 (Linear)      [[1, 4096]]               [[1, 1000]]               4.10M         0.18358
```

Remark: for now, Other is empty. More info will be added in the future.

## Contributing

We welcome contributions to the AutoOpInspect project. Whether it's reporting issues, improving documentation, or contributing code, your input is valuable.