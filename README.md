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
- Automated Dummy Input/Output Data Generation:
  - Effortlessly generate dummy input or output data for any operator within your model.
  - Allows for rapid testing and verification of individual operators, speeding up the debugging process and ensuring the stability of your models.
- Model visualization
  - Offers a clear overview of all the operators in your model along with their detailed information, by printing the model structure in a readable format.
  - Facilitates a deeper understanding of the model's architecture, helping in fine-tuning and making informed adjustments during the development stage.

## Installation

You can install AutoOpInspect using pip. Use the following command to install AutoOpInspect:

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

### Getting Dummy Data

Retrieve dummy input and output data for any operator:

``` python
operator = model.features.6
dummy_input, dummy_output = ops_info_provider.get_dummy(operator, mode='both')
# available modes are input, output and both
```

### Visualize the model

``` python
print(ops_info_provider)
```

result:

```
Layer (type)                  Input Shape               Output Shape              Param #              Module Details
=================================================================================================================================
target_module (VGG)           [[1, 3, 224, 224]]        [[1000]]                  132863336
features (Sequential)         [[1, 3, 224, 224]]        [[512, 7, 7]]             9220480
├─ features.0 (Conv2d)        [[1, 3, 224, 224]]        [[64, 224, 224]]          1792
├─ features.1 (ReLU)          [[1, 64, 224, 224]]       [[64, 224, 224]]          0
├─ features.2 (MaxPool2d)     [[1, 64, 224, 224]]       [[64, 112, 112]]          0
├─ features.3 (Conv2d)        [[1, 64, 112, 112]]       [[128, 112, 112]]         73856
├─ features.4 (ReLU)          [[1, 128, 112, 112]]      [[128, 112, 112]]         0
├─ features.5 (MaxPool2d)     [[1, 128, 112, 112]]      [[128, 56, 56]]           0
├─ features.6 (Conv2d)        [[1, 128, 56, 56]]        [[256, 56, 56]]           295168
├─ features.7 (ReLU)          [[1, 256, 56, 56]]        [[256, 56, 56]]           0
├─ features.8 (Conv2d)        [[1, 256, 56, 56]]        [[256, 56, 56]]           590080
├─ features.9 (ReLU)          [[1, 256, 56, 56]]        [[256, 56, 56]]           0
├─ features.10 (MaxPool2d)    [[1, 256, 56, 56]]        [[256, 28, 28]]           0
├─ features.11 (Conv2d)       [[1, 256, 28, 28]]        [[512, 28, 28]]           1180160
├─ features.12 (ReLU)         [[1, 512, 28, 28]]        [[512, 28, 28]]           0
├─ features.13 (Conv2d)       [[1, 512, 28, 28]]        [[512, 28, 28]]           2359808
├─ features.14 (ReLU)         [[1, 512, 28, 28]]        [[512, 28, 28]]           0
├─ features.15 (MaxPool2d)    [[1, 512, 28, 28]]        [[512, 14, 14]]           0
├─ features.16 (Conv2d)       [[1, 512, 14, 14]]        [[512, 14, 14]]           2359808
├─ features.17 (ReLU)         [[1, 512, 14, 14]]        [[512, 14, 14]]           0
├─ features.18 (Conv2d)       [[1, 512, 14, 14]]        [[512, 14, 14]]           2359808
├─ features.19 (ReLU)         [[1, 512, 14, 14]]        [[512, 14, 14]]           0
├─ features.20 (MaxPool2d)    [[1, 512, 14, 14]]        [[512, 7, 7]]             0
avgpool (AdaptiveAvgPool2d)   [[1, 512, 7, 7]]          [[512, 7, 7]]             0
classifier (Sequential)       [[1, 25088]]              [[1000]]                  123642856
├─ classifier.0 (Linear)      [[1, 25088]]              [[4096]]                  102764544
├─ classifier.1 (ReLU)        [[1, 4096]]               [[4096]]                  0
├─ classifier.2 (Dropout)     [[1, 4096]]               [[4096]]                  0
├─ classifier.3 (Linear)      [[1, 4096]]               [[4096]]                  16781312
├─ classifier.4 (ReLU)        [[1, 4096]]               [[4096]]                  0
├─ classifier.5 (Dropout)     [[1, 4096]]               [[4096]]                  0
├─ classifier.6 (Linear)      [[1, 4096]]               [[1000]]                  4097000
```

Remark: for now, Module Details is empty. More info will be added soon.

## Contributing

We welcome contributions to the AutoOpInspect project. Whether it's reporting issues, improving documentation, or contributing code, your input is valuable.