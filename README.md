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

Basic Usage

Create an OpsInfoProvider instance using a PyTorch model and input data:

``` python
from AutoOpInspect import OpsInfoProvider
import torchvision
import torch

model = torchvision.models.vgg11()
input_data = [torch.randn(1, 3, 224, 224)] # make a list of inputs (supports multiple inputs)
ops_info_provider = OpsInfoProvider(model, input_data)
```

Specifying a Target Module

You can specify a target operator to inspect, if you do not need to inspect the whole model:

``` python
target_module = model.features
ops_info_provider = OpsInfoProvider(model, input_data, target=target_module)
```

Getting Dummy Data

Retrieve dummy input and output data for a specified mode:

``` python
# available modes are input, output and both
dummy_input, dummy_output = ops_info_provider.get_dummy(target_module, mode='both')
```

## Contributing

We welcome contributions to the AutoOpInspect project. Whether it's reporting issues, improving documentation, or contributing code, your input is valuable.