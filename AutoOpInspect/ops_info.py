from typing import List, Optional, Union, Tuple
from collections import OrderedDict
import torch
import torch.nn as nn


class OpInfo:
    """
    A class to store and manage the attributes and information pertinent to an individual nn.Module.

    This class facilitates the retrieval of various attributes of a module including its input and output 
    dimensions and values, without necessitating recalculation each time, thereby enhancing efficiency.

    Attributes:
        module (nn.Module): The module whose information is to be stored.
        input_val (List): A list to store the input values. It stores non-tensor values at indices corresponding to 
                          'None' values in the `input_dim` attribute.
        output_val (List): A list to store the output values of the module.
        input_dim (Optional[List[Tuple]]): A list of tuples to represent the dimensions of each input value. 'None' at any
                                           index indicates a non-tensor value, further details of which can be found in `input_val`.
        output_dim (Optional[List[Tuple]]): A list of tuples representing the dimensions of each output value.
    """

    def __init__(self, module: nn.Module, name: str):
        """
        Initializes the OperatorInfo instance with the given module and prepares attributes for storage of module details.

        Args:
            module (nn.Module): The module whose details are to be stored.
        """
        self.module = module
        try:
            self.params = sum(p.numel() for p in module.parameters())
        except AttributeError:
            self.params = 0
        self.name = name
        self.input_val = []
        self.output_val = []
        self.input_dim: Optional[List[Tuple]] = []
        self.output_dim: Optional[List[Tuple]] = []


class OpsInfoProvider:
    """
    A wrapper for a list of operators.
    Allows to collect data for all operators in one feed-forward.
    Args:
        - model: the model to collect data from
        - operators_info: a list of OperatorInfo objects
        - device: the device of the model
        - input: a list of tuple for the sizes of the input tensor (one tuple per input), for the feed-forwarding
    """


    def __init__(self, model: nn.Module, model_input: list, target: Optional[Union[str, nn.Module]] = None, inspect_children=True):
        """
        Initializes the OpsInfoProvider with a model to analyze.
        Args:
            model: The model to analyze.
            model_input: list of the model inputs
            target: if None, all modules in the model will be inspected. If a module is 

        """
        self.model = model
        # extract the target nn.Module form the model
        if target is None:
            self.target_module = model
        elif isinstance(target, str):
            m = target
            for m_name in target.split("."):
                m = m[int(m_name)] if m_name.isdigit() else getattr(m, m_name)
            self.target_module = m
        else:
            self.target_module = target
        self._collect_info(model_input, inspect_children)  # Automatically collect information on initialization


    def _collect_info(self, model_input: list, inspect_children=True, module_filter_func=None):
        """
        Collects information on each module in the target module and stores it in self.operators_info.
        """

        # get the list of operators
        def get_modules(module, module_filter_func=None, prefix=None):
            target_modules = []
            for name, sub_module in module.named_children():
                full_name = f"{prefix}.{name}" if prefix else name
                if module_filter_func is None or module_filter_func(sub_module):
                    target_modules.append((sub_module, full_name))  # store module along with its full name
                target_modules.extend(get_modules(sub_module, module_filter_func, full_name))
            return target_modules
        target_modules = [(self.target_module, "target_module")]
        if inspect_children:
            target_modules += get_modules(self.target_module)
        self.operators_info = OrderedDict({name: OpInfo(module, name) for module, name in target_modules})

        # use hook to collect info on operators
        hook_handles = [operator_info.module.register_forward_hook(self._make_feed_hook(name)) for (name, operator_info) in self.operators_info.items()]
        self.model(*model_input)
        for handle in hook_handles:
            handle.remove()

    def _make_feed_hook(self, name: str):
        def collect_tensor_info(value):
            if isinstance(value, torch.Tensor):
                return list(value.shape)
            elif isinstance(value, tuple):
                return tuple(collect_tensor_info(v) for v in value)
            elif isinstance(value, list):
                return [collect_tensor_info(v) for v in value]
            else:
                return None
            
        def collect_val_info(value):
            if isinstance(value, torch.Tensor):
                return 'Tensor'
            elif isinstance(value, tuple):
                return tuple(collect_val_info(v) for v in value)
            elif isinstance(value, list):
                return [collect_val_info(v) for v in value]
            else:
                return value

        def hook(m: nn.Module, x: Tuple[torch.Tensor], z: Union[torch.Tensor, List[torch.Tensor]]):
            input_dims = collect_tensor_info(list(x))
            input_vals = collect_val_info(list(x))
            self.operators_info[name].input_dim = input_dims
            self.operators_info[name].input_val = input_vals  

            output_dims = collect_tensor_info([z[0]])
            output_vals = collect_val_info([z[0]])

            self.operators_info[name].output_dim = output_dims
            self.operators_info[name].output_val = output_vals
        return hook

    def get_opinfo(self, index: Union[int, str, nn.Module]) -> OpInfo:
        if isinstance(index, int):
            operator_info = list(self.operators_info.values())[index]
        elif isinstance(index, str):
            operator_info = self.operators_info[index]
        elif isinstance(index, nn.Module):
            for _, op_info in self.operators_info.items():
                if op_info.module is index:
                    operator_info = op_info
                    break
        else:
            raise TypeError(f"Index must be an integer, an str or a Module (got {type(index).__name__})")
        return operator_info

    def __getitem__(self, index: Union[int, str, nn.Module]) -> dict:
        operator_info = self.get_opinfo(index)
        return {
            'module': operator_info.module,
            'name': operator_info.name,
            'params': operator_info.params,
            'input_dim': operator_info.input_dim,
            'input_val': operator_info.input_val,
            'output_dim': operator_info.output_dim,
            'output_val': operator_info.output_val
        }

    def get_dummy(self, module, mode='input'):
        """
        Generates a dummy input or output for a specified module.
        Args:
            module: The module to generate dummy data for, or its name.
            mode (str): The mode specifying the type of dummy data to generate. 
                        It can be 'input', 'output', or 'both'. Default is 'input'.

        Returns:
            dummy_data: The generated dummy data.
        """
        module_info = self[module]
        
        def generate_dummy(data_vals, data_dims):
            return [val if dim is None else torch.randn(tuple(dim)) for val, dim in zip(data_vals, data_dims)]

        if mode == 'input':
            input_vals = module_info['input_val']
            input_dims = module_info['input_dim']
            return generate_dummy(input_vals, input_dims)
        
        elif mode == 'output':
            output_vals = module_info['output_val']
            output_dims = module_info['output_dim']
            return generate_dummy(output_vals, output_dims)
        
        elif mode == 'both':
            input_vals = module_info['input_val']
            input_dims = module_info['input_dim']
            output_vals = module_info['output_val']
            output_dims = module_info['output_dim']
            return generate_dummy(input_vals, input_dims), generate_dummy(output_vals, output_dims)
        else:
            raise ValueError("Invalid mode. It should be one of 'input', 'output', or 'both'.")


    def __str__(self):
        # Initially, find the maximum length for the Layer (type) column
        max_layer_type_length = max(
            len(name) + 4 * name.count('.') + len(operator_info.module.__class__.__name__) + 3 * name.count('.')
            for name, operator_info in self.operators_info.items()
        ) + 2

        # Creating a dynamic header using the maximum length found
        header = f"{ 'Layer (type)':<{max_layer_type_length}} {'Input Shape':<25} {'Output Shape':<25} {'Param #':<20} {'Module Details':<25}"
        lines = [header]
        lines.append("=" * (max_layer_type_length + 100))  # Adjust total length here

        for name, operator_info in self.operators_info.items():
            # Get the hierarchical level based on the number of dots in the name
            level = name.count('.')
            indent = '│ ' * (level - 1)  # Hierarchical visualization using '│ '

            if level > 0:
                indent += '├─ '  # Adding '├─' to represent a node

            # Getting just the class name of the module for a more concise display
            module_class_name = operator_info.module.__class__.__name__

            # Creating a row for the current module with necessary indentations
            row = f"{indent + name + ' (' + module_class_name + ')':<{max_layer_type_length}} {str(operator_info.input_dim):<25} {str(operator_info.output_dim):<25} {str(operator_info.params):<20} { '':<25}"
            lines.append(row)
        
        return "\n".join(lines)




    def __eq__(self, other) -> bool:
        """
        Overrides the equality operator to compare two OpsInfoProvider instances.

        Args:
            other (OpsInfoProvider): The other OpsInfoProvider instance to compare with.

        Returns:
            bool: True if the instances are equal, False otherwise.
        """
        # Ensure the other object is an instance of OpsInfoProvider
        if not isinstance(other, OpsInfoProvider):
            print("different class")
            return False

        # Check if both instances have the same number of operators
        if len(self.operators_info) != len(other.operators_info):
            print("operators_info have different length")
            return False

        # Iterating over each operator info in self and other and comparing attributes
        for (self_name, self_info), (other_name, other_info) in zip(self.operators_info.items(), other.operators_info.items()):
            # Check if both instances have the same operator names
            if self_name != other_name or self_info.name != other_info.name:
                print(f"different name: {self_name} and {other_name}")
                return False
            # Check if the stored params in OpInfo instances are the same
            if self_info.params != other_info.params:
                print(f"different number of parameters: {self_info.params} and {other_info.params}")
                return False
            # Check if the stored dims in OpInfo instances are the same
            if (not all(map(lambda x, y: x == y, self_info.input_dim, other_info.input_dim)) or 
                not all(map(lambda x, y: x == y, self_info.output_dim, other_info.output_dim))):
                print(f"different dims for {self_name}")
                return False
            # Check if the stored vals in OpInfo instances are the same
            if (
                not all(map(lambda x, y: x == y, self_info.input_val, other_info.input_val)) or
                not all(map(lambda x, y: x == y, self_info.output_val, other_info.output_val))
            ):
                print(f"different vals for {self_name}")
                return False

        # If all the checks passed, the instances are considered equivalent
        return True


















