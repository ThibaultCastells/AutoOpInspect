from typing import List, Optional, Union, Tuple
from collections import OrderedDict
import torch
import torch.nn as nn
import time
from tqdm import tqdm
from .module_print import get_other_info
from .utils import ascii_barplot_horizontal

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
        self.speed = None 
        self.error = None # allows to collect error when forwarding
        self.hash = hash(str(module).encode())

    def __getitem__(self, key: str) -> dict:
        attributes = {
            'module': self.module,
            'name': self.name,
            'params': self.params,
            'input_dim': self.input_dim,
            'input_val': self.input_val,
            'output_dim': self.output_dim,
            'output_val': self.output_val,
            'speed': self.speed,
            'hash': self.hash,
        }
        return attributes[key]
    
class OpsInfoProvider:
    """
    A wrapper for a list of operators.
    Allows to collect data for all operators in one feed-forward.
    """

    def __init__(self, model: nn.Module, model_input: list, target: Optional[Union[str, nn.Module]] = None, inspect_children=True):
        """
        Initializes the OpsInfoProvider with a model to analyze.
        Args:
            model: The model to analyze.
            model_input: list of the model inputs
            target: if None, all modules in the model will be inspected.
            inspect_children: weither the children of the target are considered or not
        """
        self.model_hash = hash(str(model).encode())
        self.model = model
        self.model_input = model_input
        self.inspect_children = inspect_children
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
        self.generations = self._calculate_generations()

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

            output_dims = collect_tensor_info([z])
            output_vals = collect_val_info([z])
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
    
    def benchmark_speed(self, operator: Optional[Union[int, str, nn.Module]]=None, device=None, iterations: int=10):
        self.refresh()
        if device is None:
            device = next(self.target_module.parameters()).device
        else:
            device = torch.device(device)

        operators_to_benchmark = [operator] if operator is not None else self.operators_info.keys()

        for op in tqdm(operators_to_benchmark):
            operator_info = self.get_opinfo(op)

            # Save the original mode and device, to put it back after the benchmark
            original_mode = operator_info['module'].training
            try:
                original_device = next(operator_info['module'].parameters()).device
            except StopIteration:
                # If the module has no parameters, use the default device
                original_device = next(self.target_module.parameters()).device
            
            module = operator_info['module']
            if not hasattr(module, "forward"):
                continue
            
            module.eval()
            module.to(device)
            
            input_data = self.get_dummy(op, mode='input', device=device)

            # Disable gradient computation
            with torch.no_grad():
                # warmup
                try:
                    module(*input_data)
                except NotImplementedError:
                    # skip modules for which the forward is not implemented, such as ModuleList
                    operator_info.error = f"The forward method for {module.__class__.__name__} is not implemented and will be skipped."
                    continue
                except Exception as e:
                    operator_info.error = e
                    continue

                total_time = 0.0

                for _ in range(iterations):
                    if device.type == 'cuda':
                        torch.cuda.synchronize()
                    start_time = time.perf_counter()
                    module(*input_data)
                    if device.type == 'cuda':
                        torch.cuda.synchronize()
                    end_time = time.perf_counter()
                    total_time += end_time - start_time

                total_time = total_time / iterations

                operator_info.speed = total_time * 1000 # convert to ms

                # Restore the original device and mode
                module.to(original_device)
                module.train(original_mode)

    def __getitem__(self, index: Union[int, str, nn.Module]) -> dict:
        return self.get_opinfo(index)

    def get_dummy(self, operator, mode='input', device=None, dtype=None):
        """
        Generates a dummy input or output for a specified module.
        Args:
            operator: The operator to generate dummy data for, or its name.
            mode (str): The mode specifying the type of dummy data to generate. 
                        It can be 'input', 'output', or 'both'. Default is 'input'.

        Returns:
            dummy_data: The generated dummy data.
        """
        operator_info = self[operator]
        if device is None:
            device = next(self.target_module.parameters()).device
        if dtype is None:
            dtype = next(self.target_module.parameters()).dtype

        def generate_dummy(data_vals, data_dims, device, dtype):

            def build_input_data(input_val, input_dim, device, dtype):
                if isinstance(input_val, tuple):
                    return tuple(build_input_data(v, d, device, dtype) for v, d in zip(input_val, input_dim))
                elif isinstance(input_val, list):
                    return [build_input_data(v, d, device, dtype) for v, d in zip(input_val, input_dim)]
                elif input_val == 'Tensor' and input_dim is not None:
                    return torch.randn(tuple(input_dim)).to(device).to(dtype)
                else:
                    return input_val  # for non-Tensor values

            out = build_input_data(data_vals, data_dims, device, dtype)

            return out
        
        if mode == 'input':
            input_vals = operator_info['input_val']
            input_dims = operator_info['input_dim']
            return generate_dummy(input_vals, input_dims, device, dtype)
        
        elif mode == 'output':
            output_vals = operator_info['output_val']
            output_dims = operator_info['output_dim']
            return generate_dummy(output_vals, output_dims, device, dtype)
        
        elif mode == 'both':
            input_vals = operator_info['input_val']
            input_dims = operator_info['input_dim']
            output_vals = operator_info['output_val']
            output_dims = operator_info['output_dim']
            return generate_dummy(input_vals, input_dims, device, dtype), generate_dummy(output_vals, output_dims, device, dtype)
        else:
            raise ValueError("Invalid mode. It should be one of 'input', 'output', or 'both'.")

    def refresh(self):
        """
        Refresh the model representation if the model architecture has been modified.
        """
        if hash(str(self.target_module).encode()) == self.model_hash:
            # efficient check to see if a refresh is needed
            return
        self.model_hash = hash(str(self.target_module).encode())

        update_required = False
        save_time = {}
        for name, module in self.target_module.named_modules():
            update_curr = False
            if name == '':
                continue
            elif name not in self.operators_info:
                # print(f'New operator: {name}')
                update_required = True
                update_curr = True
            else:
                current_hash = hash(str(module).encode())
                if self.operators_info[name]['hash'] != current_hash:
                    # print(f"Operator {name} has been updated")
                    update_required = True
                    update_curr = True
            if not update_curr:
                save_time[name] = self.operators_info[name]['speed']
        if update_required:
            self._collect_info(self.model_input, self.inspect_children)
            self.generations = self._calculate_generations()
            for name, speed in save_time.items():
                self.operators_info[name].speed = speed


    def __str__(self):
        self.refresh()
        # Initially, find the maximum length for the Layer (type) column
        max_layer_type_length = max(
            name.count('.')*2 + len(name) + len(operator_info.module.__class__.__name__)
            for name, operator_info in self.operators_info.items()
        ) + 6

        # Creating a dynamic header using the maximum length found
        header = f"{ 'Layer (type)':<{max_layer_type_length}} {'Input Shape':<25} {'Output Shape':<25} {'Param #':<13} {'Inference (ms)':<15} {'Other':<15}"
        lines = [header]
        lines.append("=" * (max_layer_type_length + 120))  # Adjust total length here

        for name, operator_info in self.operators_info.items():
            # Get the hierarchical level based on the number of dots in the name
            level = 0 if name == 'target_module' else name.count('.') + 1
            indent = '│ ' * (level - 1)  # Hierarchical visualization using '│ '

            if level > 0:
                indent += '├─ '  # Adding '├─' to represent a node

            # Getting just the class name of the module for a more concise display
            module_class_name = operator_info.module.__class__.__name__

            # Creating a row for the current module with necessary indentations
            speed = operator_info.speed  # Use get method to avoid KeyError for not benchmarked operators
            speed = f"{speed:.5f}" if isinstance(speed, float) else 'N/A'

            params = operator_info.params
            formatted_params = f"{params:,}"
            if params >= 1_000_000_000:
                formatted_params = f"{params / 1_000_000_000:.2f}B"
            elif params >= 1_000_000:
                formatted_params = f"{params / 1_000_000:.2f}M"
            params = formatted_params

            other_info = get_other_info(operator_info.module)

            row = f"{indent + name + ' (' + module_class_name + ')':<{max_layer_type_length}} {str(operator_info.input_dim):<25} {str(operator_info.output_dim):<25} {params:<13} {speed:<15} {other_info:<15}"
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
            # Check if the stored params and speed in OpInfo instances are the same
            if self_info.params != other_info.params:
                print(f"different number of parameters for {self_name}: {self_info.params} and {other_info.params}")
                return False
            if self_info.speed != other_info.speed:
                print(f"different inference speed for {self_name}: {self_info.speed} and {other_info.speed}")
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
    
    def __contains__(self, operator: Union[str, nn.Module]):
        if isinstance(operator, str):
            return operator in self.operators_info
        elif isinstance(operator, nn.Module):
            return any(op_info.module is operator for op_info in self.operators_info.values())
        else:
            raise TypeError(f"Operator must be a str or a Module (got {type(operator).__name__})")


    def _calculate_generations(self):
        """
        Calculates the generation of each operator in the model.
        """
        generations = {name: 0 for name in self.operators_info}
        for name in self.operators_info:
            parts = name.split('.')
            for depth in range(1, len(parts)):
                parent = '.'.join(parts[:depth])
                generations[parent] = max(generations[parent], len(parts) - depth)
        return generations

    def barplot_speed(self, mode: str = 'sum', generation: int = 0):
        """
        Creates a bar plot of the speed of operators at a specific generation.

        Args:
            mode (str): 'mean' or 'sum' to plot average or total speed.
            generation (int): The generation of the operators. By default, 0 (i.e. only operators with no children are considered).
        """
        self.refresh()
        module_speeds = {}
        module_counts = {}

        for name, op_info in self.operators_info.items():
            if self.generations[name] <= generation and name!='target_module':
                module_type = type(op_info.module).__name__
                if op_info.speed is not None:
                    if module_type not in module_speeds:
                        module_speeds[module_type] = 0
                        module_counts[module_type] = 0
                    module_speeds[module_type] += op_info.speed
                    module_counts[module_type] += 1

        speeds = []
        labels = []
        for module_type, total_speed in module_speeds.items():
            if mode == 'mean':
                speed = total_speed / module_counts[module_type]
            elif mode == 'sum':
                speed = total_speed
            else:
                raise ValueError("Mode must be 'mean' or 'sum'")
            speeds.append(speed)
            labels.append(module_type)

        ascii_barplot_horizontal(speeds, labels, title=f'Operator Speed in ms ({mode})')

    def barplot_quantity(self, generation: int = 0):
        """
        Creates a bar plot of the quantity of operators at a specific generation.

        Args:
            generation (int): The generation of the operators. By default, 0 (i.e. only operators with no children are considered).
        """
        self.refresh()
        module_counts = {}

        for name, op_info in self.operators_info.items():
            if self.generations[name] <= generation and name!='target_module':
                module_type = type(op_info.module).__name__
                module_counts[module_type] = module_counts.get(module_type, 0) + 1

        quantities = list(module_counts.values())
        labels = list(module_counts.keys())

        ascii_barplot_horizontal(quantities, labels, title=f'Operator Quantity')