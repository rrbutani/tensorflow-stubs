from typing import Callable, Dict, Iterable, List, Optional

from . import experimental

from numpy import ndarray

# Delegate class defined here:
# (Not public)
# https://github.com/tensorflow/tensorflow/blob/2bb4f008ff48630dd516a637232ade00ffd018cf/tensorflow/lite/python/interpreter.py#L52-L136
class Delegate:
  def __init__(self, library: str, options: Optional[Dict[str, str]]=None):
    pass

# Interpreter class defined here:
# https://github.com/tensorflow/tensorflow/blob/2bb4f008ff48630dd516a637232ade00ffd018cf/tensorflow/lite/python/interpreter.py#L171-L455

TensorDetails = Dict[str, Any]

class Interpreter:
  def __init__(self,
               model_path: Optional[str]=None,
               model_content: Optional[str]=None,
               experimental_delegates: Optional[List[Delegate]]=None) -> None: ...

  def __del__(self): ...

  def allocate_tensors(self) -> None: ...

  def _safe_to_run(self) -> bool: ...

  def _ensure_safe(self) -> None: ...

  def _get_tensor_details(self, tensor_index: int) -> TensorDetails: ...

  def get_tensor_details(self) -> List[TensorDetails]: ...

  def get_input_details(self) -> List[TensorDetails]: ...

  def set_tensor(self, tensor_index: int, value: Any) -> None: ...

  def resize_tensor_input(self, input_index: int, tensor_size: Iterable[int]) -> None: ...

  def get_output_details(self) -> List[TensorDetails]: ...

  def get_tensor(self, tensor_index: int) -> ndarray: ...

  def tensor(self, tensor_index: int) -> Callable[[], ndarray]: ...

  def invoke(self) -> None: ...

  def reset_all_variables(self) -> None: ...
