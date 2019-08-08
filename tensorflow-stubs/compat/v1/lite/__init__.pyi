from typing import Any, Callable, Dist, List, Optional, Set, Tuple, Union

from numpy import ndarray

from tensorflow import Session

GraphDef = Any
Node = Tuple[Callable[[Any], Any], str]

# TFLiteConverter class defined here:
# https://github.com/tensorflow/tensorflow/blob/d30a41fcb05e87d597052213f74ad629e8d39933/tensorflow/lite/python/lite.py#L455-L1033
class TFLiteConverter:
  def __init__(
    self,
    graph_def: GraphDef,
    input_tensors: Optional[ndarray],
    output_tensors: Optional[ndarray],
    input_arrays_with_shape: Optional[List[Tuple[str, List[int]]]]=None,
    output_arrays: Optional[List[Tuple[str, List[int]]]]=None,
    experimental_debug_info_func: Optional[Callable[[List[Node]], str]]=None
    ): ...

  @classmethod
  def from_session(cls,
                   sess: Session,
                   input_tensors: List[ndarray],
                   output_tensors: List[ndarray]
                   ) -> TFLiteConverter: ...

  @classmethod
  def from_frozen_graph(cls,
                        graph_def_file: str,
                        input_arrays: List[ndarray],
                        output_arrays: List[ndarray],
                        input_shapes: Optional[Dict[str, List[int]]]=None
                        ) -> TFLiteConverter: ...

  @classmethod
  def from_saved_model(cls,
                       saved_model_dir: str,
                       input_arrays: Optional[List[ndarray]]=None,
                       input_shapes: Optional[Dict[str, List[int]]]=None,
                       output_arrays: Optional[List[ndarray]]=None,
                       tag_set: Optional[Set[str]]=None,
                       signature_key: Optional[str]=None
                       ) -> TFLiteConverter: ...

  @classmethod
  def from_keras_model_file(cls,
    model_file: str,
    input_arrays: Optional[List[ndarray]]=None,
    input_shapes: Optional[Dict[str, List[int]]]=None,
    output_arrays: Optional[List[ndarray]]=None,
    custom_objects: Optional[Dict[str, Union[Any, Callable[[Any], Any]]]]=None
    ) -> TFLiteConverter: ...

  def __setattr__(self, name: str, value: Any) -> None: ...

  def __getattribute__(self, name: str) -> Optional[Any]: ...

  def convert(self) -> bytes: ...

  def get_input_arrays(self) -> List[str]: ...

  def _has_valid_tensors(self) -> bool: ...

  def _set_batch_size(self, batch_size: int) -> None: ...
