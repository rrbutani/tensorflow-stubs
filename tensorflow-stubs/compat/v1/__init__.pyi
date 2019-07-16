from typing import Optional, TypeVar

from . import logging

# It seems as if these three types are present within tensorflow but are not
# exported, meaning that this function essentially takes no arguments (to any
# user operating outside of the library).

# For now we'll leave TypeVars in their place.

ConfigProto = TypeVar('ConfigProto')

# Possible values for this are here: https://github.com/tensorflow/tensorflow/blob/2a555cd91f7cdc6d384e6887ddb6af7843bdb996/tensorflow/python/eager/context.py#L51-L55
DevicePolicy = TypeVar('DevicePolicy')

# Possible values for this are here: https://github.com/tensorflow/tensorflow/blob/2a555cd91f7cdc6d384e6887ddb6af7843bdb996/tensorflow/python/eager/context.py#L56-L57
ExecutionMode = TypeVar('ExecutionMode')


def enable_eager_execution(config: Optional[ConfigProto]=None,
                           device_policy: Optional[DevicePolicy]=None,
                           execution_mode: Optional[ExecutionMode]=None) -> None:
    # enable_eager_execution implemented here:
    # https://github.com/tensorflow/tensorflow/blob/179b3884e1118a0a18e03334e848e4b9053d17a1/tensorflow/python/framework/ops.py#L5524-L5593
    ...
