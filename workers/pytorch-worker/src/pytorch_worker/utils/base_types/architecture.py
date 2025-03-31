import torch
from abc import ABC, abstractmethod
from typing import Any, Iterable, TypeVar, Optional, TypedDict, Generic
from spandrel import Architecture as SpandrelArchitecture, ImageModelDescriptor
from .common import StateDict, TorchDevice

T = TypeVar("T", bound=torch.nn.Module, covariant=True)


class ComponentMetadata(TypedDict):
    display_name: str
    input_space: str
    output_space: str


# TO DO: in the future, maybe we can compare sets of keys, rather than use
# a detect method? That might be more optimized.


class Architecture(ABC, Generic[T]):
    """
    The abstract-base-class that all cozy-creator Architectures should implement.
    """

    @property
    def display_name(self) -> str:
        return self._display_name

    @property
    def input_space(self) -> str:
        return self._input_space

    @property
    def output_space(self) -> str:
        return self._output_space

    @property
    def model(self) -> T:
        """Access the underlying PyTorch model."""
        return self._model  # type: ignore

    @property
    def config(self) -> Any:
        return self._config

    def __init__(
        self,
        *,
        state_dict: Optional[StateDict] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        """
        Constructor signature should look like this, although this abstract-base
        class does not (and cannot) enforce your constructor signature.
        """
        self._display_name = "default"
        self._input_space = "default"
        self._output_space = "default"
        self._config = {}
        pass

    @classmethod
    @abstractmethod
    def detect(
        cls,
        *,
        state_dict: Optional[StateDict] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> Optional[ComponentMetadata]:
        """
        Detects whether the given state dictionary matches the architecture.

        Args:
            state_dict (StateDict): The state dictionary from a PyTorch model.
            metadata (dict[str, Any]): optional additional metadata to help identify the model

        Returns:
            bool: True if the state dictionary matches the architecture, False otherwise.
        """
        pass

    @abstractmethod
    def load(
        self,
        state_dict: StateDict,
        device: Optional[TorchDevice] = None,
    ) -> None:
        """
        Loads a model from the given state dictionary according to the architecture.

        Args:
            state_dict (StateDict): The state dictionary from a PyTorch model.
            device: The device the loaded model is sent to.
        """
        pass



class SpandrelArchitectureAdapter(Architecture):
    """
    This class converts architectures from the spandrel library to our own
    Architecture interface.
    """

    def __init__(self, arch: SpandrelArchitecture):
        super().__init__()
        if not isinstance(arch, SpandrelArchitecture):
            raise TypeError("'arch' must be an instance of spandrel Architecture")

        self.inner = arch
        self._model = None
        self._display_name = self.inner.name

    def load(self, state_dict: StateDict, device: Optional[TorchDevice] = None) -> None:
        descriptor = self.inner.load(state_dict)
        if not isinstance(descriptor, ImageModelDescriptor):
            raise TypeError("descriptor must be an instance of ImageModelDescriptor")

        self._model = descriptor.model
        if device is not None:
            self._model.to(device)
        elif descriptor.supports_half:
            self._model.to(torch.float16)
        elif descriptor.supports_bfloat16:
            self._model.to(torch.bfloat16)
        else:
            raise Exception("Device not provided and could not be inferred")

    @classmethod
    def detect(
        cls,
        state_dict: StateDict = None,
        metadata: dict[str, Any] = None,
    ) -> Optional[ComponentMetadata]:
        pass


def architecture_validator(plugin: Any) -> bool:
    try:
        if isinstance(plugin, Iterable):
            return all(architecture_validator(p) for p in plugin)
        return issubclass(plugin, Architecture)
    except TypeError:
        print(f"Invalid plugin type: {plugin}")
        return False
