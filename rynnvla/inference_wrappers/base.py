from __future__ import annotations

from abc import ABC, ABCMeta, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, List, Union

import torch

if TYPE_CHECKING:
    from transformers import BatchFeature
    from transformers.image_utils import ImageInput


class BaseInferenceWrapper(ABC):
    def __init__(
        self,
        model_path: str,
        dtype: torch.dtype,
        attn_implementation: str,
        device: str = "cuda:0",
        local_files_only: bool = True,
    ):
        self.model_path = model_path
        self.dtype = dtype
        self.attn_implementation = attn_implementation
        self.device = torch.device(device)
        self.local_files_only = local_files_only

        self._model = None
        self._processor = None

    @property
    def model(self):
        if self._model is None:
            self._model = self.load_model()
        return self._model

    @property
    def processor(self):
        if self._processor is None:
            self._processor = self.load_processor()
        return self._processor

    @abstractmethod
    def load_model(self):
        pass

    @abstractmethod
    def load_processor(self):
        pass


class BaseVLMInferenceWrapper(BaseInferenceWrapper, metaclass=ABCMeta):
    @abstractmethod
    def apply_chat_template(self, conversation: Dict[str, Any], enable_thinking: bool) -> str:
        pass

    @abstractmethod
    def load_images(
        self,
        images: ImageInput,
        processing_params: Dict[str, Any],
    ):
        pass

    @abstractmethod
    def load_videos(
        self,
        videos: Union[List[str], List[List[str]]],
        processing_params: Dict[str, Any],
    ):
        pass

    @abstractmethod
    def process_images(
        self,
        images: ImageInput,
        processing_params: Dict[str, Any],
    ):
        pass

    @abstractmethod
    def process_videos(
        self,
        videos: Union[List[str], List[List[str]]],
        processing_params: Dict[str, Any],
    ):
        pass

    @abstractmethod
    def process_text(
        self,
        text: str,
        image_inputs: Dict[str, Any],
        video_inputs: Dict[str, Any],
    ) -> BatchFeature:
        pass

    @abstractmethod
    def generate(
        self,
        model_inputs: Dict[str, Any],
        sampling_params: Dict[str, Any],
    ) -> List[str]:
        pass


class BaseVLAInferenceWrapper(BaseInferenceWrapper, metaclass=ABCMeta):
    @abstractmethod
    def process(
        self,
        text: str,
        state: Dict[str, Any],
        images: Dict[str, ImageInput],
    ) -> Dict[str, Any]:
        pass

    @abstractmethod
    def collate(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        pass

    @abstractmethod
    def prefill(self, model_inputs: Dict[str, Any]) -> Dict[str, Any]:
        pass

    @abstractmethod
    def decode(
        self,
        model_inputs: Dict[str, Any],
        cache: Dict[str, Any],
        num_steps: int,
        robot_type=None,
    ) -> torch.Tensor:
        pass

    @abstractmethod
    def post_process(
        self,
        action: torch.Tensor,
        state: Dict[str, Any],
        robot_type: str,
    ) -> Dict[str, Any]:
        pass
