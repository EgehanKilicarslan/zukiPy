from functools import lru_cache
from typing import Any, Final, TypeAlias

import requests
from requests.exceptions import RequestException

from . import API_BASE
from .enum import ModelType

# Type aliases with proper Any type
ModelData: TypeAlias = dict[str, Any]
ModelList: TypeAlias = list[ModelData]


class ZukiModels:
    """Helper class to interact with Zuki Models API."""

    _API_URL: Final[str] = f"{API_BASE}/v1/models"
    _models: dict[str, ModelData] = {}
    _endpoint_to_type: Final[dict[str, str]] = {
        model_type.endpoint: model_type.type_name for model_type in ModelType
    }
    _cache_initialized: bool = False
    _session: Final[requests.Session] = requests.Session()

    @classmethod
    def clear_cache(cls) -> None:
        """Clear the cache of models data."""
        cls._models.clear()
        cls._cache_initialized = False

    @classmethod
    @lru_cache(maxsize=128)
    def get_model(cls, model_name: str | None = None) -> ModelList | ModelData | None:
        """
        Retrieve model data from Zuki API with caching support.

        Args:
            model_name: Optional name of the specific model to retrieve

        Returns:
            List of all models if model_name is None,
            Dictionary with specific model data if model_name is provided,
            None if the model is not found or there's an error
        """
        if not cls._cache_initialized and not cls._initialize_cache():
            return None

        try:
            if model_name is None:
                return list(cls._models.values())
            return cls._models.get(model_name)
        except Exception:
            return None

    @classmethod
    def _initialize_cache(cls) -> bool:
        """Initialize the model cache from the API."""
        try:
            response = cls._session.get(cls._API_URL, timeout=10)
            response.raise_for_status()
            data = response.json()
            if not isinstance(data, dict) or "data" not in data:
                return False

            cls._models = {
                model["id"]: model
                for model in data.get("data", [])
                if isinstance(model, dict) and "id" in model
            }
            cls._cache_initialized = True
            return True
        except (RequestException, ValueError, KeyError):
            return False

    @classmethod
    def get_model_endpoint(cls, model_name: str) -> str | None:
        """
        Retrieve the endpoint of a specific model.

        Args:
            model_name: Name of the model to retrieve the endpoint for

        Returns:
            Endpoint URL of the model if found, None otherwise
        """
        model_data = cls.get_model(model_name)
        return model_data.get("endpoint") if isinstance(model_data, dict) else None

    @classmethod
    def get_model_type(cls, model_name: str) -> str | None:
        """
        Retrieve the type of a specific model.

        Args:
            model_name: Name of the model to retrieve the type for

        Returns:
            Model type if found, None otherwise
        """
        if not model_name:
            return None

        endpoint = cls.get_model_endpoint(model_name)
        return cls._endpoint_to_type.get(endpoint) if endpoint else None
