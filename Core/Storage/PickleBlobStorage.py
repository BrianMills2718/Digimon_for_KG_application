import pickle
import os
from typing import Optional, Any
from pydantic import BaseModel, Field, ConfigDict # Import Pydantic components

from Core.Common.Logger import logger
from Core.Storage.BaseBlobStorage import BaseBlobStorage # This should now be a Pydantic BaseModel
from Core.Storage.NameSpace import NameSpace


class PickleBlobStorage(BaseBlobStorage):
    """
    Stores a single Python object (blob) and can persist it to/load it from a pickle file.
    Now a Pydantic BaseModel.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Class variable, not a Pydantic field for the instance schema itself
    RESOURCE_NAME_CONST: str = "blob_data.pkl" 
    
    # Instance field, will be initialized
    internal_data: Optional[Any] = None

    # The 'namespace' attribute is inherited from BaseStorage (which should be a Pydantic BaseModel)

    def __init__(self, **data: Any):
        """
        Initialize PickleBlobStorage.
        'namespace' should be passed in 'data' if it's part of BaseStorage/BaseBlobStorage fields.
        """
        super().__init__(**data)
        # If internal_data needs to be initialized based on something passed in 'data', do it here.
        # For now, it defaults to None as per the field definition.

    async def get(self) -> Optional[Any]:
        """Retrieves the stored blob data."""
        return self.internal_data

    async def set(self, blob: Any) -> None:
        """Sets the blob data to be stored."""
        self.internal_data = blob

    async def load(self, force: bool = False) -> bool:
        """
        Loads the blob data from a pickle file.
        The filename is determined by the namespace and RESOURCE_NAME_CONST.
        """
        if self.internal_data is not None and not force:
            log_path_info = self.namespace.get_load_path(self.RESOURCE_NAME_CONST) if self.namespace else 'volatile storage (no namespace)'
            logger.info(f"Data already in memory for {log_path_info}. Skipping load unless forced.")
            return True

        if force:
            log_path_info = self.namespace.get_load_path(self.RESOURCE_NAME_CONST) if self.namespace else 'volatile storage (no namespace)'
            logger.info(f"Forcing reload for: {log_path_info}.")
            self.internal_data = None

        if self.namespace:
            data_file_name = self.namespace.get_load_path(self.RESOURCE_NAME_CONST)
            if data_file_name and os.path.exists(data_file_name):
                try:
                    with open(data_file_name, "rb") as f:
                        self.internal_data = pickle.load(f)
                    logger.info(f"Successfully loaded data file for blob storage: {data_file_name}.")
                    return True
                except Exception as e:
                    logger.error(f"Error loading data file for blob storage {data_file_name}: {e}")
                    self.internal_data = None 
                    return False
            else:
                logger.info(f"No data file found for blob storage: {data_file_name}. Loading empty storage.")
                self.internal_data = None
                return False 
        else:
            self.internal_data = None
            logger.info("Namespace not set. Creating new volatile blob storage (no file to load).")
            return False

    async def persist(self) -> None:
        """
        Persists the current blob data to a pickle file.
        """
        if self.namespace:
            data_file_name = self.namespace.get_save_path(self.RESOURCE_NAME_CONST)
            try:
                if data_file_name:
                    os.makedirs(os.path.dirname(data_file_name), exist_ok=True)
                    with open(data_file_name, "wb") as f:
                        pickle.dump(self.internal_data, f)
                    logger.info(
                        f"Successfully saved blob storage to '{data_file_name}'."
                    )
                else:
                    logger.error("Cannot persist blob storage: data_file_name is None (namespace issue?).")
            except Exception as e:
                logger.error(f"Error saving data file for blob storage {data_file_name}: {e}")
        else:
            logger.warning("Namespace not set for PickleBlobStorage. Cannot persist to file.")

