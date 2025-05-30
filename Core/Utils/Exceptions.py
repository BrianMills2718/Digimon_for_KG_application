#!/usr/bin/env python
# -*- coding: utf-8 -*-



import asyncio
import functools
import traceback
from typing import Any, Callable, Tuple, Type, TypeVar, Union

from Core.Common.Logger import logger

ReturnType = TypeVar("ReturnType")

class NotSupportedError(Exception):
    """Exception raised when a feature or operation is not supported."""
    pass

class InvalidStorageError(Exception):
    """Exception raised for errors in the storage operations."""

    def __init__(self, message: str = "Invalid storage operation"):
        self.message = message
        super().__init__(self.message)

def log_and_reraise(retry_state) -> None:
    """Log error and reraise, for tenacity retry_error_callback."""
    if retry_state.outcome.failed:
        logger.error(
            f"Tenacity retry attempt {retry_state.attempt_number} failed with {retry_state.outcome.exception()!r}, "
            f"raising."
        )
    # Tenacity will handle the actual re-raise if reraise=True in the decorator.

def handle_exception(
    _func: Callable[..., ReturnType] = None,
    *,
    exception_type: Union[Type[Exception], Tuple[Type[Exception], ...]] = Exception,
    exception_msg: str = "",
    default_return: Any = None,
) -> Callable[..., ReturnType]:
    """handle exception, return default value"""

    def decorator(func: Callable[..., ReturnType]) -> Callable[..., ReturnType]:
        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> ReturnType:
            try:
                return await func(*args, **kwargs)
            except exception_type as e:
                logger.opt(depth=1).error(
                    f"{e}: {exception_msg}, "
                    f"\nCalling {func.__name__} with args: {args}, kwargs: {kwargs} "
                    f"\nStack: {traceback.format_exc()}"
                )
                return default_return

        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> ReturnType:
            try:
                return func(*args, **kwargs)
            except exception_type as e:
                logger.opt(depth=1).error(
                    f"Calling {func.__name__} with args: {args}, kwargs: {kwargs} failed: {e}, "
                    f"stack: {traceback.format_exc()}"
                )
                return default_return

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    if _func is None:
        return decorator
    else:
        return decorator(_func)