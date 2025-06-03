"""
Structured error types for better error handling and recovery.
"""

from typing import Optional, Dict, Any, List
from enum import Enum
from dataclasses import dataclass
import traceback


class ErrorSeverity(Enum):
    """Error severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Error categories for different types of failures."""
    # LLM related
    LLM_TIMEOUT = "llm_timeout"
    LLM_RATE_LIMIT = "llm_rate_limit"
    LLM_INVALID_RESPONSE = "llm_invalid_response"
    LLM_API_ERROR = "llm_api_error"
    
    # Data related
    DATA_NOT_FOUND = "data_not_found"
    DATA_INVALID_FORMAT = "data_invalid_format"
    DATA_CORRUPTION = "data_corruption"
    
    # Configuration related
    CONFIG_MISSING = "config_missing"
    CONFIG_INVALID = "config_invalid"
    
    # Tool related
    TOOL_NOT_FOUND = "tool_not_found"
    TOOL_EXECUTION_FAILED = "tool_execution_failed"
    TOOL_INVALID_PARAMS = "tool_invalid_params"
    
    # Graph related
    GRAPH_NOT_BUILT = "graph_not_built"
    GRAPH_INVALID_OPERATION = "graph_invalid_operation"
    
    # Index related
    INDEX_NOT_BUILT = "index_not_built"
    INDEX_BUILD_FAILED = "index_build_failed"
    INDEX_QUERY_FAILED = "index_query_failed"
    
    # System related
    SYSTEM_RESOURCE_EXHAUSTED = "system_resource_exhausted"
    SYSTEM_DEPENDENCY_MISSING = "system_dependency_missing"
    
    # Unknown
    UNKNOWN = "unknown"


@dataclass
class ErrorContext:
    """Context information for an error."""
    operation: str
    component: str
    input_data: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None
    stack_trace: Optional[str] = None


@dataclass
class RecoveryStrategy:
    """Recovery strategy for an error."""
    action: str
    description: str
    params: Optional[Dict[str, Any]] = None


class StructuredError(Exception):
    """Base class for structured errors with recovery strategies."""
    
    def __init__(
        self,
        message: str,
        category: ErrorCategory,
        severity: ErrorSeverity = ErrorSeverity.ERROR,
        context: Optional[ErrorContext] = None,
        cause: Optional[Exception] = None,
        recovery_strategies: Optional[List[RecoveryStrategy]] = None
    ):
        super().__init__(message)
        self.message = message
        self.category = category
        self.severity = severity
        self.context = context or ErrorContext(
            operation="unknown",
            component="unknown",
            stack_trace=traceback.format_exc() if cause else None
        )
        self.cause = cause
        self.recovery_strategies = recovery_strategies or []
        
    def add_recovery_strategy(self, strategy: RecoveryStrategy):
        """Add a recovery strategy."""
        self.recovery_strategies.append(strategy)
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary for logging/serialization."""
        return {
            "message": self.message,
            "category": self.category.value,
            "severity": self.severity.value,
            "context": {
                "operation": self.context.operation,
                "component": self.context.component,
                "input_data": self.context.input_data,
                "metadata": self.context.metadata,
                "stack_trace": self.context.stack_trace
            },
            "cause": str(self.cause) if self.cause else None,
            "recovery_strategies": [
                {
                    "action": s.action,
                    "description": s.description,
                    "params": s.params
                }
                for s in self.recovery_strategies
            ]
        }


# Specific error types

class LLMError(StructuredError):
    """Base class for LLM-related errors."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(message, ErrorCategory.LLM_API_ERROR, **kwargs)


class LLMTimeoutError(LLMError):
    """LLM request timeout error."""
    
    def __init__(
        self, 
        message: str,
        model: str,
        timeout_seconds: int,
        estimated_tokens: Optional[int] = None,
        **kwargs
    ):
        super().__init__(
            message,
            category=ErrorCategory.LLM_TIMEOUT,
            **kwargs
        )
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.estimated_tokens = estimated_tokens
        
        # Add default recovery strategies
        self.add_recovery_strategy(RecoveryStrategy(
            action="increase_timeout",
            description=f"Increase timeout from {timeout_seconds}s to {timeout_seconds * 2}s",
            params={"new_timeout": timeout_seconds * 2}
        ))
        self.add_recovery_strategy(RecoveryStrategy(
            action="use_streaming",
            description="Use streaming mode for long responses",
            params={"stream": True}
        ))
        self.add_recovery_strategy(RecoveryStrategy(
            action="reduce_context",
            description="Reduce context size to decrease response time",
            params={"max_context_tokens": 2000}
        ))


class LLMRateLimitError(LLMError):
    """LLM rate limit error."""
    
    def __init__(
        self,
        message: str,
        model: str,
        retry_after: Optional[int] = None,
        **kwargs
    ):
        super().__init__(
            message,
            category=ErrorCategory.LLM_RATE_LIMIT,
            **kwargs
        )
        self.model = model
        self.retry_after = retry_after
        
        # Add recovery strategies
        if retry_after:
            self.add_recovery_strategy(RecoveryStrategy(
                action="wait_and_retry",
                description=f"Wait {retry_after}s before retrying",
                params={"wait_seconds": retry_after}
            ))
        self.add_recovery_strategy(RecoveryStrategy(
            action="use_fallback_model",
            description="Use a fallback model with lower rate limits",
            params={"fallback_models": ["gpt-3.5-turbo", "claude-instant"]}
        ))


class DataError(StructuredError):
    """Base class for data-related errors."""
    
    def __init__(self, message: str, category: ErrorCategory = ErrorCategory.DATA_INVALID_FORMAT, **kwargs):
        super().__init__(message, category, **kwargs)


class DataNotFoundError(DataError):
    """Data not found error."""
    
    def __init__(
        self,
        message: str,
        data_type: str,
        identifier: str,
        **kwargs
    ):
        super().__init__(
            message,
            category=ErrorCategory.DATA_NOT_FOUND,
            **kwargs
        )
        self.data_type = data_type
        self.identifier = identifier
        
        # Add recovery strategies
        self.add_recovery_strategy(RecoveryStrategy(
            action="check_path",
            description="Verify the data path is correct",
            params={"suggested_paths": []}
        ))
        self.add_recovery_strategy(RecoveryStrategy(
            action="rebuild_data",
            description=f"Rebuild {data_type} from source",
            params={"data_type": data_type}
        ))


class ConfigError(StructuredError):
    """Base class for configuration errors."""
    
    def __init__(self, message: str, category: ErrorCategory = ErrorCategory.CONFIG_INVALID, **kwargs):
        super().__init__(message, category, **kwargs)


class ConfigMissingError(ConfigError):
    """Configuration missing error."""
    
    def __init__(
        self,
        message: str,
        config_key: str,
        config_file: Optional[str] = None,
        **kwargs
    ):
        super().__init__(
            message,
            category=ErrorCategory.CONFIG_MISSING,
            **kwargs
        )
        self.config_key = config_key
        self.config_file = config_file
        
        # Add recovery strategies
        self.add_recovery_strategy(RecoveryStrategy(
            action="use_default",
            description=f"Use default value for {config_key}",
            params={"default_value": None}
        ))
        if config_file:
            self.add_recovery_strategy(RecoveryStrategy(
                action="create_config",
                description=f"Create {config_file} from template",
                params={"template_file": f"{config_file}.example"}
            ))


class ToolError(StructuredError):
    """Base class for tool-related errors."""
    
    def __init__(self, message: str, category: ErrorCategory = ErrorCategory.TOOL_EXECUTION_FAILED, **kwargs):
        super().__init__(message, category, **kwargs)


class ToolExecutionError(ToolError):
    """Tool execution error."""
    
    def __init__(
        self,
        message: str,
        tool_name: str,
        tool_params: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        super().__init__(
            message,
            category=ErrorCategory.TOOL_EXECUTION_FAILED,
            **kwargs
        )
        self.tool_name = tool_name
        self.tool_params = tool_params
        
        # Add recovery strategies
        self.add_recovery_strategy(RecoveryStrategy(
            action="retry_with_defaults",
            description=f"Retry {tool_name} with default parameters",
            params={"use_defaults": True}
        ))
        self.add_recovery_strategy(RecoveryStrategy(
            action="use_alternative_tool",
            description="Use an alternative tool for the same purpose",
            params={"alternatives": []}
        ))


class GraphError(StructuredError):
    """Base class for graph-related errors."""
    
    def __init__(self, message: str, category: ErrorCategory = ErrorCategory.GRAPH_INVALID_OPERATION, **kwargs):
        super().__init__(message, category, **kwargs)


class GraphNotBuiltError(GraphError):
    """Graph not built error."""
    
    def __init__(
        self,
        message: str,
        graph_type: str,
        dataset: str,
        **kwargs
    ):
        super().__init__(
            message,
            category=ErrorCategory.GRAPH_NOT_BUILT,
            **kwargs
        )
        self.graph_type = graph_type
        self.dataset = dataset
        
        # Add recovery strategies
        self.add_recovery_strategy(RecoveryStrategy(
            action="build_graph",
            description=f"Build {graph_type} graph for {dataset}",
            params={
                "graph_type": graph_type,
                "dataset": dataset
            }
        ))
        self.add_recovery_strategy(RecoveryStrategy(
            action="load_cached",
            description="Load pre-built graph from cache",
            params={"cache_path": f"./results/{dataset}/{graph_type}"}
        ))


class IndexError(StructuredError):
    """Base class for index-related errors."""
    
    def __init__(self, message: str, category: ErrorCategory = ErrorCategory.INDEX_QUERY_FAILED, **kwargs):
        super().__init__(message, category, **kwargs)


class IndexNotBuiltError(IndexError):
    """Index not built error."""
    
    def __init__(
        self,
        message: str,
        index_type: str,
        data_source: str,
        **kwargs
    ):
        super().__init__(
            message,
            category=ErrorCategory.INDEX_NOT_BUILT,
            **kwargs
        )
        self.index_type = index_type
        self.data_source = data_source
        
        # Add recovery strategies
        self.add_recovery_strategy(RecoveryStrategy(
            action="build_index",
            description=f"Build {index_type} index from {data_source}",
            params={
                "index_type": index_type,
                "data_source": data_source
            }
        ))
        self.add_recovery_strategy(RecoveryStrategy(
            action="use_fallback_index",
            description="Use a simpler index type as fallback",
            params={"fallback_type": "faiss"}
        ))


# Error handler utility

class ErrorHandler:
    """Utility class for handling structured errors."""
    
    @staticmethod
    def handle_error(
        error: StructuredError,
        logger = None,
        auto_recover: bool = False
    ) -> Optional[Any]:
        """
        Handle a structured error.
        
        Args:
            error: The structured error to handle
            logger: Logger instance to use
            auto_recover: Whether to attempt automatic recovery
            
        Returns:
            Recovery result if auto_recover is True and recovery succeeds
        """
        # Log the error with appropriate level
        if logger:
            log_method = getattr(logger, error.severity.value, logger.error)
            log_method(
                f"{error.category.value}: {error.message}",
                extra=error.to_dict()
            )
        
        # Display recovery strategies
        if error.recovery_strategies:
            if logger:
                logger.info("Available recovery strategies:")
                for i, strategy in enumerate(error.recovery_strategies):
                    logger.info(f"  {i+1}. {strategy.action}: {strategy.description}")
        
        # Attempt auto-recovery if enabled
        if auto_recover and error.recovery_strategies:
            # Try the first strategy by default
            # In a real implementation, this would execute the recovery action
            first_strategy = error.recovery_strategies[0]
            if logger:
                logger.info(f"Attempting automatic recovery: {first_strategy.action}")
            # Return recovery params for the caller to handle
            return first_strategy
        
        return None