"""
Progress bar utilities for long-running operations.
Uses tqdm if available, falls back to simple logging.
"""

import sys
from typing import Optional, Iterable, Any
import time

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

from .LoggerConfig import get_logger

logger = get_logger(__name__)


class ProgressBar:
    """Progress bar wrapper that gracefully degrades if tqdm not available."""
    
    def __init__(
        self,
        iterable: Optional[Iterable] = None,
        total: Optional[int] = None,
        desc: Optional[str] = None,
        unit: str = "it",
        disable: bool = False,
        leave: bool = True,
        ncols: Optional[int] = None,
        mininterval: float = 0.1,
        maxinterval: float = 10.0,
        miniters: Optional[int] = None,
        ascii: Optional[bool] = None,
        position: Optional[int] = None,
        bar_format: Optional[str] = None,
        postfix: Optional[dict] = None,
        unit_scale: Optional[bool] = None,
        dynamic_ncols: bool = True
    ):
        """
        Initialize progress bar.
        
        Args match tqdm interface for compatibility.
        """
        self.desc = desc or ""
        self.total = total
        self.unit = unit
        self.disable = disable
        self.leave = leave
        self._count = 0
        self._start_time = time.time()
        self._last_update = 0
        
        if TQDM_AVAILABLE and not disable:
            self._tqdm = tqdm(
                iterable=iterable,
                total=total,
                desc=desc,
                unit=unit,
                disable=disable,
                leave=leave,
                ncols=ncols,
                mininterval=mininterval,
                maxinterval=maxinterval,
                miniters=miniters,
                ascii=ascii,
                position=position,
                bar_format=bar_format,
                postfix=postfix,
                unit_scale=unit_scale,
                dynamic_ncols=dynamic_ncols
            )
        else:
            self._tqdm = None
            self._iterable = iterable
            if not disable and total:
                logger.info(f"{desc}: Starting (total={total} {unit}s)")
    
    def __iter__(self):
        """Iterate with progress updates."""
        if self._tqdm is not None:
            return iter(self._tqdm)
        elif self._iterable is not None:
            for item in self._iterable:
                self.update(1)
                yield item
        else:
            return iter([])
    
    def __enter__(self):
        """Context manager entry."""
        if self._tqdm is not None:
            self._tqdm.__enter__()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if self._tqdm is not None:
            self._tqdm.__exit__(exc_type, exc_val, exc_tb)
        else:
            self.close()
    
    def update(self, n: int = 1):
        """Update progress by n steps."""
        self._count += n
        
        if self._tqdm is not None:
            self._tqdm.update(n)
        elif not self.disable:
            current_time = time.time()
            # Log update every second
            if current_time - self._last_update >= 1.0:
                elapsed = current_time - self._start_time
                rate = self._count / elapsed if elapsed > 0 else 0
                
                if self.total:
                    pct = self._count / self.total * 100
                    eta = (self.total - self._count) / rate if rate > 0 else 0
                    logger.info(
                        f"{self.desc}: {self._count}/{self.total} "
                        f"({pct:.1f}%) [{rate:.1f} {self.unit}/s, "
                        f"ETA: {eta:.0f}s]"
                    )
                else:
                    logger.info(
                        f"{self.desc}: {self._count} {self.unit}s "
                        f"[{rate:.1f} {self.unit}/s]"
                    )
                self._last_update = current_time
    
    def set_description(self, desc: str, refresh: bool = True):
        """Update description."""
        self.desc = desc
        if self._tqdm is not None:
            self._tqdm.set_description(desc, refresh=refresh)
    
    def set_postfix(self, ordered_dict=None, refresh=True, **kwargs):
        """Set postfix dictionary."""
        if self._tqdm is not None:
            self._tqdm.set_postfix(ordered_dict, refresh=refresh, **kwargs)
    
    def close(self):
        """Close progress bar."""
        if self._tqdm is not None:
            self._tqdm.close()
        elif not self.disable:
            elapsed = time.time() - self._start_time
            rate = self._count / elapsed if elapsed > 0 else 0
            logger.info(
                f"{self.desc}: Completed {self._count} {self.unit}s "
                f"in {elapsed:.1f}s ({rate:.1f} {self.unit}/s)"
            )
    
    def clear(self):
        """Clear progress bar display."""
        if self._tqdm is not None:
            self._tqdm.clear()
    
    def refresh(self):
        """Force refresh display."""
        if self._tqdm is not None:
            self._tqdm.refresh()
    
    def reset(self, total=None):
        """Reset progress bar."""
        self._count = 0
        self._start_time = time.time()
        if self._tqdm is not None:
            self._tqdm.reset(total=total)
        elif total is not None:
            self.total = total
    
    @property
    def n(self):
        """Current progress count."""
        if self._tqdm is not None:
            return self._tqdm.n
        return self._count
    
    def write(self, s: str, file=None, end="\n", nolock=False):
        """Write message without interfering with progress bar."""
        if self._tqdm is not None:
            self._tqdm.write(s, file=file, end=end, nolock=nolock)
        else:
            logger.info(s)


def progress_bar(*args, **kwargs) -> ProgressBar:
    """Create a progress bar (alias for ProgressBar constructor)."""
    return ProgressBar(*args, **kwargs)


def track(
    iterable: Iterable,
    description: str = "Processing",
    total: Optional[int] = None,
    disable: bool = False
) -> ProgressBar:
    """
    Track progress of an iterable.
    
    Args:
        iterable: Items to iterate over
        description: Progress bar description
        total: Total number of items (auto-detected if possible)
        disable: Whether to disable progress bar
        
    Returns:
        ProgressBar wrapping the iterable
    """
    # Try to get total from iterable
    if total is None and hasattr(iterable, "__len__"):
        try:
            total = len(iterable)
        except:
            pass
    
    return ProgressBar(
        iterable=iterable,
        desc=description,
        total=total,
        disable=disable
    )


# Convenience functions for common use cases

def track_batch_processing(
    items: list,
    batch_size: int,
    description: str = "Processing batches"
) -> ProgressBar:
    """Track batch processing progress."""
    num_batches = (len(items) + batch_size - 1) // batch_size
    return ProgressBar(
        total=len(items),
        desc=description,
        unit="item"
    )


def track_async_tasks(
    num_tasks: int,
    description: str = "Processing tasks"
) -> ProgressBar:
    """Track async task completion."""
    return ProgressBar(
        total=num_tasks,
        desc=description,
        unit="task"
    )