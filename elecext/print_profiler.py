# Print statement profiling and optimization
import time
import threading
import sys
import io
from collections import deque
import queue
from contextlib import contextmanager

class PrintProfiler:
    """Profile print statement overhead and contention."""
    
    def __init__(self):
        self.print_times = deque(maxlen=1000)
        self.contention_events = deque(maxlen=100)
        self.lock = threading.Lock()
        self.original_stdout = sys.stdout
        self.print_count = 0
        self.total_print_time = 0.0
        
    def profile_print(self, message):
        """Measure time taken by a print operation."""
        thread_id = threading.current_thread().ident
        start_time = time.perf_counter()
        
        # Detect contention by checking if other threads are waiting
        lock_acquired = self.lock.acquire(blocking=False)
        if not lock_acquired:
            # Another thread has the lock - this is contention
            contention_start = time.perf_counter()
            self.lock.acquire()  # Block until available
            contention_time = time.perf_counter() - contention_start
            self.contention_events.append({
                'thread': thread_id,
                'wait_time': contention_time,
                'timestamp': time.perf_counter()
            })
        
        try:
            # Actual print
            print(message)
            print_duration = time.perf_counter() - start_time
            
            self.print_times.append({
                'duration': print_duration,
                'thread': thread_id,
                'message_len': len(str(message)),
                'timestamp': time.perf_counter()
            })
            
            self.print_count += 1
            self.total_print_time += print_duration
            
        finally:
            self.lock.release()
    
    def get_stats(self):
        """Get print performance statistics."""
        if not self.print_times:
            return {}
        
        durations = [p['duration'] for p in self.print_times]
        contention_times = [c['wait_time'] for c in self.contention_events]
        
        return {
            'total_prints': self.print_count,
            'total_print_time': self.total_print_time,
            'avg_print_time': sum(durations) / len(durations),
            'max_print_time': max(durations),
            'min_print_time': min(durations),
            'contention_events': len(self.contention_events),
            'total_contention_time': sum(contention_times) if contention_times else 0,
            'avg_contention_time': sum(contention_times) / len(contention_times) if contention_times else 0,
            'max_contention_time': max(contention_times) if contention_times else 0
        }


# Global print profiler
print_profiler = PrintProfiler()


class AsyncLogger:
    """Thread-safe asynchronous logging system to replace print statements."""
    
    def __init__(self, max_buffer_size=1000):
        self.message_queue = queue.Queue(maxsize=max_buffer_size)
        self.worker_thread = None
        self.running = False
        self.lock = threading.Lock()
        
    def start(self):
        """Start the async logging worker thread."""
        with self.lock:
            if not self.running:
                self.running = True
                self.worker_thread = threading.Thread(
                    target=self._worker,
                    name="AsyncLogger",
                    daemon=True
                )
                self.worker_thread.start()
    
    def stop(self):
        """Stop the async logging worker thread."""
        with self.lock:
            if self.running:
                self.running = False
                self.message_queue.put(None)  # Signal to stop
                if self.worker_thread:
                    self.worker_thread.join(timeout=1.0)
    
    def log(self, message, level="INFO"):
        """Log a message asynchronously (non-blocking)."""
        if not self.running:
            self.start()
        
        timestamp = time.perf_counter()
        thread_name = threading.current_thread().name
        thread_id = threading.current_thread().ident
        
        log_entry = {
            'timestamp': timestamp,
            'thread_name': thread_name,
            'thread_id': thread_id,
            'level': level,
            'message': message
        }
        
        try:
            # Non-blocking put - drops message if buffer full (prevents blocking worker threads)
            self.message_queue.put_nowait(log_entry)
        except queue.Full:
            # Buffer full - print directly as fallback (better than losing message)
            print(f"[BUFFER FULL] {message}")
    
    def _worker(self):
        """Worker thread that processes log messages."""
        while self.running:
            try:
                # Block for up to 0.1s waiting for messages
                log_entry = self.message_queue.get(timeout=0.1)
                
                if log_entry is None:  # Stop signal
                    break
                
                # Format and print the message
                formatted_msg = self._format_message(log_entry)
                print(formatted_msg, flush=True)
                
                self.message_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"AsyncLogger error: {e}")
    
    def _format_message(self, log_entry):
        """Format a log message."""
        return f"[{log_entry['thread_name']:>15}] {log_entry['message']}"
    
    def flush(self):
        """Wait for all queued messages to be processed."""
        if self.running:
            self.message_queue.join()


# Global async logger instance
async_logger = AsyncLogger()


class SmartPrint:
    """Smart print replacement that can switch between modes."""
    
    def __init__(self):
        self.mode = "profiled"  # "profiled", "async", "disabled", "original"
        self.async_logger = async_logger
        
    def __call__(self, message):
        """Handle print calls based on current mode."""
        if self.mode == "profiled":
            print_profiler.profile_print(message)
        elif self.mode == "async":
            self.async_logger.log(message)
        elif self.mode == "disabled":
            pass  # Drop the message
        else:  # original
            print(message)
    
    def set_mode(self, mode):
        """Set the print mode: 'profiled', 'async', 'disabled', 'original'."""
        self.mode = mode
        if mode == "async":
            self.async_logger.start()
    
    def get_stats(self):
        """Get performance statistics."""
        if self.mode == "profiled":
            return print_profiler.get_stats()
        return {}
    
    def flush(self):
        """Flush pending messages."""
        if self.mode == "async":
            self.async_logger.flush()


# Global smart print instance
smart_print = SmartPrint()


# Context manager for temporary print mode changes
@contextmanager
def print_mode(mode):
    """Temporarily change print mode."""
    old_mode = smart_print.mode
    smart_print.set_mode(mode)
    try:
        yield smart_print
    finally:
        smart_print.set_mode(old_mode)


def analyze_print_overhead():
    """Analyze and report print statement overhead."""
    stats = smart_print.get_stats()
    
    if not stats:
        print("No print profiling data available")
        return
    
    print("\n" + "="*60)
    print("🖨️  PRINT STATEMENT PERFORMANCE ANALYSIS")
    print("="*60)
    
    print(f"Total print statements: {stats['total_prints']}")
    print(f"Total time spent in print(): {stats['total_print_time']:.3f}s")
    print(f"Average print time: {stats['avg_print_time']*1000:.2f}ms")
    print(f"Max print time: {stats['max_print_time']*1000:.2f}ms")
    print(f"Min print time: {stats['min_print_time']*1000:.2f}ms")
    
    if stats['contention_events'] > 0:
        print(f"\n🚫 STDOUT CONTENTION DETECTED:")
        print(f"Contention events: {stats['contention_events']}")
        print(f"Total time lost to contention: {stats['total_contention_time']:.3f}s")
        print(f"Average contention wait: {stats['avg_contention_time']*1000:.2f}ms")
        print(f"Max contention wait: {stats['max_contention_time']*1000:.2f}ms")
        
        efficiency_loss = (stats['total_contention_time'] / stats['total_print_time']) * 100
        print(f"Efficiency loss due to contention: {efficiency_loss:.1f}%")
    else:
        print("✅ No stdout contention detected")
    
    # Performance recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    if stats['avg_print_time'] > 0.001:  # > 1ms average
        print("• Print statements are slow - consider async logging")
    if stats['contention_events'] > stats['total_prints'] * 0.1:  # >10% contention
        print("• High stdout contention - switch to async logging immediately")
    if stats['total_print_time'] > 0.1:  # >100ms total
        print("• Significant time spent in print() - optimize logging strategy")
    
    print("="*60)