import threading
import queue

class ConcurrentQueue:
    def __init__(self):
        self._queue = queue.Queue()

    def put(self, item):
            self._queue.put(item)

    def get(self, timeout=None):
        try:
            if self._queue.empty():
                return None
            return self._queue.get(timeout=timeout)
        except Exception as e:
            print("ConcurrentQueue: Get exception occurred." + e.__doc__ )
            return None

    def empty(self):
        return self._queue.empty()

