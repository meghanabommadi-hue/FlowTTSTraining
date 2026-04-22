import os
import sys


class Tee:
    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()

    def flush(self):
        for f in self.files:
            f.flush()


def setup(log_path: str):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    f = open(log_path, "w")
    sys.stdout = Tee(sys.stdout, f)
    sys.stderr = Tee(sys.stderr, f)
    return f
