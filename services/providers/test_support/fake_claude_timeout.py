#!/usr/bin/env python3
import os
import pathlib
import subprocess
import sys
import time

ready = pathlib.Path(os.environ["READY_FILE"])
child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
ready.write_text(f"{os.getpid()}:{child.pid}")
time.sleep(30)
