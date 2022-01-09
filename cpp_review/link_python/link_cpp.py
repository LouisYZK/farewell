import ctypes
import sys
import os

def load_library(path, name):
    if sys.platform == 'win32':       # *.dll
        return ctypes.cdll.LoadLibrary(os.path.join(path, name + '.dll'))
    elif sys.platform == 'linux':     # lib*.so
        return ctypes.cdll.LoadLibrary(os.path.join(path, 'lib' + name + '.so'))
    elif sys.platform == 'darwin':    # lib*.dylib
        return ctypes.cdll.LoadLibrary(os.path.join(path, 'lib' + name + '.dylib'))
    else:
        raise ImportError('Unsupported platform: ' + sys.platform)

mylib = load_library('build', 'mylib')
mylib.print_hello()

mylib.double_int.argtypes = [ctypes.c_int]
mylib.double_int.restype = ctypes.c_int
mylib.double_float.argtypes = [ctypes.c_float]
mylib.double_float.restype = ctypes.c_float
mylib.print_str.argtypes = [ctypes.c_char_p]
mylib.print_str.restype = None

print (mylib.double_int(10))
print (mylib.double_float(3.14))
mylib.print_str(b'Hello Ctypes!') # char p should be in bytes

import numpy as np
arr = np.random.rand(32).astype(np.float32)
print (type(arr.ctypes.data))

mylib.test_numpy.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
mylib.test_numpy.restype = None
mylib.test_numpy(arr.ctypes.data, arr.shape[0])

