from importlib.metadata import version

from .editor import Editor
from .ready import ReadyPipeline
from .validcomplex import ValidComplex
from .diagnosis import Diagnosis

__version__  = version('mdworks')