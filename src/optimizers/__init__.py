from .nag import NAG
from .sgd import SGD
from .adam import ADAM
from .adam_max import ADAM_MAX
from .sgd_momentum import SGD_Momentum

from os.path import dirname, basename, isfile, join
import glob
modules = glob.glob(join(dirname(__file__), "*.py"))
__all__ = [ basename(f)[:-3] for f in modules if isfile(f) and not f.endswith('__init__.py')]
