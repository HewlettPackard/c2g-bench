# c2g_env/simulators/__init__.py — backward-compatibility shim
# The physics engines have moved to c2g_env.physics.
# Old imports (``from c2g_env.simulators import X``) still work transparently.
# Use ``c2g_env.physics`` in new code.
from c2g_env.physics.bess import *        # noqa: F401, F403
from c2g_env.physics.thermal import *     # noqa: F401, F403
from c2g_env.physics.electrical import *  # noqa: F401, F403
from c2g_env.physics.macro_grid import *  # noqa: F401, F403
from c2g_env.physics.renewable import *   # noqa: F401, F403
from c2g_env.physics.weather import *     # noqa: F401, F403
from c2g_env.physics.workload import *    # noqa: F401, F403
