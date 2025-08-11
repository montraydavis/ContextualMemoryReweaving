class CMRTransformer:
    """Minimal stub for the base transformer used in demos/tests.

    This is intentionally lightweight to provide a stable import path.
    Full behavior should be implemented separately.
    """

    def __init__(self, config, memory_config=None):
        self.config = config
        self.memory_config = memory_config or {}


