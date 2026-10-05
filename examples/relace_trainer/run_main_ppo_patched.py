#!/usr/bin/env python3
"""
Wrapper script that applies the WebshopEnvironmentManager patch before running main_ppo.

This ensures the patch is active in the same process where the training runs.
"""

import sys
import os
import runpy

# Add paths
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.dirname(__file__))

# Apply the patch BEFORE importing anything that might create a WebshopEnvironmentManager
import patch_env_manager
patch_env_manager.apply_patch()

# Now run the main_ppo module using runpy (same as python -m)
# This ensures proper Hydra argument handling
if __name__ == "__main__":
    sys.argv[0] = "verl.trainer.main_ppo"  # Set the module name for proper arg parsing
    runpy.run_module("verl.trainer.main_ppo", run_name="__main__", alter_sys=True)
