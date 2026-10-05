#!/usr/bin/env python3
"""
Monkey-patch for WebshopEnvironmentManager.extract_task to handle multiple observation formats.

This must be imported BEFORE any code that creates a WebshopEnvironmentManager instance.
"""

import logging
from typing import List


def robust_extract_task(self, text_obs: List[str]):
    """
    Robust task extraction that handles WebShop's [SEP]-delimited observations.

    WebShop observations use ' [SEP] ' as separators between HTML text elements.
    The instruction appears after "Instruction:" somewhere in the observation.

    This handles cases where "Instruction:" is not at a fixed position in the
    [SEP]-separated parts (the original code assumed parts[1]=='Instruction:',
    but the HTML structure can vary).
    """
    tasks = []
    for obs in text_obs:
        task = None

        # WebShop format: find "Instruction:" in [SEP]-delimited parts
        if " [SEP] " in obs:
            parts = obs.split(" [SEP] ")
            # Find the part that contains "Instruction:"
            for i, part in enumerate(parts):
                if part.strip() == 'Instruction:' and i + 1 < len(parts):
                    # The task is the next part after "Instruction:"
                    task = parts[i + 1].strip()
                    break
                elif part.startswith('Instruction:'):
                    # Sometimes "Instruction:" and the task are in the same part
                    task = part[len('Instruction:'):].strip()
                    break

        # Fallback: Look for "Instruction:" as a substring (no [SEP])
        if task is None and 'Instruction:' in obs:
            inst_start = obs.find('Instruction:')
            task_text = obs[inst_start + len('Instruction:'):].strip()
            # Extract until the next [SEP] or newline
            if ' [SEP] ' in task_text:
                task = task_text.split(' [SEP] ')[0].strip()
            elif '\n' in task_text:
                task = task_text.split('\n')[0].strip()
            else:
                task = task_text

        # Final fallback: extract first meaningful text
        if task is None or task == '':
            # Try to get something from the observation
            if " [SEP] " in obs:
                parts = obs.split(" [SEP] ")
                # Skip empty parts and get the first substantial text
                for part in parts:
                    if part.strip() and len(part.strip()) > 10:
                        task = part.strip()[:200]
                        break
            else:
                task = obs.split('\n')[0] if '\n' in obs else obs[:200]

            logging.warning(f"Could not extract task from observation, using fallback: {task[:100]}")

        tasks.append(task)
    return tasks


def apply_patch():
    """Apply the monkey-patch to WebshopEnvironmentManager."""
    try:
        from agent_system.environments.env_manager import WebshopEnvironmentManager

        # Store the original method in case we need it
        if not hasattr(WebshopEnvironmentManager, '_original_extract_task'):
            WebshopEnvironmentManager._original_extract_task = WebshopEnvironmentManager.extract_task

        # Apply the patch
        WebshopEnvironmentManager.extract_task = robust_extract_task

        logging.info("Successfully patched WebshopEnvironmentManager.extract_task")
        return True
    except Exception as e:
        logging.error(f"Failed to patch WebshopEnvironmentManager: {e}")
        return False


# Auto-apply patch when this module is imported
if __name__ != "__main__":
    apply_patch()
