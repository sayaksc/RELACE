# Copyright 2025 Nanyang Technological University (NTU), Singapore
# and the verl-agent (GiGPO) team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# --------------------- WebShop --------------------- #
WEBSHOP_TEMPLATE_NO_HIS = """
You are an expert autonomous agent operating in the WebShop e‑commerce environment. 
Your task is to: {task_description}.
Your current observation is: {current_observation}.
Your admissible actions of the current situation are: 
[
{available_actions}
].

Now it's your turn to take one action for the current step.
You should first reason step-by-step about the current situation, then think carefully which admissible action best advances the shopping goal. This reasoning process MUST be enclosed within <think> </think> tags. 
Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags.
"""

WEBSHOP_TEMPLATE = """
You are an expert autonomous agent operating in the WebShop e‑commerce environment.
Your task is to: {task_description}.
Prior to this step, you have already taken {step_count} step(s). Below are the most recent {history_length} observations and the corresponding actions you took: {action_history}
You are now at step {current_step} and your current observation is: {current_observation}.
Your admissible actions of the current situation are: 
[
{available_actions}
].

Now it's your turn to take one action for the current step.
You should first reason step-by-step about the current situation, then think carefully which admissible action best advances the shopping goal. This reasoning process MUST be enclosed within <think> </think> tags. 
Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags.
"""


WEBSHOP_TEMPLATE_HINDSIGHT_YES_NO = """
You are an expert evaluator for an autonomous shopping agent operating in the WebShop environment.

The episode has now ended. FINAL OBSERVATION: {outcome}.

The agent took {step_count} step(s) in total. Below are the most recent {history_length} observation–action pairs leading up to the current step:
{action_history}

At step {current_step}, CURRENT OBSERVATION:
{current_observation}

The admissible actions at this step were:
[
{available_actions}
]

The agent chose CURRENT ACTION: {current_action}

Your task is to assess whether this specific action meaningfully contributed to achieving the goal.

Think step-by-step and answer the question:
Was the CURRENT ACTION necessary for the agent to make genuine progress from the CURRENT OBSERVATION to the FINAL OBSERVATION?

Your reasoning MUST be enclosed within <think> </think> tags.
Then provide your final answer within <answer> </answer> tags using ONLY Yes or No.
"""