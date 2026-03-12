import os
import sys
import shutil

# Apply AMD Compatibility Layer before any other imports
try:
    from .shared_utils.compatibility_layer import apply_compatibility_layer
    apply_compatibility_layer()
except Exception as e:
    import traceback
    print(f"[Comfy3D] Critical: Failed to apply AMD compatibility layer: {e}")
    traceback.print_exc()

import folder_paths as comfy_paths
from pyhocon import ConfigFactory
import logging

ROOT_PATH = os.path.dirname(os.path.realpath(__file__))
MODULE_PATH = os.path.join(ROOT_PATH, "Gen_3D_Modules")
MV_ALGO_PATH = os.path.join(ROOT_PATH, "MVs_Algorithms")

sys.path.append(ROOT_PATH)
sys.path.append(MODULE_PATH)
sys.path.append(MV_ALGO_PATH)

import __main__
import importlib
import inspect
from .webserver.server import server, set_web_conf
from .shared_utils.log_utils import setup_logger

conf_path = os.path.join(ROOT_PATH, "Configs/system.conf")
# Configuration
with open(conf_path) as f:
    conf_text = f.read()
sys_conf = ConfigFactory.parse_string(conf_text)

set_web_conf(sys_conf['web'])

# Log into huggingface if given user specificed token
hf_token = sys_conf['huggingface.token']
if isinstance(hf_token, str) and len(hf_token) > 0:
    from huggingface_hub import login
    login(token=hf_token)

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

print("[Comfy3D] Importing nodes.py")
nodes_filename = "nodes"
try:
    module = importlib.import_module(f".{nodes_filename}", package=__name__)
    print("[Comfy3D] nodes.py imported successfully")
except Exception as e:
    import traceback
    print(f"[Comfy3D] ERROR: Failed to import nodes.py: {e}")
    traceback.print_exc()
    module = None

if module:
    for name, cls in inspect.getmembers(module, inspect.isclass):
        if cls.__module__ == module.__name__:
            name = name.replace("_", " ")

            node = f"[Comfy3D] {name}"
            disp = f"{name}"

            NODE_CLASS_MAPPINGS[node] = cls
            NODE_DISPLAY_NAME_MAPPINGS[node] = disp
else:
    print("[Comfy3D] Skipping node registration due to import error")

WEB_DIRECTORY = "./web"
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', 'WEB_DIRECTORY']
# Cleanup old extension folder
try:
    if "__file__" in __main__.__dict__:
        folder_web = os.path.join(os.path.dirname(os.path.realpath(__main__.__file__)), "web")
    else:
        # Fallback if __main__ has no __file__ (e.g. some headless modes)
        folder_web = os.path.join(os.path.dirname(ROOT_PATH), "..", "web")

    extensions_folder = os.path.join(folder_web, 'extensions', 'ComfyUI-3D-Pack')

    def cleanup():
        if os.path.exists(extensions_folder):
            try:
                shutil.rmtree(extensions_folder)
                print('\033[34mComfy3D: \033[92mRemoved old extension folder\033[0m')
            except:
                pass

    cleanup()
except Exception as e:
    print(f"[Comfy3D] Warning during cleanup: {e}")

