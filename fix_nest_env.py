"""Patch a conda-forge NEST install so that NESTML models compile.

Run once, inside the activated conda environment, after installing the packages:

    python fix_nest_env.py

It fixes two problems of the conda-forge ``nest-simulator`` package:
1. ``nest-config --compiler`` points to the compiler of the conda-forge build machine,
   which does not exist on your computer. NESTML uses it to compile models.
2. NEST's ``numerics.h`` uses ``size_t`` without including ``<cstddef>``, which recent
   compilers reject.
Both files are backed up (``.orig``) before being changed. Running it twice is harmless.
"""
import os
import re
import shutil
import sys

prefix = sys.prefix
nest_config = os.path.join(prefix, "bin", "nest-config")
numerics_h = os.path.join(prefix, "include", "nest", "numerics.h")

for path in (nest_config, numerics_h):
    if not os.path.isfile(path):
        sys.exit(f"Not found: {path}\nActivate the conda environment that contains NEST and run again.")


def backup(path):
    if not os.path.exists(path + ".orig"):
        shutil.copy2(path, path + ".orig")


# 1. compiler path in nest-config
text = open(nest_config).read()
match = re.search(r'--compiler\)\s*\n\s*echo "([^"]+)"', text)
if match is None:
    sys.exit("Could not find the compiler entry in nest-config; please fix it by hand.")
compiler = match.group(1)
if os.path.isfile(compiler):
    print(f"nest-config compiler OK: {compiler}")
else:
    local = os.path.join(prefix, "bin", os.path.basename(compiler))
    if not os.path.isfile(local):
        sys.exit(f"Compiler {local} not found. Install it with: conda install -c conda-forge cxx-compiler")
    backup(nest_config)
    open(nest_config, "w").write(text.replace(compiler, local))
    print(f"nest-config compiler fixed: {compiler}\n                        -> {local}")

# 2. missing <cstddef> in numerics.h
text = open(numerics_h).read()
if "#include <cstddef>" in text:
    print("numerics.h OK")
else:
    backup(numerics_h)
    open(numerics_h, "w").write(text.replace("#include <cassert>", "#include <cassert>\n#include <cstddef>\nusing std::size_t;", 1))
    print("numerics.h fixed (added #include <cstddef>)")

print("Done. NESTML models can now be compiled in this environment.")
