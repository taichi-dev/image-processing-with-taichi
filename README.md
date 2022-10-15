# (GPU) Image processing with Taichi

Every Python file in this repo is runnable if you `pip3 install -U taichi opencv-python`.

For example, if you run `python3 bilateral_grid_hdr.py`, you get the following UI:

![](images/bilateral_grid_hdr.jpg)

### Developer note: enforcing code format

We use the `pre-commit` Python package, which invokes `yapf` automatically format Python code.

Usage:
1. Install `pre-commit`: `pip install pre-commit`.
2. Run code format: `pre-commit run -a`.
3. Install as pre-commit hook: `pre-commit install`.
