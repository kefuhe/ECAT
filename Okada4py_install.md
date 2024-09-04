## Install the [okada4py](https://github.com/jolivetr/okada4py) from Romain Jolivet in ***myecat***

```bash
cd path_to_okada4py
export CC=gcc
python setup.py build
python setup.py install --user --prefix=
```

After installing `okada4py` in the `myecat` environment (activated using `conda activate myecat`), the package should be located in the following directories depending on the operating system:

- **Linux**: The package should be located in a directory similar to `./.local/lib/python3.10/site-packages/okada4py`.
- **Windows**: The package should be located in a directory similar to `python310\Lib\site-packages\okada4py`.

**Note**: When using the `--prefix` option, make sure there is no content after it, including spaces.

3. If you encounter the issue of not finding the `okada4py` package during installation, you can solve it by following these steps (from commit of **[wenyuyangit](https://github.com/wenyuyangit)**):

   1. Clone the project to `~/anaconda3/envs/cutde/lib/python3.10/site-packages`, so you would have a folder named `okada4py`.
   2. Run `python setup.py build` and `python setup.py install`, so you could get `okada4py-12.0.2-py3.12-linux-x86_64.egg` in "`~/.local/lib/python3.10/site-packages/`".
   3. Copy all files in `okada4py-12.0.2-py3.12-linux-x86_64.egg` to the folder `okada4py`.
   4. Run `test.py`, and you can get two figures and no error toasted.
