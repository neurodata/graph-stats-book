# Network Machine Learning in Python

This book provides an introduction to graph statistics, with a focus on useful representations of graphs and their applications on real data.

[Google Drive Brainstorm](https://drive.google.com/drive/folders/1mcMQWv0WfydQSdgchhla7XHj2PBPD-b9?usp=sharing)
[Book Proposal](https://docs.google.com/document/d/1VTlNaogB-WPyex9LYVh7PvmED9S8rNqHoTXlTre_ASA/edit?usp=sharing)


## Usage

### Building the book

If you'd like to develop on and build the Network Machine Learning in Python book, you should:

- Clone this repository and run
- Run `pip install -r requirements.txt` (it is recommended you do this within a virtual environment)
- (Recommended) Remove the existing `Network Machine Learning in Python/_build/` directory
- Run `jupyter-book build Network Machine Learning in Python/`

A fully-rendered HTML version of the book will be built in `Network Machine Learning in Python/_build/html/`.

### Hosting the book

The html version of the book is hosted on the `gh-pages` branch of this repo. A GitHub actions workflow has been created that automatically builds and pushes the book to this branch on a push or pull request to main.

If you wish to disable this automation, you may remove the GitHub actions workflow and build the book manually by:

- Navigating to your local build; and running,
- `ghp-import -n -p -f Network Machine Learning in Python/_build/html`

This will automatically push your build to the `gh-pages` branch. More information on this hosting process can be found [here](https://jupyterbook.org/publish/gh-pages.html#manually-host-your-book-with-github-pages).

## Contributors

We welcome and recognize all contributions. You can see a list of current contributors in the [contributors tab](https://github.com/jovo/network_machine_learning_in_python/graphs/contributors).

## Credits

This project is created using the excellent open source [Jupyter Book project](https://jupyterbook.org/) and the [executablebooks/cookiecutter-jupyter-book template](https://github.com/executablebooks/cookiecutter-jupyter-book).
