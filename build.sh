#!/bin/bash

# brute force...
git clone https://github.com/ebridge2/graspologic.git /graspologic
cp /graspologic/graspologic/layouts/include/colors-100.json /opt/hostedtoolcache/Python/3.8.12/x64/lib/python3.8/site-packages/graspologic/layouts/include/colors-100.json
# actual stuff...
jupyter-book build network_machine_learning_in_python/
open network_machine_learning_in_python/_build/html/index.html 
