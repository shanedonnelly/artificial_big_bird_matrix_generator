#!/bin/bash -i
jupyter nbconvert --to html --TagRemovePreprocessor.remove_cell_tags='{"hide_code"}' big_bird.ipynb
html_pdf big_bird.html
# rm big_bird.html