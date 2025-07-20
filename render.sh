#!/bin/bash -i
jupyter nbconvert --to html big_bird.ipynb
html_pdf big_bird.html
rm big_bird.html