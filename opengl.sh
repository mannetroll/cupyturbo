#!/bin/bash

python -m pip install -U pip
python -m pip install -U PyQt6 PyQt6-Qt6 PyQt6-sip

python -c "import PyQt6.QtOpenGL, PyQt6.QtOpenGLWidgets; print('QtOpenGL OK')"
python -c "import PyQt6.QtGui as g; print('QOpenGLFunctions in QtGui:', hasattr(g,'QOpenGLFunctions'))"
python -c "import PyQt6.QtOpenGL as o; print([n for n in dir(o) if 'QOpenGLFunctions' in n])"
python -c "from PyQt6.QtOpenGLWidgets import QOpenGLWidget; print('QOpenGLWidget OK')"
