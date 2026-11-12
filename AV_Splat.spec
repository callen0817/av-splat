# Build on the target OS: python -m PyInstaller --noconfirm AV_Splat.spec
import os
os.environ['PYQTGRAPH_QT_LIB']='PySide6'
from PyInstaller.utils.hooks import collect_all, collect_submodules, copy_metadata

datas=[('logos','logos'),('licenses','licenses')]
for dependency in ['PySide6','PySide6_Essentials','PySide6_Addons','shiboken6']:
    datas += copy_metadata(dependency)
binaries=[]
hiddenimports=collect_submodules('av_common') + collect_submodules('core') + collect_submodules('gui') + collect_submodules('formats') + collect_submodules('batch')
for package in ['pye57','pyqtgraph','OpenGL','spz']:
    data, binary, hidden = collect_all(package, filter_submodules=lambda name: not any(part in name.split(".") for part in ["examples", "tests", "ml"]))
    datas += data
    binaries += binary
    hiddenimports += hidden

a=Analysis(['launch.py'],pathex=['.'],binaries=binaries,datas=datas,hiddenimports=hiddenimports,
    excludes=['torch','tensorflow','tensorboard','bpy','PyQt5','PyQt6','pytest','IPython','jupyterlab','notebook'],noarchive=False)
pyz=PYZ(a.pure)
exe=EXE(pyz,a.scripts,[],exclude_binaries=True,name='AV_Splat',debug=False,icon='logos/av_splat.png' if os.path.exists('logos/av_splat.png') else None,
    bootloader_ignore_signals=False,strip=False,upx=False,console=True,disable_windowed_traceback=False)
coll=COLLECT(exe,a.binaries,a.datas,strip=False,upx=False,name='AV_Splat')
