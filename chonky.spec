# -*- mode: python ; coding: utf-8 -*-
"""Package Chonky and its dynamically loaded application dependencies."""
from pathlib import Path
from PyInstaller.utils.hooks import collect_all, collect_submodules, copy_metadata

root = Path( SPECPATH )
datas = [(str( root / 'app.py' ), '.'),
         (str( root / 'resources' ), 'resources'),
         (str( root / '.streamlit' / 'config.toml' ), '.streamlit')]
# Runtime databases are created in user storage; development data and secrets are excluded.
hiddenimports = ['config', 'boogr', 'loaders', 'processors', 'embedders', 'enums']
binaries = []
for package in ('streamlit', 'streamlit_extras', 'streamlit_pdf', 'webview',
                'altair', 'langchain', 'langchain_core', 'langchain_community',
                'langchain_text_splitters', 'langchain_google_community',
                'chromadb', 'sqlite_vec', 'sklearn', 'spacy', 'nltk',
                'torch', 'torchvision', 'sentence_transformers', 'transformers',
                'tiktoken', 'unstructured', 'rapidocr_onnxruntime'):
    package_datas, package_bins, package_hidden = collect_all( package )
    datas += package_datas
    binaries += package_bins
    hiddenimports += package_hidden
for package in ('google.genai', 'google.cloud'):
    hiddenimports += collect_submodules( package )
# Dependencies query distribution versions dynamically at runtime.
for distribution in ('streamlit', 'langchain-community', 'sentence-transformers',
                     'transformers', 'chromadb'):
    datas += copy_metadata( distribution, recursive=True )

a = Analysis( [str( root / 'desktop.py' )], pathex=[str( root )],
    binaries=binaries, datas=datas, hiddenimports=hiddenimports,
    hookspath=[], runtime_hooks=[], excludes=[], noarchive=False )
pyz = PYZ( a.pure )
exe = EXE( pyz, a.scripts, [], exclude_binaries=True, name='Chonky',
    console=False, disable_windowed_traceback=False )
coll = COLLECT( exe, a.binaries, a.datas, strip=False, upx=False, name='Chonky' )
