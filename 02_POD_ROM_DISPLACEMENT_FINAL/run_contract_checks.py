from pathlib import Path
import py_compile

files=[p for p in Path('.').glob('*.py')]
for p in files: py_compile.compile(str(p),doraise=True)
print(f'Python syntax PASS: {len(files)} files')
print('FINAL POD displacement contract: direct/global_residual/local_residual | grid64x28 | rank16 ROM | enhanced 10D process features')
