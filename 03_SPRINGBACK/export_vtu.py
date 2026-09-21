# -*- coding: utf-8 -*-
from __future__ import annotations

import os
from typing import Dict, Optional

import numpy as np

VTK_CELL_TYPES = {
    3: 5,   # triangle
    4: 9,   # quad
}


def _fmt_array(arr, fmt="{:.8g}"):
    a = np.asarray(arr)
    if a.ndim == 1:
        return " ".join(fmt.format(float(x)) for x in a)
    return "\n".join(" ".join(fmt.format(float(x)) for x in row) for row in a)


def write_vtu(
    path: str,
    points: np.ndarray,
    cells: np.ndarray,
    point_data: Optional[Dict[str, np.ndarray]] = None,
) -> None:
    """Write a minimal ASCII VTU file for triangle/quad shell meshes."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    points = np.asarray(points, dtype=np.float64)
    cells = np.asarray(cells, dtype=np.int64)
    if cells.ndim != 2:
        raise ValueError(f"cells must be 2D, got {cells.shape}")
    npts = points.shape[0]
    ncells = cells.shape[0]
    cell_n = cells.shape[1]
    vtk_type = VTK_CELL_TYPES.get(cell_n)
    if vtk_type is None:
        raise ValueError(f"Only triangle/quad cells are supported by this writer, got {cell_n} nodes/cell.")

    connectivity = cells.reshape(-1)
    offsets = np.arange(1, ncells + 1, dtype=np.int64) * cell_n
    types = np.full((ncells,), vtk_type, dtype=np.int64)

    with open(path, "w", encoding="utf-8") as f:
        f.write('<?xml version="1.0"?>\n')
        f.write('<VTKFile type="UnstructuredGrid" version="0.1" byte_order="LittleEndian">\n')
        f.write('  <UnstructuredGrid>\n')
        f.write(f'    <Piece NumberOfPoints="{npts}" NumberOfCells="{ncells}">\n')
        f.write('      <Points>\n')
        f.write('        <DataArray type="Float64" NumberOfComponents="3" format="ascii">\n')
        f.write(_fmt_array(points) + "\n")
        f.write('        </DataArray>\n')
        f.write('      </Points>\n')
        f.write('      <Cells>\n')
        f.write('        <DataArray type="Int64" Name="connectivity" format="ascii">\n')
        f.write(" ".join(str(int(x)) for x in connectivity) + "\n")
        f.write('        </DataArray>\n')
        f.write('        <DataArray type="Int64" Name="offsets" format="ascii">\n')
        f.write(" ".join(str(int(x)) for x in offsets) + "\n")
        f.write('        </DataArray>\n')
        f.write('        <DataArray type="UInt8" Name="types" format="ascii">\n')
        f.write(" ".join(str(int(x)) for x in types) + "\n")
        f.write('        </DataArray>\n')
        f.write('      </Cells>\n')
        if point_data:
            f.write('      <PointData>\n')
            for name, arr in point_data.items():
                a = np.asarray(arr)
                if a.ndim == 1:
                    ncomp = 1
                    a2 = a.reshape(-1, 1)
                else:
                    ncomp = a.shape[1]
                    a2 = a
                if a2.shape[0] != npts:
                    continue
                f.write(f'        <DataArray type="Float64" Name="{name}" NumberOfComponents="{ncomp}" format="ascii">\n')
                f.write(_fmt_array(a2) + "\n")
                f.write('        </DataArray>\n')
            f.write('      </PointData>\n')
        f.write('    </Piece>\n')
        f.write('  </UnstructuredGrid>\n')
        f.write('</VTKFile>\n')
