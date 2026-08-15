from __future__ import annotations

import argparse
from pathlib import Path
import os
import shutil

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


input_path = Path("/home/nahuel/Documentos/Python/DNI/data/ds/ds_arg_ID_card_det_v1/images/test")
labels_in_path = Path("/home/nahuel/Documentos/Python/DNI/data/ds/ds_arg_ID_card_det_v1/labels/train")
labels_out_path = Path("/home/nahuel/Documentos/Python/DNI/data/ds/ds_arg_ID_card_det_v1/labels/test")
if not input_path.is_dir():
    raise NotADirectoryError(f"No existe el directorio de entrada: {input_path}")
else:
    print(f"Directorio de entrada: {input_path}")   

image_paths = sorted(
    path for path in input_path.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS
)
created_count = 0
for image_path in image_paths:
    # print(f"Revisando: {image_path.name}")
    label_src = labels_in_path / f"{image_path.stem}.txt"
    label_dst = labels_out_path / f"{image_path.stem}.txt"

    print(f"Etiqueta fuente: {label_src.name} - Etiqueta destino: {label_dst.name} - image: {image_path.name}")
    shutil.move(str(label_src), str(label_dst))

print(f"Imágenes revisadas: {len(image_paths)}")
# print(f"Archivos .txt creados: {created_count}")