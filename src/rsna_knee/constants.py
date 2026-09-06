from __future__ import annotations

ID_COLUMN = "StudyInstanceUID"

TARGETS = (
    "ACL",
    "MCL",
    "Medial Meniscus",
    "Lateral Meniscus",
    "Medial OA",
    "Lateral OA",
    "PF OA",
    "Effusion",
    "Synovitis",
    "Baker's",
    "Contusion",
    "Fracture",
)

PLANES = ("Sagittal", "Coronal", "Axial")
SLOTS = tuple((plane, fluid) for plane in PLANES for fluid in (1, 0))
N_TARGETS = len(TARGETS)

