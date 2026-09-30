"""Static catalog of what each appliance experiment contains in the dataset.

Source: Home-Appliance-Control-Dataset README + per-appliance info.md files.
Subject counts: TV 30, Door Lock 15, Electric Light 15, BT Speaker 14, AC 10.
"""

from __future__ import annotations

from typing import Dict, List

APPLIANCES: List[Dict] = [
    {
        "name": "Air Conditioner",
        "class_id": 1,
        "folder": "AirConditioner",
        "subjects": 10,
        "paradigm": "4-class AR oddball",
        "environment": "Microsoft HoloLens 1 (AR)",
        "stimuli": 4,
        "target_ratio": "1:3",
        "repetitions_per_block": 10,
        "commands": [
            {"stimulus_id": 1, "label": "Power On"},
            {"stimulus_id": 2, "label": "Power Off"},
            {"stimulus_id": 3, "label": "Temperature +"},
            {"stimulus_id": 4, "label": "Temperature -"},
        ],
        "note": "Icons shown in blue, flicker light-green 0.625 s, ISI 0.125 s.",
    },
    {
        "name": "Bluetooth Speaker",
        "class_id": 2,
        "folder": "BluetoothSpeaker",
        "subjects": 14,
        "paradigm": "6-class LCD oddball",
        "environment": "Monitor screen (no device shown)",
        "stimuli": 6,
        "target_ratio": "1:5",
        "repetitions_per_block": 10,
        "commands": [
            {"stimulus_id": 1, "label": "Power On"},
            {"stimulus_id": 2, "label": "Power Off"},
            {"stimulus_id": 3, "label": "Play"},
            {"stimulus_id": 4, "label": "Pause"},
            {"stimulus_id": 5, "label": "Next Track"},
            {"stimulus_id": 6, "label": "Previous Track"},
        ],
        "note": "Only paradigm with 6 real commands (no dummies needed).",
    },
    {
        "name": "Door Lock",
        "class_id": 3,
        "folder": "Doorlock",
        "subjects": 15,
        "paradigm": "4-class LCD oddball (2 real + 2 dummy)",
        "environment": "Tablet PC",
        "stimuli": 4,
        "target_ratio": "1:3",
        "repetitions_per_block": 10,
        "commands": [
            {"stimulus_id": 1, "label": "Lock"},
            {"stimulus_id": 2, "label": "Unlock"},
            {"stimulus_id": 3, "label": "Dummy stimulus 3"},
            {"stimulus_id": 4, "label": "Dummy stimulus 4"},
        ],
        "note": "Dummy stimuli pad the 2 real icons to 4 classes.",
    },
    {
        "name": "Electric Light",
        "class_id": 4,
        "folder": "ElectricLight",
        "subjects": 15,
        "paradigm": "4-class LCD oddball (3 real + 1 dummy)",
        "environment": "Tablet PC",
        "stimuli": 4,
        "target_ratio": "1:3",
        "repetitions_per_block": 10,
        "commands": [
            {"stimulus_id": 1, "label": "On"},
            {"stimulus_id": 2, "label": "Off"},
            {"stimulus_id": 3, "label": "Dim"},
            {"stimulus_id": 4, "label": "Dummy stimulus 4"},
        ],
        "note": "Dummy stimulus pads the 3 real icons to 4 classes.",
    },
    {
        "name": "TV",
        "class_id": 5,
        "folder": "TV",
        "subjects": 30,
        "paradigm": "4-class LCD oddball (multiview channels)",
        "environment": "50-inch UHD TV @ 2.5 m, 8 Hz red surround flicker",
        "stimuli": 4,
        "target_ratio": "1:3",
        "repetitions_per_block": 10,
        "commands": [
            {"stimulus_id": 1, "label": "Channel 1"},
            {"stimulus_id": 2, "label": "Channel 2"},
            {"stimulus_id": 3, "label": "Channel 3"},
            {"stimulus_id": 4, "label": "Channel 4"},
        ],
        "note": "Four preview video clips in screen quadrants double as distractors.",
    },
]

BY_NAME: Dict[str, Dict] = {a["name"]: a for a in APPLIANCES}
