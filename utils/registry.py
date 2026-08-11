from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Literal, List
from datetime import datetime

import json


LOG_DIR="/Users/pierredamienkemkadouanla/Desktop/Life/_uoft/research/buchsbaum's lab/Projects/BayCrest's Featalang/langfeat-analysis/logs"

@dataclass
class Action:
    name: Literal["file_created", "file_updated", "file_deleted"]
    status: str

@dataclass
class Event:
    path: str
    output_path: str
    file: str
    output_file: str
    action: str
    func_name: str
    create_at: str
    

def register_event(path: str, action: Action, func_name: str, out_path: str="") -> None:

    log_book_title = str(datetime.now().strftime("%Y-%m-%d"))
    log_book_path = Path(f"{LOG_DIR}/{log_book_title}.jsonl")

    instance_of_event =  Event(
        path="/".join(Path(path).parts[-4:]),
        output_path=out_path,
        file="/".join(Path(path).parts[-2:]),
        output_file=str(Path(out_path).stem) if out_path else "",
        action=action.name,
        func_name=func_name,
        create_at=str(datetime.now())
    )

    if log_book_path.is_file():
        with open(log_book_path, "a", encoding="utf-8") as f:
            json.dump(asdict(instance_of_event), f)
            f.write("\n")
    else: 
        with open(log_book_path, "w", encoding="utf-8") as f:
            json.dump(asdict(instance_of_event), f)
            f.write("\n")

def store_stimuli(stimuli_list: List, output_dir: str):
    for stim in stimuli_list:

        output_file = f"{output_dir}/{stim['title']}.json"

        with open(output_file, "w") as f:
            json.dump(stim, f)

            register_event(
                path=__file__,
                action=Action(name="file_created", status="success"),
                func_name=__name__,
                out_path=output_file)

