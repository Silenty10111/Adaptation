"""Safe morphology transformations for generated robot descriptions."""

from __future__ import annotations

import copy
import re
from typing import Dict, List


_LEG_TOKEN = re.compile(r"\bleg_(\d+)_")


def amputate_legs(description: dict, remove_leg_ids: List[int]) -> dict:
    """Remove legs and atomically renumber all remaining topology references.

    Filtering is performed against the *original* link graph before any names
    are rewritten.  This avoids the duplicate-joint bug where a removed
    ``leg_1`` was confused with an old ``leg_2`` renamed to ``leg_1``.
    Mesh paths intentionally retain their original filenames.
    """
    desc = copy.deepcopy(description)
    remove = {int(x) for x in remove_leg_ids}
    original_ids = sorted({
        int(link["leg_id"])
        for link in desc.get("links", [])
        if link.get("leg_id") is not None
    })
    unknown = remove.difference(original_ids)
    if unknown:
        raise ValueError(f"unknown leg ids: {sorted(unknown)}")
    remaining = [lid for lid in original_ids if lid not in remove]
    old_to_new = {old: new for new, old in enumerate(remaining)}

    removed_link_names = {
        str(link.get("name", ""))
        for link in desc.get("links", [])
        if link.get("leg_id") is not None and int(link["leg_id"]) in remove
    }

    def rename(value: str) -> str:
        def replace(match: re.Match) -> str:
            old = int(match.group(1))
            return f"leg_{old_to_new.get(old, old)}_"
        return _LEG_TOKEN.sub(replace, value)

    links = []
    for link in desc.get("links", []):
        lid = link.get("leg_id")
        if lid is not None and int(lid) in remove:
            continue
        item = copy.deepcopy(link)
        if lid is not None:
            old = int(lid)
            item["original_leg_id"] = old
            item["leg_id"] = old_to_new[old]
            item["name"] = rename(str(item["name"]))
        links.append(item)

    joints = []
    for joint in desc.get("joints", []):
        parent = str(joint.get("parent", ""))
        child = str(joint.get("child", ""))
        if parent in removed_link_names or child in removed_link_names:
            continue
        item = copy.deepcopy(joint)
        item["name"] = rename(str(item.get("name", "")))
        item["parent"] = rename(parent)
        item["child"] = rename(child)
        joints.append(item)

    link_names = [str(link["name"]) for link in links]
    joint_names = [str(joint["name"]) for joint in joints]
    if len(link_names) != len(set(link_names)):
        raise ValueError("amputation produced duplicate link names")
    if len(joint_names) != len(set(joint_names)):
        raise ValueError("amputation produced duplicate joint names")
    valid_links = set(link_names)
    dangling = [
        joint["name"] for joint in joints
        if joint["parent"] not in valid_links or joint["child"] not in valid_links
    ]
    if dangling:
        raise ValueError(f"amputation produced dangling joints: {dangling}")

    desc["links"] = links
    desc["joints"] = joints
    desc["num_legs"] = len(remaining)
    desc["robot_name"] = f"{description.get('robot_name', 'robot')}_{len(remaining)}legs"
    desc["amputation"] = {
        "removed_original_leg_ids": sorted(remove),
        "old_to_new_leg_ids": {str(k): v for k, v in old_to_new.items()},
    }
    return desc
