from typing import Any


def _remove_datablock(collection: Any, datablock: Any) -> None:
    try:
        collection.remove(datablock, do_unlink=True)
    except TypeError:
        collection.remove(datablock)


def _purge_unused_datablocks(bpy: Any) -> None:
    for collection_name in ("meshes", "materials", "lights", "cameras"):
        collection = getattr(bpy.data, collection_name)
        for datablock in list(collection):
            if getattr(datablock, "users", 0) == 0:
                _remove_datablock(collection, datablock)


def reset_scene(payload: dict) -> dict:
    import bpy

    scene = bpy.context.scene
    scene_name = payload.get("scene_name")
    removed_objects = len(list(bpy.data.objects))

    for obj in list(bpy.data.objects):
        _remove_datablock(bpy.data.objects, obj)

    _purge_unused_datablocks(bpy)

    scene.frame_start = 1
    scene.frame_end = 1
    if hasattr(scene, "frame_set"):
        scene.frame_set(1)
    else:
        scene.frame_current = 1

    if scene_name:
        scene.name = str(scene_name)

    return {
        "removed_objects": removed_objects,
        "remaining_objects": len(list(bpy.data.objects)),
        "scene_name": scene.name,
    }

