GEOMETRY_TOOLS = ("create_object", "set_transform", "assign_material", "inspect_scene")
MATERIAL_TOOLS = ("create_object", "assign_material", "set_material_properties", "inspect_scene")
LIGHTING_TOOLS = ("create_object", "create_light", "set_transform", "set_light_properties", "inspect_scene")
CAMERA_TOOLS = ("create_object", "create_camera", "set_camera", "set_transform", "inspect_scene")
EXPORT_TOOLS = ("create_object", "assign_material", "export_scene", "inspect_scene")

TOOL_CATALOG = {
    "geometry": GEOMETRY_TOOLS,
    "materials": MATERIAL_TOOLS,
    "lighting": LIGHTING_TOOLS,
    "camera": CAMERA_TOOLS,
    "export": EXPORT_TOOLS,
}

