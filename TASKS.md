# Benchmark Task Set

| id | category | difficulty | checked properties |
| --- | --- | --- | --- |
| `geometry_001_basic_primitives` | geometry | easy | primitive creation, object existence, placement |
| `geometry_002_positions` | geometry | easy | fixed cube coordinates, placement accuracy |
| `geometry_003_dimensions` | geometry | medium | object dimensions, scaled primitives |
| `geometry_004_rotation` | geometry | medium | object rotation, transform accuracy |
| `geometry_005_composition` | geometry | medium | multi-object composition, floor/table/object placement |
| `materials_001_basic_colors` | materials | easy | RGB material assignment, object-to-material mapping |
| `materials_002_roughness` | materials | easy | roughness values, material parameter correctness |
| `materials_003_metallic` | materials | easy | metallic values, material parameter correctness |
| `materials_004_multiple_objects` | materials | medium | multiple objects, distinct material mapping |
| `materials_005_material_composition` | materials | medium | coordinated scene materials, roughness/metallic values |
| `lighting_001_area_light` | lighting | easy | AREA light existence, energy, location |
| `lighting_002_sun_light` | lighting | easy | SUN light existence, rotation direction, energy |
| `lighting_003_three_point_lighting` | lighting | medium | key/fill/back lights, placement, energy |
| `camera_001_front_view` | camera | easy | camera existence, front view, focal length, target visibility |
| `camera_002_top_view` | camera | easy | top-down camera, focal length, multi-object visibility |
| `camera_003_composition_view` | camera | medium | angled framing, focal length, target visibility |
| `export_001_blend_file` | export | easy | object existence, light presence, `result.blend` export |
| `export_002_glb_file` | export | medium | low-poly scene, materials, `result.glb` export |

