bl_info = {
    "name"       : "Dots in objects",
    "author"     : "Clement H. Benedetti",
    "version"    : (0, 2, 0),
    "blender"    : (4, 5, 0),
    "location"   : "3D Viewport > Properties panel > Object Segmentation",
    "description": "Segment bloboid objects (cell, nuclei, ...) and dots (RNA, foci, ...) from 3D microscopy images and import them as meshes/points in Blender.",
    "category"   : "Analysis",
}

import bpy
from bpy.props import (
    StringProperty, 
    IntProperty, 
    FloatVectorProperty, 
    IntVectorProperty, 
    PointerProperty,
    EnumProperty
)
from bpy.types import (
    PropertyGroup, 
    Operator, 
    Panel
)

from pathlib import Path
from .blender_callbacks import (
    segment_and_import, 
    detect_and_import,
    get_cp_models
)
from .count_dots import dots_per_object

class DotsObjects_Props(PropertyGroup):
    objects_path: StringProperty(
        name="Objects image",
        description="Path to a 3D single-channel TIFF image representing objects (ZYX).",
        subtype='FILE_PATH',
        default=""
    )
    secondary_objects_path: StringProperty(
        name="Secondary objects image",
        description="Path to a 3D single-channel TIFF image representing secondary objects (ZYX).",
        subtype='FILE_PATH',
        default=""
    )
    obj_size: IntProperty(
        name="Object size",
        description="Median objects diameter in voxels (used by CellPose).",
        min=1, default=15
    )
    min_obj_size: IntProperty(
        name="Min object size",
        description="Minimum object size in number of voxels (used for post-filtering).",
        min=1, default=100
    )
    model: EnumProperty(
        name="CellPose model",
        description="Pre-trained CellPose model to use for segmentation.",
        items=get_cp_models(),
        default='cyto3'
    )

    dots_path: StringProperty(
        name="Dots image",
        description="Path to a 3D single-channel TIFF image representing dots (ZYX).",
        subtype='FILE_PATH',
        default=""
    )
    dots_threshold: IntProperty(
        name="Dots threshold",
        description="Intensity threshold for dot detection (applied to the LoG filtered image).",
        min=-3000, max=0, default=-1550
    )
    
    calib: FloatVectorProperty(
        name="Calibration",
        description="Voxel size (aka 'sampling disntace') in microns.",
        size=3, default=(0.116, 0.116, 0.122), subtype='XYZ',
        precision=4
    )
    patch: IntVectorProperty(
        name="Patch size (X, Y, Z)",
        description="Read/processing patch size; not required for simple Otsu, reserved for large data",
        size=3, default=(300, 140, 140), min=1, subtype='XYZ'
    )
    
    

# ---- The worker operator ----

class OBJSEG_OT_Launch(Operator):
    bl_idname  = "objseg.launch"
    bl_label   = "Launch Objects Segmentation"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        props = context.scene.objseg_props
        path = Path(bpy.path.abspath(props.objects_path) if props.objects_path else "")
        if not path.is_file():
            self.report({'ERROR'}, "Invalid input image path.")
            return {'CANCELLED'}
        
        print("=== Launching objects segmentation ===")
        calib = (props.calib[2], props.calib[1], props.calib[0])  # Z, Y, X
        obj_size_yx = props.obj_size
        chunk_size = (props.patch[2], props.patch[1], props.patch[0])  # Z, Y, X
        cp_model = props.model
        min_obj_size = props.min_obj_size
        path_secondary = Path(bpy.path.abspath(props.secondary_objects_path) if props.secondary_objects_path else "")
        if not path_secondary.is_file():
            print("No valid secondary objects image provided, proceeding with single-channel segmentation.")
            path_secondary = None
        print(f"Image path: {path.name}")
        print(f"Calib: {calib}")
        print(f"Obj size: {obj_size_yx}")
        print(f"Chunks: {chunk_size}")
        segment_and_import(
            img_path=path,
            secondary=path_secondary,
            calib=calib,
            obj_size_yx=obj_size_yx,
            chunk_size=chunk_size,
            model=cp_model,
            min_obj_size=min_obj_size
        )
        return {'FINISHED'}
    
class DOTSDEC_OT_Launch(Operator):
    bl_idname  = "dotsdec.launch"
    bl_label   = "Launch Dots Detection"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        props = context.scene.objseg_props
        path = Path(bpy.path.abspath(props.dots_path) if props.dots_path else "")
        if not path.is_file():
            self.report({'ERROR'}, "Invalid input image path.")
            return {'CANCELLED'}
        
        print("=== Launching dots detection ===")
        calib = (props.calib[2], props.calib[1], props.calib[0])  # Z, Y, X
        chunk_size = (props.patch[2], props.patch[1], props.patch[0])  # Z, Y, X
        dots_threshold = props.dots_threshold
        print(f"Image path: {path.name}")
        print(f"Calib: {calib}")
        print(f"Chunks: {chunk_size}")
        detect_and_import(
            img_path=path,
            calib=calib,
            thr=dots_threshold,
            chunk_size=chunk_size
        )
        return {'FINISHED'}
    
class COUNT_DOTS_OT_Launch(Operator):
    bl_idname = "count_dots.launch"
    bl_label = "Dots per object"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        dots_per_object()
        return {'FINISHED'}

# ---- UI Panel (N-panel, 3D View) ----

class OBJSEG_PT_Panel(Panel):
    bl_idname = "OBJSEG_PT_panel"
    bl_label = "Object Segmentation"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Object Segmentation"

    def draw(self, context):
        layout = self.layout
        props = context.scene.objseg_props

        col = layout.column(align=True)
        col.label(text="Objects")
        col.prop(props, "objects_path", text="Main channel")
        col.prop(props, "secondary_objects_path", text="Secondary channel (optional)")
        col.prop(props, "obj_size", text="Object size")
        col.prop(props, "min_obj_size", text="Min object size")
        col.prop(props, "model", text="Model")

        col.separator()

        col.label(text="Dots")
        col.prop(props, "dots_path", text="Dots")
        col.prop(props, "dots_threshold", text="Dots Thr.")

        col.separator()

        col.label(text="General")
        row = col.row(align=True)
        row.prop(props, "calib", text="Calibration")
        row = col.row(align=True)
        row.prop(props, "patch", text="Patches")

        col.separator()

        col.operator("objseg.launch", text="Segment objects", icon='PLAY')
        col.operator("dotsdec.launch", text="Detect dots", icon='PLAY')
        col.operator("count_dots.launch", text="Count Dots/Nucleus", icon='SORTSIZE')

# ---- register ----

classes = (
    DotsObjects_Props,
    OBJSEG_OT_Launch,
    DOTSDEC_OT_Launch,
    COUNT_DOTS_OT_Launch,
    OBJSEG_PT_Panel,
)

def register():
    for c in classes:
        bpy.utils.register_class(c)
    bpy.types.Scene.objseg_props = PointerProperty(type=DotsObjects_Props)

def unregister():
    del bpy.types.Scene.objseg_props
    for c in reversed(classes):
        bpy.utils.unregister_class(c)

if __name__ == "__main__":
    register()
