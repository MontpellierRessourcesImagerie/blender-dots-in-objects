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
    FloatProperty,
    FloatVectorProperty,
    IntVectorProperty,
    PointerProperty,
    EnumProperty,
    BoolProperty,
    PointerProperty
)
from bpy.types import (
    PropertyGroup,
    Operator,
    Panel
)

import os
from .blender_callbacks import (
    segment_and_import,
    detect_and_import,
    get_cp_models
)
from .count_dots import (
    dots_to_closest_object,
    dots_per_object,
    remove_dots_outside_objects,
    count_co_occurrences
)
from .lib.dots_finder import prefilters_as_enum

def update_files_list(self, context):
    imgs_list = [("---", "---", "No file selected")]
    folder = self.root_folder
    if not folder:
        return imgs_list
    
    folder = os.path.abspath(bpy.path.abspath(folder))
    if os.path.isdir(folder):
        valid_ext = {'.tif', '.tiff', '.zar', '.zarr'}
        files = sorted([
            f for f in os.listdir(folder)
            if os.path.isfile(os.path.join(folder, f))
            and os.path.splitext(f)[1].lower() in valid_ext
        ])
        imgs_list += [(os.path.join(folder, f), f, "") for f in files]
    
    return imgs_list


class DotsObjects_Props(PropertyGroup):
    root_folder: StringProperty(
        name="Root folder",
        description="Folder in which all images are located.",
        subtype='DIR_PATH',
        default=""
    )
    objects_path: EnumProperty(
        name="Objects image",
        description="Path to a 3D single-channel TIFF image representing objects (ZYX).",
        items=update_files_list
    )
    secondary_objects_path: EnumProperty(
        name="Secondary objects image",
        description="Path to a 3D single-channel TIFF image representing secondary objects (ZYX).",
        items=update_files_list
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
        items=get_cp_models("lib"),
        default='cyto3'
    )

    dots_path: EnumProperty(
        name="Dots image",
        description="Path to a 3D single-channel TIFF image representing dots (ZYX).",
        items=update_files_list
    )
    dots_prefilter: EnumProperty(
        name="Dots pre-filter",
        description="Pre-filter applied before detecting local maxima for dot detection.",
        items=prefilters_as_enum(),
        default='Laplacian of Gaussian'
    )
    dots_sigma: FloatProperty(
        name="Dots pre-filter sigma",
        description="Sigma of the pre-filter applied before detecting local maxima for dot detection.",
        default=1.0, min=0.1, max=10.0
    )   
    dots_threshold: FloatProperty(
        name="Dots threshold",
        description="Intensity threshold for dot detection (applied to the LoG filtered image).",
        min=-1.0, max=1.0, default=0.5
    )
    auto_threshold: BoolProperty(
        name="Auto threshold (Otsu)",
        description="Whether to automatically compute the dots detection threshold using Otsu's method.",
        default=False
    )
    
    calib: FloatVectorProperty(
        name="Calibration",
        description="Voxel size (aka 'sampling distance') in microns.",
        size=3, default=(0.116, 0.116, 0.122), subtype='XYZ',
        precision=4
    )
    patch: IntVectorProperty(
        name="Patch size (X, Y, Z)",
        description="Read/processing patch size; not required for simple Otsu, reserved for large data",
        size=3, default=(300, 140, 140), min=1, subtype='XYZ'
    )
    full_image: BoolProperty(
        name="Full image",
        description="Whether to process the full image at once (no chunking). Not recommended for large images.",
        default=False
    )

    objects_collection: PointerProperty(
        name="Objects collection",
        description="Collection in which segmented objects are imported.",
        type=bpy.types.Collection
    )
    co_occ_per_obj: BoolProperty(
        name="Count co-occurrences per object",
        description="Whether to count co-occurrences of dots per object (instead of globally).",
        default=True
    )
    co_occ_dist_threshold: FloatProperty(
        name="Co-occurrence distance threshold",
        description="Maximum distance (in microns) between dots to be considered co-occurring.",
        default=0.5, min=0.001, max=1000.0
    )
    
    

# ---- The worker operator ----

class OBJSEG_OT_Launch(Operator):
    bl_idname  = "objseg.launch"
    bl_label   = "Launch Objects Segmentation"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        props = context.scene.objseg_props
        path_main = props.objects_path
        if not path_main or not os.path.isfile(path_main):
            self.report({'ERROR'}, "Invalid input image path.")
            return {'CANCELLED'}
        
        print("=== Launching objects segmentation ===")
        calib = ( # Z, Y, X
            props.calib[2], 
            props.calib[1], 
            props.calib[0]
        ) 
        obj_size_yx = props.obj_size
        chunk_size = ( # Z, Y, X
            props.patch[2], 
            props.patch[1], 
            props.patch[0]
        )
        cp_model = props.model
        min_obj_size = props.min_obj_size
        use_full = props.full_image

        path_secondary = props.secondary_objects_path
        if not path_secondary or not os.path.isfile(path_secondary):
            print("No secondary objects image provided, proceeding with single-channel segmentation.")
            path_secondary = None
        
        print(f"Calib: {calib}")
        print(f"Obj size: {obj_size_yx}")
        print(f"Chunks: {chunk_size}")

        segment_and_import(
            img_path=path_main,
            secondary=path_secondary,
            calib=calib,
            obj_size_yx=obj_size_yx,
            chunk_size=chunk_size,
            model=cp_model,
            min_obj_size=min_obj_size,
            use_full_image=use_full
        )
        return {'FINISHED'}
    
class DOTSDEC_OT_Launch(Operator):
    bl_idname  = "dotsdec.launch"
    bl_label   = "Launch Dots Detection"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        props = context.scene.objseg_props
        path = props.dots_path
        if not path or not os.path.isfile(path):
            self.report({'ERROR'}, "Invalid input image path.")
            return {'CANCELLED'}
        
        print("=== Launching dots detection ===")
        
        calib = (props.calib[2], props.calib[1], props.calib[0])  # Z, Y, X
        chunk_size = (props.patch[2], props.patch[1], props.patch[0])  # Z, Y, X
        dots_threshold = props.dots_threshold
        sigma = props.dots_sigma
        prefilter = props.dots_prefilter
        use_full_image = props.full_image

        auto_thr = props.auto_threshold
        if auto_thr:
            dots_threshold = -1.0
        
        print(f"Calib: {calib}")
        print(f"Chunks: {chunk_size}")

        detect_and_import(
            img_path=path,
            calib=calib,
            thr=dots_threshold,
            chunk_size=chunk_size,
            sigma=sigma,
            prefilter=prefilter,
            use_full_image=use_full_image
        )
        return {'FINISHED'}
    
class ASSIGN_DOTS_OT_Launch(Operator):
    bl_idname = "assign_dots.launch"
    bl_label = "Assign dots to objects"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        collection = context.scene.objseg_props.objects_collection
        if collection is None:
            self.report({'ERROR'}, "Please select an objects collection.")
            return {'CANCELLED'}
        dots_to_closest_object(collection)
        return {'FINISHED'}
    
class COUNT_DOTS_PER_OBJECT_OT_Launch(Operator):
    bl_idname = "count_dots_per_object.launch"
    bl_label = "Count dots per object"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        collection = context.scene.objseg_props.objects_collection
        if collection is None:
            self.report({'ERROR'}, "Please select an objects collection.")
            return {'CANCELLED'}
        dots_per_object(collection)
        return {'FINISHED'}
    
class COUNT_CO_OCCUR_OT_Launch(Operator):
    bl_idname = "count_co_occur.launch"
    bl_label = "Count co-occurrences"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        max_dist = context.scene.objseg_props.co_occ_dist_threshold
        per_obj = context.scene.objseg_props.co_occ_per_obj
        obj_collection = context.scene.objseg_props.objects_collection
        count_co_occurrences(max_dist, per_obj, obj_collection)
        return {'FINISHED'}
    
class REMOVE_DOTS_OUTSIDE_OT_Launch(Operator):
    bl_idname = "remove_dots_outside.launch"
    bl_label = "Remove dots outside objects"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        remove_dots_outside_objects()
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

        layout.prop(props, "root_folder", text="Root folder")

        col = layout.box()

        col.label(text="General", icon='EMPTY_DATA')
        row = col.row(align=True)
        row.prop(props, "calib", text="Calibration")
        row = col.row(align=True)
        row.prop(props, "patch", text="Patches")
        row = col.row(align=True)
        row.prop(props, "full_image", text="Full image (no chunking)")

        col.separator()
        col = layout.box()

        col.label(text="Objects", icon='OBJECT_DATA')
        col.prop(props, "objects_path", text="Main channel")
        col.prop(props, "secondary_objects_path", text="Secondary channel (optional)")
        col.prop(props, "obj_size", text="Object size")
        col.prop(props, "min_obj_size", text="Min object size")
        col.prop(props, "model", text="Model")

        col.operator("objseg.launch", text="Segment objects", icon='PLAY')

        col.separator()
        col = layout.box()

        col.label(text="Dots channel", icon='GROUP_VERTEX')
        col.prop(props, "dots_path", text="Dots")
        
        row = col.row(align=True)
        row.prop(props, "dots_prefilter", text="Pre-filter")
        row.prop(props, "dots_sigma", text="Sigma")
        
        row = col.row(align=True)
        row.prop(props, "auto_threshold", text="Auto (Otsu)")
        row.prop(props, "dots_threshold", text="Dots threshold")
        col.operator("dotsdec.launch", text="Detect dots", icon='PLAY')

        col.separator()
        col = layout.column()

        row = col.row(align=True)
        row.prop(props, "objects_collection", text="")
        row.operator("assign_dots.launch", text="Assign dots to objects", icon='PARTICLES')

        row = col.row(align=True)
        row.operator("remove_dots_outside.launch", text="Remove dots outside objects", icon='CANCEL')
        row.operator("count_dots_per_object.launch", text="Count dots per object", icon='SORTBYEXT')

        row = col.row(align=True)
        row.prop(props, "co_occ_per_obj", text="Per object?")
        row.prop(props, "co_occ_dist_threshold", text="Max distance")
        row.operator("count_co_occur.launch", text="Count co-occurrences", icon='SORTBYEXT')

# ---- register ----

classes = (
    DotsObjects_Props,
    OBJSEG_OT_Launch,
    DOTSDEC_OT_Launch,
    ASSIGN_DOTS_OT_Launch,
    OBJSEG_PT_Panel,
    COUNT_DOTS_PER_OBJECT_OT_Launch,
    COUNT_CO_OCCUR_OT_Launch,
    REMOVE_DOTS_OUTSIDE_OT_Launch
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
