from robosuite.utils.mjcf_utils import new_site
from ..bddl_base_domain import BDDLBaseDomain, register_problem
from ..robots import *
from ..objects import *
from ..predicates import *
from ..regions import *
from ..utils import rectangle2xyrange

from ...envs.arenas.table_arena import TableArena
from ...envs.arenas.table_arena import TableArenaExtended as TableArena
from robosuite.utils.mjcf_utils import array_to_string
from robosuite.models.tasks import ManipulationTask

@register_problem
class Libero_Tabletop_Manipulation(BDDLBaseDomain):
    def __init__(self, bddl_file_name, *args, **kwargs):
        self.workspace_name = "main_table"
        self.visualization_sites_list = []
        if "table_full_size" in kwargs:
            self.table_full_size = table_full_size
        else:
            self.table_full_size = (1.0, 1.2, 0.05)
        self.table_offset = (0, 0, 0.90)
        # For z offset of environment fixtures
        self.z_offset = 0.01 - self.table_full_size[2]
        kwargs.update(
            {"robots": [f"Mounted{robot_name}" for robot_name in kwargs["robots"]]}
        )
        kwargs.update({"workspace_offset": self.table_offset})
        kwargs.update({"arena_type": "table"})

        if "scene_xml" not in kwargs or kwargs["scene_xml"] is None:
            kwargs.update({"scene_xml": "scenes/libero_tabletop_base_style.xml"})
        if "scene_properties" not in kwargs or kwargs["scene_properties"] is None:
            kwargs.update(
                {
                    "scene_properties": {
                        "floor_style": "light-gray",
                        "wall_style": "light-gray-plaster",
                    }
                }
            )
        
        ### viewpoint perturb
        self.view_angle = kwargs['view_angle'] if 'view_angle' in kwargs else 0
        self.camera_distance = kwargs['camera_distance'] if 'camera_distance' in kwargs and kwargs['camera_distance'] is not None else 1.3
        # multi-view collection: list of dicts {name, azimuth, elevation, distance}
        self.multi_view_camera_configs = kwargs.pop('multi_view_camera_configs', None)

        super().__init__(bddl_file_name, *args, **kwargs)

    def _load_fixtures_in_arena(self, mujoco_arena):
        """Nothing extra to load in this simple problem."""
        for fixture_category in list(self.parsed_problem["fixtures"].keys()):
            if fixture_category == "table":
                continue

            for fixture_instance in self.parsed_problem["fixtures"][fixture_category]:
                self.fixtures_dict[fixture_instance] = get_object_fn(fixture_category)(
                    name=fixture_instance,
                    joints=None,
                )

    def _load_objects_in_arena(self, mujoco_arena):
        objects_dict = self.parsed_problem["objects"]
        for category_name in objects_dict.keys():
            for object_name in objects_dict[category_name]:
                self.objects_dict[object_name] = get_object_fn(category_name)(
                    name=object_name
                )

    def _load_sites_in_arena(self, mujoco_arena):
        # Create site objects
        object_sites_dict = {}
        region_dict = self.parsed_problem["regions"]
        for object_region_name in list(region_dict.keys()):

            if "main_table" in object_region_name:
                ranges = region_dict[object_region_name]["ranges"][0]
                assert ranges[2] >= ranges[0] and ranges[3] >= ranges[1]
                zone_size = ((ranges[2] - ranges[0]) / 2, (ranges[3] - ranges[1]) / 2)
                zone_centroid_xy = (
                    (ranges[2] + ranges[0]) / 2,
                    (ranges[3] + ranges[1]) / 2,
                )
                target_zone = TargetZone(
                    name=object_region_name,
                    rgba=region_dict[object_region_name]["rgba"],
                    zone_size=zone_size,
                    zone_centroid_xy=zone_centroid_xy,
                )
                object_sites_dict[object_region_name] = target_zone

                mujoco_arena.table_body.append(
                    new_site(
                        name=target_zone.name,
                        pos=target_zone.pos,
                        quat=target_zone.quat,
                        rgba=target_zone.rgba,
                        size=target_zone.size,
                        type="box",
                    )
                )
                continue
            # Otherwise the processing is consistent
            for query_dict in [self.objects_dict, self.fixtures_dict]:
                for name, body in query_dict.items():
                    try:
                        if "worldbody" not in list(body.__dict__.keys()):
                            # This is a special case for CompositeObject, we skip this as this is very rare in our benchmark
                            continue
                    except:
                        continue
                    for part in body.worldbody.find("body").findall(".//body"):
                        sites = part.findall(".//site")
                        joints = part.findall("./joint")
                        if sites == []:
                            break
                        for site in sites:
                            site_name = site.get("name")
                            if site_name == object_region_name:
                                object_sites_dict[object_region_name] = SiteObject(
                                    name=site_name,
                                    parent_name=body.name,
                                    joints=[joint.get("name") for joint in joints],
                                    size=site.get("size"),
                                    rgba=site.get("rgba"),
                                    site_type=site.get("type"),
                                    site_pos=site.get("pos"),
                                    site_quat=site.get("quat"),
                                    object_properties=body.object_properties,
                                )
        self.object_sites_dict = object_sites_dict

        # Keep track of visualization objects
        for query_dict in [self.fixtures_dict, self.objects_dict]:
            for name, body in query_dict.items():
                if body.object_properties["vis_site_names"] != {}:
                    self.visualization_sites_list.append(name)

    def _add_placement_initializer(self):
        """Very simple implementation at the moment. Will need to upgrade for other relations later."""
        super()._add_placement_initializer()

    def _check_success(self):
        """
        Check if the goal is achieved. Consider conjunction goals at the moment
        """
        goal_state = self.parsed_problem["goal_state"]
        result = True
        for state in goal_state:
            result = self._eval_predicate(state) and result
        return result

    def _eval_predicate(self, state):
        if len(state) == 3:
            # Checking binary logical predicates
            predicate_fn_name = state[0]
            object_1_name = state[1]
            object_2_name = state[2]
            return eval_predicate_fn(
                predicate_fn_name,
                self.object_states_dict[object_1_name],
                self.object_states_dict[object_2_name],
            )
        elif len(state) == 2:
            # Checking unary logical predicates
            predicate_fn_name = state[0]
            object_name = state[1]
            return eval_predicate_fn(
                predicate_fn_name, self.object_states_dict[object_name]
            )

    def _setup_references(self):
        super()._setup_references()

    def _post_process(self):
        super()._post_process()

        self.set_visualization()

    def set_visualization(self):

        for object_name in self.visualization_sites_list:
            for _, (site_name, site_visible) in (
                self.get_object(object_name).object_properties["vis_site_names"].items()
            ):
                vis_g_id = self.sim.model.site_name2id(site_name)
                if ((self.sim.model.site_rgba[vis_g_id][3] <= 0) and site_visible) or (
                    (self.sim.model.site_rgba[vis_g_id][3] > 0) and not site_visible
                ):
                    # We toggle the alpha value
                    self.sim.model.site_rgba[vis_g_id][3] = (
                        1 - self.sim.model.site_rgba[vis_g_id][3]
                    )

    def _setup_camera(self, mujoco_arena):
        # breakpoint()
        # print(xml[36400:36999])
        mujoco_arena.set_camera(
            camera_name="agentview",
            # pos=[0.6586131746834771, 0.0, 1.6103500240372423], # axis: [forth-back, left-right, down-up]
            # quat=[
            #     0.6380177736282349,
            #     0.3048497438430786,
            #     0.30484986305236816,
            #     0.6380177736282349,
            # ],
            # 45
            # pos=[0.5, 0.6, 1.6103500240372423],
            # quat=[ # z, 45, counterclockwise
            #     0.34529270255348316,
            #     0.16498343332932022,
            #     0.3983054625805496,
            #     0.8336103254930782
            # ],
            # -45
            pos=[0.5, -0.6, 1.6103500240372423],
            quat=[ # z, 45, clockwise
                0.8336103254930782,
                0.3983053980649523,
                0.16498358908375027,
                0.34529270255348316
            ],
            # 90
            # pos=[-0.2, 0.8, 1.6103500240372423],
            # quat=[ # z, 90, counterclockwise
            #     5.551114667225805e-17,
            #     -8.429369007185494e-08,
            #     0.4311226911154512,
            #     0.9022933143969116
            # ],
            # -90
            # pos=[-0.1, -0.8, 1.6103500240372423],
            # quat=[ # z, 90, clockwise
            #     0.9022933143969116,
            #     0.4311226911154512,
            #     8.429369012736608e-08,
            #     5.551114667225805e-17
            # ],
            # 135
            # pos=[-0.7, 0.6, 1.6103500240372423],
            # quat=[ # z, 135, counterclockwise
            #     -0.34529270255348316,
            #     -0.16498358908375024,
            #     0.3983053980649523,
            #     0.8336103254930782
            # ],
            # -135
            # pos=[-0.7, -0.6, 1.6103500240372423],
            # quat=[ # z, 135, clockwise
            #     0.8336103254930782,
            #     0.39830546258054966,
            #     -0.16498343332932022,
            #     -0.34529270255348316
            # ],
            # 180
            # pos=[-1.2, 0.0, 1.6103500240372423],
            # quat=[ # z, 180, clockwise
            #     0.6380177212293417,
            #     0.30484983801576876,
            #     -0.30484971880648903,
            #     -0.6380177212293417
            # ],
            # pos=[-0.6, 0.0, 2.1], # better version of 180
            # quat=[ # z, 180, clockwise
            #     0.6888757091555809, 0.15953137035740664, -0.15953125397386284, -0.6888757349571913
            # ],
        )

        # For visualization purpose
        mujoco_arena.set_camera(
            camera_name="frontview", pos=[1.0, 0.0, 1.48], quat=[0.56, 0.43, 0.43, 0.56]
        )
        mujoco_arena.set_camera(
            camera_name="galleryview",
            pos=[2.844547668904445, 2.1279684793440667, 3.128616846013882],
            quat=[
                0.42261379957199097,
                0.23374411463737488,
                0.41646939516067505,
                0.7702690958976746,
            ],
        )

    def _load_model(self):
        """
        Loads an xml model, puts it in self.model
        """
        super()._load_model()
        # Adjust base pose accordingly

        # if self._arena_type == "table":
        xpos = self.robots[0].robot_model.base_xpos_offset["table"](
            self.table_full_size[0]
        )
        self.robots[0].robot_model.set_base_xpos(xpos)
        mujoco_arena = TableArena(
            table_full_size=self.table_full_size,
            table_offset=self.workspace_offset,
            table_friction=(0.6, 0.005, 0.0001),
            xml=self._arena_xml,
            **self._arena_properties,
        )
        
        # Arena always gets set to zero origin
        mujoco_arena.set_origin([0, 0, 0])

        self._setup_camera_specific_spherical(mujoco_arena, self.view_angle, self.camera_distance)
        print(f'setup camera specific spherical: angle={self.view_angle} distance={self.camera_distance}', flush=True)

        # Add extra cameras for multi-view data collection
        # center_pos already has +0.4 z offset applied by _setup_camera_specific_spherical
        if self.multi_view_camera_configs:
            target_coord = mujoco_arena.center_pos
            for cam_cfg in self.multi_view_camera_configs:
                pos, xyaxes = compute_camera_xyaxes(
                    target_coord, cam_cfg['distance'], cam_cfg['azimuth'], cam_cfg['elevation']
                )
                mujoco_arena.set_camera(
                    camera_name=cam_cfg['name'],
                    pos=pos,
                    quat=None,
                    camera_attribs={'xyaxes': array_to_string(xyaxes)},
                )

        self._load_custom_material()

        self._load_fixtures_in_arena(mujoco_arena)

        self._load_objects_in_arena(mujoco_arena)

        self._load_sites_in_arena(mujoco_arena)

        self._generate_object_state_wrapper()

        self._setup_placement_initializer(mujoco_arena)

        self.objects = list(self.objects_dict.values())
        self.fixtures = list(self.fixtures_dict.values())

        # task includes arena, robot, and objects of interest
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=self.objects + self.fixtures,
        )

        for fixture in self.fixtures:
            self.model.merge_assets(fixture)

    def _setup_camera_specific_spherical(self, mujoco_arena, view_angle=None, distance=None):

        # target_coord = [0.03, 0.0, 1.2] # fixed
        target_coord = mujoco_arena.center_pos
        target_coord[2] += 0.4 # slightly above table center
        # print(f'setup camera spherical at target coord: {target_coord}', flush=True)
        
        elevation = 30
        # pos, quat = compute_camera_quaternion(target_coord, distance, view_angle, elevation)
        # pos, zaxis = spherical_to_cartesian_camera(target_coord, distance, 0, view_angle)
        pos, xyaxes = compute_camera_xyaxes(target_coord, distance, view_angle, elevation)

        mujoco_arena.set_camera(
            camera_name="agentview",
            pos=pos,
            # quat=quat
            quat=None,
            camera_attribs={
                'xyaxes': array_to_string(xyaxes),
                # 'fovy': '100.0'
            }
        )

import numpy as np

def compute_camera_xyaxes(target_coord, distance, azimuth_deg, elevation_deg):
    """
    Compute camera position and xyaxes for MuJoCo camera specification.
    
    MuJoCo conventions:
    - Right-handed coordinate system with Z-axis vertical
    - xyaxes: 6D vector [x_right, y_right, z_right, x_down, y_down, z_down]
      where first 3 elements define camera X-axis (right)
      and last 3 elements define camera Y-axis (down)
    - Camera Z-axis (forward/optical axis) is implicit: Z = X × Y
    
    Args:
        target_coord: (3,) array of the target's XYZ position (m).
        distance: Radial distance from target to camera (m).
        azimuth_deg: Angle around the Z-axis (yaw/lateral) (degrees, starts from X towards Y).
        elevation_deg: Angle above the XY plane (pitch/vertical) (degrees).
    
    Returns:
        Tuple: (camera_position as list, xyaxes_vector as list)
    """
    target_coord = np.asarray(target_coord)
    
    # Convert spherical coordinates to Cartesian (relative to target)
    azimuth_rad = np.radians(azimuth_deg) # hard coded adjustment
    elevation_rad = np.radians(elevation_deg)
    
    # Compute camera position relative to target
    x = distance * np.cos(elevation_rad) * np.cos(azimuth_rad)
    y = distance * np.cos(elevation_rad) * np.sin(azimuth_rad)
    z = distance * np.sin(elevation_rad)
    
    camera_position = target_coord + np.array([x, y, z])
    
    # Compute camera orientation (look-at direction)
    # Camera Z-axis points from camera to target (looking direction)
    # forward = target_coord - camera_position
    forward = camera_position - target_coord
    forward = forward / np.linalg.norm(forward)
    
    # Define world up vector (Z-axis in MuJoCo)
    world_up = np.array([0, 0, 1])
    
    # Compute camera right vector (X-axis)
    right = np.cross(world_up, forward)
    right_norm = np.linalg.norm(right)
    
    # Handle gimbal lock when camera is directly above/below target
    if right_norm < 1e-6:
        # Use alternative reference when looking straight up/down
        world_ref = np.array([1, 0, 0])  # Use X-axis as reference
        right = np.cross(world_ref, forward)
    
    right = right / np.linalg.norm(right)
    
    # Compute camera down vector (Y-axis)
    # Z = X × Y, so Y = Z × X
    down = np.cross(forward, right)
    down = down / np.linalg.norm(down)
    
    # Construct xyaxes: [x_right, y_right, z_right, x_down, y_down, z_down]
    xyaxes_vector = np.concatenate([right, down])
    
    return camera_position.tolist(), xyaxes_vector.tolist()
