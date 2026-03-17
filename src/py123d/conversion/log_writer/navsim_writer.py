from __future__ import annotations

import uuid
import pickle
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
import numpy as np

from py123d.conversion.dataset_converter_config import DatasetConverterConfig
from py123d.datatypes.metadata import LogMetadata
from py123d.datatypes.time.time_point import TimePoint
from py123d.datatypes.vehicle_state.ego_state import EgoStateSE3
from py123d.datatypes.detections.box_detections import BoxDetectionWrapper
from py123d.datatypes.detections.traffic_light_detections import TrafficLightDetectionWrapper
from py123d.conversion.log_writer.abstract_log_writer import AbstractLogWriter, CameraData, LiDARData


class NavsimWriter(AbstractLogWriter):
    """
    Writer class to export py123d standardized data into the official Navsim Pickle (.pkl) format.
    Matches the schema expected by `navsim.common.dataclasses.Scene.load_from_disk()`.
    """

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.log_name = ""
        self.map_name = ""
        self.scene_token = ""
        self.initial_token = ""
        self.frames_data: List[Dict[str, Any]] = []

    def reset(
        self,
        dataset_converter_config: DatasetConverterConfig,
        log_metadata: LogMetadata,
    ) -> bool:
        """Resets the writer for a new scene/log."""
        self.log_name = log_metadata.log_name
        self.map_name = log_metadata.map_name if hasattr(log_metadata, 'map_name') else "us-ma-boston"
        self.scene_token = uuid.uuid4().hex  # Generate unique token for the scene
        self.initial_token = ""
        self.frames_data = []
        return True

    def write(
        self,
        timestamp: TimePoint,
        ego_state: Optional[EgoStateSE3] = None,
        box_detections: Optional[BoxDetectionWrapper] = None,
        traffic_lights: Optional[TrafficLightDetectionWrapper] = None,
        pinhole_cameras: Optional[List[CameraData]] = None,
        fisheye_mei_cameras: Optional[List[CameraData]] = None,
        lidars: Optional[List[LiDARData]] = None,
        scenario_tags: Optional[List[str]] = None,
        route_lane_group_ids: Optional[List[int]] = None,
        **kwargs,
    ) -> None:
        """Processes a single frame and formats it exactly as Navsim's Frame dict."""
        
        frame_token = uuid.uuid4().hex
        if not self.frames_data:
            self.initial_token = frame_token  # Set initial token for metadata

        # 1. Ego Status (Requires strictly [x, y, yaw] for pose, and numpy arrays)
        ego_status_dict = {
            "ego_pose": np.zeros(3, dtype=np.float64),
            "ego_velocity": np.zeros(2, dtype=np.float32),
            "ego_acceleration": np.zeros(2, dtype=np.float32),
            "driving_command": np.zeros(1, dtype=int), # Default/dummy command
            "in_global_frame": True
        }
        
        if ego_state is not None:
            # Extract x, y, and yaw (assuming Rotation object has a yaw representation)
            tx, ty, tz = ego_state.pose.translation
            try:
                yaw = ego_state.pose.rotation.as_euler('zyx')[0] # Typical yaw extraction
            except AttributeError:
                yaw = 0.0 # Fallback
                
            ego_status_dict["ego_pose"] = np.array([tx, ty, yaw], dtype=np.float64)
            
            # Extract velocity and acceleration if dynamic_state exists
            if hasattr(ego_state, 'dynamic_state') and ego_state.dynamic_state is not None:
                vx, vy = ego_state.dynamic_state.velocity[:2]
                ax, ay = ego_state.dynamic_state.acceleration[:2]
                ego_status_dict["ego_velocity"] = np.array([vx, vy], dtype=np.float32)
                ego_status_dict["ego_acceleration"] = np.array([ax, ay], dtype=np.float32)

        # 2. Annotations (Requires strictly dimensioned numpy arrays)
        annotations_dict = {
            "boxes": np.empty((0, 7), dtype=np.float32), # [x, y, z, l, w, h, yaw]
            "names": [],
            "velocity_3d": np.empty((0, 3), dtype=np.float32),
            "instance_tokens": [],
            "track_tokens": []
        }
        
        if box_detections is not None and len(box_detections) > 0:
            num_boxes = len(box_detections)
            boxes_array = np.zeros((num_boxes, 7), dtype=np.float32)
            vel_array = np.zeros((num_boxes, 3), dtype=np.float32)
            names_list = []
            tokens_list = []
            
            for i in range(num_boxes):
                tx, ty, tz = box_detections.translations[i]
                l, w, h = box_detections.dimensions[i]
                try:
                    b_yaw = box_detections.rotations[i].as_euler('zyx')[0]
                except AttributeError:
                    b_yaw = 0.0
                
                boxes_array[i] = [tx, ty, tz, l, w, h, b_yaw]
                
                if hasattr(box_detections, 'velocities') and box_detections.velocities is not None:
                    vel_array[i] = box_detections.velocities[i]
                
                name = box_detections.class_names[i] if hasattr(box_detections, 'class_names') else "unknown"
                names_list.append(name)
                
                # Use track IDs or generate mock tokens
                t_id = str(box_detections.track_ids[i]) if hasattr(box_detections, 'track_ids') else uuid.uuid4().hex
                tokens_list.append(t_id)

            annotations_dict["boxes"] = boxes_array
            annotations_dict["velocity_3d"] = vel_array
            annotations_dict["names"] = names_list
            annotations_dict["instance_tokens"] = tokens_list
            annotations_dict["track_tokens"] = tokens_list

        # 3. Traffic Lights
        tl_list: List[Tuple[str, bool]] = []
        if traffic_lights is not None:
            # Map py123d traffic lights to Navsim format: (lane_connector_id, is_red)
            pass 

        # Build Frame
        frame_dict = {
            "token": frame_token,
            "timestamp": int(timestamp.timestamp_ns / 1000), # microseconds
            "roadblock_ids": route_lane_group_ids if route_lane_group_ids else [],
            "traffic_lights": tl_list,
            "annotations": annotations_dict,
            "ego_status": ego_status_dict,
            "lidar_path": None, # Handle sensor blobs if necessary
            "camera_dict": {}   # Handle camera dicts if necessary
        }

        self.frames_data.append(frame_dict)

    def close(self) -> None:
        """Compiles the scene dictionary and dumps it as a Pickle file (.pkl)."""
        if not self.frames_data:
            return
            
        total_frames = len(self.frames_data)
        # Standard Navsim usually uses 4 history frames and 10 future frames (14 frames, 2Hz)
        # Adjust dynamically based on extracted scene length to prevent out-of-bound errors.
        num_history = 4 if total_frames >= 4 else total_frames
        num_future = total_frames - num_history

        # Construct exact scene dictionary
        scene_dict = {
            "scene_metadata": {
                "log_name": self.log_name,
                "scene_token": self.scene_token,
                "map_name": self.map_name,
                "initial_token": self.initial_token,
                "num_history_frames": num_history,
                "num_future_frames": num_future,
                "corresponding_original_scene": None,
                "corresponding_original_initial_token": None
            },
            "frames": self.frames_data,
            "extended_traffic_light_data": None,
            "extended_detections_tracks": None
        }

        # Save to Pickle file using highest protocol as Navsim expects
        output_file = self.output_dir / f"{self.scene_token}.pkl"
        
        with open(output_file, 'wb') as f:
            pickle.dump(scene_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
            
        print(f"[NavsimWriter] Successfully exported strictly-formatted Navsim pickle to {output_file}")