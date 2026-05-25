"""
map_overhead_widget.py
======================
A standalone Tkinter overhead (top-down) map widget for visualising a UAV
person-search mission in real time.

The widget can be embedded in any tk container (Frame, Toplevel, Tk root) by
passing it as the *parent* argument, or used standalone via the demo at the
bottom of this file.

Coordinate system
-----------------
World coordinates are in Unreal Engine cm-scale (x = forward, y = right).
The widget transforms them to canvas pixel space via world_to_canvas().

Color scheme
------------
Canvas background : #1e1e2e  (dark navy)
Grid lines        : #2a2a3e  (slightly lighter navy)
Text labels       : #e0e0f0  (off-white)
UAV marker        : red triangle + small circle
House status      :
    UNSEARCHED    鈥?gray fill,   gray outline
    IN_PROGRESS   鈥?yellow fill, orange outline (thick)
    EXPLORED      鈥?green fill,  dark-green outline
    PERSON_FOUND  鈥?red fill,    dark-red outline
Target indicator  鈥?double concentric red rings around the house circle

Usage
-----
    import tkinter as tk
    from map_overhead_widget import OverheadMapWidget

    root = tk.Tk()
    bounds = (1000, -500, 5000, 3000)   # (min_x, min_y, max_x, max_y) in cm
    widget = OverheadMapWidget(root, world_bounds=bounds)
    widget.canvas.pack(fill="both", expand=True)

    widget.update_uav(x=2400.0, y=100.0, yaw_deg=-90.0)
    widget.update_houses([
        {"id": "house_A", "name": "House A", "center_x": 2400.0, "center_y": 100.0,
         "radius_cm": 700.0, "status": "UNSEARCHED", "is_target": False},
    ])
    root.mainloop()
"""

from __future__ import annotations

import math
import tkinter as tk
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageTk

# ---------------------------------------------------------------------------
# Color constants
# ---------------------------------------------------------------------------

BG_COLOR      = "#1e1e2e"
GRID_COLOR    = "#2a2a3e"
TEXT_COLOR    = "#e0e0f0"
UAV_COLOR     = "#ff4444"
UAV_DOT_COLOR = "#ffaaaa"
CURRENT_RING_COLOR = "#55ccff"
ROUTE_COLOR = "#66ddff"
ROUTE_PLAN_COLOR = "#ffd166"
ROUTE_PLAN_ACTIVE_COLOR = "#ff5c5c"
ROUTE_PLAN_VISITED_COLOR = "#6ee7a8"
ROUTE_PLAN_TEXT_COLOR = "#fff3bf"
TRAJECTORY_COLOR = "#2dd4bf"
TRAJECTORY_DOT_COLOR = "#ccfbf1"
POINT_OVERLAY_COLOR = "#f97316"
POINT_OVERLAY_TEXT_COLOR = "#fed7aa"

# House status 鈫?(fill, outline, outline_width)
_STATUS_STYLE: Dict[str, Tuple[str, str, int]] = {
    "UNSEARCHED":   ("#888888", "#888888", 1),
    "IN_PROGRESS":  ("#ffdd44", "#ff8800", 3),
    "EXPLORED":     ("#44cc66", "#228844", 2),
    "PERSON_FOUND": ("#ee3333", "#990000", 2),
}
_DEFAULT_STYLE = ("#888888", "#888888", 1)

# Minimum pixel radius for drawn house circles
_MIN_CIRCLE_PX = 8


# ---------------------------------------------------------------------------
# Widget
# ---------------------------------------------------------------------------

class OverheadMapWidget:
    """
    Top-down overhead map widget.

    Parameters
    ----------
    parent      : tk widget that owns this canvas.
    world_bounds: (min_x, min_y, max_x, max_y) in world cm units.
    canvas_w    : canvas width in pixels.
    canvas_h    : canvas height in pixels.
    """

    def __init__(
        self,
        parent: tk.Widget,
        world_bounds: Tuple[float, float, float, float],
        canvas_w: int = 480,
        canvas_h: int = 380,
    ) -> None:
        self._world_min_x, self._world_min_y, \
        self._world_max_x, self._world_max_y = world_bounds
        self._canvas_w = canvas_w
        self._canvas_h = canvas_h

        # Canvas
        self.canvas = tk.Canvas(
            parent,
            width=canvas_w,
            height=canvas_h,
            bg=BG_COLOR,
            highlightthickness=0,
        )
        self.canvas.bind("<Button-1>", self._on_canvas_click)
        self.canvas.bind("<ButtonPress-1>", self._on_canvas_press)
        self.canvas.bind("<B1-Motion>", self._on_canvas_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_canvas_release)

        # Internal state
        self._uav_x: float   = 0.0
        self._uav_y: float   = 0.0
        self._uav_yaw: float = 0.0
        self._houses: List[dict] = []
        self._background_bgr: Optional[np.ndarray] = None
        self._background_photo: Optional[ImageTk.PhotoImage] = None
        self._route_target: Optional[Tuple[float, float]] = None
        self._route_plan: List[dict] = []
        self._trajectory_points: List[Tuple[float, float]] = []
        self._point_overlay_points: List[dict] = []
        self._image_size: Optional[Tuple[int, int]] = None
        self._image_canvas_rect: Optional[Tuple[float, float, float, float]] = None
        self._image_layer_offset_px: Tuple[float, float] = (0.0, 0.0)
        self._affine_world_to_image: Optional[np.ndarray] = None
        self._homography_world_to_image: Optional[np.ndarray] = None
        self._calibration_anchors: List[dict] = []
        self._calibration_anchor_canvas_points: Dict[str, Tuple[float, float, float]] = {}
        self._calibration_anchor_select_callback: Optional[Callable[[str, float, float], None]] = None
        self._calibration_anchor_drag_callback: Optional[Callable[[str, float, float], None]] = None
        self._drag_anchor_label: Optional[str] = None
        self._drag_anchor_start_canvas: Optional[Tuple[float, float]] = None
        self._drag_anchor_moved: bool = False
        self._house_boxes: List[dict] = []
        self._rect_select_enabled: bool = False
        self._rect_select_callback: Optional[Callable[[dict], None]] = None
        self._rect_start_canvas: Optional[Tuple[float, float]] = None
        self._rect_preview_canvas: Optional[Tuple[float, float, float, float]] = None

        # House bounding boxes in canvas pixels, keyed by house id,
        # used for click-hit detection: {id: (cx, cy, r_px)}
        self._house_canvas_circles: Dict[str, Tuple[float, float, float]] = {}

        # Optional callback: fn(house_id: str)
        self._click_callback: Optional[Callable[[str], None]] = None
        self._map_click_callback: Optional[Callable[[float, float], None]] = None

        # Draw initial background
        self._draw_grid()

    def _canvas_alive(self) -> bool:
        """Return whether the Tk canvas is still valid and not destroyed."""
        try:
            return bool(self.canvas is not None and int(self.canvas.winfo_exists()) == 1)
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def update_uav(self, x: float, y: float, yaw_deg: float) -> None:
        """Update the UAV marker position and heading, then redraw."""
        self._uav_x   = x
        self._uav_y   = y
        self._uav_yaw = yaw_deg
        self._redraw()

    def update_houses(self, houses: List[dict]) -> None:
        """
        Replace the house list and redraw.

        Each dict must contain:
            id, name, center_x, center_y, radius_cm, status, is_target
        """
        self._houses = list(houses)
        self._redraw()

    def set_click_callback(self, fn: Callable[[str], None]) -> None:
        """
        Register a callback that fires when the user clicks a house circle.
        The callback receives the house id as its sole argument.
        """
        self._click_callback = fn

    def clear(self) -> None:
        """Clear all drawn items (UAV, houses) but keep the grid background."""
        if not self._canvas_alive():
            return
        self.canvas.delete("dynamic")
        self._house_canvas_circles.clear()

    def resize_canvas(self, canvas_w: int, canvas_h: int) -> None:
        """Resize the drawing area and redraw map layers."""
        self._canvas_w = max(1, int(canvas_w))
        self._canvas_h = max(1, int(canvas_h))
        if self._canvas_alive():
            self.canvas.configure(width=self._canvas_w, height=self._canvas_h)
        self._redraw()

    def set_background_image(self, image_bgr: Optional[np.ndarray]) -> None:
        """
        Set an optional background image for the overhead map.

        The image is interpreted as a top-down map already aligned to the
        configured world bounds and is scaled to the canvas size.
        """
        self._background_bgr = None if image_bgr is None else image_bgr.copy()
        self._redraw()

    def set_image_layer_offset(self, dx_px: float, dy_px: float) -> None:
        """
        Shift only image-space layers in source-image pixels.

        The background image, saved house boxes, and calibration anchors move
        by this offset. World-space objects such as the UAV marker, trajectory,
        Route6_entrance_search plan, and world house circles do not move.
        """
        self._image_layer_offset_px = (float(dx_px), float(dy_px))
        self._redraw()

    def set_route_target(self, world_xy: Optional[Tuple[float, float]]) -> None:
        """
        Set an optional Route6_entrance_search target in world coordinates.

        When present, a dashed line is drawn from the UAV to the target house.
        """
        self._route_target = None if world_xy is None else (float(world_xy[0]), float(world_xy[1]))
        self._redraw()

    def set_route_plan(self, route_plan: Optional[Any]) -> None:
        """
        Set an optional multi-waypoint Route6_entrance_search plan in world coordinates.

        The accepted shape is either a list of point dictionaries or a dict
        containing route_points / waypoints. Each point may use x/y or
        world_x/world_y and optional label/status fields.
        """
        if route_plan is None:
            self._route_plan = []
            self._redraw()
            return
        active_index = None
        raw_points: Any = []
        if isinstance(route_plan, dict):
            raw_points = route_plan.get("route_points") or route_plan.get("waypoints") or []
            active_index = route_plan.get("active_waypoint_index")
        elif isinstance(route_plan, list):
            raw_points = route_plan
        points: List[dict] = []
        if isinstance(raw_points, list):
            for idx, point in enumerate(raw_points):
                if not isinstance(point, dict):
                    continue
                x_value = point.get("x", point.get("world_x"))
                y_value = point.get("y", point.get("world_y"))
                try:
                    wx = float(x_value)
                    wy = float(y_value)
                except Exception:
                    continue
                status = str(point.get("status", "") or "")
                if active_index is not None:
                    try:
                        if int(active_index) == idx and status not in {"visited", "done"}:
                            status = "active"
                    except Exception:
                        pass
                points.append(
                    {
                        "x": wx,
                        "y": wy,
                        "label": str(point.get("label", "") or f"R{idx}"),
                        "status": status,
                        "route_point_type": str(point.get("route_point_type", "") or ""),
                        "scan_id": str(point.get("scan_id", "") or ""),
                        "facade": str(point.get("facade", "") or ""),
                        "height_band": str(point.get("height_band", "") or ""),
                        "floor_index": point.get("floor_index"),
                        "safe_interval_index": point.get("safe_interval_index"),
                        "color": str(point.get("color", "") or ""),
                        "outline_color": str(point.get("outline_color", "") or ""),
                    }
                )
        self._route_plan = points
        self._redraw()

    def set_trajectory(self, points: Optional[Any]) -> None:
        """
        Set an optional UAV trajectory in world coordinates.

        The accepted shape is a list of dicts with x/y or world_x/world_y, or
        two-value tuples/lists.
        """
        normalized: List[Tuple[float, float]] = []
        if isinstance(points, list):
            for point in points:
                try:
                    if isinstance(point, dict):
                        x_value = point.get("x", point.get("world_x"))
                        y_value = point.get("y", point.get("world_y"))
                    else:
                        x_value = point[0]
                        y_value = point[1]
                    normalized.append((float(x_value), float(y_value)))
                except Exception:
                    continue
        self._trajectory_points = normalized
        self._redraw()

    def set_point_overlay_points(self, points: Optional[Any]) -> None:
        """Set unconnected world-space point overlays such as local obstacles."""
        normalized: List[dict] = []
        if isinstance(points, list):
            for idx, point in enumerate(points):
                if not isinstance(point, dict):
                    continue
                try:
                    wx = float(point.get("x", point.get("world_x")))
                    wy = float(point.get("y", point.get("world_y")))
                except Exception:
                    continue
                normalized.append(
                    {
                        "x": wx,
                        "y": wy,
                        "label": str(point.get("label", "") or f"P{idx}"),
                        "status": str(point.get("status", "") or ""),
                        "color": str(point.get("color", "") or POINT_OVERLAY_COLOR),
                        "outline_color": str(point.get("outline_color", "") or "#111827"),
                        "radius_px": point.get("radius_px", 4),
                    }
                )
        self._point_overlay_points = normalized
        self._redraw()

    def set_calibration(
        self,
        affine_world_to_image: Optional[List[List[float]]],
        image_size: Optional[Tuple[int, int]],
        anchors: Optional[List[dict]] = None,
        homography_world_to_image: Optional[List[List[float]]] = None,
    ) -> None:
        """Set optional world->image calibration for the background map."""
        self._affine_world_to_image = None if affine_world_to_image is None else np.asarray(affine_world_to_image, dtype=np.float32)
        self._homography_world_to_image = (
            None
            if homography_world_to_image is None
            else np.asarray(homography_world_to_image, dtype=np.float32)
        )
        self._image_size = None if image_size is None else (int(image_size[0]), int(image_size[1]))
        self._calibration_anchors = list(anchors or [])
        self._redraw()

    def set_map_click_callback(self, fn: Callable[[float, float], None]) -> None:
        """Register a callback for raw background clicks in image-pixel space."""
        self._map_click_callback = fn

    def set_calibration_anchor_callbacks(
        self,
        select_fn: Optional[Callable[[str, float, float], None]] = None,
        drag_fn: Optional[Callable[[str, float, float], None]] = None,
    ) -> None:
        """Register callbacks for selecting and dragging calibration anchors."""
        self._calibration_anchor_select_callback = select_fn
        self._calibration_anchor_drag_callback = drag_fn

    def set_house_boxes(self, house_boxes: List[dict]) -> None:
        """Set map-image rectangle annotations for houses."""
        self._house_boxes = list(house_boxes or [])
        self._redraw()

    def set_rect_select_callback(self, fn: Optional[Callable[[dict], None]]) -> None:
        """Register a callback for drag-rectangle selection in image-pixel space."""
        self._rect_select_callback = fn

    def set_rect_select_enabled(self, enabled: bool) -> None:
        """Enable or disable drag-rectangle selection mode."""
        self._rect_select_enabled = bool(enabled)
        self._rect_start_canvas = None
        self._rect_preview_canvas = None
        self._redraw()

    def world_to_canvas(self, wx: float, wy: float) -> Tuple[float, float]:
        """
        Transform world (wx, wy) in cm to canvas pixel coordinates.

        World x (forward/east)  鈫?canvas x (left 鈫?right)
        World y (right/south)   鈫?canvas y (top 鈫?bottom)
        A small margin of 5 % is kept on each side.
        """
        if self._image_size is not None:
            if self._homography_world_to_image is not None:
                point = self._homography_world_to_image @ np.asarray([float(wx), float(wy), 1.0], dtype=np.float32)
                if abs(float(point[2])) > 1e-9:
                    image_x = float(point[0] / point[2])
                    image_y = float(point[1] / point[2])
                    return self.image_to_canvas(image_x, image_y, apply_layer_offset=False)
            if self._affine_world_to_image is not None:
                image_x = (
                    float(self._affine_world_to_image[0, 0]) * float(wx)
                    + float(self._affine_world_to_image[0, 1]) * float(wy)
                    + float(self._affine_world_to_image[0, 2])
                )
                image_y = (
                    float(self._affine_world_to_image[1, 0]) * float(wx)
                    + float(self._affine_world_to_image[1, 1]) * float(wy)
                    + float(self._affine_world_to_image[1, 2])
                )
                return self.image_to_canvas(image_x, image_y, apply_layer_offset=False)

        margin_x = self._canvas_w * 0.05
        margin_y = self._canvas_h * 0.05
        avail_w  = self._canvas_w - 2 * margin_x
        avail_h  = self._canvas_h - 2 * margin_y

        world_w = self._world_max_x - self._world_min_x
        world_h = self._world_max_y - self._world_min_y

        # Avoid division by zero
        sx = avail_w / world_w if world_w > 0 else 1.0
        sy = avail_h / world_h if world_h > 0 else 1.0

        cx = margin_x + (wx - self._world_min_x) * sx
        cy = margin_y + (wy - self._world_min_y) * sy

        return cx, cy

    def image_to_canvas(
        self,
        image_x: float,
        image_y: float,
        *,
        apply_layer_offset: bool = True,
    ) -> Tuple[float, float]:
        """Convert background-image pixel coordinates to canvas coordinates."""
        if apply_layer_offset:
            image_x = float(image_x) + float(self._image_layer_offset_px[0])
            image_y = float(image_y) + float(self._image_layer_offset_px[1])
        if self._image_size is None:
            return float(image_x), float(image_y)
        if self._image_canvas_rect is not None:
            offset_x, offset_y, draw_w, draw_h = self._image_canvas_rect
            image_w = max(1, int(self._image_size[0]))
            image_h = max(1, int(self._image_size[1]))
            return (
                offset_x + float(image_x) * draw_w / float(image_w),
                offset_y + float(image_y) * draw_h / float(image_h),
            )
        image_w = max(1, int(self._image_size[0]))
        image_h = max(1, int(self._image_size[1]))
        return (
            float(image_x) * self._canvas_w / float(image_w),
            float(image_y) * self._canvas_h / float(image_h),
        )

    def canvas_to_image(
        self,
        canvas_x: float,
        canvas_y: float,
        *,
        apply_layer_offset: bool = True,
    ) -> Tuple[float, float]:
        """Convert canvas coordinates to background-image pixel coordinates."""
        if self._image_size is None:
            image_x, image_y = float(canvas_x), float(canvas_y)
            if apply_layer_offset:
                image_x -= float(self._image_layer_offset_px[0])
                image_y -= float(self._image_layer_offset_px[1])
            return image_x, image_y
        if self._image_canvas_rect is not None:
            offset_x, offset_y, draw_w, draw_h = self._image_canvas_rect
            image_w = max(1, int(self._image_size[0]))
            image_h = max(1, int(self._image_size[1]))
            image_x, image_y = (
                (float(canvas_x) - offset_x) * float(image_w) / max(1.0, draw_w),
                (float(canvas_y) - offset_y) * float(image_h) / max(1.0, draw_h),
            )
        else:
            image_w = max(1, int(self._image_size[0]))
            image_h = max(1, int(self._image_size[1]))
            image_x, image_y = (
                float(canvas_x) * float(image_w) / self._canvas_w,
                float(canvas_y) * float(image_h) / self._canvas_h,
            )
        if apply_layer_offset:
            image_x -= float(self._image_layer_offset_px[0])
            image_y -= float(self._image_layer_offset_px[1])
        return image_x, image_y

    # ------------------------------------------------------------------
    # Private drawing helpers
    # ------------------------------------------------------------------

    def _draw_grid(self) -> None:
        """Draw a subtle 10x10 grid on the canvas background."""
        if not self._canvas_alive():
            return
        self.canvas.delete("grid")
        cols = 10
        rows = 10
        for i in range(1, cols):
            x = self._canvas_w * i / cols
            self.canvas.create_line(
                x, 0, x, self._canvas_h,
                fill=GRID_COLOR, width=1, tags="grid"
            )
        for j in range(1, rows):
            y = self._canvas_h * j / rows
            self.canvas.create_line(
                0, y, self._canvas_w, y,
                fill=GRID_COLOR, width=1, tags="grid"
            )

    def _redraw(self) -> None:
        """Clear dynamic items and redraw everything."""
        if not self._canvas_alive():
            return
        try:
            self.canvas.delete("background")
            self.canvas.delete("dynamic")
        except tk.TclError:
            return
        self._house_canvas_circles.clear()
        self._calibration_anchor_canvas_points.clear()
        self._image_canvas_rect = None

        if self._background_bgr is not None:
            self._draw_background_image()
        else:
            self._background_photo = None

        # Draw houses first (so UAV appears on top)
        for house in self._houses:
            self._draw_house(house)

        if self._route_plan:
            self._draw_route_plan()
        elif self._route_target is not None:
            self._draw_route_line(*self._route_target)

        if self._trajectory_points:
            self._draw_trajectory()

        if self._point_overlay_points:
            self._draw_point_overlays()

        for house_box in self._house_boxes:
            self._draw_house_box(house_box)

        if self._rect_preview_canvas is not None:
            self._draw_rect_preview(*self._rect_preview_canvas)

        for anchor in self._calibration_anchors:
            self._draw_calibration_anchor(anchor)

        # Draw UAV marker
        cx, cy = self.world_to_canvas(self._uav_x, self._uav_y)
        self._draw_uav_marker(cx, cy, self._uav_yaw)

    def _draw_background_image(self) -> None:
        """Draw the optional top-down background image behind the grid/markers."""
        if not self._canvas_alive():
            return
        if self._background_bgr is None or self._background_bgr.size == 0:
            return
        image_h, image_w = self._background_bgr.shape[:2]
        scale = min(self._canvas_w / max(1, image_w), self._canvas_h / max(1, image_h))
        draw_w = max(1, int(round(image_w * scale)))
        draw_h = max(1, int(round(image_h * scale)))
        offset_x = 0.5 * (self._canvas_w - draw_w)
        offset_y = 0.5 * (self._canvas_h - draw_h)
        self._image_canvas_rect = (offset_x, offset_y, float(draw_w), float(draw_h))
        preview = cv2.resize(
            self._background_bgr,
            (draw_w, draw_h),
            interpolation=cv2.INTER_AREA,
        )
        preview_rgb = cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)
        self._background_photo = ImageTk.PhotoImage(Image.fromarray(preview_rgb))
        layer_offset_x = float(self._image_layer_offset_px[0]) * float(draw_w) / float(max(1, image_w))
        layer_offset_y = float(self._image_layer_offset_px[1]) * float(draw_h) / float(max(1, image_h))
        self.canvas.create_image(
            offset_x + layer_offset_x,
            offset_y + layer_offset_y,
            image=self._background_photo,
            anchor="nw",
            tags="background",
        )
        self.canvas.tag_lower("background", "grid")

    def _draw_house(self, house: dict) -> None:
        """Draw a single house circle, label, and optional target rings."""
        if not self._canvas_alive():
            return
        hid     = house.get("id", "?")
        name    = house.get("name", hid)
        wx      = float(house.get("center_x", 0.0))
        wy      = float(house.get("center_y", 0.0))
        r_cm    = float(house.get("radius_cm", 700.0))
        status  = str(house.get("status", "UNSEARCHED"))
        is_tgt  = bool(house.get("is_target", False))
        is_current = bool(house.get("is_current", False))

        cx, cy = self.world_to_canvas(wx, wy)

        # Scale radius: use the x scale factor (world_w 鈫?canvas_w)
        world_w = self._world_max_x - self._world_min_x
        margin_x = self._canvas_w * 0.05
        avail_w  = self._canvas_w - 2 * margin_x
        scale_x  = avail_w / world_w if world_w > 0 else 1.0
        r_px = max(_MIN_CIRCLE_PX, r_cm * scale_x)

        fill_color, outline_color, outline_width = _STATUS_STYLE.get(
            status, _DEFAULT_STYLE
        )

        # If this is the target, draw double concentric red rings first
        # (behind the main circle so they appear as a halo)
        if is_tgt:
            ring_gap = 4
            for ring_r in (r_px + ring_gap * 2, r_px + ring_gap):
                self.canvas.create_oval(
                    cx - ring_r, cy - ring_r,
                    cx + ring_r, cy + ring_r,
                    outline="#ff2222",
                    width=1,
                    tags="dynamic",
                )

        if is_current:
            current_r = r_px + 6
            self.canvas.create_oval(
                cx - current_r, cy - current_r,
                cx + current_r, cy + current_r,
                outline=CURRENT_RING_COLOR,
                width=2,
                dash=(5, 3),
                tags="dynamic",
            )

        # Main house circle
        self.canvas.create_oval(
            cx - r_px, cy - r_px,
            cx + r_px, cy + r_px,
            fill=outline_color if status == "IN_PROGRESS" else fill_color,
            outline=outline_color,
            width=outline_width,
            stipple="" if status != "IN_PROGRESS" else "",
            tags="dynamic",
        )

        # For IN_PROGRESS use a lighter inner fill to distinguish from outline
        if status == "IN_PROGRESS":
            inner_r = r_px - outline_width
            if inner_r > 2:
                self.canvas.create_oval(
                    cx - inner_r, cy - inner_r,
                    cx + inner_r, cy + inner_r,
                    fill=fill_color,
                    outline="",
                    tags="dynamic",
                )

        # House name label below the circle
        self.canvas.create_text(
            cx,
            cy + r_px + 8,
            text=name,
            fill=TEXT_COLOR,
            font=("Consolas", 8),
            anchor="n",
            tags="dynamic",
        )

        # Store canvas-space bounding circle for click detection
        self._house_canvas_circles[hid] = (cx, cy, r_px)

    def _draw_route_line(self, wx: float, wy: float) -> None:
        """Draw a dashed Route6_entrance_search line from the UAV to the current target."""
        if not self._canvas_alive():
            return
        ux, uy = self.world_to_canvas(self._uav_x, self._uav_y)
        tx, ty = self.world_to_canvas(wx, wy)
        self.canvas.create_line(
            ux, uy, tx, ty,
            fill=ROUTE_COLOR,
            width=2,
            dash=(8, 6),
            tags="dynamic",
        )
        dot_r = 4
        self.canvas.create_oval(
            tx - dot_r, ty - dot_r,
            tx + dot_r, ty + dot_r,
            fill=ROUTE_COLOR,
            outline="",
            tags="dynamic",
        )

    def _draw_route_plan(self) -> None:
        """Draw a multi-waypoint Route6_entrance_search plan from the UAV/map planner."""
        if not self._canvas_alive() or not self._route_plan:
            return
        canvas_points: List[Tuple[float, float, dict]] = []
        for point in self._route_plan:
            try:
                cx, cy = self.world_to_canvas(float(point["x"]), float(point["y"]))
            except Exception:
                continue
            canvas_points.append((cx, cy, point))
        if len(canvas_points) >= 2:
            for idx in range(len(canvas_points) - 1):
                x1, y1, _ = canvas_points[idx]
                x2, y2, next_point = canvas_points[idx + 1]
                point = canvas_points[idx][2]
                point_type = str(point.get("route_point_type", "") or "")
                next_type = str(next_point.get("route_point_type", "") or "")
                if "observation" in point_type or "observation" in next_type:
                    continue
                point_is_scan = point.get("route_point_type") == "scan_point" or bool(point.get("scan_id"))
                next_is_scan = next_point.get("route_point_type") == "scan_point" or bool(next_point.get("scan_id"))
                if point_is_scan and next_is_scan and point.get("facade") != next_point.get("facade"):
                    continue
                if point_is_scan and next_is_scan and point.get("facade") == next_point.get("facade"):
                    if (
                        point.get("safe_interval_index") is not None
                        and next_point.get("safe_interval_index") is not None
                        and str(point.get("safe_interval_index")) != str(next_point.get("safe_interval_index"))
                    ):
                        continue
                self.canvas.create_line(
                    x1,
                    y1,
                    x2,
                    y2,
                    fill=ROUTE_PLAN_COLOR,
                    width=3,
                    dash=(10, 5) if idx == 0 else (),
                    arrow=tk.LAST if idx == len(canvas_points) - 2 else tk.NONE,
                    tags="dynamic",
                )
        for idx, (cx, cy, point) in enumerate(canvas_points):
            status = str(point.get("status", "") or "")
            is_scan_point = point.get("route_point_type") == "scan_point" or bool(point.get("scan_id"))
            floor_index = 0
            try:
                floor_index = int(point.get("floor_index") or 0)
            except Exception:
                floor_index = 0
            if status in {"visited", "done", "captured"}:
                color = ROUTE_PLAN_VISITED_COLOR
            elif status == "active":
                color = ROUTE_PLAN_ACTIVE_COLOR
            elif status == "blocked":
                color = "#ef4444"
            elif is_scan_point:
                band_palette = ["#ffd166", "#73d2de", "#c084fc", "#fb7185"]
                color = band_palette[(max(1, floor_index) - 1) % len(band_palette)] if floor_index else "#ffd166"
            else:
                color = ROUTE_PLAN_COLOR
            custom_color = str(point.get("color", "") or "")
            if custom_color:
                color = custom_color
            outline_color = str(point.get("outline_color", "") or "#1a1a1a")
            radius = 6 if status == "active" else (4 if is_scan_point else 5)
            draw_cx = cx
            draw_cy = cy
            if is_scan_point and floor_index > 1:
                draw_cx += float(((floor_index - 1) % 4) - 1.5) * 3.0
                draw_cy -= float((floor_index - 1) // 4) * 3.0
            self.canvas.create_oval(
                draw_cx - radius,
                draw_cy - radius,
                draw_cx + radius,
                draw_cy + radius,
                fill=color,
                outline=outline_color,
                width=1,
                tags="dynamic",
            )
            label = str(point.get("label", "") or f"R{idx}")
            if is_scan_point and status not in {"active", "visited", "done", "captured"}:
                label = label if idx % 5 == 0 else ""
            if label:
                self.canvas.create_text(
                    draw_cx + 8,
                    draw_cy - 8,
                    text=label,
                    fill=ROUTE_PLAN_TEXT_COLOR,
                    font=("Consolas", 8, "bold"),
                    anchor="w",
                    tags="dynamic",
                )

    def _draw_trajectory(self) -> None:
        """Draw the recent UAV trajectory in world coordinates."""
        if not self._canvas_alive() or not self._trajectory_points:
            return
        canvas_points: List[Tuple[float, float]] = []
        for wx, wy in self._trajectory_points:
            try:
                canvas_points.append(self.world_to_canvas(float(wx), float(wy)))
            except Exception:
                continue
        if len(canvas_points) >= 2:
            flattened: List[float] = []
            for cx, cy in canvas_points:
                flattened.extend([cx, cy])
            self.canvas.create_line(
                *flattened,
                fill=TRAJECTORY_COLOR,
                width=2,
                smooth=True,
                tags="dynamic",
            )
        for cx, cy in canvas_points[-12:]:
            self.canvas.create_oval(
                cx - 2,
                cy - 2,
                cx + 2,
                cy + 2,
                fill=TRAJECTORY_DOT_COLOR,
                outline="",
                tags="dynamic",
            )

    def _draw_point_overlays(self) -> None:
        """Draw unconnected point overlays without affecting Route6_entrance_search lines."""
        if not self._canvas_alive() or not self._point_overlay_points:
            return
        for idx, point in enumerate(self._point_overlay_points):
            try:
                cx, cy = self.world_to_canvas(float(point["x"]), float(point["y"]))
            except Exception:
                continue
            try:
                radius = max(2.0, min(10.0, float(point.get("radius_px", 4) or 4)))
            except Exception:
                radius = 4.0
            color = str(point.get("color", "") or POINT_OVERLAY_COLOR)
            outline = str(point.get("outline_color", "") or "#111827")
            self.canvas.create_oval(
                cx - radius,
                cy - radius,
                cx + radius,
                cy + radius,
                fill=color,
                outline=outline,
                width=1,
                tags="dynamic",
            )
            label = str(point.get("label", "") or "")
            if label and idx % 20 == 0:
                self.canvas.create_text(
                    cx + 7,
                    cy + 6,
                    text=label,
                    fill=POINT_OVERLAY_TEXT_COLOR,
                    font=("Consolas", 7, "bold"),
                    anchor="w",
                    tags="dynamic",
                )

    def _draw_calibration_anchor(self, anchor: dict) -> None:
        """Draw a numbered calibration anchor marker on the map."""
        if not self._canvas_alive():
            return
        ix = float(anchor.get("image_x", 0.0))
        iy = float(anchor.get("image_y", 0.0))
        label = str(anchor.get("label", anchor.get("index", "")))
        status = str(anchor.get("status", "pending") or "pending").lower()
        cx, cy = self.image_to_canvas(ix, iy)
        if status == "active":
            fill = "#ffd166"
            outline = "#ff5c5c"
            text_fill = "#ffe08a"
            r = 9
            self.canvas.create_oval(
                cx - r - 5, cy - r - 5, cx + r + 5, cy + r + 5,
                outline="#ff5c5c", width=3, dash=(4, 3), tags="dynamic",
            )
        elif status == "done":
            fill = "#22c55e"
            outline = "#ffffff"
            text_fill = "#86efac"
            r = 8
        else:
            fill = "#00d5ff"
            outline = "#ffffff"
            text_fill = "#00f0ff"
            r = 7
        self._calibration_anchor_canvas_points[label] = (float(cx), float(cy), float(r) + 8.0)
        self.canvas.create_oval(
            cx - r - 3, cy - r - 3, cx + r + 3, cy + r + 3,
            fill="#101018", outline="#101018", width=1, tags="dynamic",
        )
        self.canvas.create_oval(
            cx - r, cy - r, cx + r, cy + r,
            fill=fill, outline=outline, width=2, tags="dynamic",
        )
        if label:
            self.canvas.create_text(
                cx + 9, cy - 1,
                text=label,
                fill="#101018",
                font=("Consolas", 9, "bold"),
                anchor="w",
                tags="dynamic",
            )
            self.canvas.create_text(
                cx + 8, cy - 2,
                text=label,
                fill=text_fill,
                font=("Consolas", 9, "bold"),
                anchor="w",
                tags="dynamic",
            )

    def _draw_house_box(self, house_box: dict) -> None:
        """Draw a saved house rectangle annotation in image-pixel space."""
        if not self._canvas_alive():
            return
        bbox = house_box.get("map_bbox_image", house_box)
        try:
            x1 = float(bbox["x1"]); y1 = float(bbox["y1"])
            x2 = float(bbox["x2"]); y2 = float(bbox["y2"])
        except Exception:
            return
        cx1, cy1 = self.image_to_canvas(x1, y1)
        cx2, cy2 = self.image_to_canvas(x2, y2)
        label = str(house_box.get("name") or house_box.get("id") or "")
        is_target = bool(house_box.get("is_target", False))
        is_current = bool(house_box.get("is_current", False))
        if is_target:
            outline = "#ff2222"
            width = 3
            label_fill = "#ff6666"
        elif is_current:
            outline = "#33ccff"
            width = 3
            label_fill = "#66ddff"
        else:
            outline = "#ff8800"
            width = 2
            label_fill = "#ffb347"
        self.canvas.create_rectangle(
            cx1, cy1, cx2, cy2,
            outline=outline,
            width=width,
            tags="dynamic",
        )
        if label:
            self.canvas.create_text(
                min(cx1, cx2) + 4,
                min(cy1, cy2) - 4,
                text=label,
                fill=label_fill,
                font=("Consolas", 9, "bold"),
                anchor="sw",
                tags="dynamic",
            )

    def _draw_rect_preview(self, x1: float, y1: float, x2: float, y2: float) -> None:
        """Draw the live drag-selection rectangle."""
        if not self._canvas_alive():
            return
        self.canvas.create_rectangle(
            x1, y1, x2, y2,
            outline="#00ffff",
            width=2,
            dash=(6, 4),
            tags="dynamic",
        )

    def _draw_uav_marker(self, cx: float, cy: float, yaw_deg: float) -> None:
        """
        Draw a compact UAV body marker with a forward arrow.

        For the current map convention, yaw_deg = 0 points to canvas-left.
        Positive yaw rotates clockwise in canvas space.
        """
        if not self._canvas_alive():
            return
        body_r = 5
        arrow_len = 18
        angle_rad = math.radians(yaw_deg + 180.0)
        base_x = cx + (body_r + 2) * math.cos(angle_rad)
        base_y = cy + (body_r + 2) * math.sin(angle_rad)
        tip_x = cx + arrow_len * math.cos(angle_rad)
        tip_y = cy + arrow_len * math.sin(angle_rad)

        self.canvas.create_line(
            base_x,
            base_y,
            tip_x,
            tip_y,
            fill="#ffffff",
            width=2,
            arrow="last",
            arrowshape=(10, 12, 4),
            tags="dynamic",
        )

        self.canvas.create_oval(
            cx - body_r,
            cy - body_r,
            cx + body_r,
            cy + body_r,
            fill=UAV_COLOR,
            outline="#ffffff",
            width=1,
            tags="dynamic",
        )

        dot_r = 2
        self.canvas.create_oval(
            cx - dot_r,
            cy - dot_r,
            cx + dot_r,
            cy + dot_r,
            fill=UAV_DOT_COLOR,
            outline="",
            tags="dynamic",
        )

    # ------------------------------------------------------------------
    # Click handling
    # ------------------------------------------------------------------

    def _hit_calibration_anchor(self, canvas_x: float, canvas_y: float) -> Optional[str]:
        for label, (cx, cy, radius) in self._calibration_anchor_canvas_points.items():
            if math.hypot(float(canvas_x) - cx, float(canvas_y) - cy) <= max(10.0, radius):
                return label
        return None

    def _emit_calibration_anchor_select(self, label: str, canvas_x: float, canvas_y: float) -> None:
        if self._calibration_anchor_select_callback is None:
            return
        image_x, image_y = self.canvas_to_image(float(canvas_x), float(canvas_y))
        self._calibration_anchor_select_callback(str(label), float(image_x), float(image_y))

    def _emit_calibration_anchor_drag(self, label: str, canvas_x: float, canvas_y: float) -> None:
        if self._calibration_anchor_drag_callback is None:
            return
        image_x, image_y = self.canvas_to_image(float(canvas_x), float(canvas_y))
        self._calibration_anchor_drag_callback(str(label), float(image_x), float(image_y))

    def _on_canvas_press(self, event: tk.Event) -> None:
        if not self._rect_select_enabled:
            label = self._hit_calibration_anchor(float(event.x), float(event.y))
            if label:
                self._drag_anchor_label = label
                self._drag_anchor_start_canvas = (float(event.x), float(event.y))
                self._drag_anchor_moved = False
                self._emit_calibration_anchor_select(label, float(event.x), float(event.y))
            return
        if not self._rect_select_enabled:
            return
        self._rect_start_canvas = (float(event.x), float(event.y))
        self._rect_preview_canvas = (float(event.x), float(event.y), float(event.x), float(event.y))
        self._redraw()

    def _on_canvas_drag(self, event: tk.Event) -> None:
        if self._drag_anchor_label:
            if self._drag_anchor_start_canvas is not None:
                sx, sy = self._drag_anchor_start_canvas
                if math.hypot(float(event.x) - sx, float(event.y) - sy) < 2.0 and not self._drag_anchor_moved:
                    return
            self._drag_anchor_moved = True
            self._emit_calibration_anchor_drag(self._drag_anchor_label, float(event.x), float(event.y))
            return
        if not self._rect_select_enabled or self._rect_start_canvas is None:
            return
        sx, sy = self._rect_start_canvas
        self._rect_preview_canvas = (sx, sy, float(event.x), float(event.y))
        self._redraw()

    def _on_canvas_release(self, event: tk.Event) -> None:
        if self._drag_anchor_label:
            if self._drag_anchor_moved:
                self._emit_calibration_anchor_drag(self._drag_anchor_label, float(event.x), float(event.y))
            self._drag_anchor_label = None
            self._drag_anchor_start_canvas = None
            self._drag_anchor_moved = False
            return
        if not self._rect_select_enabled or self._rect_start_canvas is None:
            return
        sx, sy = self._rect_start_canvas
        ex, ey = float(event.x), float(event.y)
        self._rect_start_canvas = None
        self._rect_preview_canvas = None
        self._redraw()
        if abs(ex - sx) < 4 or abs(ey - sy) < 4:
            return
        ix1, iy1 = self.canvas_to_image(min(sx, ex), min(sy, ey))
        ix2, iy2 = self.canvas_to_image(max(sx, ex), max(sy, ey))
        if self._rect_select_callback is not None:
            self._rect_select_callback(
                {
                    "x1": float(ix1),
                    "y1": float(iy1),
                    "x2": float(ix2),
                    "y2": float(iy2),
                }
            )

    def _on_canvas_click(self, event: tk.Event) -> None:
        """Find which house (if any) was clicked and fire the callback."""
        if self._rect_select_enabled:
            return
        ex, ey = event.x, event.y
        label = self._hit_calibration_anchor(float(ex), float(ey))
        if label:
            self._emit_calibration_anchor_select(label, float(ex), float(ey))
            return
        if self._click_callback is not None:
            for house_id, (cx, cy, r_px) in self._house_canvas_circles.items():
                dist = math.sqrt((ex - cx) ** 2 + (ey - cy) ** 2)
                if dist <= r_px:
                    self._click_callback(house_id)
                    return
        if self._map_click_callback is not None:
            image_x, image_y = self.canvas_to_image(ex, ey)
            self._map_click_callback(image_x, image_y)


# ---------------------------------------------------------------------------
# Standalone demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import random

    root = tk.Tk()
    root.title("Overhead Map 鈥?Demo")
    root.configure(bg=BG_COLOR)

    bounds = (1000.0, -500.0, 5000.0, 3000.0)
    widget = OverheadMapWidget(root, world_bounds=bounds, canvas_w=640, canvas_h=480)
    widget.canvas.pack(padx=8, pady=8)

    HOUSES = [
        {"id": "house_A", "name": "House A", "center_x": 2400.0, "center_y": 100.0,
         "radius_cm": 700.0, "status": "IN_PROGRESS", "is_target": True},
        {"id": "house_B", "name": "House B", "center_x": 3800.0, "center_y": 800.0,
         "radius_cm": 750.0, "status": "UNSEARCHED",  "is_target": False},
        {"id": "house_C", "name": "House C", "center_x": 2100.0, "center_y": 2200.0,
         "radius_cm": 680.0, "status": "EXPLORED",    "is_target": False},
    ]

    uav_x, uav_y, uav_yaw = 2400.0, 100.0, -90.0

    def click_handler(hid: str) -> None:
        print(f"Clicked house: {hid}")

    widget.set_click_callback(click_handler)

    def animate() -> None:
        global uav_x, uav_y, uav_yaw
        uav_x   += random.uniform(-20, 20)
        uav_y   += random.uniform(-20, 20)
        uav_yaw  = (uav_yaw + 5) % 360
        widget.update_uav(uav_x, uav_y, uav_yaw)
        widget.update_houses(HOUSES)
        root.after(150, animate)

    animate()
    root.mainloop()

