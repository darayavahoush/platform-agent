"""Member 1: CNN (frame -> entity/player detections) + LSTM (detections over
time -> tracked entities & predicted trajectories for moving platforms/enemies).
Implement PerceptionModule from core.interfaces. Do not import from the other
3 member packages.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from torch import nn

from core.interfaces import Entity, Frame, PerceptionModule, PerceptionOutput


class SmallColorCNN(nn.Module):
    """Tiny CNN feature extractor for the demo scene. It is intentionally small
    so it can run quickly on the 40x800 synthetic RGB frames used in this project.
    The feature output is kept separate from the temporary synthetic detection
    step so a trained detector can replace the color logic later without altering
    the public `process()` interface.
    """

    def __init__(self, in_channels: int = 3, hidden_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(hidden_dim, hidden_dim * 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net(x)
        return x.flatten(1)


class CNNLSTMPerception(PerceptionModule):
    def __init__(self, history_len: int = 8, horizon: int = 10):
        self.history_len = history_len
        self.horizon = horizon
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cnn = SmallColorCNN().to(self.device)
        self.lstm = nn.LSTM(input_size=64, hidden_size=64, num_layers=1, batch_first=True).to(self.device)
        # Separate forecasting head: uses the temporal embedding + recent motion
        # trend to predict short-horizon future positions. It is intentionally
        # lightweight and independent of ground-truth state.
        self.forecast_head = nn.Sequential(
            nn.Linear(66, 32),
            nn.ReLU(),
            nn.Linear(32, 2 * self.horizon),
        ).to(self.device)
        self._track_counter = 0
        self._active_tracks: dict[str, dict] = {}
        self._object_history: dict[str, list[dict]] = {}
        self._frame_history: list[Frame] = []

    def _new_track_id(self, kind: str) -> str:
        self._track_counter += 1
        return f"{kind}_{self._track_counter}"

    def _match_detections_to_tracks(self, detections: list[dict], previous_tracks: dict[str, dict], tick: int) -> dict[int, str]:
        assignments: dict[int, str] = {}
        used_ids: set[str] = set()

        for det_idx, det in enumerate(detections):
            best_track_id = None
            best_score = None
            best_threshold = 80.0

            for track_id, prev in previous_tracks.items():
                if track_id in used_ids:
                    continue
                if prev["kind"] != det["kind"]:
                    continue

                dx = det["x"] - prev["x"]
                dy = det["y"] - prev["y"]
                dist = float(np.hypot(dx, dy))
                size_penalty = abs(det["width"] - prev["width"]) + abs(det["height"] - prev["height"])
                score = dist + 0.1 * size_penalty
                threshold = 80.0 if det["kind"] != "platform" else 150.0
                if best_score is None or score < best_score:
                    best_track_id = track_id
                    best_score = score
                    best_threshold = threshold

            if best_track_id is not None and best_score is not None and best_score <= best_threshold:
                assignments[det_idx] = best_track_id
                used_ids.add(best_track_id)

        for det_idx, det in enumerate(detections):
            if det_idx in assignments:
                continue
            assignments[det_idx] = self._new_track_id(det["kind"])

        return assignments

    def _frame_to_tensor(self, frame: Frame) -> torch.Tensor:
        rgb = np.asarray(frame.rgb, dtype=np.float32) / 255.0
        rgb = np.transpose(rgb, (2, 0, 1))
        tensor = torch.from_numpy(rgb).unsqueeze(0).to(self.device)
        return tensor

    def _encode_frame(self, frame: Frame) -> np.ndarray:
        tensor = self._frame_to_tensor(frame)
        with torch.no_grad():
            embedding = self.cnn(tensor)
        return embedding.detach().cpu().numpy().reshape(-1).astype(np.float32)

    def _temporal_embedding(self, history: list[Frame]) -> np.ndarray:
        seq_frames = history[-self.history_len:] if history else [history[-1]] if history else []
        if not seq_frames:
            return np.zeros(64, dtype=np.float32)

        with torch.no_grad():
            features = []
            for hist_frame in seq_frames:
                feat = self.cnn(self._frame_to_tensor(hist_frame)).squeeze(0)
                features.append(feat)
            sequence = torch.stack(features, dim=0).unsqueeze(0).to(self.device)  # [1, T, 64]
            lstm_out, _ = self.lstm(sequence)
            embedding = lstm_out[:, -1, :].squeeze(0)
        return embedding.detach().cpu().numpy().astype(np.float32)

    def _forecast_entity_trajectory(self, entity: dict, embedding: np.ndarray) -> list[tuple[float, float]]:
        hist = self._object_history.get(entity["id"], [])
        if hist:
            recent = hist[-min(len(hist), 5):]
            recent_vx = float(np.mean([p["vx"] for p in recent]))
            recent_vy = float(np.mean([p["vy"] for p in recent]))
            x0 = float(recent[-1]["x"])
            y0 = float(recent[-1]["y"])
        else:
            recent_vx = float(entity.get("vx", 0.0))
            recent_vy = float(entity.get("vy", 0.0))
            x0 = float(entity["x"])
            y0 = float(entity["y"])

        motion = np.array([recent_vx, recent_vy], dtype=np.float32)
        model_input = torch.tensor(np.concatenate([embedding.astype(np.float32), motion]), dtype=torch.float32, device=self.device)
        with torch.no_grad():
            forecast = self.forecast_head(model_input)
        deltas = forecast.detach().cpu().numpy().astype(np.float32)
        dx = deltas[:self.horizon]
        dy = deltas[self.horizon:]
        dx = dx * (0.5 + abs(recent_vx) * 1.5)
        dy = dy * (0.5 + abs(recent_vy) * 1.5)

        traj: list[tuple[float, float]] = []
        x = x0
        y = y0
        for i in range(self.horizon):
            x += float(dx[i])
            y += float(dy[i])
            traj.append((x, y))
        return traj

    def _forecast_trajectories(self, embedding: np.ndarray, active_tracks: dict[str, dict]) -> dict[str, list[tuple[float, float]]]:
        trajs: dict[str, list[tuple[float, float]]] = {}
        for track_id, entity in active_tracks.items():
            if entity["kind"] not in {"moving_platform", "enemy"}:
                continue
            trajs[track_id] = self._forecast_entity_trajectory(entity, embedding)
        return trajs

    def _color_mask(self, rgb: np.ndarray, target: tuple[int, int, int], tol: int = 25) -> np.ndarray:
        target_arr = np.asarray(target, dtype=np.int16)
        diff = np.abs(rgb.astype(np.int16) - target_arr[None, None, :])
        return np.max(diff, axis=2) <= tol

    def _bbox_from_mask(self, mask: np.ndarray) -> Optional[tuple[int, int, int, int]]:
        ys, xs = np.where(mask)
        if len(xs) == 0:
            return None
        x0, y0 = int(xs.min()), int(ys.min())
        x1, y1 = int(xs.max()) + 1, int(ys.max()) + 1
        return x0, y0, x1 - x0, y1 - y0

    def _synthetic_detections(self, rgb: np.ndarray) -> list[tuple[str, float, float, float, float, float]]:
        detections: list[tuple[str, float, float, float, float, float]] = []

        # Platform / ground
        ground_mask = self._color_mask(rgb, (34, 139, 34), tol=30)
        ground_box = self._bbox_from_mask(ground_mask)
        if ground_box is not None:
            x, y, w, h = ground_box
            detections.append(("platform", float(x + w / 2), float(y + h / 2), float(w), float(h), 0.95))

        # Player
        player_mask = self._color_mask(rgb, (255, 215, 0), tol=30)
        player_box = self._bbox_from_mask(player_mask)
        if player_box is not None:
            x, y, w, h = player_box
            detections.append(("player", float(x + w / 2), float(y + h / 2), float(w), float(h), 0.96))

        # Moving platform
        moving_mask = self._color_mask(rgb, (160, 82, 45), tol=35)
        moving_box = self._bbox_from_mask(moving_mask)
        if moving_box is not None:
            x, y, w, h = moving_box
            detections.append(("moving_platform", float(x + w / 2), float(y + h / 2), float(w), float(h), 0.93))

        # Enemy
        enemy_mask = self._color_mask(rgb, (220, 50, 50), tol=35)
        enemy_box = self._bbox_from_mask(enemy_mask)
        if enemy_box is not None:
            x, y, w, h = enemy_box
            detections.append(("enemy", float(x + w / 2), float(y + h / 2), float(w), float(h), 0.92))
            detections.append(("hazard", float(x + w / 2), float(y + h / 2), float(w), float(h), 0.9))

        # Collectible
        collectible_mask = self._color_mask(rgb, (255, 255, 0), tol=25)
        collectible_box = self._bbox_from_mask(collectible_mask)
        if collectible_box is not None:
            x, y, w, h = collectible_box
            detections.append(("collectible", float(x + w / 2), float(y + h / 2), float(w), float(h), 0.94))

        return detections

    def process(self, frame: Frame, history: list[Frame]) -> PerceptionOutput:
        if frame is None:
            raise ValueError("frame must not be None")

        frame_history = history if history else [frame]
        self._frame_history = frame_history[-self.history_len:]

        rgb = np.asarray(frame.rgb, dtype=np.uint8)
        detections = self._synthetic_detections(rgb)

        previous_tracks = dict(self._active_tracks)
        assignments = self._match_detections_to_tracks(
            [{"kind": kind, "x": x, "y": y, "width": w, "height": h, "confidence": conf}
             for kind, x, y, w, h, conf in detections],
            previous_tracks,
            frame.tick,
        )

        entities: list[Entity] = []
        player_pose: Optional[Entity] = None
        confidences: list[float] = []
        active_tracks: dict[str, dict] = {}

        for det_idx, (kind, x, y, w, h, conf) in enumerate(detections):
            track_id = assignments[det_idx]
            prev = previous_tracks.get(track_id)
            if prev is not None:
                tick_delta = max(int(frame.tick - prev["tick"]), 1)
                vx = (float(x) - float(prev["x"])) / float(tick_delta)
                vy = (float(y) - float(prev["y"])) / float(tick_delta)
            else:
                vx = 0.0
                vy = 0.0

            entity = Entity(
                x=float(x),
                y=float(y),
                vx=vx,
                vy=vy,
                kind=kind,
                extra={
                    "id": track_id,
                    "width": float(w),
                    "height": float(h),
                    "confidence": float(conf),
                    "tick": int(frame.tick),
                },
            )
            entities.append(entity)
            confidences.append(float(conf))
            if kind == "player":
                player_pose = entity

            track_state = {
                "id": track_id,
                "kind": kind,
                "x": float(x),
                "y": float(y),
                "width": float(w),
                "height": float(h),
                "tick": int(frame.tick),
                "confidence": float(conf),
                "vx": vx,
                "vy": vy,
            }
            active_tracks[track_id] = track_state
            self._object_history.setdefault(track_id, []).append({
                "tick": int(frame.tick),
                "x": float(x),
                "y": float(y),
                "vx": vx,
                "vy": vy,
            })
            self._object_history[track_id] = self._object_history[track_id][-self.history_len:]

        self._active_tracks = active_tracks

        if player_pose is None:
            player_pose = Entity(x=0.0, y=0.0, kind="player", extra={"id": "player_0", "confidence": 0.0, "tick": int(frame.tick)})

        embedding_vector = self._temporal_embedding(frame_history)
        predicted_trajectories = self._forecast_trajectories(embedding_vector, active_tracks)

        confidence = float(np.mean(confidences)) if confidences else 0.0
        return PerceptionOutput(
            entities=entities,
            player_pose=player_pose,
            predicted_trajectories=predicted_trajectories,
            confidence=confidence,
            embedding=embedding_vector.astype(np.float32),
        )
