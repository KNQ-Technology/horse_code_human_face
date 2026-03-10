import cv2
import numpy as np


class FaceDetector:

    def __init__(self, detect_model, recognize_model, probability):
        self._face_detect_model = detect_model
        self._recognize_model = recognize_model
        self._probability = probability

    def detect_boxes(self, face_img):
        if face_img is None or face_img.size == 0:
            return None
        detections = self._face_detect_model.detect(face_img, input_size=(640, 640), thresh=self._probability)
        return detections

    def detect_with_keypoints(self, face_img):
        """Detect faces and return both bounding boxes and keypoints."""
        if face_img is None or face_img.size == 0:
            return np.empty((0, 5)), None
        bboxes, kpss = self._face_detect_model.detect(
            face_img, input_size=(640, 640), thresh=self._probability,
        )
        if bboxes is None or len(bboxes) == 0:
            return np.empty((0, 5)), None
        return bboxes, kpss

    def get_embedding_from_kps(self, frame, kps):
        """Align face using existing keypoints and extract ArcFace embedding
        (skips redundant re-detection)."""
        feat, _ = self._recognize_model.get_feat_and_face(frame, kps)
        return np.array(feat)

    def detect_features(self, face_img):
        """Detect single largest face and return its embedding."""
        if face_img is None or face_img.size == 0:
            return np.array([])
        bboxes, kpss = self._face_detect_model.autodetect(face_img, max_num=1)
        if bboxes.shape[0] == 0:
            return np.array([])
        feat, _ = self._recognize_model.get_feat_and_face(face_img, kpss[0])
        return np.array(feat)
