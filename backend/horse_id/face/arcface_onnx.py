import cv2
import onnx
import onnxruntime

__all__ = [
    'ArcFaceONNX',
]

from horse_id.face.face_align import norm_crop


class ArcFaceONNX:
    def __init__(self, model_file, providers=None):
        if providers is None:
            providers = ["CPUExecutionProvider"]
        find_sub = False
        find_mul = False
        model = onnx.load(model_file)
        graph = model.graph # 获取模型结构
        for nid, node in enumerate(graph.node[:8]):
            # print(nid, node.name)
            if node.name.startswith('Sub') or node.name.startswith('_minus'):
                find_sub = True
            if node.name.startswith('Mul') or node.name.startswith('_mul'):
                find_mul = True
        if find_sub and find_mul:
            # mxnet arcface model
            input_mean = 0.0
            input_std = 1.0
        else:
            input_mean = 127.5
            input_std = 127.5
        self.input_mean = input_mean
        self.input_std = input_std
        self.session = onnxruntime.InferenceSession(model_file, providers=providers)
        input_cfg = self.session.get_inputs()[0]
        input_shape = input_cfg.shape
        input_name = input_cfg.name
        self.input_size = tuple(input_shape[2:4][::-1])
        outputs = self.session.get_outputs()
        output_names = []
        for out in outputs:
            output_names.append(out.name)
        self.input_name = input_name
        self.output_names = output_names
        assert len(self.output_names) == 1
        self.output_shape = outputs[0].shape

    def get_feat_and_face(self, img, kps):
        """
        获取人脸特征值和人脸图片
        :param img: 传入人脸图片
        :param kps: 人脸识别点位
        :return:
        """
        crop_img = norm_crop(img, landmark=kps, image_size=self.input_size[0])
        embedding = self.get_feat(crop_img)
        return embedding, crop_img

    def get_feat(self, imgs):
        """
        获取人脸特征值
        :param imgs:
        :return:
        """
        if not isinstance(imgs, list):
            imgs = [imgs]

        blob = cv2.dnn.blobFromImages(imgs, 1.0 / self.input_std, self.input_size,
                                      (self.input_mean, self.input_mean, self.input_mean), swapRB=True)
        net_out = self.session.run(self.output_names, {self.input_name: blob})[0]
        return net_out.flatten()
