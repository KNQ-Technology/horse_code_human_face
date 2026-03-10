import atexit
import os
import shutil
import tempfile
from multiprocessing import current_process

import numpy as np

from config import Config


def create_process_tempfile(original_path: str) -> str:
    """创建进程专用的临时文件副本

    Args:
        original_path: 原始文件路径

    Returns:
        临时文件路径 (进程退出时自动删除)
    """
    # 生成进程唯一标识
    proc_id = current_process().ident
    suffix = f"{proc_id}_{os.path.basename(original_path)}"

    # 创建带进程标识的临时文件
    temp_file = tempfile.NamedTemporaryFile(
        suffix=suffix,
        delete=False  # 手动控制删除
    )
    temp_path = temp_file.name
    temp_file.close()

    # 高效拷贝大文件
    with open(original_path, 'rb') as f_src, open(temp_path, 'wb') as f_dst:
        shutil.copyfileobj(f_src, f_dst, length=1024 * 1024 * 8)  # 8MB块

    # 注册退出清理
    def cleanup():
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except:
                pass

    atexit.register(cleanup)
    return temp_path


# 使用示例
# FACE_EMBEDDING_DB = create_process_tempfile(
#     Config.get_config_val(Config.DATA, 'face_embedding_db')
# )


def compute_sim(feat1, feat2):
    # 将特征展平为一维向量
    feat1 = feat1.flatten()
    feat2 = feat2.flatten()
    # 计算余弦相似度
    sim = np.dot(feat1, feat2) / (np.linalg.norm(feat1) * np.linalg.norm(feat2))
    return sim


def normalize_vector(vector):
    norm = np.linalg.norm(vector)  # 计算向量的 L2 范数
    if norm > 0:
        return vector / norm
    else:
        return vector.copy()  # 或者直接 return vecto


"""
人脸识别器
"""

class FaceRecognizer:
    def __init__(self, threshold=0.55):
        self.threshold = threshold
        # self.temp_file_path = FACE_EMBEDDING_DB  # 保存临时文件路径
        # self.collection_name = "face_collection"  # 显式定义集合名称

    # def release_resources(self):
    #     """清理内存中的文件副本并断开客户端连接"""
    #     # 卸载集合（释放内存和锁）
    #     if self.client.has_collection(self.collection_name):
    #         self.client.release_collection(self.collection_name)  # 显式释放集合
    #     # 清理临时文件
    #     if os.path.exists(self.temp_file_path):
    #         try:
    #             os.remove(self.temp_file_path)
    #         except Exception as e:
    #             print(f'删除临时文件失败: {str(e)}')

    def recognize_face(self, feature):
        normalized_feature = normalize_vector(feature)
        # 加载集合到内存
        self.client.load_collection(collection_name=self.collection_name)

        # 执行相似性搜索
        search_params = {
            "metric_type": "IP",  # 使用内积（余弦相似度）
            "params": {"nprobe": 32}  # 搜索参数，nprobe 是 IVF 索引的探针数
        }
        results = self.client.search(
            collection_name=self.collection_name,  # 集合名称
            data=[normalized_feature],  # 查询向量，必须是二维数组
            search_params=search_params,  # 查询参数
            anns_field="embedding",  # 指定向量字段
            limit=5,  # 返回最相似的 5 条记录
            output_fields=["face_id", "org_code", "user_name"]  # 返回的字段
        )
        rst = []
        # 解析结果
        for hits in results:
            for hit in hits:
                score = hit['distance']
                if score > self.threshold:
                    entity = {
                        "face_id": hit['entity']["face_id"],
                        "user_name": hit['entity']["user_name"],
                        "score": score
                    }
                    rst.append(entity)
        return rst
