# 视频上传处理系统 - 接口文档 (API Interface)

本文档定义了前端 Vue 项目与后端服务之间的通信协议。

## 1. 基础信息
- **Base URL**: `http://<server-ip>:端口/api`
- **数据格式**: 请求 `multipart/form-data` (上传) 或 `application/json`；响应均为 `application/json`

---

## 2. 接口定义

### 2.1 视频上传
- **URL**: `/upload`
- **Method**: `POST`
- **Content-Type**: `multipart/form-data`
- **请求参数**:
  | 参数名 | 类型 | 必选 | 说明 |
  | :--- | :--- | :--- | :--- |
  | video | File | 是 | 视频二进制文件 |
- **成功响应**:
  ```json
  {
    "code": 200,
    "message": "上传成功",
    "data": {
      "task_id": "uuid-123456789"
    }
  }
  ```

### 2.2 处理进度与结果查询
- **URL**: `/status`
- **Method**: `GET`
- **请求参数**:
  | 参数名 | 类型 | 必选 | 说明 |
  | :--- | :--- | :--- | :--- |
  | task_id | String | 是 | 上传接口返回的任务唯一 ID |
- **响应格式 (处理中)**:
  ```json
  {
    "code": 200,
    "data": {
      "status": "processing",
      "progress": 45,
      "message": "正在进行 AI 识别..."
    }
  }
  ```
- **响应格式 (已完成)**:
  ```json
  {
    "code": 200,
    "data": {
      "status": "completed",
      "progress": 100,
      "processed_video_url": "http://cdn.example.com/output.mp4",
      "result": {
        "filename": "video.mp4",
        "duration": "00:30",
        "resolution": "1920x1080",
        "processed_at": "2026-03-06 14:00:00",
        "detections": [
          {
            "timestamp": "00:05",
            "horse_id": "H001",
            "person_name": "张三",
            "confidence": "0.98_0.95"
          },
          {
            "timestamp": "00:12",
            "horse_id": "H005",
            "person_name": "李四",
            "confidence": "0.92_0.88"
          }
        ]
      }
    }
  }
  ```

---

## 3. 前端逻辑映射 (App.vue)

### 上传逻辑
前端使用 `axios.post` 配合 `onUploadProgress` 钩子实时更新**上传进度条**。

### 处理逻辑
前端在上传成功后，会开启一个定时器（或使用 WebSocket）轮询 `/status` 接口，直到状态变为 `completed`。
- 状态 `completed` 后，前端会自动：
  1. 解锁“处理后视频”切换按钮。
  2. 自动切换播放器至处理后的 URL。
  3. 将“开始处理”按钮变更为“上传新视频”。
