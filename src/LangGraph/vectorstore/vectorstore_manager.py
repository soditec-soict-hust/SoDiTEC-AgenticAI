import json
import os
from datetime import datetime
from typing import Dict, List, Optional

class VectorStoreManager:
    """Quản lý lịch sử các vectorstore đã tạo"""
    
    def __init__(self, base_path: str = "src/LangGraph/vectorstore"):
        self.base_path = base_path
        self.history_file = os.path.join(base_path, "vectorstore_history.json")
        self.ensure_base_path()
    
    def ensure_base_path(self):
        """Đảm bảo thư mục base tồn tại"""
        if not os.path.exists(self.base_path):
            os.makedirs(self.base_path)
    
    def load_history(self) -> List[Dict]:
        """Load lịch sử từ file JSON"""
        if not os.path.exists(self.history_file):
            return []
        
        try:
            with open(self.history_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading history: {e}")
            return []
    
    def save_history(self, history: List[Dict]):
        """Lưu lịch sử vào file JSON"""
        try:
            with open(self.history_file, 'w', encoding='utf-8') as f:
                json.dump(history, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Error saving history: {e}")
    
    def add_file_record(self, filename: str, vectorstore_path: str, embedding_model: str) -> Dict:
        """Thêm record mới cho file đã upload"""
        history = self.load_history()
        
        record = {
            "id": len(history) + 1,
            "filename": filename,
            "vectorstore_path": vectorstore_path,
            "embedding_model": embedding_model,
            "upload_time": datetime.now().isoformat(),
            "status": "active"
        }
        
        history.append(record)
        self.save_history(history)
        return record
    
    def get_active_files(self) -> List[Dict]:
        """Lấy danh sách các file đang active"""
        history = self.load_history()
        active_files = []
        
        for record in history:
            if record.get("status") == "active" and os.path.exists(record.get("vectorstore_path", "")):
                active_files.append(record)
        
        return active_files
    
    def get_file_by_path(self, vectorstore_path: str) -> Optional[Dict]:
        """Tìm file theo vectorstore path"""
        history = self.load_history()
        for record in history:
            if record.get("vectorstore_path") == vectorstore_path:
                return record
        return None
    
    def delete_file_record(self, vectorstore_path: str) -> bool:
        """Xóa record và vectorstore của file"""
        history = self.load_history()
        updated_history = []
        deleted = False
        
        for record in history:
            if record.get("vectorstore_path") == vectorstore_path:
                # Xóa thư mục vectorstore
                try:
                    import shutil
                    import time
                    if os.path.exists(vectorstore_path):
                        shutil.rmtree(vectorstore_path)
                        time.sleep(0.5)
                    deleted = True
                except Exception as e:
                    print(f"Error deleting vectorstore {vectorstore_path}: {e}")
                    # Đánh dấu là inactive thay vì xóa hoàn toàn
                    record["status"] = "inactive"
                    updated_history.append(record)
            else:
                updated_history.append(record)
        
        self.save_history(updated_history)
        return deleted
    
    def cleanup_invalid_records(self):
        """Dọn dẹp các record không hợp lệ (vectorstore không tồn tại)"""
        history = self.load_history()
        valid_history = []
        
        for record in history:
            vectorstore_path = record.get("vectorstore_path", "")
            if os.path.exists(vectorstore_path):
                valid_history.append(record)
            else:
                print(f"Removing invalid record: {record.get('filename')} - {vectorstore_path}")
        
        self.save_history(valid_history)
