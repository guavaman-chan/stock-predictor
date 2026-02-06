"""
雲端儲存模組
使用 Supabase Storage 來持久化儲存模型和回饋資料
"""

import os
import io
import json
from datetime import datetime
from typing import Optional, Dict, Any

try:
    from supabase import create_client, Client
    SUPABASE_AVAILABLE = True
except ImportError:
    SUPABASE_AVAILABLE = False
    print("Warning: supabase package not installed, cloud storage disabled")


class CloudStorage:
    """Supabase 雲端儲存管理器"""
    
    def __init__(self):
        self.client: Optional[Client] = None
        self.enabled = False
        self._init_client()
    
    def _init_client(self):
        """初始化 Supabase 客戶端"""
        if not SUPABASE_AVAILABLE:
            return
        
        # 嘗試從 Streamlit secrets 或環境變數獲取設定
        url = None
        key = None
        
        try:
            import streamlit as st
            if hasattr(st, 'secrets') and 'supabase' in st.secrets:
                url = st.secrets['supabase']['url']
                key = st.secrets['supabase']['key']
        except:
            pass
        
        # 備用：從環境變數讀取
        if not url:
            url = os.environ.get('SUPABASE_URL')
            key = os.environ.get('SUPABASE_KEY')
        
        if url and key:
            try:
                self.client = create_client(url, key)
                self.enabled = True
                print("✅ Supabase 雲端儲存已啟用")
            except Exception as e:
                print(f"⚠️ Supabase 連線失敗: {e}")
    
    def upload_file(self, bucket: str, file_path: str, file_name: str) -> bool:
        """
        上傳檔案到 Supabase Storage
        
        Args:
            bucket: Storage bucket 名稱 (models 或 feedback)
            file_path: 本地檔案路徑
            file_name: 遠端檔案名稱
            
        Returns:
            是否上傳成功
        """
        if not self.enabled or not self.client:
            return False
        
        try:
            with open(file_path, 'rb') as f:
                file_data = f.read()
            
            # 先嘗試刪除舊檔案（如果存在）
            try:
                self.client.storage.from_(bucket).remove([file_name])
            except:
                pass
            
            # 上傳新檔案
            result = self.client.storage.from_(bucket).upload(
                file_name,
                file_data,
                file_options={"content-type": "application/octet-stream"}
            )
            print(f"✅ 已上傳 {file_name} 到 {bucket}")
            return True
            
        except Exception as e:
            print(f"⚠️ 上傳失敗: {e}")
            return False
    
    def download_file(self, bucket: str, file_name: str, local_path: str) -> bool:
        """
        從 Supabase Storage 下載檔案
        
        Args:
            bucket: Storage bucket 名稱
            file_name: 遠端檔案名稱
            local_path: 本地儲存路徑
            
        Returns:
            是否下載成功
        """
        if not self.enabled or not self.client:
            return False
        
        try:
            # 確保目錄存在
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            
            # 下載檔案
            response = self.client.storage.from_(bucket).download(file_name)
            
            with open(local_path, 'wb') as f:
                f.write(response)
            
            print(f"✅ 已下載 {file_name} 到 {local_path}")
            return True
            
        except Exception as e:
            print(f"⚠️ 下載失敗 ({file_name}): {e}")
            return False
    
    def file_exists(self, bucket: str, file_name: str) -> bool:
        """檢查遠端檔案是否存在"""
        if not self.enabled or not self.client:
            return False
        
        try:
            files = self.client.storage.from_(bucket).list()
            return any(f['name'] == file_name for f in files)
        except:
            return False
    
    def list_files(self, bucket: str) -> list:
        """列出 bucket 中的所有檔案"""
        if not self.enabled or not self.client:
            return []
        
        try:
            files = self.client.storage.from_(bucket).list()
            return [f['name'] for f in files]
        except:
            return []
    
    def upload_json(self, bucket: str, file_name: str, data: dict) -> bool:
        """上傳 JSON 資料"""
        if not self.enabled or not self.client:
            return False
        
        try:
            json_bytes = json.dumps(data, ensure_ascii=False, default=str).encode('utf-8')
            
            # 先嘗試刪除舊檔案
            try:
                self.client.storage.from_(bucket).remove([file_name])
            except:
                pass
            
            result = self.client.storage.from_(bucket).upload(
                file_name,
                json_bytes,
                file_options={"content-type": "application/json"}
            )
            print(f"✅ 已上傳 JSON {file_name}")
            return True
            
        except Exception as e:
            print(f"⚠️ JSON 上傳失敗: {e}")
            return False
    
    def download_json(self, bucket: str, file_name: str) -> Optional[dict]:
        """下載 JSON 資料"""
        if not self.enabled or not self.client:
            return None
        
        try:
            response = self.client.storage.from_(bucket).download(file_name)
            return json.loads(response.decode('utf-8'))
        except Exception as e:
            print(f"⚠️ JSON 下載失敗: {e}")
            return None


# 全域實例
_cloud_storage = None

def get_cloud_storage() -> CloudStorage:
    """獲取雲端儲存實例（單例模式）"""
    global _cloud_storage
    if _cloud_storage is None:
        _cloud_storage = CloudStorage()
    return _cloud_storage
