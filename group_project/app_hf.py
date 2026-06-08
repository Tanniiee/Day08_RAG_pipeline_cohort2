"""
Entry point cho Hugging Face Spaces.
HF Spaces tìm file app.py ở root của Space repo.
File này redirect về chatbot.py chính.

Deploy steps:
  1. Tạo Space mới tại https://huggingface.co/spaces
  2. SDK: Streamlit
  3. Push code lên Space repo
  4. Set Secrets: OPENAI_API_KEY
"""

# Re-export chatbot app — HF Spaces chỉ cần import là chạy
import sys
from pathlib import Path

# Thêm root vào sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Chạy chatbot
exec(open(Path(__file__).parent / "chatbot.py").read())
