# logging_config.py
import logging

# 配置日志记录器
logging.basicConfig(level=logging.INFO)  # 默认级别为INFO
logger = logging.getLogger(__name__)