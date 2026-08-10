"""pytest 配置：默认离线 + 复用 backtest_report_data 的 stub 注入，开箱即用

效果：`cd scripts && pytest` 即可跑，无需手动设 PANDA_DATA_OFFLINE / SDK 凭证。
"""
import os
import sys

# 兄弟模块 import（factor/validate/backtest 与 test 同目录，插入 sys.path 保证任意启动目录可解析）
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# 默认离线：读 fixtures 不联网（测试不依赖网络与凭证）
os.environ.setdefault("PANDA_DATA_OFFLINE", "1")
# 复用 backtest_report_data 的 _inject_panda_data_stub_for_offline：SDK 缺失时注入空 stub，
# 让 import 链（data_loader → panda_data）在无 SDK 环境也通过；已装 SDK 则无副作用
import backtest_report_data  # noqa: F401,E402
