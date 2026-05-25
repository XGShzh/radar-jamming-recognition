"""
字体配置辅助模块
在有中文字体的系统上启用中文，否则使用英文标签回退。
在 main.py / visualizer.py 开头调用 setup_fonts() 即可。
"""
import matplotlib
import matplotlib.pyplot as plt
import subprocess, sys


def setup_fonts():
    """
    尝试配置中文字体（SimHei / WenQuanYi Zen Hei 等）。
    若失败则回退到英文模式（所有中文标签自动转英文）。
    """
    candidates = [
        'SimHei', 'WenQuanYi Zen Hei', 'WenQuanYi Micro Hei',
        'Noto Sans CJK SC', 'PingFang SC', 'Heiti SC',
        'Microsoft YaHei', 'FZShuTi', 'STSong',
    ]
    from matplotlib import font_manager
    available = {f.name for f in font_manager.fontManager.ttflist}

    for candidate in candidates:
        if candidate in available:
            plt.rcParams['font.family'] = candidate
            plt.rcParams['axes.unicode_minus'] = False
            print(f"[Font] 使用中文字体：{candidate}")
            return True

    # 尝试安装（Linux）
    try:
        subprocess.run(['apt-get', 'install', '-y', '-q',
                        'fonts-wqy-zenhei'], capture_output=True, timeout=30)
        font_manager._load_fontmanager(try_read_cache=False)
        plt.rcParams['font.family'] = 'WenQuanYi Zen Hei'
        print("[Font] 安装并启用 WenQuanYi Zen Hei")
        return True
    except Exception:
        pass

    # 回退：英文模式
    plt.rcParams['font.family'] = 'DejaVu Sans'
    print("[Font] 回退英文模式（中文标签将以拼音/英文显示）")
    return False
