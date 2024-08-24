"""matplotlib head for t480s - wsl2 ubuntu"""
import matplotlib as mpl
font_name = "simhei"
mpl.rcParams['font.family'] = font_name # 指定字体，实际上相当于修改 matplotlibrc 文件　只不过这样做是暂时的　下次失效
mpl.rcParams['axes.unicode_minus'] = False # 正确显示负号，防止变成方框
import matplotlib.pyplot as plt
plt.rc("figure", figsize=(6, 3))
plt.rc("font", size=9)
plt.rc("savefig", dpi=100)
