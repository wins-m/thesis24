import os
os.chdir('/home/winston/Thesis')
import sys
sys.path.append('/home/winston/Thesis/bin')
from plt_head import *
import numpy as np
import pandas as pd

src = 'results/daily_avg_corr.xlsx'
data = pd.read_excel(src, index_col=0)
data = data.drop(columns=['次日超额收益率'], index=['次日超额收益率'])
# 使用 seaborn 绘制热力图
import seaborn as sns
sns.heatmap(data, cmap='Greys', annot=True, fmt='.3f', annot_kws={'size': 9}, cbar=False)
# plt.xticks(rotation=0)
plt.xticks(rotation=20, rotation_mode='anchor', ha='right', va='center')
# plt.yticks(rotation=30, rotation_mode='anchor', ha='right', va='center')
# 显示图形
plt.ylabel('')
plt.tight_layout()
tgt = 'results/corr_heatmap.png'
plt.savefig(tgt, transparent=True)
print(tgt)
plt.close()
# plt.show()