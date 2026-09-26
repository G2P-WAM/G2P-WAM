# G2P-WAM — Anonymous Project Website

这是 G2P-WAM 论文的匿名项目主页源码。解压后可直接上传 GitHub，无需 npm、Node.js 或构建命令。

页面参考 Spatial Forcing 的灰色居中标题区、圆角资源按钮和单页论文展示结构，重新编写 HTML/CSS/JS，使用本论文的文字、图像和实验数据。

## 1. 本地查看

双击根目录的 `index.html` 即可查看。图片、视频、样式、交互脚本、参考代码与匿名论文 PDF 均在本包内，不依赖外部字体、CDN、统计脚本或后端。

也可在该目录运行 `python -m http.server 8000`，然后在浏览器打开 `http://localhost:8000`。

## 2. 上传到 GitHub

1. 创建用于匿名展示的仓库。免费 Pages 方案通常使用匿名账号下的公开仓库；账号资料和提交者信息应与匿名用途一致。
2. 解压本包，把 `index.html`、`static/`、`code/`、`README.md` 等内容上传到仓库根目录。**不要只上传 ZIP，也不要在外面再套一层文件夹。**
3. 仓库打开 **Settings → Pages**。
4. 在 **Build and deployment** 中设置 **Source: Deploy from a branch**，选择 **main** 分支和 **/(root)**，点击 **Save**。
5. 等待 GitHub 页面显示部署完成，再打开生成的项目链接。

包内 `.nojekyll` 用于直接发布静态文件。如果系统隐藏了点号文件，可在 GitHub 上新建一个名为 `.nojekyll` 的空文件。本网站没有以下划线开头的资源目录，遗漏它通常不影响当前页面，但建议保留。

## 3. 使用 anonymous.4open.science

1. **先按上节开启原仓库的 GitHub Pages。** 当前 Anonymous GitHub 页面选项需要识别到 Pages 配置。
2. 访问 https://anonymous.4open.science/ ，登录后创建匿名化项目，填入 GitHub 仓库地址，选择与 Pages 相同的 `main` 分支。
3. 设置匿名 ID、需要替换的身份关键词，以及覆盖审稿期的有效期。
4. 开启 **GitHub Pages** 页面展示选项，并允许页面需要的 **Keep links、Display images、Display PDFs**。**Keep links 本身不是网站托管开关。**
5. 创建后从项目菜单选择 **View page**，复制平台实际生成的网页链接。主页通常为 `https://anonymous.4open.science/w/匿名ID/`；`/r/匿名ID/` 是仓库文件浏览页。
6. 用未登录窗口检查最终主页、视频播放、图片、论文与代码下载按钮及手机布局。后续更新 GitHub 文件后，可在平台中按需使用 **Force update**。

所有站内资源使用相对路径，适用于 GitHub Pages 项目子目录以及匿名页面子目录。若 “GitHub Pages” 选项不可用，先检查原仓库的 Pages 部署状态和源分支。私有仓库能否启用 GitHub Pages 取决于 GitHub 套餐；私有仓库本身并不保证其 Pages 网站私有。

## 4. 内容与文件

| 文件 | 用途 |
|---|---|
| `index.html` | 全部英文页面文字和结果表格 |
| `static/css/style.css` | 桌面、手机和打印样式 |
| `static/js/main.js` | 图片放大；播放一条视频时暂停其他视频 |
| `static/images/overview.png` | 方法概览 |
| `static/images/framework.png` | 三阶段训练框架 |
| `static/images/reward-diagnosis.png` | 奖励诊断 |
| `static/images/teacher-targets.png` | 静态和动态教师目标 |
| `static/images/simulation-rollouts.png` | 仿真执行序列 |
| `static/images/piper-results.png` | PiPER X 执行序列与结果 |
| `static/images/a1c-rollouts.png` | A1C 执行序列 |
| `static/files/paper.pdf` | 28 页匿名论文，已优化文件大小 |
| `static/files/results.json` | 页面展示数值的便于读取副本 |
| `static/videos/` | 15 条 H.264 MP4 视频 |
| `static/videos/posters/` | 15 张视频封面 |
| `static/files/media.json` | 视频内容、时长、帧率与来源类型 |
| `static/files/g2p-wam-reference.zip` | 提供给读者下载的原始参考代码包 |
| `code/` | 相同参考代码的可浏览文件 |

当前包含方法、摘要、奖励诊断、三个仿真基准、两个真实机器人平台和消融结果。正文与附录来自当前修订稿。

关键数值：RoboTwin 92.95%，LIBERO 98.58%，LIBERO-Plus 79.5%；PiPER X 85.6%，UNT David A1C 92.2%。GeoSFT 的 RoboTwin 均值为 92.46%。更新数据时应同步修改 `index.html` 与 `static/files/results.json`。

## 5. 视频与参考代码

已整合 **6 条真机执行视频、6 条仿真执行视频、3 条几何可视化视频**。所有视频使用浏览器常见的 H.264/yuv420p MP4，按需加载，有封面、播放控制和手机内联播放。播放一条时会暂停其他视频。视频完整时长、帧率和分辨率与所提供素材一致；网页用副本去掉了音轨和设备元数据。Object stacking 的原素材为 HEVC，已转为 H.264；其余兼容视频保持原编码画面。

| 页面分组 | 内容 | 素材来源标签 |
|---|---|---|
| AgileX PiPER X | 存放水果、放置盘子、放置香蕉 | 真机执行 |
| UNT David A1C | 双臂交接、物体堆叠、打开抽屉 | 真机执行 |
| LIBERO | 摩卡壶上炉、奶酪与黄油入篮 | Base policy |
| LIBERO-Plus | 黑碗入抽屉、杯子入微波炉 | DPO checkpoint |
| RoboTwin | 挂杯子、拨动开关 | GeoSFT |
| 方法部分 | VGGT 特征、SpatialTrackerV2 轨迹 | 几何教师输出 |
| 奖励诊断 | 4RC 置信度与重建点云 | 采样输出可视化 |

仿真策略标签依据原素材清单中的来源目录；并非全部都来自最终 G2P-WAM 检查点。VGGT 和 4RC 分别用 8 个和 6 个采样时间点，以 2fps 展示，页面已注明。SpatialTrackerV2 轨迹视频使用原素材的 20fps。网站没有把这些教师可视化标为真实执行速度。

`code/` 包含 GeoSFT、奖励诊断、偏好筛选、GeoDPO 和教师张量接口的参考实现。它是 **tensor-level reference implementation**，不含 WAM 主干、外部教师推理、权重、训练主循环、数据加载或完整基准评测脚本。网页的 **Code** 按钮跳到代码介绍，**Download Reference Code** 下载原始参考包，代码文件未改写。

在 Python 3.10+ 环境中可按原代码说明运行 CPU 检查：

```sh
cd code
python -m pip install -r requirements.txt
python -B -m unittest discover -s tests -v
```

以上命令供使用者运行；网站打包过程没有运行训练、基准评测或代码自带的 CPU 检查。页面本身不需要安装 Python 或 PyTorch。

以后更换视频时，保持 `index.html` 中的 MP4 与 poster 路径对应，并同步修改 `static/files/media.json`。视频采用 `preload="none"`，读者按播放按钮后加载。当前无公开模型或 arXiv 地址，因此未放置这些链接。

## 6. 匿名版与素材说明

- 页面和 PDF 作者均为 Anonymous Authors，没有作者、机构、联系邮箱或个人账号链接。
- 没有 Git 历史、跟踪脚本和外部字体请求；图片、视频、代码与 PDF 为本地资源。
- `noindex` 只表达不希望搜索引擎收录，不是访问控制。
- 论文 PDF 与图片属于二进制素材，Anonymous GitHub 的文字替换不能代替其内容检查。添加新图片/视频时检查姓名、实验室标识和水印。
- 页面实现独立编写；底部保留布局参考来源。未复制参考项目的科研图片、作者信息、实验数据或论文文字。

参考与操作文档（2026-09-26 核对）：
- 布局参考：https://spatial-forcing.github.io/
- GitHub Pages 发布源：https://docs.github.com/en/pages/getting-started-with-github-pages/configuring-a-publishing-source-for-your-github-pages-site
- Anonymous GitHub FAQ：https://anonymous.4open.science/faq
- Anonymous GitHub 实现与页面选项：https://github.com/tdurieux/anonymous_github
