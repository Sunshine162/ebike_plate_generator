## 车牌数据生成

### 车牌生成

车牌元素：

1. 城市名21个（广州、深圳、珠海、汕头、韶关、河源、梅州、惠州、汕尾、东莞、中山、江门、佛山、阳江、湛江、茂名、肇庆、清远、潮州、揭阳、云浮）
2. 二维码
3. 号码（6位，可能为6位全数字，也可能参杂大写字母，无英文字母 I 与 O ）
4. 安装孔2个

其他：

1. 城市名字体不一致
2. 边框样式不一致
3. 二维码位置不一致
4. 颜色不是纯白色，随机变化；受光照与角度影响，车牌中不同位置的颜色也不知完全一样的。
5. 车牌固定螺丝帽样式不一致。

### 背景生成

1. 车牌尺寸
2. 车牌角度
3. 整图光照强度
4. 在背景中位置
5. 车牌数量

### 相关链接
1. [佛山电动自行车牌照有哪些颜色](http://m.wenda.bendibao.com/live/114493.shtm)
2. [《电动自行车安全技术规范》（GB17761-2018）](https://gdga.gd.gov.cn/xxgk/wgk/glgk/content/post_2286895.html)
3. [广州电动车车牌分几类](http://m.wenda.bendibao.com/live/175160.shtm)
4. [广州启动电动自行车登记上牌](http://gd.people.com.cn/n2/2021/1103/c123932-34987496.html)
5. [仿射变换与透视变换](https://codec.wang/docs/opencv/start/extra-05-warpaffine-warpperspective)
6. [opencv 图像特效处理 素描、怀旧、光照、流年、滤镜 原理及实现](https://www.cnblogs.com/wojianxin/p/12757953.html)


### TODO LIST

1. 安装孔与安装钉
2. 边框
    + 无边框
    + 内黑线
    + 透明塑料包边
    + 黑色塑料包边
    + 边缘涂黑色
3. 变形
    + 弯曲变形
    + 折曲变形
    + 凹陷与凸起
    + 扭转变形
4. 残缺
5. 反光
6. 强光
7. 模糊
8. 阴影

