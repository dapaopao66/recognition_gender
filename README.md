# recognition_gender
从数据收集到，训练模型，全套流程。

需要安装numpy,sklearn,keras,opencv,tensorflow，如果你有好的gpu推荐安装tensorflow-gpu,速度比cpu快很多

### 1.如果你没有数据
执行image_spider.py 下载图片 代码来源 [百度爬虫](https://github.com/kong36088/BaiduImageSpider )

### 2.从下载的人图中，获取脸部图片
执行get_face.py

### 3.进行训练
执行image_train.py

### 4.查看训练结果
执行 image_show.py

可以在我的网站上试试：[我的网站](http://www.aidump.com/CoolImage/GuessGender) 试一试结果 
可能是训练数据样本集不多的原因，正确率不是很高，可通过增加样本集，增加训练次数来提高正确率
