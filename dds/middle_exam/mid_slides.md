---
marp: true
theme: gaia
paginate: true
# header: '中期进展汇报-基于多模态抑郁症量表的辅助诊断-杨智凯-2021-11-23'
style: | 
  section header{color:black;font-size: 20px;text-align: center} 
---
<style scoped>
section h1 {text-align: center;font-size: 70px;color:black;font-family:STSong}
section {
  background-image:url('img/fm.png');
  background-size:cover
}
header{color:black;font-size: 20px; } 
</style>
<!-- _class: lead gaia -->

# 基于多模态抑郁症量表的辅助诊断
## 答辩人: 杨智凯 
## 导师：常象宇
## 2021-11-25
---
<style >

section{
  background-image:url('img/bg.png');
  background-size:cover;
  position: absolute;
  }
section h1 {font-size:40px;color:black;margin-top:px;}
section h2 {font-size:30px;color:black;margin-top:px;}
section h3 {font-size:25px;color:black;margin-top:px;}
section h4 {font-size:20px;color:black;margin-top:px;}
section p {font-size: 25px;color:black;}
section table {text-align: center;font-size: 25px;color:black;}
/* section a {font-size: 25px;color:black;} */
li {font-size: 28px;text-align: left;}

img {
    margin-left: auto; 
    margin-right:auto; 
    display:block;
    margin:0 auto;
    width:25cm;
    }
</style>

<style scoped>
section h1 {font-size:60px;color:black;margin-top:px;}
li {font-size: 50px;text-align: left;}
</style>

# 总览
1. 研究背景
2. 研究内容
3. 研究创新
4. 中期进度
5. 未来计划
![bg right fit](img/bg02.png)
---
# 题目解释

## **多模态抑郁症量表（Multi Modal Self-rating Depression Scale, MMSDS）**，是本研究提出的一种借助软件平台，病人在自测抑郁量表时同时收集题目作答与思考转化过程的**面部、头部**模态与传统量表内容中的**心理模态**和**躯体模态**结合的量表。
- 其中心理模态和躯体模态来自传统SDS量表的条目测试，面部模态和头部模态来源被试测评时的客观捕捉。
- 编制原理结合了传统心理测量理论的因子解释结构和多模态机器学习特征非线性融合的优势
- 期望能增强抑郁症自评检测量表的信度、效度和筛选效率。
## **辅助诊断**：MMSDS替代抑郁症门诊传统自评量表环节，期望能提升量表筛选的准确性，其精度达到辅助诊断的目的，以减轻门诊的医疗资源压力，优化病人就诊、病情监测流程。
---
# 一、研究背景
## 1. 抑郁症（Depression Disorder）患者激增
抑郁症作为常见的精神类疾病，其发病率高、负面影响大--如个人功能受损、社会经济负担等[1], 已经成为严重的社会问题。随着社会竞争压力不断增大，社会对抑郁症的认识不断加深，抑郁症门诊需求数量逐年攀升。**2021年11月9日**，教育部对政协 **《关于进一步落实青少年抑郁症防治措施的提案》** 进行了答复，其中明确将抑郁症筛查纳入学生健康体检内容，建立学生心理健康档案，评估学生心理健康状况，对测评结果异常的学生给予重点关注。

## 2. 我国精神医疗资源紧张
中国作为 14 亿人口大国，全国 1000 多所精神医疗机构应对近7,000万的精神疾病患 者群体，服务容量有限，市场处于供不应求状态。特别是声望高，专业性强的精神疾 病医疗机构更是日常爆满，单个患者的问诊时间被大大压缩。从就诊数量看，居民的 生活压力与精神疾病的患病率与国家经济增长成正相关，21 世纪以来中国经济高速增长，随之而来的将是精神疾病患者的日益增加，相应医疗资源压力继续增大。[2]

#### [1] Hammen C. Stress and depression[J]. Annu. Rev. Clin. Psychol., 2005, 1: 293-319.
#### [2] 中国精神医疗行业概览。2019 https://www.leadleo.com/report/details?id=5c889be8f272472733b147ca

---
# 一、研究背景
## 3. 门诊流程不合理
- 检查流程长 
- 病人平均问诊时间少（< 10min）
- 资源费配不合理：重症病人医疗资源被轻症、待查病人挤占
- 资源匮乏，无法进行心理治疗
- 自评量表的效率不高
![bg right:62% fit](img/就诊流程.jpg)

---
# 一、研究背景
## 4. 自评量表的筛选效率亟待提升
![bg right:50% 68%](img/量表结果.jpeg)
- SDS (Self-rating Depression Scale)存在误差，既有来自施测、评分和解释过程等客观因素，也有被试的主观因素。
- SDS采用简单的阶段分数区间来诊断病情，自1965年正式用于临床以来，已有不少研究证明此量表的信度、效度存在提升空间。[3]

#### [3] Zung WW, Richards CB, Short MJ. Self-rating depression scale in an outpatient clinic. Further validation of the SDS. Arch Gen Psychiatry. 1965 Dec;13(6):508-15.
---
<style scoped>
section h2 {
    margin-bottom: -20px;
} 
section li {
    margin-bottom: -10px;
}
</style>
# 二、选题动机 Motivation 
## 1. 基于经典测量理论（CTT）的抑郁量表(SDS、BID-2、PHQ-9等)存在理论缺陷
- CTT假定的观测变量呈正态分布可能不成立
- CTT假定的因子结构与目标构念（Construct）的**线性关系**可能不成立
- CTT假定的**存在可重复性平行试验**存在缺陷 [4]

## 2. 多模态深度学习对抑郁症检测表现优秀
- 多类模态特征可以预测患者抑郁状态 
    - 多媒体模态：图像、声音、文本内容、躯体动作等
    - 生物体征模态：心率、血压、脑电、心电等
    - 心理模态：人格测试、情绪状态等
- 深度网络结构可以充分利用Transformer、Attention等技术进行模态间的交叉融合[5]
#### [4] Raykov T, Marcoulides G A. Introduction to psychometric theory[M]. Routledge, 2011.
#### [5] Ringeval F, Schuller B, Valstar M, et al. AVEC 2019 workshop and challenge: state-of-mind, detecting depression with AI, and cross-cultural affect recognition[C]
---
<style scoped>
section {
    padding: 30px;
}

section h2 {
    margin-bottom: -20px;
} 
</style>
# 二、选题动机 Motivation 
## 3. 机器学习方法与心理测量优势互补
- 传统心理测量测重解释、一般使用因子分析结构编制
- 机器学习侧重数据与算法、强调泛化能力
    - 特征工程：特征能组合交叉，客观特征信度较高、误差小
    - 多模态深度学习：能处理非线性关系、多领域特征、预训练模型解决样本少量问题
- 二者结合能兼顾解释性和泛化能力[6]

![bg right:45% 80%](img/sds结构.jpg)
#### [6] 余嘉元.粗糙集和神经网络在心理测量中的应用[J].心理学报,2008,40(8):939-946.

---
# 三、研究内容

## （一）面部、头部模态对抑郁症诊断影响的实证模型研究
采用文献搜集的研究方法，针对本研究提出的**新增客观模态**找出心理学、生物医学、计算机科学等方面的理论依据，同时应考虑进其他调节因素，并通过实证数据验证。
## （二）多模态抑郁症量表（MMSDS）的设计和信效度检验
基于研究（一）的实证模型，结合传统 SDS 自评量表的条目，提出基于多模态深度学习方 法的量表设计。同时基于采集数据进行量表结构的优化探索与参数训练。此外，心理测量理论和 机器学习理论相结合进行 MMSDS 的信度和效度检验。
## （三）MMSDS 辅助诊断的优化效果研究
MMSDS 参数、结构确定后，将其融入到门诊流程测试，探究其真实筛选效果，能否有效缓 解门诊资源紧张、重症病人的时间能否有效延长等问题。

---
<style scoped>
    section {
        padding: 30px;
    }
    section h2 {
        margin-top: 10px;
        margin-bottom: -20px;
    }
    section ul {
        margin-bottom: -5px;
    }
</style>
# 四、国内外研究现状
## （一）抑郁量表筛选效果改善研究
- 主要体现在心理测量和精神医学领域对量表内容的迭代。但均基于的是经典测量理论。
## （二）机器学习与心理量表结合研究
- 有研究通过将量表作为特征输入机器学习模型进行量表的效度检验、抑郁症预测模型实验等。[1]
- 有研究证明单纯使用量表条目作为特征训练出的模型在筛选效率上优与传统量表。[2]
## （三）多模态抑郁检测研究
- 在开源抑郁症检测数据集（DAIC-WOZ）上，众多研究围绕音频、视频、文本模态展开多模态深度深度学习模型的训练，目前最优秀的AUC达到0.82. [3]

#### [1] Gonzalez O. Psychometric and machine learning approaches for diagnostic assessment and tests of individual classification[J]. Psychological Methods, 2021, 26(2): 236.
#### [2] 成也，杨镇恺，姚力，王新波，赵小杰．量表大数据的深度神经网络抑郁分类模型[J/OL]．北京师范大学学报(自然科学版).
#### [3] Ringeval F, Schuller B, Valstar M, et al. AVEC 2019 workshop and challenge: state-of-mind, detecting depression with AI, and cross-cultural affect recognition[C]//Proceedings of the 9th International on Audio/Visual Emotion Challenge and Workshop. 2019: 3-12.

---
# 四、国内外研究现状：总结
## 1. 量表 、多模态学习组合可能表现更优
| 特征 | 模型 | 编号 |
|--- | --- | --- |
| SDS量表条目  | SVM/随机森林 | M1 |
| 单模态（面部）| 复杂网络结构 |  M2 |
| 多模态 (图像、声音、文本、生理等) | 复杂网络结构 |  M3 |

以ROC评价二分类，目前文献中在开源数据集（DAIC-WOZ）总结
$$ ROC(M3) > ROC(M2) \ge ROC(M1) $$

## 2. 暂未搜索到使用量表和多模态特征共同考虑的抑郁量表
$$
ROC(M3 + M1) > ROC(M3) > ROC(M2) \ge ROC(M1)
$$

---
# 五、研究进度
![bg 60%](img/研究路线.jpg)

---
<style scoped>
section h2 {
    margin-bottom: -20px;
}
section {
    padding: 40px;
}
</style>
# 五、研究进度
## 1. 数据采集与特征处理
门诊病人数据采集平台开发完成，能够使病人在平台作答自评量表SDS时捕捉到人脸模态，同时后台程序自动处理生成所需各类特征。已经启动在空军军医大学西京医院心身科现场部署采集工作。
| | |
| --- | --- |
| ![w:15cm](img/cap01.gif) |  ![w:15cm](img/cap02.gif)|
| MMSDS操作界面 | 医生管理后台-查看结果-图表等|

---
# 五、研究进度
## 1. 数据采集与特征处理：人脸特征, 采集与处理工具`OpenFace`
| | |
|---| --- |
| ![w:12cm](img/eye.png) | ![w:18cm](img/AU.png)|
| 面部标志点位置追踪 | 面部动作单元（Facical Action Units）|


---
# 五、研究进度
## 1. 数据采集与特征处理：头部特征，头部位移速率、位移频次等
![w:20cm](img/head_pose.png)

## 样本标签
- 基本信息（性别、年龄、传统SDS量表作答情况）
- **他评量表分数**，主要是HAMD（汉密尔顿抑郁他评量表）
- **主治医生最终诊断结果**

---
<style scoped>
section {
    padding: 30px;
}
section h2 {
    margin-bottom: -30px
}
section h4 {
    margin-top: 0px
}
section li {
    font-size: 24px;
}
section ul {
    margin-bottom: -5px;
}
</style>
# 五、研究进度
## 2. 人脸与头部模态的实证模型研究
- **面部动作单元--人类情绪**：研究发现面部动作单元的组合可以作为情绪的测量[1]
- **人类情绪--抑郁状态**： 根据目前国内外通用的抑郁症诊断手册`ICD-10`和`DSM-5`，都将心情上的厌恶、恐惧、悲伤情绪作为抑郁症状态的重要表征。此三类情绪对人类抑郁状态有正向影响具备临床上的科学依据。
- **躯体惰性--抑郁状态**： 在多模态预测抑郁症相关研究中，发现头部动作的幅度、频次、速度等反应的头部躯体惰性是抑郁症的重要表征。具体的验证方法是将头部特征移除模型后预测效果发生明显变化。[2]
#### [1] Karthick(2013). Survey of Advanced Facial Feature Tracking and Facial Expression Recognition. 
#### [2] S. Alghowinem "Head Pose and Movement Analysis as an Indicator of Depression," 2013 Humaine Association Conference on Affective Computing and Intelligent Interaction, 2013, pp. 283-288

![bg right:50% 90% ](img/实证模型.jpg)

---
<style scoped>
section {
    padding: 30px;
}
section h2 {
    margin-bottom: -30px
}
section h4 {
    margin-top: 0px
}
section ul {
    margin-bottom: -5px;
}
</style>
# 五、研究进度
## 2. 人脸与头部模态的实证模型研究
![bg right:50% 90% ](img/实证模型.jpg)
- **性别**的调节作用
    - 女性患抑郁症概率高于男性[1]
    - 面部动作性别差异显著，男性面部动作变化更加非对称，而女性更加均匀，有心理研究表明女性似乎更倾向释放情绪. [2]
    - Stratou.G 等 人将性别因素考虑进多模态抑郁症识别模型中，效果得到了显著提升[3]
#### [1] Sex differences in unipolar depression: Evidence and theory. Psychological Bulletin 101(2), 259
#### [2] Shields, S. A. (2000). Thinking about gender, thinking about theory: Gender and emotional experience. 
#### [3] Stratou, G. et al. Automatic nonverbal behavior indicators of depression and PTSD: the effect of gender. J Multimodal User Interfaces 9, 17–29 (2015)
---
<style scoped>
section {
    padding: 40px;
}
section h2 {
    margin-bottom: -20px
}
section ul {
    margin-bottom: -5px;
    font-size: 25px;
}
</style>
# 五、研究进度
## 3. MMSDS的设计方案
![bg right:65% 90%](img/mmsds.jpg)
- 面部、头部模态作为SDS补充条目
- SDS李克特五点计分看做分类变量，采用向量嵌入（embedding）处理
- 面部模态具有时序特征，采用Bi-LSTM处理
- 视数据质量和数量决定采用特征增强or **预训练模型**处理
- 特征交叉方式以实证模型为指导设计

---
<style scoped>
section {
    padding: 30px;
}

</style>
# 五、研究进度
## 3. MMSDS的设计

| 符号 | 含义 |
| --- | --- |
| $\bold{m} \in R^d,\bold{s} \in R^d,~~\bold{f} \in R^d,~~\bold{h} \in R^d$  | 心理、躯体、面部、头部特征 |
| $\bold{X} = (\bold{x}_1,...,\bold{x}_t)$ | 面部的时刻序列特征|
| $LSTM(·)$| Bi-LSTM算子|
| fc(·) | 全连接网络层算子|

$$
\bold{f} = LSTM(\bold{X_t}), ~~~~~~X_t = (\bold{x}_1, ..., \bold{x}_t)
$$
$$
\bold{x} = (\bold{m} \oplus \bold {s}) \oplus fc(\bold{f}) \oplus \bold{h} 
$$
$$
\bold {x}' = fc(\bold{x}), ~~~~
y = sigmod(\bold{x'})
$$
![bg right:40% 100%](img/mmsds.jpg)

---
<style scoped>
    section h2 {
        margin-bottom: -30px;
    }
    </style>
# 五、研究进度
## 3. MMSDS的信度、效度检验方案
- 信度检验
    - Cronbach-alpha系数，可以针对各模态进行检验，期望系数在0.8以上，表示每个模态内的条目有较好的相关性
    - 组内相关系数，按照提出的理论模型（面部模态实证解释 + SDS量表编制模型）进行CFA（验证性因子分析），计算出组内相关系数
    - 重测信度检验，假如收集到的数据有复诊，可以对其进行重测信度检验（比较前后两次测验相关性）
    - 半分信度检验，讲题目简单分半，检测两部分的信度
- 效度检验
    - 效标检验法: 选择患者病历的最终诊断结果、患者他评量表的结果（HAMD得分）等作为校标检验
    - ROC比较法: 分别对比传统SDS、面部+头部量表、MMSDS的ROC情况

---
# 六、总结: 研究意义与创新
## 1. 筛选效率、信度效度更优的抑郁自评量表--MMSDS
MMSDS的设计思路为传统心理测量理论与机器学习方法相结合，因子结构+数据算法期望得到筛选效果更优的量表。
## 2. 心理测量量表编制的新思路
本研究通过探究 "传统心理量表条目和多模态客观特征条目 + 深度学习模型" 的方法编制测评量表。期望为机器学习在心理测量领域的应用添加新思路。
## 3. MMSDS期望能优化门诊流程、合理化医疗资源配置
MMSDS期望能收获优秀的诊断能力:
对于患者的病情管理，可以作为一个可信赖的抑郁症检测工具，达到健康管控的目的。
对于医院管理，能够优化门诊自评量表检查环节，优化各类病人的就诊时间，使精神医疗资源更合理地配置。

---
# 七、未来工作计划
- [P0] 数据采集工作继续推进，保证数据质量与特征维护工作。
- [P0] 开展研究内容（一）中的实证模型验证，适当调整模态选择
- [P0] 开展MMSDS结构和参数的训练与调优工作
- [P1] 继续发掘相关文献
    - 寻找更多模态特征解释抑郁症的心理学、医学等相关文献、理论依据
    - "心理测量+机器学习"创新量表编制方法探索
    - 探索更多量表信度、效度提升或检验的方法
- [P1] MMSDS医院门诊测试与实践
    - 探索MMSDS与医院门诊流程融合操作，检验MMSDS在门诊资源优化、流程优化与真实检测效率上的表现。

---
<style scoped>
section h1 {text-align: center;font-size: 100px;color:black;}
section {
  background-image:url('img/fm.png');
  background-size:cover
}
footer{color:black;font-size: 20px;} 
</style>
<!-- _class: lead gaia -->


# 谢谢 ：）
## 请各位老师批评指正~
