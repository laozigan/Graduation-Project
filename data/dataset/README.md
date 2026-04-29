---
license: CC0
technical_domain:
  - OCR Recognition
  - Object Detection
  - 自然语言处理
  - 计算机视觉
---

<div><strong>数据集简介：</strong></div>
<div>人才数据很多是以非结构化文档形式存储，其中存储了大量的价值数据，比如人才基本信息、人才职业情况、人才专业技能、人才工作履历、人才学历情况等，而这些数据因为存储形式，很难直接应用于各种人才分析场景。为了有效利用这些数据，需要将其抽取识别出来，构建以人为主题的人才知识图谱库，形成人与人、人与企业、人与学校等之间的关联关系，进而为后续的人才社群分析、人才精准推荐、企业招聘推荐等业务场景提供精准的价值数据支撑。</div>
<div>&nbsp;</div>
<div><strong>数据集描述：</strong></div>
<div>
<div>脱敏之后的中文人才简历数据和标注数据。</div>
<div>标注类别包括：姓名、出生年月、性别、电话、最高学历、籍贯、落户市县、政治面貌、毕业院校、工作单位、工作内容、职务、项目名称、项目责任、学位、毕业时间、工作时间、项目时间共18个字段。</div>
<div>在训练数据集中，每个&ldquo;毕业院校、学位、毕业时间&rdquo;为一组，以&ldquo;教育经历&rdquo;列表给出；每个&ldquo;工作单位、工作内容、职务、工作时间&rdquo;为一组，以&ldquo;工作经历&rdquo;列表给出；每个&ldquo;项目名称、项目责任、项目时间&rdquo;为一组，以&ldquo;项目经历&rdquo;列表给出；具体格式见训练数据示例。</div>
<div>为了保护用户隐私，本次训练数据仅提供几类常见非标准简历格式模板的人工构造数据，共2000份。</div>
<div>&nbsp;</div>
</div>
<div><strong>数据集来源：</strong></div>
<div>https://tianchi.aliyun.com/competition/entrance/231771/information</div>