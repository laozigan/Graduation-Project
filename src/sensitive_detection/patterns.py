# 敏感信息模式库
REGEX_PATTERNS = {
    'bank_card': r'^62\d{14,17}$',
    'id_card': r'[1-9]\d{5}(18|19|20)?\d{2}(0[1-9]|1[0-2])(0[1-9]|[12]\d|3[01])\d{3}[\dXx]',
    'phone': r'1[3-9]\d{9}',
    'email': r'\w+([.-]?\w+)*@\w+([.-]?\w+)*(\.\w{2,3})+',
    # 可以继续添加：护照号、社保号等
}

KEYWORD_CATEGORIES = {
    'name': ['姓名', '名字', '名称', '病人姓名', '客户姓名', '学生姓名', '员工姓名', 'name', 'full name',  'user name', 'username'],
    'address': ['地址', '住址', '家庭住址', '通讯地址', '现居住地', 'address', 'location', 'residence'],
    'id': ['身份证', '身份证号', '公民身份号码', '居民身份证号', 'id number', 'identification'],
    'phone': ['手机', '电话', '联系电话', '手机号', 'phone', 'mobile', 'telephone', 'tel'],
    'bank_card': ['银行', '账号', '银行卡号', '开户行', '信用卡', '借记卡', 'bank', 'account'],
    'medical': ['病历号', '就诊号', '住院号', '医保号'],   # 医疗敏感
}

COMMON_SURNAMES = {}