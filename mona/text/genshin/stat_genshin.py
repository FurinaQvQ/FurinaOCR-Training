"""
原神属性数据生成器
包含圣遗物主属性、副属性的名称和数值生成器

主要功能：
- 生成圣遗物主属性名称和数值
- 生成圣遗物副属性文本
- 支持不同星级和等级的属性数值计算
- 基于游戏内真实数据进行生成

用途：为OCR训练提供圣遗物属性识别数据
"""

import json
import random
from mona.text.text_generator import TextGenerator

# 属性信息映射表
# 将游戏内部属性名称映射为中文显示名称和百分比标识
stat_info = {
    "FIGHT_PROP_CRITICAL_HURT": {
        "percent": True,           # 是否为百分比属性
        "chs": "暴击伤害",          # 中文显示名称
    },
    "FIGHT_PROP_CRITICAL": {
        "percent": True,
        "chs": "暴击率",
    },
    "FIGHT_PROP_HP": {
        "percent": False,          # 固定数值属性
        "chs": "生命值",
    },
    "FIGHT_PROP_HP_PERCENT": {
        "percent": True,           # 百分比生命值
        "chs": "生命值",
    },
    "FIGHT_PROP_ATTACK": {
        "percent": False,          # 固定攻击力
        "chs": "攻击力",
    },
    "FIGHT_PROP_ATTACK_PERCENT": {
        "percent": True,           # 百分比攻击力
        "chs": "攻击力",
    },
    "FIGHT_PROP_DEFENSE": {
        "percent": False,          # 固定防御力
        "chs": "防御力",
    },
    "FIGHT_PROP_DEFENSE_PERCENT": {
        "percent": True,           # 百分比防御力
        "chs": "防御力",
    },
    "FIGHT_PROP_CHARGE_EFFICIENCY": {
        "percent": True,
        "chs": "元素充能效率",
    },
    "FIGHT_PROP_ELEMENT_MASTERY": {
        "percent": False,
        "chs": "元素精通",
    },
    "FIGHT_PROP_HEAL_ADD": {
        "percent": True,
        "chs": "治疗加成",
    },
    # 各元素伤害加成属性
    "FIGHT_PROP_FIRE_ADD_HURT": {
        "percent": True,
        "chs": "火元素伤害加成",
    },
    "FIGHT_PROP_ELEC_ADD_HURT": {
        "percent": True,
        "chs": "雷元素伤害加成",
    },
    "FIGHT_PROP_WATER_ADD_HURT": {
        "percent": True,
        "chs": "水元素伤害加成",
    },
    "FIGHT_PROP_WIND_ADD_HURT": {
        "percent": True,
        "chs": "风元素伤害加成",
    },
    "FIGHT_PROP_ROCK_ADD_HURT": {
        "percent": True,
        "chs": "岩元素伤害加成",
    },
    "FIGHT_PROP_GRASS_ADD_HURT": {
        "percent": True,
        "chs": "草元素伤害加成",
    },
    "FIGHT_PROP_ICE_ADD_HURT": {
        "percent": True,
        "chs": "冰元素伤害加成",
    },
    "FIGHT_PROP_PHYSICAL_ADD_HURT": {
        "percent": True,
        "chs": "物理伤害加成",
    }
}

# 提取所有不重复的属性名称
stat_name_set = set()
for item in stat_info:
    stat_name_set.add(stat_info[item]["chs"])
stat_name = list(stat_name_set)

# 不同部位圣遗物的主属性范围定义
# 花：固定生命值
# 羽：固定攻击力  
# 沙：元素精通/充能效率/防御力%/攻击力%/生命值%
# 杯：各种伤害加成/元素精通/防御力%/攻击力%/生命值%
# 冠：治疗加成/元素精通/防御力%/攻击力%/生命值%/暴击率/暴击伤害
main_stat_names = {
    "flower": ["FIGHT_PROP_HP"],                                    # 花：固定生命值
    "feather": ["FIGHT_PROP_ATTACK"],                               # 羽：固定攻击力
    "sand": ["FIGHT_PROP_ELEMENT_MASTERY", "FIGHT_PROP_CHARGE_EFFICIENCY",
             "FIGHT_PROP_DEFENSE_PERCENT", "FIGHT_PROP_ATTACK_PERCENT", "FIGHT_PROP_HP_PERCENT"],
    "cup": ["FIGHT_PROP_PHYSICAL_ADD_HURT", "FIGHT_PROP_ICE_ADD_HURT", "FIGHT_PROP_ROCK_ADD_HURT",
            "FIGHT_PROP_WIND_ADD_HURT", "FIGHT_PROP_WATER_ADD_HURT", "FIGHT_PROP_ELEC_ADD_HURT",
            "FIGHT_PROP_FIRE_ADD_HURT", "FIGHT_PROP_ELEMENT_MASTERY", "FIGHT_PROP_DEFENSE_PERCENT",
            "FIGHT_PROP_ATTACK_PERCENT", "FIGHT_PROP_HP_PERCENT", "FIGHT_PROP_GRASS_ADD_HURT"],
    "head": ["FIGHT_PROP_HEAL_ADD", "FIGHT_PROP_ELEMENT_MASTERY", "FIGHT_PROP_DEFENSE_PERCENT",
             "FIGHT_PROP_ATTACK_PERCENT", "FIGHT_PROP_HP_PERCENT", "FIGHT_PROP_CRITICAL",
             "FIGHT_PROP_CRITICAL_HURT"]
}

# 副属性可能的属性类型（不包含元素伤害加成和治疗加成）
sub_stat_keys = [
    "FIGHT_PROP_CRITICAL_HURT", "FIGHT_PROP_CRITICAL", "FIGHT_PROP_HP", "FIGHT_PROP_HP_PERCENT",
    "FIGHT_PROP_ATTACK", "FIGHT_PROP_ATTACK_PERCENT", "FIGHT_PROP_DEFENSE_PERCENT", "FIGHT_PROP_DEFENSE",
    "FIGHT_PROP_CHARGE_EFFICIENCY", "FIGHT_PROP_ELEMENT_MASTERY",
]

# 主属性可能的属性类型（包含所有属性）
main_stat_keys = [
    "FIGHT_PROP_CRITICAL_HURT", "FIGHT_PROP_CRITICAL", "FIGHT_PROP_HP", "FIGHT_PROP_HP_PERCENT",
    "FIGHT_PROP_ATTACK", "FIGHT_PROP_ATTACK_PERCENT", "FIGHT_PROP_DEFENSE_PERCENT", "FIGHT_PROP_DEFENSE",
    "FIGHT_PROP_CHARGE_EFFICIENCY", "FIGHT_PROP_ELEMENT_MASTERY", "FIGHT_PROP_HEAL_ADD",
    "FIGHT_PROP_PHYSICAL_ADD_HURT", "FIGHT_PROP_ICE_ADD_HURT", "FIGHT_PROP_ROCK_ADD_HURT",
    "FIGHT_PROP_WIND_ADD_HURT", "FIGHT_PROP_WATER_ADD_HURT", "FIGHT_PROP_ELEC_ADD_HURT",
    "FIGHT_PROP_FIRE_ADD_HURT", "FIGHT_PROP_GRASS_ADD_HURT"
]

# 副属性数值范围定义（大致范围，允许超过理论最大值）
# 用于生成连续数值而非依赖离散的游戏数据
sub_stat_range = {
    "FIGHT_PROP_CRITICAL_HURT": (0.001, 0.5),      # 暴击伤害：0.1%-50%
    "FIGHT_PROP_CRITICAL": (0.001, 0.25),          # 暴击率：0.1%-25%
    "FIGHT_PROP_HP": (10, 1800),                   # 生命值：10-1800
    "FIGHT_PROP_HP_PERCENT": (0.001, 0.4),         # 生命值%：0.1%-40%
    "FIGHT_PROP_ATTACK": (1, 120),                 # 攻击力：1-120
    "FIGHT_PROP_ATTACK_PERCENT": (0.001, 0.4),     # 攻击力%：0.1%-40%
    "FIGHT_PROP_DEFENSE_PERCENT": (0.001, 0.5),    # 防御力%：0.1%-50%
    "FIGHT_PROP_DEFENSE": (1, 150),                # 防御力：1-150
    "FIGHT_PROP_CHARGE_EFFICIENCY": (0.001, 0.4),  # 元素充能效率：0.1%-40%
    "FIGHT_PROP_ELEMENT_MASTERY": (1, 150),        # 元素精通：1-150
}

# 加载游戏内真实数据文件
# 这些JSON文件包含原神游戏内的圣遗物属性数据
with open("./assets/ReliquaryLevelExcelConfigData.json") as f:
    string = f.read()
    main_stat_data = json.loads(string)    # 主属性等级数据
with open("./assets/ReliquaryAffixExcelConfigData.json") as f:
    string = f.read()
    sub_stat_data = json.loads(string)     # 副属性词缀数据

# 构建副属性数据映射表
# 结构：{属性类型: {星级: [数值列表]}}
sub_stat_map = {}
for item in sub_stat_data:
    star = item["DepotId"] // 100          # 从DepotId计算星级
    key = item["PropType"]                 # 属性类型
    value = item["PropValue"]              # 属性数值

    if key not in sub_stat_map:
        sub_stat_map[key] = {}

    if star not in sub_stat_map[key]:
        sub_stat_map[key][star] = []

    sub_stat_map[key][star].append(value)

# 构建主属性数据映射表
# 结构：{星级: {等级: {属性类型: 数值}}}
main_stat_map = {}
for item in main_stat_data:
    star = item["Rank"]                    # 圣遗物星级
    level = item["Level"] - 1              # 强化等级（从0开始）
    data = item["AddProps"]                # 属性增量数据

    if star not in main_stat_map:
        main_stat_map[star] = {}
    main_stat_map[star][level] = {}
    
    # 处理每个属性的数值
    for i in data:
        key = i["PropType"]                # 属性类型
        value = i["Value"]                 # 属性数值
        main_stat_map[star][level][key] = value


def format_value(stat_name, value):
    """
    格式化属性数值为显示文本
    
    Args:
        stat_name (str): 属性内部名称
        value (float): 属性数值
        
    Returns:
        str: 格式化后的属性值文本
    """
    if stat_info[stat_name]["percent"]:
        # 百分比属性：转换为百分比并保留1位小数
        return str(round(value * 100, 1)) + "%"
    else:
        # 固定数值属性：转换为整数
        temp = str(int(value))
        # 为大于等于4位的数字添加千位分隔符
        if len(temp) >= 4:
            temp = temp[0] + "," + temp[1:]
        return temp


class GenshinMainStatNameGenerator(TextGenerator):
    """
    原神圣遗物主属性名称生成器
    
    根据不同部位的主属性范围随机生成属性名称
    考虑了游戏内不同部位圣遗物的主属性限制
    
    用途：训练识别圣遗物主属性名称
    """
    def __init__(self):
        super(GenshinMainStatNameGenerator, self).__init__("Genshin Main Stat Name")

    def generate_text(self):
        """
        生成主属性名称文本
        
        Returns:
            str: 随机选择的主属性中文名称
        """
        # 随机选择圣遗物部位
        position = random.choice(list(main_stat_names.keys()))
        # 从该部位可能的主属性中随机选择
        entry = random.choice(main_stat_names[position])
        return stat_info[entry]["chs"]

    def get_lexicon(self):
        """
        获取生成文本所需的词汇表
        
        Returns:
            set: 包含所有属性名称字符的集合
        """
        ret = set()
        for k in stat_info:
            for char in stat_info[k]["chs"]:
                ret.add(char)
        return ret


class GenshinMainStatValueGenerator(TextGenerator):
    """
    原神圣遗物主属性数值生成器
    
    基于游戏内真实数据生成不同星级和等级的主属性数值
    考虑了星级权重分布，五星圣遗物出现概率更高
    
    用途：训练识别圣遗物主属性数值
    """
    def __init__(self):
        super(GenshinMainStatValueGenerator, self).__init__("Genshin Main Stat Value")

    def generate_text(self):
        """
        生成主属性数值文本
        
        Returns:
            str: 格式化的主属性数值（包含%符号或千位分隔符）
        """
        # 随机选择圣遗物部位和属性
        position = random.choice(list(main_stat_names.keys()))
        key = random.choice(main_stat_names[position])

        # 按权重随机选择星级（五星概率最高）
        star = random.choices(population=[1, 2, 3, 4, 5], weights=[0.05, 0.05, 0.2, 0.2, 0.5], k=1)[0]
        
        # 根据星级确定等级范围
        if star == 5:
            level = random.randint(0, 20)      # 五星：0-20级
        elif star == 4:
            level = random.randint(0, 16)      # 四星：0-16级
        elif star == 3:
            level = random.randint(0, 16)      # 三星：0-16级
        else:
            level = 0                          # 低星级：固定0级

        # 从真实数据中获取对应数值
        value = main_stat_map[star][level][key]

        return format_value(key, value)

    def get_lexicon(self):
        """
        获取生成文本所需的词汇表
        
        Returns:
            set: 包含数字、符号和特殊字符的集合
        """
        ret = set()
        # 包含数字、空格、百分号、逗号、斜杠等符号
        for char in " '0123456789.+%,/":
            ret.add(char)
        return ret


class GenshinSubStatGenerator(TextGenerator):
    """
    原神圣遗物副属性生成器
    
    生成格式为"属性名+数值"的副属性文本
    使用连续数值范围而非离散数据，避免数据局限性
    
    用途：训练识别圣遗物副属性条目
    """
    def __init__(self):
        super(GenshinSubStatGenerator, self).__init__("Genshin Sub Stat")

    def generate_text(self):
        """
        生成副属性文本
        
        Returns:
            str: 格式为"属性名+数值"的副属性文本
        """
        # 随机选择副属性类型
        key = random.choice(sub_stat_keys)
        
        # 使用连续数值生成，避免原版数据误差
        # 能生成777、299等特殊数值
        value = random.uniform(sub_stat_range[key][0], sub_stat_range[key][1])
        value_str = format_value(key, value)
        chs = stat_info[key]["chs"]

        return chs + "+" + value_str

    def get_lexicon(self):
        """
        获取生成文本所需的词汇表
        
        Returns:
            set: 包含属性名称字符和数值符号的集合
        """
        ret = set()
        
        # 添加所有属性名称字符
        for k in stat_info:
            for char in stat_info[k]["chs"]:
                ret.add(char)
        
        # 添加数值和符号字符
        for char in " '0123456789.+%,/":
            ret.add(char)
        
        return ret
