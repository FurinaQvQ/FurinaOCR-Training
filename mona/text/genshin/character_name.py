"""
原神角色装备文本生成器
包含原神游戏中所有角色的名称数据和装备状态文本生成器

主要用于生成"xxx已装备"格式的文本，用于OCR训练识别角色装备信息
"""

from mona.text.text_generator import TextGenerator
import random

# 原神角色名称列表（不包含旅行者）
# 包含游戏中所有可抽取角色，涵盖四星和五星角色
characters_name_genshin = [
    "珐露珊", "流浪者", "纳西妲", "莱依拉", "赛诺", "坎蒂丝", "妮露", "柯莱", "多莉", "提纳里",
    "久岐忍", "鹿野院平藏", "夜兰", "神里绫人", "云堇", "八重神子", "申鹤", "荒泷一斗", "五郎", "托马",
    "埃洛伊", "珊瑚宫心海", "雷电将军", "九条裟罗", "宵宫", "早柚", "神里绫华", "枫原万叶",
    "优菈", "烟绯", "罗莎莉亚", "胡桃", "魈", "甘雨", "阿贝多", "钟离", "辛焱", "达达利亚", "迪奥娜",
    "可莉", "温迪", "刻晴", "莫娜", "七七", "迪卢克", "琴", "砂糖", "重云", "诺艾尔", "班尼特", "菲谢尔",
    "凝光", "行秋", "北斗", "香菱", "雷泽", "芭芭拉", "丽莎", "凯亚", "安柏",
    "白术", "卡维", "瑶瑶", "艾尔海森", "迪希雅", "米卡",
    "琳妮特", "林尼", "菲米尼", "芙宁娜", "那维莱特", "夏沃蕾", "娜维娅", "嘉明", "闲云", "千织",
    "阿蕾奇诺", "夏洛蒂", "莱欧斯利", "克洛琳德", "希格雯", "赛索斯", "艾梅莉埃",
    "基尼奇", "卡齐娜", "玛拉妮", "希诺宁", "欧洛伦", "恰斯卡", "茜特菈莉", "玛薇卡",
    "蓝砚", "梦见月瑞希", "伊安珊", "瓦雷莎", "伊法", "爱可菲", "丝柯克", "塔利雅"
]


class GenshinCharacterEquipTextGenerator(TextGenerator):
    """
    原神角色装备文本生成器
    
    生成格式："xxx已装备"
    其中xxx为随机选择的角色名称
    
    用途：训练识别圣遗物界面中的角色装备状态信息
    """
    
    def __init__(self):
        super(GenshinCharacterEquipTextGenerator, self).__init__("Genshin Equip Name")

    def generate_text(self):
        """
        生成角色装备文本
        
        Returns:
            str: 格式为"角色名已装备"的文本
        """
        return random.choice(characters_name_genshin) + "已装备"

    def get_lexicon(self):
        """
        获取生成文本所需的词汇表
        
        Returns:
            set: 包含所有角色名称字符和"已装备"文字的字符集合
        """
        ret = set()
        
        # 添加所有角色名称中的字符
        for name in characters_name_genshin:
            for char in name:
                ret.add(char)

        # 添加装备状态文字
        for char in "已装备":
            ret.add(char)

        return ret
