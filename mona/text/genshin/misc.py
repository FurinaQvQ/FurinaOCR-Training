"""
原神杂项文本生成器
包含圣遗物数量、等级等辅助信息的文本生成器

主要用于生成圣遗物界面中的辅助文本信息，为OCR训练提供数据
"""

import random

from mona.text.text_generator import TextGenerator


class GenshinArtifactCountGenerator(TextGenerator):
    """
    圣遗物数量文本生成器
    
    生成格式："圣遗物 xxx/2100"
    其中xxx为0-2100之间的随机数，2100为游戏中圣遗物最大容量
    
    用途：训练识别圣遗物背包容量信息
    """
    def __init__(self):
        super(GenshinArtifactCountGenerator, self).__init__("Genshin Artifact Count")

    def generate_text(self):
        """
        生成圣遗物数量文本
        
        Returns:
            str: 格式为"圣遗物 xxx/2100"的文本
        """
        # 生成0-2100之间的随机数量
        flag_ac = random.randint(0, 2100)
        return f"圣遗物 {flag_ac}/2100"

    def get_lexicon(self):
        """
        获取生成文本所需的词汇表
        
        Returns:
            set: 包含数字、符号和"圣遗物"文字的字符集合
        """
        ret = set()
        # 添加数字、符号和中文字符
        for c in " 0123456789.+%,圣遗物":
            ret.add(c)
        return ret


class GenshinArtifactLevelGenerator(TextGenerator):
    """
    圣遗物等级文本生成器
    
    生成格式："+xx"
    其中xx为0-20之间的随机数，表示圣遗物强化等级
    
    用途：训练识别圣遗物强化等级信息
    """
    def __init__(self):
        super(GenshinArtifactLevelGenerator, self).__init__("Genshin Artifact Level")

    def generate_text(self):
        """
        生成圣遗物等级文本
        
        Returns:
            str: 格式为"+xx"的等级文本，xx为0-20的随机数
        """
        # 圣遗物最高强化等级为+20
        return "+" + str(random.randint(0, 20))

    def get_lexicon(self):
        """
        获取生成文本所需的词汇表
        
        Returns:
            set: 包含数字和"+"符号的字符集合
        """
        ret = set()
        # 只需要数字和加号
        for c in "0123456789+":
            ret.add(c)
        return ret
