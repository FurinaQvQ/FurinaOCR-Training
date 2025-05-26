"""
原神文本生成器模块
包含各种原神游戏相关的文本生成器类，用于OCR训练数据生成

主要功能：
- 圣遗物名称生成
- 角色装备文本生成  
- 属性名称和数值生成
- 圣遗物数量、等级等辅助信息生成
"""

# 圣遗物名称生成器
from .artifact_name import GenshinArtifactTextGenerator

# 角色装备文本生成器
from .character_name import GenshinCharacterEquipTextGenerator

# 属性相关生成器
from .stat_genshin import GenshinMainStatNameGenerator, GenshinSubStatGenerator, GenshinMainStatValueGenerator

# 辅助信息生成器
from .misc import GenshinArtifactCountGenerator, GenshinArtifactLevelGenerator
