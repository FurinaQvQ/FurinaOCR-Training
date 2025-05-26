from mona.text.genshin import *
from mona.text.common import *
from mona.text.lexicon import Lexicon


def construct_genshin_lexicon():
    """构建原神专用词汇表生成器"""
    random_artifact_name = GenshinArtifactTextGenerator()
    random_main_stat_name = GenshinMainStatNameGenerator()
    random_main_stat_value = GenshinMainStatValueGenerator()
    random_sub_stat = GenshinSubStatGenerator()
    random_level = GenshinArtifactLevelGenerator()
    random_equip = GenshinCharacterEquipTextGenerator()
    random_artifact_count = GenshinArtifactCountGenerator()
    random_number_generator = RandomNumberTextGenerator()

    weighted_generator = WeightedTextGenerator()
    weighted_generator.add_entry(0.1, random_artifact_name)
    weighted_generator.add_entry(0.05, random_main_stat_name)
    weighted_generator.add_entry(0.15, random_main_stat_value)
    weighted_generator.add_entry(0.64, random_sub_stat)
    weighted_generator.add_entry(0.02, random_level)
    weighted_generator.add_entry(0.02, random_equip)
    weighted_generator.add_entry(0.1, random_artifact_count)
    weighted_generator.add_entry(0.2, random_number_generator)

    return weighted_generator


def get_lexicon():
    """获取原神词汇表"""
    generator = construct_genshin_lexicon()
    return Lexicon(generator)
