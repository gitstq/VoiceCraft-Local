"""Text processing for TTS."""

import re
from typing import List, Dict
import numpy as np


class TextProcessor:
    """Process text for TTS synthesis."""
    
    # Phoneme vocabulary (simplified)
    PHONEME_VOCAB = {
        # Vowels
        'AA': 0, 'AE': 1, 'AH': 2, 'AO': 3, 'AW': 4, 'AY': 5,
        'EH': 6, 'ER': 7, 'EY': 8, 'IH': 9, 'IY': 10, 'OW': 11,
        'OY': 12, 'UH': 13, 'UW': 14,
        # Consonants
        'B': 15, 'CH': 16, 'D': 17, 'DH': 18, 'F': 19, 'G': 20,
        'HH': 21, 'JH': 22, 'K': 23, 'L': 24, 'M': 25, 'N': 26,
        'NG': 27, 'P': 28, 'R': 29, 'S': 30, 'SH': 31, 'T': 32,
        'TH': 33, 'V': 34, 'W': 35, 'Y': 36, 'Z': 37, 'ZH': 38,
        # Special tokens
        '<PAD>': 39, '<SOS>': 40, '<EOS>': 41, '<UNK>': 42,
    }
    
    # Chinese pinyin to phoneme mapping (simplified)
    PINYIN_MAP = {
        'a': 'AA', 'ai': 'AY', 'an': 'AN', 'ang': 'ANG',
        'ao': 'AW', 'ba': 'BA', 'bai': 'BAY', 'ban': 'BAN',
        'bang': 'BANG', 'bao': 'BAW', 'bei': 'BEY', 'ben': 'BEN',
        'beng': 'BENG', 'bi': 'BI', 'bian': 'BIAN', 'biao': 'BIAW',
        'bie': 'BIE', 'bin': 'BIN', 'bing': 'BING', 'bo': 'BO',
        'bu': 'BU', 'ca': 'CA', 'cai': 'CAY', 'can': 'CAN',
        'cang': 'CANG', 'cao': 'CAW', 'ce': 'CE', 'cen': 'CEN',
        'ceng': 'CENG', 'cha': 'CHA', 'chai': 'CHAY', 'chan': 'CHAN',
        'chang': 'CHANG', 'chao': 'CHAW', 'che': 'CHE', 'chen': 'CHEN',
        'cheng': 'CHENG', 'chi': 'CHI', 'chong': 'CHONG', 'chou': 'CHOW',
        'chu': 'CHU', 'chua': 'CHUA', 'chuai': 'CHUAY', 'chuan': 'CHUAN',
        'chuang': 'CHUANG', 'chui': 'CHUI', 'chun': 'CHUN', 'chuo': 'CHUO',
        'ci': 'CI', 'cong': 'CONG', 'cou': 'COW', 'cu': 'CU',
        'cuan': 'CUAN', 'cui': 'CUI', 'cun': 'CUN', 'cuo': 'CUO',
        'da': 'DA', 'dai': 'DAY', 'dan': 'DAN', 'dang': 'DANG',
        'dao': 'DAW', 'de': 'DE', 'dei': 'DEY', 'den': 'DEN',
        'deng': 'DENG', 'di': 'DI', 'dian': 'DIAN', 'diao': 'DIAW',
        'die': 'DIE', 'ding': 'DING', 'diu': 'DIU', 'dong': 'DONG',
        'dou': 'DOW', 'du': 'DU', 'duan': 'DUAN', 'dui': 'DUI',
        'dun': 'DUN', 'duo': 'DUO', 'e': 'E', 'ei': 'EY',
        'en': 'EN', 'eng': 'ENG', 'er': 'ER', 'fa': 'FA',
        'fan': 'FAN', 'fang': 'FANG', 'fei': 'FEY', 'fen': 'FEN',
        'feng': 'FENG', 'fo': 'FO', 'fou': 'FOW', 'fu': 'FU',
        'ga': 'GA', 'gai': 'GAY', 'gan': 'GAN', 'gang': 'GANG',
        'gao': 'GAW', 'ge': 'GE', 'gei': 'GEY', 'gen': 'GEN',
        'geng': 'GENG', 'gong': 'GONG', 'gou': 'GOW', 'gu': 'GU',
        'gua': 'GUA', 'guai': 'GUAY', 'guan': 'GUAN', 'guang': 'GUANG',
        'gui': 'GUI', 'gun': 'GUN', 'guo': 'GUO', 'ha': 'HA',
        'hai': 'HAY', 'han': 'HAN', 'hang': 'HANG', 'hao': 'HAW',
        'he': 'HE', 'hei': 'HEY', 'hen': 'HEN', 'heng': 'HENG',
        'hong': 'HONG', 'hou': 'HOW', 'hu': 'HU', 'hua': 'HUA',
        'huai': 'HUAY', 'huan': 'HUAN', 'huang': 'HUANG', 'hui': 'HUI',
        'hun': 'HUN', 'huo': 'HUO', 'ji': 'JI', 'jia': 'JIA',
        'jian': 'JIAN', 'jiang': 'JIANG', 'jiao': 'JIAW', 'jie': 'JIE',
        'jin': 'JIN', 'jing': 'JING', 'jiong': 'JIONG', 'jiu': 'JIU',
        'ju': 'JU', 'juan': 'JUAN', 'jue': 'JUE', 'jun': 'JUN',
        'ka': 'KA', 'kai': 'KAY', 'kan': 'KAN', 'kang': 'KANG',
        'kao': 'KAW', 'ke': 'KE', 'kei': 'KEY', 'ken': 'KEN',
        'keng': 'KENG', 'kong': 'KONG', 'kou': 'KOW', 'ku': 'KU',
        'kua': 'KUA', 'kuai': 'KUAY', 'kuan': 'KUAN', 'kuang': 'KUANG',
        'kui': 'KUI', 'kun': 'KUN', 'kuo': 'KUO', 'la': 'LA',
        'lai': 'LAY', 'lan': 'LAN', 'lang': 'LANG', 'lao': 'LAW',
        'le': 'LE', 'lei': 'LEY', 'leng': 'LENG', 'li': 'LI',
        'lia': 'LIA', 'lian': 'LIAN', 'liang': 'LIANG', 'liao': 'LIAW',
        'lie': 'LIE', 'lin': 'LIN', 'ling': 'LING', 'liu': 'LIU',
        'lo': 'LO', 'long': 'LONG', 'lou': 'LOW', 'lu': 'LU',
        'luan': 'LUAN', 'lun': 'LUN', 'luo': 'LUO', 'ma': 'MA',
        'mai': 'MAY', 'man': 'MAN', 'mang': 'MANG', 'mao': 'MAW',
        'me': 'ME', 'mei': 'MEY', 'men': 'MEN', 'meng': 'MENG',
        'mi': 'MI', 'mian': 'MIAN', 'miao': 'MIAW', 'mie': 'MIE',
        'min': 'MIN', 'ming': 'MING', 'miu': 'MIU', 'mo': 'MO',
        'mou': 'MOW', 'mu': 'MU', 'na': 'NA', 'nai': 'NAY',
        'nan': 'NAN', 'nang': 'NANG', 'nao': 'NAW', 'ne': 'NE',
        'nei': 'NEY', 'nen': 'NEN', 'neng': 'NENG', 'ni': 'NI',
        'nian': 'NIAN', 'niang': 'NIANG', 'niao': 'NIAW', 'nie': 'NIE',
        'nin': 'NIN', 'ning': 'NING', 'niu': 'NIU', 'nong': 'NONG',
        'nou': 'NOW', 'nu': 'NU', 'nuan': 'NUAN', 'nun': 'NUN',
        'nuo': 'NUO', 'o': 'O', 'ou': 'OW', 'pa': 'PA',
        'pai': 'PAY', 'pan': 'PAN', 'pang': 'PANG', 'pao': 'PAW',
        'pei': 'PEY', 'pen': 'PEN', 'peng': 'PENG', 'pi': 'PI',
        'pian': 'PIAN', 'piao': 'PIAW', 'pie': 'PIE', 'pin': 'PIN',
        'ping': 'PING', 'po': 'PO', 'pou': 'POW', 'pu': 'PU',
        'qi': 'QI', 'qia': 'QIA', 'qian': 'QIAN', 'qiang': 'QIANG',
        'qiao': 'QIAW', 'qie': 'QIE', 'qin': 'QIN', 'qing': 'QING',
        'qiong': 'QIONG', 'qiu': 'QIU', 'qu': 'QU', 'quan': 'QUAN',
        'que': 'QUE', 'qun': 'QUN', 'ran': 'RAN', 'rang': 'RANG',
        'rao': 'RAW', 're': 'RE', 'ren': 'REN', 'reng': 'RENG',
        'ri': 'RI', 'rong': 'RONG', 'rou': 'ROW', 'ru': 'RU',
        'ruan': 'RUAN', 'rui': 'RUI', 'run': 'RUN', 'ruo': 'RUO',
        'sa': 'SA', 'sai': 'SAY', 'san': 'SAN', 'sang': 'SANG',
        'sao': 'SAW', 'se': 'SE', 'sen': 'SEN', 'seng': 'SENG',
        'sha': 'SHA', 'shai': 'SHAY', 'shan': 'SHAN', 'shang': 'SHANG',
        'shao': 'SHAW', 'she': 'SHE', 'shei': 'SHEY', 'shen': 'SHEN',
        'sheng': 'SHENG', 'shi': 'SHI', 'shou': 'SHOW', 'shu': 'SHU',
        'shua': 'SHUA', 'shuai': 'SHUAY', 'shuan': 'SHUAN', 'shuang': 'SHUANG',
        'shui': 'SHUI', 'shun': 'SHUN', 'shuo': 'SHUO', 'si': 'SI',
        'song': 'SONG', 'sou': 'SOW', 'su': 'SU', 'suan': 'SUAN',
        'sui': 'SUI', 'sun': 'SUN', 'suo': 'SUO', 'ta': 'TA',
        'tai': 'TAY', 'tan': 'TAN', 'tang': 'TANG', 'tao': 'TAW',
        'te': 'TE', 'tei': 'TEY', 'teng': 'TENG', 'ti': 'TI',
        'tian': 'TIAN', 'tiao': 'TIAW', 'tie': 'TIE', 'ting': 'TING',
        'tong': 'TONG', 'tou': 'TOW', 'tu': 'TU', 'tuan': 'TUAN',
        'tui': 'TUI', 'tun': 'TUN', 'tuo': 'TUO', 'wa': 'WA',
        'wai': 'WAY', 'wan': 'WAN', 'wang': 'WANG', 'wei': 'WEY',
        'wen': 'WEN', 'weng': 'WENG', 'wo': 'WO', 'wu': 'WU',
        'xi': 'XI', 'xia': 'XIA', 'xian': 'XIAN', 'xiang': 'XIANG',
        'xiao': 'XIAW', 'xie': 'XIE', 'xin': 'XIN', 'xing': 'XING',
        'xiong': 'XIONG', 'xiu': 'XIU', 'xu': 'XU', 'xuan': 'XUAN',
        'xue': 'XUE', 'xun': 'XUN', 'ya': 'YA', 'yan': 'YAN',
        'yang': 'YANG', 'yao': 'YAW', 'ye': 'YE', 'yi': 'YI',
        'yin': 'YIN', 'ying': 'YING', 'yo': 'YO', 'yong': 'YONG',
        'you': 'YOW', 'yu': 'YU', 'yuan': 'YUAN', 'yue': 'YUE',
        'yun': 'YUN', 'za': 'ZA', 'zai': 'ZAY', 'zan': 'ZAN',
        'zang': 'ZANG', 'zao': 'ZAW', 'ze': 'ZE', 'zei': 'ZEY',
        'zen': 'ZEN', 'zeng': 'ZENG', 'zha': 'ZHA', 'zhai': 'ZHAY',
        'zhan': 'ZHAN', 'zhang': 'ZHANG', 'zhao': 'ZHAW', 'zhe': 'ZHE',
        'zhei': 'ZHEY', 'zhen': 'ZHEN', 'zheng': 'ZHENG', 'zhi': 'ZHI',
        'zhong': 'ZHONG', 'zhou': 'ZHOW', 'zhu': 'ZHU', 'zhua': 'ZHUA',
        'zhuai': 'ZHUAY', 'zhuan': 'ZHUAN', 'zhuang': 'ZHUANG', 'zhui': 'ZHUI',
        'zhun': 'ZHUN', 'zhuo': 'ZHUO', 'zi': 'ZI', 'zong': 'ZONG',
        'zou': 'ZOW', 'zu': 'ZU', 'zuan': 'ZUAN', 'zui': 'ZUI',
        'zun': 'ZUN', 'zuo': 'ZUO',
    }
    
    def __init__(self, language: str = "zh"):
        """Initialize text processor.
        
        Args:
            language: Default language code.
        """
        self.language = language
        self.vocab_size = len(self.PHONEME_VOCAB)
    
    def text_to_phonemes(self, text: str, language: str = None) -> List[str]:
        """Convert text to phoneme sequence.
        
        Args:
            text: Input text.
            language: Language code override.
            
        Returns:
            List of phonemes.
        """
        language = language or self.language
        
        # Normalize text
        text = self._normalize_text(text)
        
        if language == "zh":
            return self._chinese_to_phonemes(text)
        elif language == "en":
            return self._english_to_phonemes(text)
        else:
            # Fallback to character-based
            return list(text.lower())
    
    def _normalize_text(self, text: str) -> str:
        """Normalize input text."""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep punctuation
        text = re.sub(r'[^\w\s。！？.,!?;:]', '', text)
        return text.strip()
    
    def _chinese_to_phonemes(self, text: str) -> List[str]:
        """Convert Chinese text to phonemes (simplified)."""
        # In a real implementation, this would use a proper
        # Chinese text-to-pinyin library like pypinyin
        # For this demo, we'll use a simplified approach
        
        phonemes = []
        for char in text:
            if char in self.PINYIN_MAP:
                phonemes.append(self.PINYIN_MAP[char])
            elif char.isalpha():
                phonemes.append(char.upper())
            elif char in '。！？.!?':
                phonemes.append('<PAUSE>')
            else:
                phonemes.append('<UNK>')
        
        return phonemes
    
    def _english_to_phonemes(self, text: str) -> List[str]:
        """Convert English text to phonemes (simplified)."""
        # In a real implementation, this would use a proper
        # G2P (Grapheme-to-Phoneme) converter
        
        phonemes = []
        words = text.lower().split()
        
        for word in words:
            # Simple letter-based phonemization
            for char in word:
                if char.isalpha():
                    phonemes.append(char.upper())
            phonemes.append('<SPACE>')
        
        return phonemes[:-1] if phonemes else phonemes  # Remove last space
    
    def phonemes_to_ids(self, phonemes: List[str]) -> List[int]:
        """Convert phonemes to IDs.
        
        Args:
            phonemes: List of phonemes.
            
        Returns:
            List of phoneme IDs.
        """
        ids = [self.PHONEME_VOCAB['<SOS>']]
        
        for phoneme in phonemes:
            ids.append(self.PHONEME_VOCAB.get(phoneme, self.PHONEME_VOCAB['<UNK>']))
        
        ids.append(self.PHONEME_VOCAB['<EOS>'])
        return ids
    
    def ids_to_phonemes(self, ids: List[int]) -> List[str]:
        """Convert IDs back to phonemes.
        
        Args:
            ids: List of phoneme IDs.
            
        Returns:
            List of phonemes.
        """
        id_to_phoneme = {v: k for k, v in self.PHONEME_VOCAB.items()}
        return [id_to_phoneme.get(i, '<UNK>') for i in ids]
