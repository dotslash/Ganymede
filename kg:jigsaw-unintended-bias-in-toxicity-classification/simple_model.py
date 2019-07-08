import datetime
from typing import List, Tuple, Dict, Callable

import numpy as np
import pandas
import tensorflow as tf
from keras.callbacks import LearningRateScheduler
from keras.initializers import Constant
from keras.layers import Bidirectional, GlobalMaxPooling1D
from keras.layers import Embedding
from keras.layers import GlobalAveragePooling1D, CuDNNLSTM
from keras.layers import Input, Dense, SpatialDropout1D
from keras.layers import add, concatenate
from keras.models import Model
from keras.preprocessing import text, sequence

# Flags.
LSTM_UNITS: int = 128
DENSE_HIDDEN_UNITS: int = 4 * LSTM_UNITS
MAX_SEQUENCE_LENGTH: int = 200
TESTING_MODE: bool = False
ENABLE_TEXT_PROCESSING: bool = True
BATCH_SIZE: int = 1024
NUM_EPOCHS: int = 4
TOKENIZER_NUM_WORDS: int = 50000
NUM_MODELS: int = 2

# Facts.
CATEGORY_COLS: List[str] = ['severe_toxicity', 'obscene', 'identity_attack',
                            'insult', 'threat']
IDENTITY_COLS: List[str] = [
    'male', 'female', 'homosexual_gay_or_lesbian', 'christian', 'jewish',
    'muslim', 'black', 'white', 'psychiatric_or_mental_illness']

# TODO(dotslash): Make the file paths work both in kaggle and locally.
TRAIN_FILE: str = \
    '../input/jigsaw-unintended-bias-in-toxicity-classification/train.csv'
TEST_FILE: str = \
    '../input/jigsaw-unintended-bias-in-toxicity-classification/test.csv'
EMBEDDING_FILES: List[str] = [
    '../input/fasttext-crawl-300d-2m/crawl-300d-2M.vec',
    '../input/glove840b300dtxt/glove.840B.300d.txt']
# Facts in the form of Data.
# TEXT PROCESSING STUFF (NOT SURE IF THIS HELPS)
CONTRACTION_MAPPING: Dict[str, str] = {
    "'cause": "because",
    "ain't": "is not",
    "aren't": "are not",
    "can't": "cannot",
    "could've": "could have",
    "couldn't": "could not",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he would",
    "he'll": "he will",
    "he's": "he is",
    "here's": "here is",
    "how'd": "how did",
    "how'd'y": "how do you",
    "how'll": "how will",
    "how's": "how is",
    "i'd": "i would",
    "i'd've": "i would have",
    "i'll": "i will",
    "i'll've": "i will have",
    "i'm": "i am",
    "i've": "i have",
    "isn't": "is not",
    "it'd": "it would",
    "it'd've": "it would have",
    "it'll": "it will",
    "it'll've": "it will have",
    "it's": "it is",
    "let's": "let us",
    "ma'am": "madam",
    "mayn't": "may not",
    "might've": "might have",
    "mightn't": "might not",
    "mightn't've": "might not have",
    "must've": "must have",
    "mustn't": "must not",
    "mustn't've": "must not have",
    "needn't": "need not",
    "needn't've": "need not have",
    "o'clock": "of the clock",
    "oughtn't": "ought not",
    "oughtn't've": "ought not have",
    "sha'n't": "shall not",
    "shan't": "shall not",
    "shan't've": "shall not have",
    "she'd": "she would",
    "she'd've": "she would have",
    "she'll": "she will",
    "she'll've": "she will have",
    "she's": "she is",
    "should've": "should have",
    "shouldn't": "should not",
    "shouldn't've": "should not have",
    "so's": "so as",
    "so've": "so have",
    "that'd": "that would",
    "that'd've": "that would have",
    "that's": "that is",
    "there'd": "there would",
    "there'd've": "there would have",
    "there's": "there is",
    "they'd": "they would",
    "they'd've": "they would have",
    "they'll": "they will",
    "they'll've": "they will have",
    "they're": "they are",
    "they've": "they have",
    "this's": "this is",
    "to've": "to have",
    "wasn't": "was not",
    "we'd": "we would",
    "we'd've": "we would have",
    "we'll": "we will",
    "we'll've": "we will have",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what'll": "what will",
    "what'll've": "what will have",
    "what're": "what are",
    "what's": "what is",
    "what've": "what have",
    "when's": "when is",
    "when've": "when have",
    "where'd": "where did",
    "where's": "where is",
    "where've": "where have",
    "who'll": "who will",
    "who'll've": "who will have",
    "who's": "who is",
    "who've": "who have",
    "why's": "why is",
    "why've": "why have",
    "will've": "will have",
    "won't": "will not",
    "won't've": "will not have",
    "would've": "would have",
    "wouldn't": "would not",
    "wouldn't've": "would not have",
    "y'all": "you all",
    "y'all'd": "you all would",
    "y'all'd've": "you all would have",
    "y'all're": "you all are",
    "y'all've": "you all have",
    "you'd": "you would",
    "you'd've": "you would have",
    "you'll": "you will",
    "you'll've": "you will have",
    "you're": "you are",
    "you've": "you have",
}
for k, v in list(CONTRACTION_MAPPING.items()):
    CONTRACTION_MAPPING[k.capitalize()] = v.capitalize()
QUOTES = ['’', '‘', '´', '`']
__SYMBOLS_TO_ISOLATE = '.,?!-;*"…:—()%#$&_/@＼・ω+=”“[]^–>\\°<~•≠™ˈʊɒ∞§{}·τα❤☺ɡ|¢→̶`❥━┣┫┗Ｏ►★©―ɪ✔®\x96\x92●£♥➤´¹☕≈÷♡◐║▬′ɔː€۩۞†μ✒➥═☆ˌ◄½ʻπδηλσερνʃ✬ＳＵＰＥＲＩＴ☻±♍µº¾✓◾؟．⬅℅»Вав❣⋅¿¬♫ＣＭβ█▓▒░⇒⭐›¡₂₃❧▰▔◞▀▂▃▄▅▆▇↙γ̄″☹➡«φ⅓„✋：¥̲̅́∙‛◇✏▷❓❗¶˚˙）сиʿ✨。ɑ\x80◕！％¯−ﬂﬁ₁²ʌ¼⁴⁄₄⌠♭✘╪▶☭✭♪☔☠♂☃☎✈✌✰❆☙○‣⚓年∎ℒ▪▙☏⅛ｃａｓǀ℮¸ｗ‚∼‖ℳ❄←☼⋆ʒ⊂、⅔¨͡๏⚾⚽Φ×θ￦？（℃⏩☮⚠月✊❌⭕▸■⇌☐☑⚡☄ǫ╭∩╮，例＞ʕɐ̣Δ₀✞┈╱╲▏▕┃╰▊▋╯┳┊≥☒↑☝ɹ✅☛♩☞ＡＪＢ◔◡↓♀⬆̱ℏ\x91⠀ˤ╚↺⇤∏✾◦♬³の｜／∵∴√Ω¤☜▲↳▫‿⬇✧ｏｖｍ－２０８＇‰≤∕ˆ⚜☁'
__SYMBOLS_TO_REMOVE = '\n🍕\r🐵\xa0\ue014\t\uf818\uf04a\xad😢🐶️\uf0e0😜😎👊\u200b\u200e😁عدويهصقأناخلىبمغر😍💖💵Е👎😀😂\u202a\u202c🔥😄🏻💥ᴍʏʀᴇɴᴅᴏᴀᴋʜᴜʟᴛᴄᴘʙғᴊᴡɢ😋👏שלוםבי😱‼\x81エンジ故障\u2009🚌ᴵ͞🌟😊😳😧🙀😐😕\u200f👍😮😃😘אעכח💩💯⛽🚄🏼ஜ😖ᴠ🚲‐😟😈💪🙏🎯🌹😇💔😡\x7f👌ἐὶήιὲκἀίῃἴξ🙄Ｈ😠\ufeff\u2028😉😤⛺🙂\u3000تحكسة👮💙فزط😏🍾🎉😞\u2008🏾😅😭👻😥😔😓🏽🎆🍻🍽🎶🌺🤔😪\x08‑🐰🐇🐱🙆😨🙃💕𝘊𝘦𝘳𝘢𝘵𝘰𝘤𝘺𝘴𝘪𝘧𝘮𝘣💗💚地獄谷улкнПоАН🐾🐕😆ה🔗🚽歌舞伎🙈😴🏿🤗🇺🇸мυтѕ⤵🏆🎃😩\u200a🌠🐟💫💰💎эпрд\x95🖐🙅⛲🍰🤐👆🙌\u2002💛🙁👀🙊🙉\u2004ˢᵒʳʸᴼᴷᴺʷᵗʰᵉᵘ\x13🚬🤓\ue602😵άοόςέὸתמדףנרךצט😒͝🆕👅👥👄🔄🔤👉👤👶👲🔛🎓\uf0b7\uf04c\x9f\x10成都😣⏺😌🤑🌏😯ех😲Ἰᾶὁ💞🚓🔔📚🏀👐\u202d💤🍇\ue613小土豆🏡❔⁉\u202f👠》कर्मा🇹🇼🌸蔡英文🌞🎲レクサス😛外国人关系Сб💋💀🎄💜🤢َِьыгя不是\x9c\x9d🗑\u2005💃📣👿༼つ༽😰ḷЗз▱ц￼🤣卖温哥华议会下降你失去所有的钱加拿大坏税骗子🐝ツ🎅\x85🍺آإشء🎵🌎͟ἔ油别克🤡🤥😬🤧й\u2003🚀🤴ʲшчИОРФДЯМюж😝🖑ὐύύ特殊作戦群щ💨圆明园קℐ🏈😺🌍⏏ệ🍔🐮🍁🍆🍑🌮🌯🤦\u200d𝓒𝓲𝓿𝓵안영하세요ЖљКћ🍀😫🤤ῦ我出生在了可以说普通话汉语好极🎼🕺🍸🥂🗽🎇🎊🆘🤠👩🖒🚪天一家⚲\u2006⚭⚆⬭⬯⏖新✀╌🇫🇷🇩🇪🇮🇬🇧😷🇨🇦ХШ🌐\x1f杀鸡给猴看ʁ𝗪𝗵𝗲𝗻𝘆𝗼𝘂𝗿𝗮𝗹𝗶𝘇𝗯𝘁𝗰𝘀𝘅𝗽𝘄𝗱📺ϖ\u2000үսᴦᎥһͺ\u2007հ\u2001ɩｙｅ൦ｌƽｈ𝐓𝐡𝐞𝐫𝐮𝐝𝐚𝐃𝐜𝐩𝐭𝐢𝐨𝐧Ƅᴨןᑯ໐ΤᏧ௦Іᴑ܁𝐬𝐰𝐲𝐛𝐦𝐯𝐑𝐙𝐣𝐇𝐂𝐘𝟎ԜТᗞ౦〔Ꭻ𝐳𝐔𝐱𝟔𝟓𝐅🐋ﬃ💘💓ё𝘥𝘯𝘶💐🌋🌄🌅𝙬𝙖𝙨𝙤𝙣𝙡𝙮𝙘𝙠𝙚𝙙𝙜𝙧𝙥𝙩𝙪𝙗𝙞𝙝𝙛👺🐷ℋ𝐀𝐥𝐪🚶𝙢Ἱ🤘ͦ💸ج패티Ｗ𝙇ᵻ👂👃ɜ🎫\uf0a7БУі🚢🚂ગુજરાતીῆ🏃𝓬𝓻𝓴𝓮𝓽𝓼☘﴾̯﴿₽\ue807𝑻𝒆𝒍𝒕𝒉𝒓𝒖𝒂𝒏𝒅𝒔𝒎𝒗𝒊👽😙\u200cЛ‒🎾👹⎌🏒⛸公寓养宠物吗🏄🐀🚑🤷操美𝒑𝒚𝒐𝑴🤙🐒欢迎来到阿拉斯ספ𝙫🐈𝒌𝙊𝙭𝙆𝙋𝙍𝘼𝙅ﷻ🦄巨收赢得白鬼愤怒要买额ẽ🚗🐳𝟏𝐟𝟖𝟑𝟕𝒄𝟗𝐠𝙄𝙃👇锟斤拷𝗢𝟳𝟱𝟬⦁マルハニチロ株式社⛷한국어ㄸㅓ니͜ʖ𝘿𝙔₵𝒩ℯ𝒾𝓁𝒶𝓉𝓇𝓊𝓃𝓈𝓅ℴ𝒻𝒽𝓀𝓌𝒸𝓎𝙏ζ𝙟𝘃𝗺𝟮𝟭𝟯𝟲👋🦊多伦🐽🎻🎹⛓🏹🍷🦆为和中友谊祝贺与其想象对法如直接问用自己猜本传教士没积唯认识基督徒曾经让相信耶稣复活死怪他但当们聊些政治题时候战胜因圣把全堂结婚孩恐惧且栗谓这样还♾🎸🤕🤒⛑🎁批判检讨🏝🦁🙋😶쥐스탱트뤼도석유가격인상이경제황을렵게만들지않록잘관리해야합다캐나에서대마초와화약금의품런성분갈때는반드시허된사용🔫👁凸ὰ💲🗯𝙈Ἄ𝒇𝒈𝒘𝒃𝑬𝑶𝕾𝖙𝖗𝖆𝖎𝖌𝖍𝖕𝖊𝖔𝖑𝖉𝖓𝖐𝖜𝖞𝖚𝖇𝕿𝖘𝖄𝖛𝖒𝖋𝖂𝕴𝖟𝖈𝕸👑🚿💡知彼百\uf005𝙀𝒛𝑲𝑳𝑾𝒋𝟒😦𝙒𝘾𝘽🏐𝘩𝘨ὼṑ𝑱𝑹𝑫𝑵𝑪🇰🇵👾ᓇᒧᔭᐃᐧᐦᑳᐨᓃᓂᑲᐸᑭᑎᓀᐣ🐄🎈🔨🐎🤞🐸💟🎰🌝🛳点击查版🍭𝑥𝑦𝑧ＮＧ👣\uf020っ🏉ф💭🎥Ξ🐴👨🤳🦍\x0b🍩𝑯𝒒😗𝟐🏂👳🍗🕉🐲چی𝑮𝗕𝗴🍒ꜥⲣⲏ🐑⏰鉄リ事件ї💊「」\uf203\uf09a\uf222\ue608\uf202\uf099\uf469\ue607\uf410\ue600燻製シ虚偽屁理屈Г𝑩𝑰𝒀𝑺🌤𝗳𝗜𝗙𝗦𝗧🍊ὺἈἡχῖΛ⤏🇳𝒙ψՁմեռայինրւդձ冬至ὀ𝒁🔹🤚🍎𝑷🐂💅𝘬𝘱𝘸𝘷𝘐𝘭𝘓𝘖𝘹𝘲𝘫کΒώ💢ΜΟΝΑΕ🇱♲𝝈↴💒⊘Ȼ🚴🖕🖤🥘📍👈➕🚫🎨🌑🐻𝐎𝐍𝐊𝑭🤖🎎😼🕷ｇｒｎｔｉｄｕｆｂｋ𝟰🇴🇭🇻🇲𝗞𝗭𝗘𝗤👼📉🍟🍦🌈🔭《🐊🐍\uf10aლڡ🐦\U0001f92f\U0001f92a🐡💳ἱ🙇𝗸𝗟𝗠𝗷🥜さようなら🔼'
ISOLATE_DICT = {ord(c): f' {c} ' for c in __SYMBOLS_TO_ISOLATE}
REMOVE_DICT = {ord(c): ' ' for c in __SYMBOLS_TO_REMOVE}


def pretty_time_delta(delta: datetime.timedelta):
    seconds = delta.total_seconds()
    sign_string = '-' if seconds < 0 else ''
    seconds = abs(int(seconds))
    days, seconds = divmod(seconds, 86400)
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    if days > 0:
        return '%s%dd%dh%dm%ds' % (sign_string, days, hours, minutes, seconds)
    else:
        return '%s%dh%dm%ds' % (sign_string, hours, minutes, seconds)


class Logger:
    def __init__(self):
        self.start = datetime.datetime.now()

    def log(self, message: str) -> None:
        now = datetime.datetime.now()
        time_taken = pretty_time_delta(now - self.start)
        print('{} delta-{}: {}'.format(now, time_taken, message))


logger = Logger()
logger.log('Started.')


class TextCleaner:
    def __init__(self):
        self.specials: List[str] = QUOTES
        self.contractions_dict: Dict[str, str] = CONTRACTION_MAPPING
        self.clean_chars_translation_dict: Dict[int, str] = ISOLATE_DICT
        for k, v in REMOVE_DICT.items():
            self.clean_chars_translation_dict[k] = v

    def contractions(self, inp: str) -> str:
        for s in self.specials:
            inp = inp.replace(s, "'")
        return ' '.join(
            [self.contractions_dict.get(word, word) for word in inp.split(' ')])

    def contractions_lambda(self) -> Callable[[str], str]:
        return lambda x: self.contractions(x)

    def clean_chars(self, inp: str) -> str:
        return inp.translate(self.clean_chars_translation_dict)

    def clean_chars_lambda(self) -> Callable[[str], str]:
        return lambda x: self.clean_chars(x)


class EmbeddingStore:
    def __init__(self, embedding_file: str):
        f = open(embedding_file)
        logger.log('Loading embedding file:{}'.format(embedding_file))
        self.dict = dict()
        for line in f:
            if TESTING_MODE and len(self.dict) > 100000:
                # 100k words are enough if we are in test mode
                break
            values = line.strip().split(' ')
            word = values[0]
            try:
                coeffs = np.asarray(values[1:], dtype='float32')
                self.vector_length = len(coeffs)
                self.dict[word] = coeffs
            except Exception:
                logger.log('Failed parsing embedding for "{}"'.format(word))
        f.close()
        logger.log('Loaded embedding file: {}'.format(embedding_file))
        logger.log('Found %s word vectors.' % len(self.dict))

    def embedding(self, word: str) -> np.array:
        return self.dict.get(word, np.zeros(self.vector_length))


def get_top_words(tokenizer: text.Tokenizer):
    ret = [(v, k) for k, v in tokenizer.index_word.items()]
    return ret[:tokenizer.num_words]


def binary_accuracy(y_true: tf.Tensor, y_pred: tf.Tensor):
    import keras.backend as kb
    return kb.mean(kb.equal(kb.round(y_true), kb.round(y_pred)))


def build_model(embedding_matrix: np.array, num_other_results: int):
    inp = Input(shape=(MAX_SEQUENCE_LENGTH,))
    x = Embedding(embedding_matrix.shape[0], embedding_matrix.shape[1],
                  embeddings_initializer=Constant(embedding_matrix),
                  input_length=MAX_SEQUENCE_LENGTH,
                  trainable=False)(inp)
    x = SpatialDropout1D(0.2)(x)
    x = Bidirectional(CuDNNLSTM(LSTM_UNITS, return_sequences=True))(x)
    x = Bidirectional(CuDNNLSTM(LSTM_UNITS, return_sequences=True))(x)
    x = concatenate([
        GlobalMaxPooling1D()(x),
        GlobalAveragePooling1D()(x)])
    x = add([x, Dense(DENSE_HIDDEN_UNITS, activation='relu')(x)])
    x = add([x, Dense(DENSE_HIDDEN_UNITS, activation='relu')(x)])
    result = Dense(1, activation='sigmoid')(x)
    other_results = Dense(num_other_results, activation='sigmoid')(x)
    model = Model(inputs=inp, outputs=[result, other_results])
    model.compile(loss='binary_crossentropy', optimizer='adam',
                  metrics=['acc', binary_accuracy])
    return model


def print_diff(s1: pandas.Series, s2: pandas.Series) -> None:
    diff: pandas.Series = (s1 == s2)
    logger.log('diff')
    print(diff.value_counts())


# TODO(dotslash): Create a container type for return value of this
#                 function.
def load_train_data() -> Tuple[np.array, np.array,
                               np.array, np.array,
                               pandas.DataFrame, pandas.DataFrame,
                               text.Tokenizer]:
    # Load Training and Testing data.
    train_data: pandas.DataFrame = pandas.read_csv(TRAIN_FILE)
    test_data: pandas.DataFrame = pandas.read_csv(TEST_FILE)
    np.random.shuffle(train_data.values)
    logger.log('Loaded train and test data.')

    if TESTING_MODE:
        train_data = train_data.head(10000)
        test_data = test_data.head(10000)

    if ENABLE_TEXT_PROCESSING:
        # eliminate contractions
        logger.log('Processing text.')
        cleaner = TextCleaner()
        orig: pandas.Series = train_data['comment_text'].copy(deep=True)
        train_data['comment_text'] = \
            train_data['comment_text'].apply(cleaner.contractions_lambda())
        test_data['comment_text'] = \
            test_data['comment_text'].apply(cleaner.contractions_lambda())
        print_diff(train_data['comment_text'], orig)
        # remove junk chars
        train_data['comment_text'] = \
            train_data['comment_text'].apply(cleaner.clean_chars_lambda())
        test_data['comment_text'] = \
            test_data['comment_text'].apply(cleaner.clean_chars_lambda())
        print_diff(train_data['comment_text'], orig)
        logger.log('Processed text.')

    # Trim the train data and keep only the useful columns.
    useful_cols: List[str] = \
        ['id', 'comment_text', 'target'] + CATEGORY_COLS + IDENTITY_COLS
    train_data: pandas.DataFrame = train_data[useful_cols]
    print('Sample training data\n' + train_data.head().to_string())
    print('Sample test data\n' + test_data.head().to_string())

    # Create a tokenizer based on train and test data.
    tokenizer: text.Tokenizer = text.Tokenizer(num_words=TOKENIZER_NUM_WORDS)
    tokenizer.fit_on_texts(list(train_data['comment_text']) + \
                           list(test_data['comment_text']))
    logger.log('Fit text tokens.')

    # Prepare X, Y for training and testing.
    # We will convert the text to a sequence using the tokenizer.
    train_seq = tokenizer.texts_to_sequences(list(train_data['comment_text']))
    train_seq = sequence.pad_sequences(train_seq, maxlen=MAX_SEQUENCE_LENGTH)
    test_seq = tokenizer.texts_to_sequences(list(test_data['comment_text']))
    test_seq = sequence.pad_sequences(test_seq, maxlen=MAX_SEQUENCE_LENGTH)
    logger.log('Converted tokens to sequences.')

    x_train, y_train, y_other_train = \
        train_seq, train_data['target'], train_data[CATEGORY_COLS]
    x_test = test_seq
    logger.log('Prepared and train, validation and test sets.')
    return x_train, y_train, y_other_train, x_test, train_data, test_data, tokenizer


def construct_embedding_matrix(tokenizer: text.Tokenizer) -> np.array:
    # Load embeddings from disk.
    embeddings = [EmbeddingStore(embedding_file)
                  for embedding_file in EMBEDDING_FILES]
    # Construct a embedding matrix used for Embedding layer.
    embedding_dim = sum(embedding.vector_length for embedding in embeddings)
    tokenizer_words = get_top_words(tokenizer)
    embedding_matrix = np.zeros((len(tokenizer_words) + 1, embedding_dim))
    for word, ind in tokenizer_words:
        embedding_matrix[ind] = np.concatenate(
            [embedding.embedding(word) for embedding in embeddings])
    logger.log('Created embedding matrix.')
    return embedding_matrix


def main():
    x_train, y_train, y_other_train, x_test, train_data, \
    test_data, tokenizer = load_train_data()
    embedding_matrix = construct_embedding_matrix(tokenizer)
    sample_weights: pandas.Series = pandas.Series(
        data=np.ones(len(x_train), dtype=np.float32))

    for column in IDENTITY_COLS:
        train_data[column] = np.where(train_data[column] >= 0.5, True, False)
    bool_target: pandas.Series = pandas.Series(
        data=np.where(train_data['target'] > 0.5, True, False))
    inv_bool_target: pandas.Series = ~bool_target
    train_id_columns_sum: pandas.Series = train_data[IDENTITY_COLS].sum(axis=1)
    inv_train_id_columns_sum: pandas.Series = (~train_data[IDENTITY_COLS]).sum(
        axis=1)
    # Focus more on the rows that have identity columns.
    sample_weights += train_id_columns_sum
    # Focus more on the false negatives
    sample_weights += (bool_target * inv_train_id_columns_sum)
    # Focus a lot more on the false positives
    sample_weights += ((inv_bool_target * train_id_columns_sum) * 5)
    sample_weights /= sample_weights.mean()

    # Fit the model.
    logger.log('Training model.')
    checkpoint_predictions = []
    weights = []
    for model_idx in range(NUM_MODELS):
        model = build_model(embedding_matrix, y_other_train.shape[-1])
        merge = np.concatenate((x_train, y_train, y_other_train), axis=1)
        np.random.shuffle(merge)
        x_train = merge[:,len(x_train[0])]
        y_train = merge[:len(x_train[0]), len(x_train[0])+1]
        y_other_train = merge[:len(x_train[0])]
        for global_epoch in range(NUM_EPOCHS):
            model.fit(
                x_train, [y_train, y_other_train],
                validation_split=0.1,
                batch_size=BATCH_SIZE,
                epochs=1,
                # One set of sample_weights for each output
                sample_weight=[sample_weights.values,
                               sample_weights.values],
                # TODO(dotslash): How does this help?
                callbacks=[LearningRateScheduler(
                    lambda _: (0.55 ** global_epoch) / 100.0, verbose=1)])
            logger.log('Trained model: {}.'.format(model_idx))
            weights.append(2 ** global_epoch)
            checkpoint_predictions.append(
                model.predict(x_test, batch_size=2048)[0])
    y_test = np.average(checkpoint_predictions, weights=weights, axis=0)
    logger.log('Predicted test set.')
    submission = pandas.DataFrame.from_dict({
        'id': test_data.id,
        'prediction': y_test.flatten()
    })
    submission.to_csv('submission.csv', index=False)
    logger.log('Done.')


if __name__ == '__main__':
    main()
