# encoding: UTF-8
# code from
# - https://github.com/naver/ai-hackathon-2018
# - hangul_utils preprocess.py

import itertools

INITIAL = 0x001
MEDIAL = 0x010
FINAL = 0x100
CHAR_LISTS = {
    # 초성 19 "ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ"
    INITIAL: list(map(chr, [
        0x3131, 0x3132, 0x3134, 0x3137, 0x3138, 0x3139,
        0x3141, 0x3142, 0x3143, 0x3145, 0x3146, 0x3147,
        0x3148, 0x3149, 0x314a, 0x314b, 0x314c, 0x314d,
        0x314e
    ])),
    # 중성 21 "ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ"
    MEDIAL: list(map(chr, [
        0x314f, 0x3150, 0x3151, 0x3152, 0x3153, 0x3154,
        0x3155, 0x3156, 0x3157, 0x3158, 0x3159, 0x315a,
        0x315b, 0x315c, 0x315d, 0x315e, 0x315f, 0x3160,
        0x3161, 0x3162, 0x3163
    ])),
    # 종성 28(첫번째는 종성이 없는 경우) "-ㄱㄲㄳㄴㄵㄶㄷㄹㄺㄻㄼㄽㄾㄿㅀㅁㅂㅄㅅㅆㅇㅈㅊㅋㅌㅍㅎ"
    FINAL: list(map(chr, [
        0x3131, 0x3132, 0x3133, 0x3134, 0x3135, 0x3136,
        0x3137, 0x3139, 0x313a, 0x313b, 0x313c, 0x313d,
        0x313e, 0x313f, 0x3140, 0x3141, 0x3142, 0x3144,
        0x3145, 0x3146, 0x3147, 0x3148, 0x314a, 0x314b,
        0x314c, 0x314d, 0x314e
    ]))
}
CHAR_INITIALS = CHAR_LISTS[INITIAL]
CHAR_MEDIALS = CHAR_LISTS[MEDIAL]
CHAR_FINALS = CHAR_LISTS[FINAL]
CHAR_SETS = {k: set(v) for k, v in CHAR_LISTS.items()}
CHARSET = set(itertools.chain(*CHAR_SETS.values()))
CHAR_INDICES = {k: {c: i for i, c in enumerate(v)}
                for k, v in CHAR_LISTS.items()}


def is_hangul_syllable(c):
    # 가-힣
    # 한글 음절
    return 0xac00 <= ord(c) <= 0xd7a3  # Hangul Syllables


def is_hangul_jamo(c):
    # 한글자모(초,중,종성 분리) : ᄀ-ᇿ
    return 0x1100 <= ord(c) <= 0x11ff  # Hangul Jamo


def is_hangul_compat_jamo(c):
    # 한글자모(초,중,종성 미구분) : 
    return 0x3130 <= ord(c) <= 0x318f  # Hangul Compatibility Jamo


def is_hangul_jamo_exta(c):
    # 한글 확장 자모(고어)
    return 0xa960 <= ord(c) <= 0xa97f  # Hangul Jamo Extended-A


def is_hangul_jamo_extb(c):
    # 한글 확장 자모(고어)
    return 0xd7b0 <= ord(c) <= 0xd7ff  # Hangul Jamo Extended-B


def is_hangul(c):
    # 고어 포함 한글인지 확인
    return (is_hangul_syllable(c) or
            is_hangul_jamo(c) or
            is_hangul_compat_jamo(c) or
            is_hangul_jamo_exta(c) or
            is_hangul_jamo_extb(c))


def is_supported_hangul(c):
    # 현대 한글 여부 확인
    return is_hangul_syllable(c) or is_hangul_compat_jamo(c)


def check_hangul(c, jamo_only=False):
    if not ((jamo_only or is_hangul_compat_jamo(c)) or is_supported_hangul(c)):
        raise ValueError(f"'{c}' is not a supported hangul character. "
                         f"'Hangul Syllables' (0xac00 ~ 0xd7a3) and "
                         f"'Hangul Compatibility Jamos' (0x3130 ~ 0x318f) are "
                         f"supported at the moment.")


def get_jamo_type(c):
    # 한글 자모(미구분)인지 확인하고
    # 초성 중성 종성의 포함확인
    # ex) ㄱ : 초성 -> 1
    # ex) 가 : 초성 + 중성 -> 17
    # ex) 감 : 초성 + 중성 + 종성 -> 273
    check_hangul(c)
    assert is_hangul_compat_jamo(c), f"not a jamo: {ord(c):x}"
    return sum(t for t, s in CHAR_SETS.items() if c in s)


def split_syllable_char(c):
    """
    Splits a given korean syllable into its components. Each component is
    represented by Unicode in 'Hangul Compatibility Jamo' range.

    Arguments:
        c: A Korean character.

    Returns:
        A triple (initial, medial, final) of Hangul Compatibility Jamos.
        If no jamo corresponds to a position, `None` is returned there.

    Example:
        >>> split_syllable_char("안")
        ("ㅇ", "ㅏ", "ㄴ")
        >>> split_syllable_char("고")
        ("ㄱ", "ㅗ", None)
        >>> split_syllable_char("ㅗ")
        (None, "ㅗ", None)
        >>> split_syllable_char("ㅇ")
        ("ㅇ", None, None)
    """
    check_hangul(c)
    if len(c) != 1:
        raise ValueError("Input string must have exactly one character.")

    init, med, final = None, None, None
    if is_hangul_syllable(c):
        offset = ord(c) - 0xac00
        x = (offset - offset % 28) // 28
        init, med, final = x // 21, x % 21, offset % 28
        if not final:
            final = None
        else:
            final -= 1
    else:
        pos = get_jamo_type(c)
        # 초성, 중성, 종성 여부 확인 후 해당 인덱스를 출력
        if pos & INITIAL == INITIAL:
            pos = INITIAL
        elif pos & MEDIAL == MEDIAL:
            pos = MEDIAL
        elif pos & FINAL == FINAL:
            pos = FINAL
        
        idx = CHAR_INDICES[pos][c]
        if pos == INITIAL:
            init = idx
        elif pos == MEDIAL:
            med = idx
        else:
            final = idx
    return tuple(CHAR_LISTS[pos][idx] if idx is not None else "-" # 종성 공백 문자 
                 for pos, idx in
                 zip([INITIAL, MEDIAL, FINAL], [init, med, final]))


def split_syllables(s, ignore_err=True, pad=None):
    """
    Performs syllable-split on a string.

    Arguments:
        s (str): A string (possibly mixed with non-Hangul characters).
        ignore_err (bool): If set False, it ensures that all characters in
            the string are Hangul-splittable and throws a ValueError otherwise.
            (default: True)
        pad (str): Pad empty jamo positions (initial, medial, or final) with
            `pad` character. This is useful for cases where fixed-length
            strings are needed. (default: None)

    Returns:
        Hangul-split string

    Example:
        >>> split_syllables("안녕하세요")
        "ㅇㅏㄴㄴㅕㅇㅎㅏㅅㅔㅇㅛ"
        >>> split_syllables("안녕하세요~~", ignore_err=False)
        ValueError: encountered an unsupported character: ~ (0x7e)
        >>> split_syllables("안녕하세요ㅛ", pad="x")
        'ㅇㅏㄴㄴㅕㅇㅎㅏxㅅㅔxㅇㅛxxㅛx'
    """

    def try_split(c):
        try:
            return split_syllable_char(c)
        except ValueError:
            if ignore_err:
                return (c, )
            raise ValueError(f"encountered an unsupported character: "
                             f"{c} (0x{ord(c):x})")

    s = map(try_split, s)
    if pad is not None:
        tuples = map(lambda x: tuple(pad if y is None else y for y in x), s)
    else:
        tuples = map(lambda x: filter(None, x), s)
    return "".join(itertools.chain(*tuples))


def join_jamos_char(init, med, final=None):
    """
    Combines jamos into a single syllable.

    Arguments:
        init (str): Initial jao.
        med (str): Medial jamo.
        final (str): Final jamo. If not supplied, the final syllable is made
            without the final. (default: None)

    Returns:
        A Korean syllable.
    """
    chars = (init, med, final)
    for c in filter(None, chars):
        check_hangul(c, jamo_only=True)

    idx = tuple(CHAR_INDICES[pos][c] if c is not None else c
                for pos, c in zip((INITIAL, MEDIAL, FINAL), chars))
    init_idx, med_idx, final_idx = idx
    # final index must be shifted once as
    # final index with 0 points to syllables without final
    final_idx = 0 if final_idx is None else final_idx + 1
    return chr(0xac00 + 28 * 21 * init_idx + 28 * med_idx + final_idx)


def join_jamos(s, ignore_err=True):
    """
    Combines a sequence of jamos to produce a sequence of syllables.

    Arguments:
        s (str): A string (possible mixed with non-jamo characters).
        ignore_err (bool): If set False, it will ensure that all characters
            will be consumed for the making of syllables. It will throw a
            ValueError when it fails to do so. (default: True)

    Returns:
        A string

    Example:
        >>> join_jamos("ㅇㅏㄴㄴㅕㅇㅎㅏㅅㅔㅇㅛ")
        "안녕하세요"
        >>> join_jamos("ㅇㅏㄴㄴㄴㅕㅇㅎㅏㅅㅔㅇㅛ")
        "안ㄴ녕하세요"
        >>> join_jamos()
    """
    last_t = 0
    queue = []
    new_string = ""

    def flush(n=0):
        new_queue = []
        while len(queue) > n:
            new_queue.append(queue.pop())
        if len(new_queue) == 1:
            if not ignore_err:
                raise ValueError(f"invalid jamo character: {new_queue[0]}")
            result = new_queue[0]
        elif len(new_queue) >= 2:
            try:
                result = join_jamos_char(*new_queue)
            except (ValueError, KeyError):
                # Invalid jamo combination
                if not ignore_err:
                    raise ValueError(f"invalid jamo characters: {new_queue}")
                result = "".join(new_queue)
        else:
            result = None
        return result

    for c in s:
        if c not in CHARSET:
            if queue:
                new_c = flush() + c
            else:
                new_c = c
            last_t = 0
        else:
            t = get_jamo_type(c)
            new_c = None
            if t & FINAL == FINAL:
                if not (last_t == MEDIAL):
                    new_c = flush()
            elif t == INITIAL:
                new_c = flush()
            elif t == MEDIAL:
                if last_t & INITIAL == INITIAL:
                    new_c = flush(1)
                else:
                    new_c = flush()
            last_t = t
            queue.insert(0, c)
        if new_c:
            new_string += new_c
    if queue:
        new_string += flush()
    return new_string.replace("-", "")



def decompose_as_one_hot(in_char, warning=True):
  hangul_length = 67
  one_hot = []
  # print(ord('ㅣ'), chr(0xac00))
  # [0,66]: hangul / [67,194]: ASCII / [195,245]: hangul danja,danmo / [246,247]: special characters
  # Total 248 dimensions.
  if ord('가') <= in_char <= ord('힣'):  # 가:44032 , 힣: 55203
    x = in_char - 44032  # in_char - ord('가')
    y = x // 28
    z = x % 28
    x = y // 21
    y = y % 21
    # if there is jong, then is z > 0. So z starts from 1 index.
    zz = CHAR_FINALS[z - 1] if z > 0 else ''
    if x >= len(CHAR_INITIALS):
      if warning:
        print('Unknown Exception: ', in_char, chr(in_char), x, y, z, zz)

    one_hot.append(x)
    one_hot.append(len(CHAR_INITIALS) + y)
    if z > 0:
      one_hot.append(len(CHAR_INITIALS) + len(CHAR_MEDIALS) + (z - 1))
    else:
      one_hot.append(113) # "-" : 종성 공백문자 1 + 67 + 45

    return one_hot
  # 한글 아니면
  else:
    if in_char < 128:
    # https://whatisthenext.tistory.com/103
    # 유니코드 십진수 인덱스 번호 기준 128 이하에 대해 vocab 인덱스 부여
    # 십진수 인덱스 번호 기준 유니코드 정리
    # 0 ~ 32, 127 : 인쇄 및 전송 제어용 신호문자
    # 33 ~ 47, 58 ~ 64, 91 ~96, 123 ~ 126 : 특수문자
    # 48 ~ 57 : 0~9
    # 65 ~ 90, 97 ~ 122 : 영어 대문자, 소문자
      result = hangul_length + in_char  # 67~
    # 한글 자모(초중종성 구분 없음) 51개 "ㄱ" : 0x3131(12593) ~ "ㅣ" : 0x3163(12643)에 대하여
    # 194번부터 vocab 인덱스 부여 # [ㄱ:12593]~[ㅣ:12643] (len = 51) 
    elif ord('ㄱ') <= in_char <= ord('ㅣ'):
      result = hangul_length + 128 + (in_char - 12593)
    # 말뭉치 상에서 시 구분 문자는 ★, 개행 문자는 ☆로 전처리 (9733, 9734)
    # 246번부터 vocab 인덱스 부여
    elif in_char == ord('★'):
      result = hangul_length + 128 + 51   # ★
    elif in_char == ord('☆'):
      result = hangul_length + 128 + 51 + 1  # ☆
    else:
      if warning:
        print('Unhandled character:', chr(in_char), in_char)
      # unknown character
      result = hangul_length + 128 + 51 + 2  # for unknown character

    return [result]

def decompose_str_as_one_hot(string, warning=True):
  tmp_list = []
  for x in string:
    tmp = decompose_as_one_hot(ord(x), warning=warning)
    # for zero vector
    tmp = [xx + 1 for xx in tmp]
    tmp_list.extend(tmp)
  return tmp_list
