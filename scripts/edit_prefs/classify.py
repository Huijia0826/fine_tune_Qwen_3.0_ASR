"""Rule-based taxonomy of VoiceCursor edit instructions + text language detection (no LLM calls).

Categories (primary label chosen by PRIORITY; `labels()` returns all matches):
  reusable style preferences
    translate        translate / change the output language ("翻译成英文", "Make it Chinese", "Traduz pro inglês")
    concise          shorten / concise / summarize ("精简一点", "make it shorter", "resumir")
    formal           formal / professional / polite ("更正式", "more professional", "deixa formal")
    casual           casual / friendly / natural ("口语一点", "make it chill", "mais casual")
    tone_other       other tone/voice: emoji, cute, sweet, funny, sincere, persona voices
    expand           expand / elaborate / more detail
    structure        list / bullets / paragraphs / line breaks / email / headings / organize into a format
    punctuation      generic punctuation changes ("去掉句号", "no periods", "句号改成感叹号")
    grammar_fix      fix grammar / typos / cleanup fillers ("Fix the typos", "conserta a frase", "去掉语气词")
    polish           generic rewrite / polish / flow / clarity ("润色", "make this flow better", "melhore")
  one-off content edits
    casing_quotes    capitalize / uppercase / quotes / parentheses / bold on a span
    content_replace  replace / correct a specific word, name, number, date ("把周三改成周四", "it's X not Y", spelled names)
    delete_span      delete a specific part ("删除", "delete this line")
    dictated_text    instruction is just replacement text / re-reading (no edit verb)
    non_edit         questions, chat, external actions, unrelated speech
    empty            no instruction text logged
"""
from __future__ import annotations

import re

STYLE = ["translate", "concise", "formal", "casual", "tone_other", "expand", "structure", "punctuation",
         "grammar_fix", "polish"]
ONE_OFF = ["casing_quotes", "content_replace", "delete_span", "dictated_text", "non_edit", "empty"]
PRIORITY = ["translate", "concise", "formal", "casual", "tone_other", "expand", "structure", "punctuation",
            "grammar_fix", "polish", "casing_quotes", "content_replace", "delete_span"]

LANG_WORDS = (r"english|inglês|ingles|inglés|chinese|mandarin|chinês|spanish|espanhol|español|espanol|portuguese|"
              r"português|portugues|japanese|japonês|korean|coreano|french|francês|francés|german|alemão|alemán|"
              r"italian|italiano|russian|russo|arabic|árabe|hindi|vietnamese|thai|turkish|indonesian")
ZH_LANG = r"(中文|英文|英语|日语|日文|韩语|韩文|西班牙语|西语|法语|德语|葡萄牙语|葡语|俄语|粤语|繁体|简体|意大利语|泰语|越南语|阿拉伯语)"

P = {
    "translate": [
        rf"\btranslat\w*|\btradu[zçcs]\w*|翻译|翻譯|翻成|译成|\bin ({LANG_WORDS})\b|\b(to|into|para|pro|pra|en|al) "
        rf"(o |el |)({LANG_WORDS})\b|^\W*({LANG_WORDS})\W*$|make (it|this) ({LANG_WORDS})|"
        rf"(改|换|写|变|转)(成|为)?{ZH_LANG}|用{ZH_LANG}|^\W*{ZH_LANG}\W*$|{ZH_LANG}(版|翻)"],
    "concise": [
        r"concis\w*|shorte\w*|shorter|\bbrevity\b|\bbrief(er|ly)?\b|summar\w*|condens\w*|trim (it|this|down)|cut (it )?down|"
        r"less wordy|fewer words|tighten\w*|tl;?dr|fit (in|into) a tweet|too long|"
        r"简洁|精简|精炼|提炼|简短|简练|简单(一点|一些|点|些|干净)|短一[点些]|短点|缩短|压缩|总结|概括|删减|言简意赅|啰嗦|囉嗦|太长|"
        r"resum\w*|encurt\w*|mais curt\w*|mais curt|diminu\w*|enxug\w*|sintetiz\w*|mais (diret|objetiv)\w*|\d+ (to \d+ )?(words|characters|caracteres)|under \d+|"
        r"más corto|acort\w*|redundân|redundan"],
    "formal": [
        r"formal(?!at)|professional\w*|profession\w*|polite\w*|courteous|respectful|business (tone|like)|"
        r"corporate|executive|diplomatic|"
        r"正式|专业|專業|礼貌|禮貌|客气|客氣|商务|商務|得体|书面|書面|圆润|委婉|"
        r"profissional|educad\w*|formal|executiv\w*|cordial|polid\w*|respeitos\w*|deletrea|"
        r"деловой|формальн|вежлив|professionnel"],
    "casual": [
        r"casual\w*|informal\w*|friendl\w*|chill\b|relaxed|conversational|natural(ly)?\b|native\b|human(-| )?(readable|sounding|like)|"
        r"less formal|less robotic|laid[- ]back|"
        r"口语化|口语一|随意|轻松|随和|亲切|自然一|接地气|"
        r"descontra\w*|informal|mais leve|natural|coloquial|amigável|amigavel"],
    "tone_other": [
        r"emoji\w*|emoticon|funny|funnier|humor\w*|sweet\w*|lovely|romantic\w*|cute\w*|flirt\w*|sincere\w*|"
        r"warm(er)?\b|enthusias\w*|excit\w*|energetic|sarcas\w*|savage|rude|angry|aggressive|confident|empath\w*|"
        r"apologetic|persuasive|gentle|sadder|happier|seriou\w*|soft(er)?\b|tone\b|sound like|voice of|in the style|"
        r"表情|可爱|可愛|幽默|搞笑|温柔|溫柔|甜|浪漫|热情|語氣|语气(?!词)|轻盈|俏皮|真诚|霸气|高情商|情商|"
        r"engraçad\w*|carinhos\w*|fof\w*|romântic\w*|romantic\w*|amoros\w*|emoç|voz de|\btom\b|"
        r"educada e carinhosa|sedutor\w*|irônic\w*|ironic\w*"],
    "expand": [
        r"expand\w*|elaborat\w*|longer|more detail\w*|in detail|add (more|some) (detail|context|content)|flesh out|"
        r"扩写|擴寫|展开|详细|詳細|丰富|充实|多写|加长|补充一些|"
        r"expand\w*|detalh\w*|mais long\w*|aument\w*|desenvolv\w*|elabor\w*"],
    "structure": [
        r"bullet\w*|\blist\b|listing|numbered|paragraph\w*|line break|new line|separate lines|each line|"
        r"\bformat\w*|email|e-mail|headings?\b|header|(?<!title )\btitle\b(?! case)|bold\b|table\b|outline|structur\w*|"
        r"organi[sz]e (it|this|the text|into)|put (it|this) in (to )?(a )?(list|points)|steps\b|template|prompt\b|"
        r"列表|清单|清單|分点|分點|分条|分條|分段|换行|換行|段落|格式|排版|邮件|郵件|标题|標題|加粗|表格|大纲|"
        r"bullet points|条理|整理成|结构|提示词|"
        r"lista\b|tópic\w*|topic\w*|parágraf\w*|paragraf\w*|linha|formata\w*|negrito|título|titulo|cabeçalho|"
        r"organiz\w*|estrutur\w*|separ\w* em|em tópicos|abnt|citação"],
    "punctuation": [
        r"punctuat\w*|periods?\b|full stops?|commas?\b|question marks?|exclamation|semicolon|colon\b|dash(es)?\b|"
        r"标点|標點|句号|句號|逗号|逗號|问号|問號|感叹号|感嘆號|顿号|冒号|引号(?!里)|"
        r"pontua\w*|ponto final|vírgula|virgula|interroga\w*|exclama\w*|signos? de"],
    "grammar_fix": [
        r"grammar\w*|gramm\w*|typos?\b|fix (the )?spelling|spell ?check|spelling (mistake|error)s?|misspell\w*|fix (it|this|the sentence|the text|everything)|"
        r"correct (it|this|the|any|errors|mistakes)|clean (it|this|that)? ?up|proofread|filler\w*|"
        r"语法|語法|错别字|錯別字|错字|纠正|糾正|改正|病句|语气词|語氣詞|口语词|口头禅|口頭禪|多余的字|废话|重复的|嗯啊|"
        r"gramátic\w*|gramatic\w*|ortogr\w*|corri[gj]\w*|consert\w*|arrum\w*|erros?\b|correç\w*|correcc\w*|"
        r"исправ|ошибк"],
    "polish": [
        r"rewrite|re-write|easy to read|simpler|simple\b|fluent\w*|rephras\w*|reword\w*|polish\w*|improv\w*|better\b|flow\w*|clear(er|ly)?\b|clarity|"
        r"easier to read|reads? (better|well|easier)|readab\w*|makes? sense|refine|enhance|tidy|smooth\w*|"
        r"润色|潤色|改写|改寫|重写|重寫|优化|優化|通顺|通順|流畅|流暢|逻辑|邏輯|改好|改得更好|好一点|好一些|修饰|"
        r"melhor\w*|reescrev\w*|reformul\w*|aprimor\w*|refin\w*|ajust\w*|adequ\w*|revis\w*|"
        r"mejor\w*|reescrib\w*|réécri|verbessern|улучш|перепиш"],
    "casing_quotes": [
        r"title case|capitali[sz]\w*|uppercase|upper case|lowercase|lower case|all caps|caps\b|quot\w*|parenthes\w*|brackets?|"
        r"大写|大寫|小写|小寫|首字母|引号|引號|括号|括號|"
        r"maiúscul\w*|maiuscul\w*|minúscul\w*|minuscul\w*|aspas|parênte\w*|mayúscul\w*|comillas"],
    "content_replace": [
        r"\b(change|replace|substitut\w*|switch|swap|instead of|should (be|say|read)|spell\w*|it'?s not|not .{1,40},? it'?s|"
        r"add (the word|a word|\")|insert|put .{1,40} (before|after|between|at the end|at the beginning)|"
        r"make (it|that) (say|read)|rename|update (it|the)|move .{1,30} to)\b|"
        r"改成|改为|改為|换成|換成|替换|替換|不是.{0,20}是|应该是|應該是|写成|寫成|加上|加一个|添加|插入|"
        r"troc\w*|troqu\w*|substitu\w*|mud\w* (para|o|a)|não é .{1,40} é|coloc\w*|acrescent\w*|adicion\w*|bot\w* |"
        r"cambia\w*|reemplaz\w*|quita\w*|pon\b"],
    "delete_span": [
        r"\b(delete|remove|erase|get rid of|take out|drop)\b|删|刪|去掉|去除|拿掉|"
        r"\b(apag\w*|tir\w*|retir\w*|delet\w*|remov\w*|exclu\w*|limpa\w*|borra\w*|elimin\w*|cortar)\b"],
}
RX = {k: re.compile("|".join(v), re.I) for k, v in P.items()}

# words that mark an instruction as an edit command (used to tell dictated replacement text from commands)
COMMAND = re.compile(
    r"\b(make|change|fix|rewrite|translate|add|remove|delete|put|replace|format|turn|convert|write|correct|"
    r"edit|can you|please|could you)\b|帮我|幫我|把|改|让|讓|弄|写|請|请|"
    r"\b(deixa|deixe|faz|faça|coloca|coloque|escreve|escreva|muda|mude|troca|corrige|melhora|melhore|"
    r"transforma|traduz|tira|põe|pode)\b", re.I)
QUESTION = re.compile(r"\?$|？$|^(what|how|why|who|where|when|is|are|do|does|can you tell)\b|"
                      r"吗[？?]?$|什么|怎么样|为什么|多少钱|是什么", re.I)


def labels(instr: str | None) -> list[str]:
    t = (instr or "").strip()
    return [k for k in PRIORITY if RX[k].search(t)] if t else []


def primary(instr: str | None, intent_category: str | None = None) -> str:
    t = (instr or "").strip()
    if not re.search(r"\w", t):
        return "empty"
    labs = labels(t)
    # "去掉句号" is punctuation, not delete; "去掉语气词" is grammar_fix
    if labs:
        return labs[0]
    if intent_category in ("unrelated_speech", "external_action") or (QUESTION.search(t) and not COMMAND.search(t)):
        return "non_edit"
    if not COMMAND.search(t):
        return "dictated_text"  # replacement text / readback: one-off content
    return "non_edit" if intent_category == "unclear" else "dictated_text"


def is_style(cat: str) -> bool:
    return cat in STYLE


# ---------- language ----------
STOP = {
    "en": set("the and is are you to of it that this for with have was what not my me we they can make please "
              "change fix add put translate shorter more less sound like".split()),
    "pt": set("de que não e o a é um uma para com eu você isso mas está tá tô então muito também pra já deixa "
              "coloca coloque faz mais isso essa esse texto frase melhore".split()),
    "es": set("de que no el la es un una para con yo eso pero está y muy también ahora ya bueno hazlo más "
              "texto frase pon quita".split()),
}
LATIN_WORD = re.compile(r"[A-Za-zÀ-ɏ']+")
HAN = re.compile(r"[一-鿿]")
OTHER_SCRIPT = re.compile(r"[Ѐ-ӿ؀-ۿऀ-ॿঀ-৿฀-๿぀-ヿ가-힯]")


def text_lang(text: str | None) -> str | None:
    """zh / zh-en / en / other (pt, es, ru, ...). None if too little text."""
    t = text or ""
    n_han = len(HAN.findall(t))
    words = [w.lower() for w in LATIN_WORD.findall(t)]
    n_other = len(OTHER_SCRIPT.findall(t))
    if n_han:
        if n_other > n_han:
            return "other"
        # zh-en: substantial Latin words inside Chinese text (code-switching)
        return "zh-en" if len(words) >= 2 and len(words) / (n_han / 1.6 + len(words)) >= 0.2 else "zh"
    if n_other >= 2:
        return "other"
    if not words:
        return None
    sc = {k: sum(w in v for w in words) for k, v in STOP.items()}
    accents = bool(re.search(r"[À-ɏ]", t))
    if sc["en"] >= max(sc["pt"], sc["es"]) and (sc["en"] > 0 or not accents):
        return "en"
    return "other"
