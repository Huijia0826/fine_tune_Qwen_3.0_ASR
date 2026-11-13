"""Recording script v1 (2026-11-04): sentences for the owner to dictate, aimed at the terms our models get wrong.

Term choice (scripts/mine_term_errors.py, human-verified training data only — no test / val errors):
  A  observed errors in batch 1 (C step 600 drafts) and phase 1 (zero-shot large-v3), e.g. Claude → "cloud", sync →
     "think", LibTV → "lib tv", Palo Alto → "跑道上"; each 3–5 sentences, in different positions and contexts, often next
     to the word it is confused with
  B  English words most Chinese-dictation users say (distinct users in production, 2026-08-31 → 11-04); 2–3 each,
     several per sentence
  C  self-correction and spelled-out letters (rare in production, decided 2026-11-02 to record instead); labels follow
     the annotation policy: raw = verbatim (the correction stays in), spelled letters as spoken
Labels are written in our label style (Chinese punctuation, English terms in standard case, spaces around English).
Output: data/private/recording_script/script_v1.csv (id, block, text, targets) + script_v1.md (to read from).
Usage: python3 scripts/build_recording_script.py
"""

import csv
import re
from pathlib import Path

OUT = Path("data/private/recording_script")

A = [  # (targets, sentence) — observed errors first
    ("Claude", "我刚用 Claude 改了一下这段代码，你帮我看看有没有问题。"),
    ("Claude", "这个任务交给 Claude 做比较好，它写长文档比较稳。"),
    ("Claude cloud", "不是 cloud，是 Claude，就是 Anthropic 家的那个模型。"),
    ("Claude Code", "你装一下 Claude Code，然后在终端里直接让它跑测试。"),
    ("Claude", "昨天 Claude 的额度用完了，我今天先换别的模型试试。"),
    ("LibTV", "我是用 LibTV 做的视频，你看一下效果怎么样。"),
    ("LibTV 可灵", "可灵生成一条要等十几分钟，LibTV 会快一点。"),
    ("LibTV", "你在 LibTV 里面把图片放在第二个镜头，然后加一句字幕。"),
    ("LibTV live", "LibTV 的积分我砍完价之后才十几块钱，比 live 的方案便宜。"),
    ("sync", "这两天有点忙，还没来得及找你 sync，明天上午可以吗？"),
    ("sync", "我们周五下午 sync 一下进度，你把文档先发我。"),
    ("sync think", "我 think 这个问题不大，但还是想跟设计 sync 一下。"),
    ("sync", "日历那边没有 sync 过来，你手动加一下这个会。"),
    ("Palo Alto", "我下午要去 Palo Alto 见一个朋友，可能晚点回消息。"),
    ("Palo Alto Menlo Park", "他们办公室在 Menlo Park，离 Palo Alto 很近。"),
    ("Mountain View", "Google 的总部在 Mountain View，我们周三过去参观。"),
    ("Palo Alto Mountain View", "从 Palo Alto 开到 Mountain View 大概十五分钟。"),
    ("Menlo Park", "Menlo Park 那边的房子太贵了，我们再看看别的地方。"),
    ("Cupertino Sunnyvale San Jose", "Cupertino 和 Sunnyvale 都可以，San Jose 稍微远一点。"),
    ("million", "我们这一轮想融 five million，估值还没定。"),
    ("million", "如果能签下五家这种客户，一年就有 two million 的收入。"),
    ("hundred million", "他们公司估值已经到 one hundred million 了。"),
    ("million minute", "是 million 不是 minute，你再听一下那段录音。"),
    ("AI", "我们这个产品的核心卖点就是 AI 帮你整理语音。"),
    ("AI agent", "现在大家都在做 AI agent，我们得想清楚自己的差异化。"),
    ("AI ToB", "AI ToB 的生意周期很长，销售要提前布局。"),
    ("live", "今晚八点有个 live，你要不要一起来看？"),
    ("live", "他说那个时候会开 live，到时候我把链接发你。"),
    ("Typeless", "你试过 Typeless 吗？它也是做语音输入的。"),
    ("Typeless Wispr Flow", "Typeless 和 Wispr Flow 我都用过，体验差不多。"),
    ("Typeless", "我们跟 Typeless 比，最大的优势是中英混说更准。"),
    ("Voice Cursor", "Voice Cursor 是前端的 macOS app，后端是另一个 repo。"),
    ("Voice Cursor", "用 Voice Cursor 说话比打字快很多，你可以试试。"),
    ("VoiceCursor", "把 VoiceCursor 的安装包发给新同事，让他先装上。"),
    ("Codex", "我让 Codex 帮我把这个函数重构了一遍。"),
    ("Codex ChatGPT", "Codex 和 ChatGPT 是一家的，但是用法不太一样。"),
    ("Codex code", "Codex 生成的 code 还要再人工 review 一遍。"),
    ("HTML", "你直接导出一个 HTML 文件，我在浏览器里打开看看。"),
    ("HTML CSS", "这个页面的 HTML 写得很乱，CSS 也要重新整理。"),
    ("DeepSeek", "DeepSeek 新出的模型推理能力很强，价格也便宜。"),
    ("DeepSeek", "你用 DeepSeek 跑一下这个问题，看看结果和 Claude 有什么区别。"),
    ("Vercel", "前端我们部署在 Vercel 上，域名也在那边配。"),
    ("Vercel Supabase", "Vercel 加 Supabase 这套组合做原型特别快。"),
    ("Supabase", "我没理解 Supabase 里面怎么会没有这条记录。"),
    ("NVIDIA", "我四点约了朋友在 NVIDIA 见面，结束之后再找你。"),
    ("NVIDIA", "NVIDIA 的股价今天又涨了，显卡也更难买了。"),
    ("ToB ToC", "我们先做 ToB 还是先做 ToC，这个下周要定下来。"),
    ("Anthropic", "Anthropic 发了一篇新的博客，讲的是模型安全。"),
    ("Anthropic OpenAI", "Anthropic 和 OpenAI 都在做 coding agent。"),
    ("ChatGPT", "这个问题你先问一下 ChatGPT，看看它怎么说。"),
    ("ChatGPT", "我把 ChatGPT 的回答复制过来了，你帮我改得口语一点。"),
    ("PostHog", "线上的数据都在 PostHog 里面，你按时间筛一下。"),
    ("PostHog Langfuse", "先在 PostHog 找到 trace id，再去 Langfuse 看 prompt。"),
    ("context", "你把 context 给它讲清楚，它就不会答偏了。"),
    ("context contact", "我说的是 context，上下文的意思，不是 contact。"),
    ("calendar", "你看一下我的 calendar，下周二下午有没有空。"),
    ("calendar", "把这个会加到团队的 calendar 里，提醒大家准时参加。"),
    ("decision", "这个 decision 要老板来做，我们先把方案列出来。"),
    ("Whisper", "我们是在 Whisper 的基础上做的微调，不是从头训练。"),
    ("Whisper 微信", "Whisper 是语音识别模型，跟微信没有关系。"),
    ("iPhone", "我的 iPhone 升级之后，语音输入就不太灵了。"),
    ("iPhone iPad", "iPhone 和 iPad 上的界面要分别适配。"),
    ("PPT", "明天开会之前，你把 PPT 再改一版发给我。"),
    ("PPT PDF", "PPT 先别发，转成 PDF 再给客户。"),
    ("square feet", "这个房子大概两千 square feet，带一个小院子。"),
    ("full time", "他现在是 full time 在做这个项目吗？"),
    ("real time", "我们要做的是 real time 的转写，延迟不能超过一秒。"),
    ("AutoPush", "这个模型是在 AutoPush 上单独部署的一个端口。"),
    ("Seedance", "我想写一个 prompt 给 Seedance，生成一段十秒的视频。"),
    ("ElevenLabs", "配音我们用 ElevenLabs，声音比较自然。"),
    ("Gemini", "Gemini 的长上下文很好用，一次能读完整本书。"),
    ("GitHub", "你把代码推到 GitHub 上，我晚上 review。"),
    ("GitHub PR", "GitHub 上那个 PR 我已经 approve 了，你可以合并了。"),
    ("Grok", "Grok 是 xAI 做的模型，你可以对比一下。"),
    ("Mobbin", "用 Mobbin 找一下别人的设计参考，再出我们的稿子。"),
    ("Notion", "会议纪要我都放在 Notion 里了，你直接看就行。"),
    ("Kimi 豆包", "Kimi 和豆包我都试了，中文写作豆包更顺一点。"),
    ("Menlo Park NVIDIA", "下周我先去 Menlo Park，再去 NVIDIA 那边开会。"),
]

B = [  # high-frequency words of Chinese-dictation users, several per sentence
    ("app agent", "这个 app 里面内置了一个 agent，可以帮你自动回邮件。"),
    ("cursor API", "把 cursor 放在这里，然后调一下 API 看看返回什么。"),
    ("case bug", "这个 case 复现不了，你把 bug 的截图发我。"),
    ("Mac UI", "Mac 上的 UI 和 Windows 上不一样，要分开测。"),
    ("user email", "每个 user 注册的时候都要验证 email。"),
    ("token session", "token 过期以后 session 就断了，要重新登录。"),
    ("model research", "这个 model 的效果我们做了一些 research，结论下周给你。"),
    ("prompt Gmail", "你写一个 prompt，让它帮我整理 Gmail 里的未读邮件。"),
    ("PDF link", "我把 PDF 的 link 发在群里了，你下载一下。"),
    ("iOS review", "iOS 的新版本还在 review，大概要等两天。"),
    ("OpenAI key", "OpenAI 的 key 不要写在代码里，放到环境变量。"),
    ("memory profile", "它有 memory 功能，会记住你的 profile 和偏好。"),
    ("PR feedback", "PR 我提了，有什么 feedback 直接在上面评论。"),
    ("demo design", "周四给客户做 demo，design 那边今天要出图。"),
    ("YouTube TikTok", "这条视频先发 YouTube，再剪一个短版发 TikTok。"),
    ("Meta Instagram", "Meta 的广告后台可以同时投 Instagram。"),
    ("feature project", "这个 feature 先放到下一个 project 里做。"),
    ("workflow onboarding", "新用户的 onboarding workflow 要再简化一下。"),
    ("dictation evaluation", "我们在做 dictation 的 evaluation，看看哪个模型更准。"),
    ("pipeline commit", "pipeline 挂了，是上一个 commit 引入的问题。"),
    ("Slack Linear", "需求先记在 Linear 里，讨论放在 Slack。"),
    ("Excel Chrome", "那个 Excel 在 Chrome 里打不开，你换个浏览器试试。"),
    ("LinkedIn Reddit", "招聘信息发 LinkedIn，用户反馈多看 Reddit。"),
    ("Opus macOS", "Opus 模型在 macOS 的客户端里可以直接选。"),
    ("CEO HR offer", "CEO 已经同意了，HR 今天会把 offer 发给你。"),
    ("meeting deadline", "这个 meeting 主要是对 deadline，十分钟就够。"),
    ("Google Docs", "你把稿子放到 Google Docs，我们一起改。"),
    ("token cost", "这个月 token 的 cost 超预算了，要控制一下调用量。"),
    ("data dashboard", "data 都接进来了，dashboard 明天可以看到。"),
    ("context window", "这个模型的 context window 有一百万 token。"),
    ("plan review", "先出一个 plan，大家 review 完再开始做。"),
    ("search feature", "search 这个 feature 用户反馈最多。"),
    ("Windows macOS", "Windows 版本比 macOS 版本晚两周上线。"),
    ("logo brand", "logo 的颜色要跟 brand 手册保持一致。"),
    ("skill agent", "给 agent 加一个 skill，让它能直接查日程。"),
    ("follow up email", "开完会记得 follow up，发一封 email 总结一下。"),
    ("check link", "你 check 一下这个 link 还能不能打开。"),
    ("copy text", "把这段 text copy 到文档里，格式不用管。"),
    ("debug log", "debug 的时候先看 log，别急着改代码。"),
    ("deploy server", "今晚十点 deploy 到 server，大家留意一下。"),
]

C = [  # self-correction / spelled-out letters (verbatim labels)
    ("self_correction", "我们周三开会，不对，周四开会，下午三点。"),
    ("self_correction", "把这个发给小王，哦不是，发给小李。"),
    ("self_correction", "预算是五万，我是说十五万，你别写错了。"),
    ("self_correction Claude", "这个用 ChatGPT，算了，还是用 Claude 吧。"),
    ("self_correction", "明天早上九点，不对不对，九点半出发。"),
    ("self_correction", "地址写成三号楼，我的意思是三号门。"),
    ("self_correction sync", "我们下午 call 一下，不是，下午 sync 一下就行。"),
    ("spelling Vercel", "把部署平台换成 Vercel，V-E-R-C-E-L。"),
    ("spelling LibTV", "那个工具叫 LibTV，L-I-B，TV。"),
    ("spelling Mobbin", "设计网站叫 Mobbin，M-O-B-B-I-N。"),
    ("spelling PostHog", "你搜一下 PostHog，P-O-S-T，H-O-G。"),
    ("spelling Supabase", "数据库用 Supabase，S-U-P-A，base。"),
    ("spelling Typeless", "竞品叫 Typeless，T-Y-P-E，less。"),
    ("spelling", "我的邮箱是 lisa，L-I-S-A，at gmail dot com。"),
]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for block, items in (("A_observed", A), ("B_frequent", B), ("C_correction_spelling", C)):
        for targets, text in items:
            rows.append({"id": f"R{len(rows) + 1:03d}", "block": block, "text": text, "targets": targets})
    with open(OUT / "script_v1.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["id", "block", "text", "targets"])
        w.writeheader()
        w.writerows(rows)
    md = ["# 录制脚本 v1（{} 句）".format(len(rows)), "",
          "打开一个空白备忘录，用 VoiceCursor 口述，一次读一句（说完一整句再结束这次口述）。读错了就再读一遍，我按文字自动对齐，多读的会去掉。",
          "语速、语气按平时说话来，不用字正腔圆；英文词按你平时的发音读。可以分几次录，换着用 MacBook 麦克风、手机、耳机更好。", ""]
    titles = {"A_observed": "A. 我们的模型真实听错过的词（优先）", "B_frequent": "B. 中文用户最常说的英文词",
              "C_correction_spelling": "C. 改口和拼字母（照着读，改口的部分也要读出来）"}
    for block in titles:
        md += ["", f"## {titles[block]}", ""]
        md += [f"{r['id']}　{r['text']}" for r in rows if r["block"] == block]
    (OUT / "script_v1.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    terms = {t for r in rows for t in r["targets"].split() if re.search(r"[A-Za-z]", t)}
    chars = sum(len(re.sub(r"\s", "", r["text"])) for r in rows)
    print(f"{len(rows)} sentences (A {len(A)}, B {len(B)}, C {len(C)}), {len(terms)} target terms, "
          f"≈ {chars / 4.5 / 60:.0f} min of speech at 4.5 chars/s")


if __name__ == "__main__":
    main()
