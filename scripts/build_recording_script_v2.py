"""Recording script v2 (2026-11-04): ready-to-read sentences for a NEW speaker, aimed at the problems found in the trial.

The owner asked for sentences to read directly (prompt cards were not usable). To limit the read-speech effect seen in
v1 (reports/lora_large_v3_H.md), the sentences are written as casual dictation (chat messages, notes, quick requests)
and the speaker is asked to say them at a normal pace, not to read them out carefully.
Blocks
  S  spelling        a word followed by its letters, e-mail addresses, codes with letters and digits
  C  self-correction say something, then correct part of it with a marker (不对 / 我是说 / 哦不是 / 改成 / sorry, I mean …)
  E  entities        Chinese products / places / names / homophones, English tools inside Chinese
  G  general         ordinary dictation (control: overall quality must not drop)
Label policy (owner, 2026-11-04): spelled letters as spoken, upper case, joined by "-" ("Vercel，V-E-R-C-E-L"; the
post-processing LLM assembles them); self-corrections verbatim; fillers 嗯/呃/um/uh removed. The sentence text below is
the label draft; the review fixes whatever the speaker actually said.
Split: every sentence marked T is TEST only (never trained): about one in three per block, plus terms that occur only
in test (unseen), so the test measures both "learned these terms" and "generalises to new ones".
Output: data/private/recording_script/script_v2.csv (id, block, split, text, targets) + script_v2.md (to read from).
Usage: python3 scripts/build_recording_script_v2.py
"""

import csv
import re
from pathlib import Path

OUT = Path("data/private/recording_script")

# (split, text) — "T" = test only, "" = train
S = [
    ("", "前端直接部署到 Vercel 就行，V-E-R-C-E-L。"),
    ("", "数据库我们用的是 Supabase，S-U-P-A-B-A-S-E。"),
    ("T", "埋点数据都在 PostHog 里，P-O-S-T-H-O-G。"),
    ("", "trace 去 Langfuse 里看，L-A-N-G-F-U-S-E。"),
    ("", "我说的是 Claude，C-L-A-U-D-E，不是 cloud。"),
    ("", "代码让 Codex 先写一版，C-O-D-E-X。"),
    ("T", "编辑器我用 Cursor，C-U-R-S-O-R。"),
    ("", "视频是在 LibTV 上做的，L-I-B，T-V。"),
    ("", "设计参考可以去 Mobbin 找，M-O-B-B-I-N。"),
    ("T", "竞品叫 Typeless，T-Y-P-E-L-E-S-S。"),
    ("", "这家芯片公司叫 Cerebras，C-E-R-E-B-R-A-S。"),
    ("", "东南亚电商是 Shopee，S-H-O-P-E-E。"),
    ("T", "稿子在 Figma 里，F-I-G-M-A。"),
    ("", "需求先记在 Linear 上，L-I-N-E-A-R。"),
    ("", "那个写代码的 agent 叫 Devin，D-E-V-I-N。"),
    ("T", "搜索可以试试 Perplexity，P-E-R-P-L-E-X-I-T-Y。"),
    ("", "视频模型用的是 Seedance，S-E-E-D-A-N-C-E。"),
    ("", "缓存用 Redis，R-E-D-I-S。"),
    ("T", "本地跑模型用 Ollama，O-L-L-A-M-A。"),
    ("", "这个模型是 Qwen，Q-W-E-N，阿里的。"),
    ("", "公司名是 Anthropic，A-N-T-H-R-O-P-I-C。"),
    ("T", "消息队列用 Kafka，K-A-F-K-A。"),
    # unseen words: test only
    ("T", "我们用 Replit 搭的原型，R-E-P-L-I-T。"),
    ("T", "这个是用 Lovable 生成的页面，L-O-V-A-B-L-E。"),
    ("T", "向量库用 Pinecone，P-I-N-E-C-O-N-E。"),
    ("T", "视频生成也可以用 Runway，R-U-N-W-A-Y。"),
    # e-mail addresses
    ("", "我的邮箱是 L-I-S-A at gmail dot com。"),
    ("", "你发到 K-E-V-I-N dot chen at gmail dot com。"),
    ("T", "抄送一下 A-M-Y 2024 at outlook dot com。"),
    ("", "他的邮箱是 W-A-N-G dot hao at qq dot com。"),
    ("", "简历发到 J-E-S-S-I-C-A at gmail dot com 就行。"),
    ("T", "联系人邮箱是 T-O-N-Y 88 at 163 dot com。"),
    ("", "账号是 C-H-R-I-S dot liu at company dot com。"),
    ("", "发票寄到 G-R-A-C-E 99 at gmail dot com。"),
    # codes with letters and digits
    ("", "订单号是 A-B-3-7-2-9，你帮我查一下。"),
    ("T", "我的工号是 K-0-4-5。"),
    ("", "航班号 C-A-9-8-1，明天早上八点的。"),
    ("", "邀请码是 X-7-Q-2，注册的时候填一下。"),
    ("T", "车牌号是京 A-8-8-6-K-Q。"),
    ("", "报错码是 E-4-0-4，页面打不开。"),
    ("", "会议号 M-T-6-6-0，密码是 1234。"),
    ("T", "房间号是 B-1-2-0-3，在十二楼。"),
    # acronyms spelled
    ("", "这个季度的 KPI，K-P-I，下周要定下来。"),
    ("", "我们先写一个 PRD，P-R-D，再排期。"),
    ("T", "先做一个 MVP，M-V-P，验证一下需求。"),
    ("", "这是我们的 SOP，S-O-P，大家按这个来。"),
]

C = [
    ("", "我们周三下午三点开会，不对，周四上午十点。"),
    ("", "预算大概五万，我是说十五万，你别写错了。"),
    ("T", "把这个发给小王，哦不是，发给小李。"),
    ("", "明天早上九点出发，不对不对，九点半。"),
    ("", "地址写三号楼，我的意思是三号门，门口等我。"),
    ("", "这个用 ChatGPT 做吧，算了，还是用 Claude。"),
    ("T", "我们下午开个会，不是，下午 sync 一下就行。"),
    ("", "一共十二个人，不对，二十个人，订大一点的包间。"),
    ("", "餐厅订在周五晚上，改成周六中午吧。"),
    ("T", "截止日期是下周一，哦不对，是这周五。"),
    ("", "房间号是 1203，我说反了，是 1302。"),
    ("", "手机尾号 6688，不对，是 6866。"),
    ("T", "我们在望京 SOHO 见，还是改到五道口吧。"),
    ("", "我坐早上八点那班飞机，sorry，是中午十二点那班。"),
    ("", "这个卖三百九，哦，是三百九十九。"),
    ("T", "通知发到飞书群里，我是说钉钉群。"),
    ("", "文件看 PRD 第二版，不对，第三版。"),
    ("", "这个先发给设计组，哦不是，发给产品组。"),
    ("T", "我下周去杭州出差，不对，是苏州。"),
    ("", "按钮用蓝色，还是用深灰色吧。"),
    ("", "我们约了三点，等一下，是三点半。"),
    ("T", "这个需求给前端，不对，给后端来做。"),
    ("", "报销金额是八百块，我看错了，是八百五。"),
    ("", "明天先开周会，然后，不对，先跟客户开会。"),
    ("T", "我住在 1806，哦不对，1608。"),
    ("", "这个版本周二上线，我说错了，是周三上线。"),
    ("", "Let's schedule the call on Tuesday, sorry, I mean Wednesday."),
    ("T", "The budget is fifteen thousand, I mean fifty thousand."),
    ("", "Use Cursor for this one, actually, let's use Codex."),
    ("", "Send it to Kevin, scratch that, send it to Chris."),
    ("T", "The deadline is Friday, no wait, Monday."),
    ("", "We're meeting in San Jose, actually, make it Palo Alto."),
]

E = [
    # Chinese office / content / AI products
    ("", "这个通知我已经发到飞书群里了，钉钉那边你再同步一下。"),
    ("", "我们公司现在用企业微信，之前用的是钉钉。"),
    ("T", "会议纪要放在飞书文档里，链接我发你。"),
    ("", "这条视频先发小红书，抖音那边晚点再发。"),
    ("", "我在哔哩哔哩上看到一个讲大模型的视频，挺不错的。"),
    ("T", "小红书的流量比抖音稳定一点。"),
    ("", "豆包和通义千问我都试了，写中文豆包更顺。"),
    ("", "文心一言最近更新了，你可以再试试。"),
    ("T", "通义千问的长文本能力还不错。"),
    ("", "可灵生成一条视频要等十几分钟，即梦会快一点。"),
    ("", "剪辑我一般用剪映，配音用别的工具。"),
    ("T", "即梦的图片质量比上个月好了很多。"),
    ("", "拼多多上买的比闲鱼还便宜。"),
    ("", "这双鞋我是在得物上买的，你可以对比一下价格。"),
    ("T", "闲鱼上有人在卖二手显卡。"),
    ("", "酒店我在携程上订好了，打车就用滴滴。"),
    ("", "你用高德导航过去，大概二十分钟。"),
    ("T", "滴滴叫的车已经到楼下了。"),
    # places
    ("", "我们明天在望京开会，开完去五道口吃饭。"),
    ("", "国贸那边堵车太严重了，我坐地铁过去。"),
    ("T", "五道口有家咖啡店挺安静的。"),
    ("", "公司在陆家嘴，我住在张江，通勤有点远。"),
    ("", "周末去徐家汇逛一下吧。"),
    ("T", "张江那边新开了一个园区。"),
    ("", "他们团队在深圳南山科技园。"),
    ("", "杭州的办公室在西溪，离滨江也不远。"),
    ("T", "下次去杭州可以约在滨江见面。"),
    # Chinese names
    ("", "这个项目让王浩然负责，李思琪配合他。"),
    ("", "陈子涵下周入职，你带一下她。"),
    ("T", "你把这份文档转给李思琪看一下。"),
    # homophones from the trial
    ("", "这个评测集是我们自己从零搭起来的。"),
    ("", "我想看看我自己训的这个模型到底有多好用。"),
    ("T", "评测集里的专有名词还是识别得不太好。"),
    ("", "我们推动了词典上线，用户可以自己加词。"),
    ("", "中文这边我们改了一下路由的方式。"),
    ("T", "这个词典功能对专有名词帮助很大。"),
    ("", "这个 checkpoint 在验证集上效果最好。"),
    ("", "我们先微调一版，看看验证集的结果。"),
    ("T", "微调之后验证集的错误率下降了一点。"),
    # English tools inside Chinese
    ("", "我现在大部分代码都交给 Claude Code 来写。"),
    ("", "Codex 和 Cursor 我都在用，各有各的好。"),
    ("T", "Claude Code 处理长任务比其他工具稳。"),
    ("", "前端部署在 Vercel，数据库用 Supabase。"),
    ("", "本地开发用 Docker 起一个环境就行。"),
    ("T", "Vercel 上的域名已经配好了。"),
    ("", "埋点在 PostHog 里看，prompt 去 Langfuse 里查。"),
    ("", "你先找到 trace id，再看具体的调用。"),
    ("T", "PostHog 的看板我已经建好了。"),
    ("", "Anthropic 和 OpenAI 最近都发了新模型。"),
    ("", "DeepSeek 的推理模型价格很便宜。"),
    ("T", "OpenAI 的新模型下周开放。"),
    ("", "Cerebras 是做 AI 芯片的，推理速度很快。"),
    ("", "Shopee 是东南亚最大的电商平台之一。"),
    ("T", "他之前在 Hudson River Trading 实习过。"),
    ("", "我用 LibTV 做的视频，你看看首帧和尾帧。"),
    ("", "Seedance 生成的视频动作很自然。"),
    ("T", "LibTV 的积分比可灵便宜。"),
    ("", "我下午要去 Palo Alto 见个朋友。"),
    ("", "他们办公室在 Menlo Park，离 Mountain View 很近。"),
    ("T", "Mountain View 那边房租也很贵。"),
    ("", "Typeless 和 Wispr Flow 都是做语音输入的。"),
    ("", "VoiceCursor 的中英混说识别比较准。"),
    ("T", "我同事一直在用 Wispr Flow。"),
    # unseen terms: test only
    ("T", "楼下新开了一家瑞幸，旁边是蜜雪冰城。"),
    ("T", "霸王茶姬最近排队的人好多。"),
    ("T", "这个原型是用 Replit 搭的，页面用 Lovable 生成。"),
    ("T", "我们周末去成都，逛一下春熙路和太古里。"),
    ("T", "重庆解放碑那边晚上人特别多。"),
    ("T", "赵一鸣和孙雨桐下周来北京出差。"),
    ("T", "周梓萱负责这次的发布会。"),
]

G = [
    ("", "今天中午吃了一碗牛肉面，味道还不错。"),
    ("", "领导，我明天上午请半天假，去医院做个检查。"),
    ("T", "这周末打算去爬山，天气好的话就去。"),
    ("", "妈，我已经到家了，你们早点休息。"),
    ("", "早上地铁特别挤，差点没挤上去。"),
    ("T", "感谢您这段时间的支持，我们会继续努力。"),
    ("", "最近在看一部剧，挺好看的，推荐给你。"),
    ("", "记一下，明天要买牛奶、鸡蛋和面包。"),
    ("T", "晚上七点在公司楼下见，一起去吃火锅。"),
    ("", "用户希望可以把报告导出成 PDF。"),
    ("", "Hi, I'm working on the speech recognition project this quarter."),
    ("T", "Could you review my pull request when you have time?"),
    ("", "明天的 meeting 改到下午两点，大家记得参加。"),
    ("", "周末去爬山了，风景特别好。"),
    ("T", "房东你好，热水器好像坏了，麻烦找人来看一下。"),
    ("", "这个问题我晚点再回复你，现在在开会。"),
    ("", "我们下周三之前把方案定下来吧。"),
    ("T", "今天有点累，先下班了，明天再说。"),
    ("", "这个功能用户反馈挺好的，可以继续优化。"),
    ("", "Let me know if you need anything else from me."),
]


def letters_ok(text: str) -> bool:
    """Spelled sequences follow the label policy: upper-case letters / digits joined by '-'."""
    return all(re.fullmatch(r"[A-Z0-9](?:-[A-Z0-9])+", s) for s in re.findall(r"(?<![\w-])[A-Z0-9](?:-[A-Z0-9])+(?![\w-])", text))


def main() -> None:
    rows = []
    for block, items in (("S", S), ("C", C), ("E", E), ("G", G)):
        for split, text in items:
            assert letters_ok(text), text
            rows.append({"id": f"V{len(rows) + 1:03d}", "block": block, "split": "test" if split == "T" else "train",
                         "text": text, "targets": " ".join(re.findall(r"[A-Za-z][\w.]*(?: [A-Z][\w]*)*", text))})
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "script_v2.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["id", "block", "split", "text", "targets"])
        w.writeheader()
        w.writerows(rows)
    titles = {"S": "S. 拼字母（字母一个一个念，「-」不用念）", "C": "C. 改口（改口的部分也照着念出来）",
              "E": "E. 专有名词", "G": "G. 普通口述"}
    md = [f"# 录制脚本 v2（{len(rows)} 句，直接念）", "",
          "- 打开一个空白备忘录，用 VoiceCursor 口述，一次念一句（念完一整句再结束这次口述）。编号不用念。",
          "- **按平时说话的语速和语气念，像发语音消息一样，不用字正腔圆。** 念错了就再念一遍，不用删。",
          "- 拼字母：字母一个一个念出来，「-」不用念；dot 念 dot，at 念 at。",
          "- 可以分几次录，每次 20–30 分钟；能换着用电脑麦克风、手机、耳机更好。",
          "- 录完把账号的 user id 和大概的录制时间段告诉我。", ""]
    for b, t in titles.items():
        md += ["", f"## {t}", ""]
        md += [f"{r['id']}　{r['text']}" for r in rows if r["block"] == b]
    (OUT / "script_v2.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    n = {b: sum(r["block"] == b for r in rows) for b in titles}
    t = sum(r["split"] == "test" for r in rows)
    chars = sum(len(re.sub(r"\s", "", r["text"])) for r in rows)
    print(f"{len(rows)} sentences {n}; test {t} ({t / len(rows):.0%}); ≈ {chars / 4.5 / 60:.0f} min of speech")


if __name__ == "__main__":
    main()
