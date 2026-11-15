# Test sentences

Generic sentences written for this test only. They are NOT Nicholas's private study notes.

The listening comparison uses the same four sentences for every model: s01, s08, s10, s11.

## Comparison set (same for every model)

| ID | Covers | Sentence |
|----|--------|----------|
| s01 | polyphone 长 | 这条路很长，走路要花很长时间，他长大以后就明白了。 |
| s08 | tone sandhi 不, 一, third tone | 我不想去，你也不想去，我们都想好好休息一下。 |
| s10 | erhua | 下午我们去小院儿里坐坐，那儿很好玩儿，还有花儿和鸟儿。 |
| s11 | date, time, decimal | 会议定在2026年11月15日下午3点半，费用是12.5元，请提前十分钟到。 |

## Full set (14 sentences)

| ID | Covers | Sentence |
|----|--------|----------|
| s01 | polyphone 长 | 这条路很长，走路要花很长时间，他长大以后就明白了。 |
| s02 | polyphone 行, 还 | 银行今天很忙，行长说这件事还行，让我们先试一试。 |
| s03 | polyphone 重 | 这份工作很重，他重新安排了时间，重点是把事情做好。 |
| s04 | polyphone 乐 | 音乐让他很快乐，他最喜欢在周末听音乐。 |
| s05 | polyphone 得, 着 | 他觉得这个办法还得再想想，门还开着，灯也亮着。 |
| s06 | polyphone 差 | 这件事差一点就成功了，他说差不多就行了，别差太多。 |
| s07 | surname 单, 曾 | 单老师说，曾国藩的故事很有意思，请大家先读一读。 |
| s08 | tone sandhi 不, 一, third tone | 我不想去，你也不想去，我们都想好好休息一下。 |
| s09 | third-tone run | 你好，我也很好，雨水很少，我们一起去买水果。 |
| s10 | erhua | 下午我们去小院儿里坐坐，那儿很好玩儿，还有花儿和鸟儿。 |
| s11 | date, time, decimal | 会议定在2026年11月15日下午3点半，费用是12.5元，请提前十分钟到。 |
| s12 | decimal, money | 费用是12.5元，一共三杯咖啡，每杯四块五。 |
| s13 | identifier, number string | 订单号是202611150123，电话后四位是7522，请核对一下。 |
| s14 | proper name, place | 小明和小红去北京大学听讲座，回来的时候路过天安门广场。 |

## Pinyin-fix variants

`test_sentences.json` carries a `pinyin_fix_indextts` variant for each polyphone sentence, using IndexTTS inline format such as `中(zhong1)国(guo2)`. CosyVoice 3 also supports a pinyin override (pronunciation inpainting), but the exact input format was not verified, so no CosyVoice 3 variant is written here yet. Verify it in the CosyVoice 3 docs before use, do not guess it.

## How to listen

Audio clips live in `audio/<model>/<id>.wav`. Compare the same sentence ID across models, and judge by ear: correct readings first, then naturalness. No WER or CER score is used, because those scores do not catch polyphone errors.
