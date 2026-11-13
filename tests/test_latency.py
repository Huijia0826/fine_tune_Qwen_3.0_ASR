"""Exercise 11: summarize_latency buckets clips by audio length and reports n / p50 / p95 / mean / RTF."""

import pytest

from whisperflow.eval.latency import bucket_of, summarize_latency


def test_bucket_boundaries():
    assert bucket_of(4.99) == "<5 s"
    assert bucket_of(5.0) == "5-15 s"
    assert bucket_of(15.0) == "15-30 s"
    assert bucket_of(30.0) == "15-30 s"  # still one encoder window = short-form
    assert bucket_of(30.01) == ">30 s"


def test_percentiles_and_rtf():
    recs = [{"duration_s": 2.0, "latency_s": x} for x in (0.1, 0.2, 0.3, 0.4, 0.5)]
    s = summarize_latency(recs)["<5 s"]
    assert s["n"] == 5
    assert s["p50_s"] == pytest.approx(0.3)
    assert s["p95_s"] == pytest.approx(0.48)  # linear interpolation between 0.4 and 0.5
    assert s["mean_s"] == pytest.approx(0.3)
    assert s["rtf"] == pytest.approx(1.5 / 10.0)  # total latency / total audio, not mean of ratios


def test_empty_bucket_and_all():
    recs = [{"duration_s": 3.0, "latency_s": 0.2}, {"duration_s": 40.0, "latency_s": 2.0}]
    s = summarize_latency(recs)
    assert s["5-15 s"] == {"n": 0, "p50_s": None, "p95_s": None, "mean_s": None, "rtf": None}
    assert s[">30 s"]["n"] == 1
    assert s["all"]["n"] == 2
    assert s["all"]["rtf"] == pytest.approx(2.2 / 43.0, abs=1e-4)  # values are rounded to 4 decimals
